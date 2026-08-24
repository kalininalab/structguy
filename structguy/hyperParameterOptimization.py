import math
import random
import time
import sys
import traceback
import warnings
import ray
import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import MinMaxScaler

from structguy import util, trainForest
from structguy.sampleSpace import DataSAIL_cv, SampleSpace
from structman.base_utils.base_utils import deep_get_size_of, sizeof_fmt

from ray.util.queue import Queue

class Parameter:
    """A single hyperparameter and the space the optimizer searches it in.

    `log_scale` makes the optimizer work on log10(value + log_offset) instead of the raw
    value. Multiplicative parameters such as the learning rate, the number of trees or
    the regularization terms are only meaningful on a ratio scale: sampling uniformly in
    [10, 10000] puts 99% of the draws above 100 trees, and uniformly in [0, 2] puts 95%
    of the learning rate draws above 0.1. Searching in log space spreads the draws over
    the orders of magnitude that actually differ, and gives the gaussian process a
    smoother function to model. `log_offset` shifts parameters whose lower bound is 0 so
    that the logarithm stays finite.

    The conversion is confined to setValue/getValue, so everything downstream (bounds,
    the MinMaxScaler, the GP, the acquisition function) keeps working on search space
    values while `config` always holds the real value.
    """

    def __init__(self, name, param_type, half_step_limits=None, possible_values=None, regression_specific=False, classification_specific=False, transform_limits=None, log_scale=False, log_offset=0.0):
        self.name = name
        self.half_step_limits = half_step_limits
        self.possible_values = possible_values
        self.param_type = param_type
        self.regression_specific = regression_specific
        self.classification_specific = classification_specific
        self.log_scale = log_scale
        self.log_offset = log_offset

        if self.half_step_limits is None:
            self.half_step_limits = [0, (len(self.possible_values) - 1)]
        elif transform_limits is not None:
            transform_function, additional_args = transform_limits
            self.half_step_limits = transform_function(half_step_limits, additional_args)

        if self.log_scale:
            self.half_step_limits = [self.to_search_space(limit) for limit in self.half_step_limits]

    def to_search_space(self, value):
        if not self.log_scale:
            return value
        return math.log10(max(float(value) + self.log_offset, 1e-12))

    def from_search_space(self, value):
        if not self.log_scale:
            return value
        return (10.0 ** float(value)) - self.log_offset

    def setValue(self, config, val):
        if self.param_type == "categorical" and not isinstance(val, str):
            val = self.possible_values[round(val)]
        elif self.log_scale:
            val = self.from_search_space(val)
        config.setByString(self.name, val)

    def getValue(self, config):
        if self.param_type == "categorical":
            category = config.getByString(self.name)
            val = None
            for pos, cat in enumerate(self.possible_values):
                if cat == category:
                    val = pos
            if val is None:
                # The configured category is not in possible_values; fall back to the
                # first one instead of raising an UnboundLocalError further down.
                val = 0
        else:
            val = config.getByString(self.name)
            if self.log_scale:
                val = self.to_search_space(val)
        return val


class HpoTrace:
    """Per-subspace diagnostics for one bayesian_optimisation call.

    The log used to record only the resulting objective score, which makes it impossible
    to tell afterwards whether a candidate came from the gaussian process or from the
    uniform fallback, which hyperparameter vector produced it, whether the surrogate had
    any predictive power, and whether an accepted improvement was larger than the
    evaluation noise. This collects exactly that and prints a summary at the end.
    """

    def __init__(self, config, param_names, n_iters):
        self.config = config
        self.param_names = param_names
        self.n_iters = n_iters
        self.t0 = time.time()
        self.provenance = {}          # sample key -> (source, predicted mu, predicted sigma)
        self.scores = []              # every finite candidate score
        self.by_source = {}           # source -> list of scores
        self.gp_pred_err = []         # |predicted - observed| for gp proposals
        self.accepts = []             # (score, margin over the previous incumbent)
        self.blocked = []             # (score, margin, required margin) rejected by the noise gate
        self.n_dispatched = 0
        self.repeat = int(getattr(config, "repeat_training", 1) or 1)
        self.repeat_stds = []         # direct noise measurements from repeated trainings
        self.gp_noise = None          # noise std the gaussian process learned

    @staticmethod
    def key(sample):
        return tuple(round(float(v), 12) for v in np.ravel(sample))

    def describe(self, sample):
        return ", ".join(f"{n}={float(v):.6g}" for n, v in zip(self.param_names, np.ravel(sample)))

    def record_dispatch(self, sample, source, mu=None, sigma=None):
        self.provenance[self.key(sample)] = (source, mu, sigma)
        self.n_dispatched += 1
        if self.config.verbosity >= 2:
            pred = "" if mu is None else f" | GP predicted {mu:.6f} +- {sigma:.6f}"
            self.config.logger.info(f"HPO dispatch #{self.n_dispatched} [{source}]{pred}: {self.describe(sample)}")

    def record_result(self, sample, cv_score, interrupted=False, scores=None):
        source, mu, sigma = self.provenance.pop(self.key(sample), ("unknown", None, None))
        usable = cv_score is not None and cv_score == cv_score

        # With repeat_training > 1 every evaluation is already a mean over that many
        # independent trainings, and mean_scores() records the spread across them in
        # mean_spear_repeat_std. That is a direct measurement of the evaluation noise, so
        # prefer it over anything inferred from the candidate scores. The standard error
        # of the reported mean is that spread divided by sqrt(repeat). Only valid for the
        # Mean Spearman objective and only when there really were several repeats: with
        # repeat == 1 the field holds the spread across CV folds, which is not noise.
        if usable and self.repeat > 1 and self.config.objective_function == "Mean Spearman":
            rep_std = getattr(scores, "mean_spear_repeat_std", None)
            if rep_std is not None and rep_std == rep_std and rep_std > 0:
                self.repeat_stds.append(rep_std / math.sqrt(self.repeat))
        if usable:
            self.scores.append(cv_score)
            self.by_source.setdefault(source, []).append(cv_score)
            if mu is not None:
                self.gp_pred_err.append(abs(mu - cv_score))
        if self.config.verbosity >= 1:
            if mu is None:
                pred = ""
            elif usable:
                pred = f" | GP predicted {mu:.6f} +- {sigma:.6f} (error {abs(mu - cv_score):.6f})"
            else:
                pred = f" | GP predicted {mu:.6f} +- {sigma:.6f}"
            self.config.logger.info(f"HPO result [{source}] score={cv_score} {interrupted=}{pred}")
        return source

    def record_accept(self, cv_score, previous_best):
        margin = None
        if previous_best is not None and cv_score is not None and cv_score == cv_score:
            margin = cv_score - previous_best
        self.accepts.append((cv_score, margin))
        spread = self.noise_hint()
        if margin is not None and spread is not None and margin < spread:
            self.config.logger.info(
                f"NOTE: accepted a new optimum on a margin of {margin:+.6f}, which is below the "
                f"observed candidate spread of {spread:.6f} in this subspace. This improvement is "
                f"not distinguishable from evaluation noise."
            )

    def accept_gate(self, cv_score, previous_best):
        """Decide whether an improvement is large enough to be believed.

        The objective is a noisy estimate, so `score > best` alone accepts pure luck.
        Adopting such a candidate moves the whole configuration onto a value that was
        never actually shown to be better, and because the incumbent then holds an
        inflated score nothing can beat it afterwards. Require the margin to exceed a
        multiple of the estimated evaluation noise instead.

        Returns (allowed, message). Falls through to the old behaviour when the gate is
        disabled or there is not enough data yet to estimate the noise.
        """
        factor = getattr(self.config, "hpo_noise_gate", 1.0)
        if not factor or factor <= 0:
            return True, None
        if previous_best is None or cv_score is None or cv_score != cv_score:
            return True, None
        noise = self.noise_hint()
        if noise is None:
            return True, None

        margin = cv_score - previous_best
        required = factor * noise
        if margin >= required:
            return True, f"margin {margin:+.6f} >= {factor:g} x noise {noise:.6f}"

        self.blocked.append((cv_score, margin, required))
        return False, (
            f"margin {margin:+.6f} is below the noise gate ({factor:g} x {noise:.6f} = {required:.6f}); "
            f"not adopting this candidate"
        )

    def noise_hint(self):
        """Estimated standard deviation of a single evaluation of the objective.

        Three sources, in decreasing order of trustworthiness:

        1. Repeated trainings of the *same* hyperparameters (repeat_training > 1). This
           is the only one that measures noise directly, because everything except the
           random seeds is held fixed.
        2. The noise level the gaussian process learned. The WhiteKernel term absorbs
           exactly the part of the observed scatter that the smooth Matern component
           cannot explain, which is the definition we want. It is fitted on normalized
           targets, so it has to be scaled back by the spread of the observations.
        3. The scatter of all candidates. A crude fallback: it mixes real hyperparameter
           effects into the estimate and therefore overestimates. That is the safe
           direction for a gate, because overestimating only makes acceptance stricter,
           whereas underestimating reproduces the noise chasing this gate exists to stop.

        Deliberately *not* the scatter of the top few scores: selecting the maximum of a
        noisy sample pulls the top order statistics together, so their spread understates
        the noise badly (on the goldstandard_strict run the top three scores had a spread
        of 0.00015 while the true evaluation noise was around 0.0011).

        Run with repeat_training > 1 to get source 1. It is the only one that measures
        the noise instead of inferring it, and it is what makes this gate trustworthy.
        """
        if self.repeat_stds:
            return sum(self.repeat_stds) / len(self.repeat_stds)

        if self.gp_noise is not None and self.gp_noise > 0:
            return self.gp_noise

        if len(self.scores) < 6:
            return None
        mean = sum(self.scores) / len(self.scores)
        return (sum((s - mean) ** 2 for s in self.scores) / (len(self.scores) - 1)) ** 0.5

    def noise_source(self):
        if self.repeat_stds:
            return f"repeated trainings (n={len(self.repeat_stds)}, repeat={self.repeat})"
        if self.gp_noise is not None and self.gp_noise > 0:
            return "gaussian process WhiteKernel"
        if len(self.scores) >= 6:
            return "scatter of all candidates (crude fallback, overestimates)"
        return "unavailable"

    def log_kernel(self, model, yp=None):
        try:
            kernel = model.kernel_
        except AttributeError:
            return
        if self.config.verbosity >= 2:
            self.config.logger.info(f"HPO fitted GP kernel: {kernel}")

        # Recover the learned noise level and undo the normalize_y scaling, so that it is
        # expressed in the same units as the objective.
        try:
            noise_level = kernel.k2.noise_level
            lower_bound = kernel.k2.noise_level_bounds[0]
        except (AttributeError, IndexError, TypeError):
            return
        if yp is None or len(yp) < 2:
            return

        # A noise level sitting on the optimizer's lower bound means the marginal
        # likelihood preferred to explain every observation as signal, which happens
        # whenever the points are still sparse relative to the dimensionality. That is
        # not a measurement of zero noise, it is a failure to identify the noise, and
        # using it would silently disable the acceptance gate.
        if noise_level <= lower_bound * 10.0:
            self.gp_noise = None
            if self.config.verbosity >= 2:
                self.config.logger.info(
                    f"HPO GP noise level {noise_level:.3g} is at its lower bound "
                    f"({lower_bound:.3g}); the noise is not identifiable from this data yet"
                )
            return

        y_std = float(np.std(np.asarray(yp, dtype=float)))
        if y_std > 0 and noise_level > 0:
            self.gp_noise = math.sqrt(noise_level) * y_std

    def summary(self, best_scores, new_optimimum):
        c = self.config
        c.logger.info("================ HPO subspace summary ================")
        c.logger.info(f"Parameters ({len(self.param_names)}): {self.param_names}")
        c.logger.info(f"Wall time: {(time.time() - self.t0) / 3600.0:.2f} h, evaluations: {len(self.scores)}, n_iters budget: {self.n_iters}")
        for source in sorted(self.by_source):
            vals = self.by_source[source]
            c.logger.info(
                f"  from {source:<20} n={len(vals):4d}  min/mean/max = "
                f"{min(vals):.6f} / {sum(vals)/len(vals):.6f} / {max(vals):.6f}"
            )
        if self.scores:
            best_cand = max(self.scores)
            incumbent = util.get_objective_score(c, best_scores, feature_penalty=c.feature_penalty)
            c.logger.info(f"Best candidate: {best_cand:.6f}   incumbent now: {incumbent}")
            if isinstance(incumbent, float) and incumbent == incumbent:
                c.logger.info(f"Best candidate - incumbent: {best_cand - incumbent:+.6f}")
        else:
            c.logger.info("No usable candidate scores were produced in this subspace")
        hint = self.noise_hint()
        if hint is not None:
            c.logger.info(f"Estimated evaluation noise: {hint:.6f}  [source: {self.noise_source()}]")
            gate = getattr(c, "hpo_noise_gate", 1.0)
            if gate and gate > 0:
                c.logger.info(f"Acceptance threshold was {gate:g} x noise = {gate * hint:.6f}")
        else:
            c.logger.info("Estimated evaluation noise: unavailable, the acceptance gate was inactive")
        if self.gp_pred_err:
            mean_err = sum(self.gp_pred_err) / len(self.gp_pred_err)
            c.logger.info(f"GP surrogate mean absolute prediction error: {mean_err:.6f} over {len(self.gp_pred_err)} gp proposals")
            if hint is not None:
                c.logger.info(
                    f"  -> surrogate error / noise = {mean_err / hint:.2f} "
                    f"(near 1 means the GP is as good as the data allows, much larger means it is not learning)"
                )
        if self.blocked:
            c.logger.info(f"Improvements rejected by the noise gate: {len(self.blocked)}")
            for score, margin, required in self.blocked:
                c.logger.info(f"  rejected {score:.6f} (margin {margin:+.6f}, needed {required:+.6f})")
        c.logger.info(f"Accepted improvements: {len(self.accepts)}")
        for score, margin in self.accepts:
            if margin is not None:
                c.logger.info(f"  accepted {score:.6f} (margin {margin:+.6f})")
            else:
                c.logger.info(f"  accepted {score}")
        c.logger.info(f"new_optimimum={new_optimimum}")
        c.logger.info("======================================================")


def draw_random_sample(config, bounds, scaler, center_values=None):
    """Draw a random hyperparameter vector for the non-GP part of the search.

    Drawing uniformly from the whole box is close to useless once the incumbent is
    already a good configuration: every coordinate is randomized at once, so essentially
    every draw is far worse than the incumbent and the gaussian process only ever learns
    what the bad regions look like. Instead most draws are taken from a gaussian ball
    around the incumbent (in the [0, 1] scaled space, so the width is relative to each
    parameter's own range), with a configurable fraction still taken from the whole box
    so that distant regions stay reachable.

    Returns the sample in search space, i.e. the same space as `bounds`.
    """
    explore_prob = getattr(config, "hpo_explore_prob", 0.25)
    local_sigma = getattr(config, "hpo_local_sigma", 0.15)

    if center_values is None or local_sigma <= 0 or np.random.random() < explore_prob:
        return np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0]), "uniform-explore"

    center = scaler.transform([np.asarray(center_values, dtype=float)])[0]
    local = np.clip(center + np.random.normal(0.0, local_sigma, bounds.shape[0]), 0.0, 1.0)
    return scaler.inverse_transform([local])[0], "local-around-incumbent"


def fit_gp(model, scaled_xp, yp):
    """Fit the gaussian process, silencing the hyperparameter bound warnings.

    With an ARD kernel a length scale that runs into its upper bound is the expected
    result for a hyperparameter the objective does not depend on, and a noise level at
    its lower bound is the expected result while there are still too few observations to
    identify the noise. Both raise a ConvergenceWarning per dimension per fit, which
    would bury the actual optimization log.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        model.fit(scaled_xp, yp)
    return model


# Taken from https://github.com/thuijskens/bayesian-optimization
def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """expected_improvement
    Expected improvement acquisition function.
    Arguments:
    ----------
        x: array-like, shape = [n_samples, n_hyperparams]
            The point for which the expected improvement needs to be computed.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: Numpy array.
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        n_params: int.
            Dimension of the hyperparameter space.
    """

    x_to_predict = x.reshape(-1, n_params)

    mu, sigma = gaussian_process.predict(x_to_predict, return_std=True)

    if greater_is_better:
        loss_optimum = np.max(evaluated_loss)
    else:
        loss_optimum = np.min(evaluated_loss)

    scaling_factor = (-1) ** (not greater_is_better)

    # In case sigma equals zero
    with np.errstate(divide="ignore"):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] = 0.0

    return -1 * expected_improvement


# Taken from https://github.com/thuijskens/bayesian-optimization
def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False, bounds=(0, 10), n_restarts=25) -> None | np.ndarray:
    """sample_next_hyperparameter
    Proposes the next hyperparameter to sample the loss function for.
    Arguments:
    ----------
        acquisition_func: function.
            Acquisition function to optimise.
        gaussian_process: GaussianProcessRegressor object.
            Gaussian process trained on previously evaluated hyperparameters.
        evaluated_loss: array-like, shape = [n_obs,]
            Numpy array that contains the values off the loss function for the previously
            evaluated hyperparameters.
        greater_is_better: Boolean.
            Boolean flag that indicates whether the loss function is to be maximised or minimised.
        bounds: Tuple.
            Bounds for the L-BFGS optimiser.
        n_restarts: integer.
            Number of times to run the minimiser with different starting points.
    """
    best_x: np.ndarray | None = None
    best_acquisition_value = np.inf
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
        x0 = starting_point.reshape(1, -1)[0]
        # config.logger.info(f'In sample_next_hyperparameter, x0: {x0}, bounds: {bounds}')
        res = minimize(fun=acquisition_func, x0=x0, bounds=bounds, method="L-BFGS-B", args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        fun_value = float(np.ravel(res.fun)[0])
        if fun_value < best_acquisition_value:
            best_acquisition_value = fun_value
            best_x = res.x

    if best_x is None:
        # No restart produced a usable acquisition value. Fall back to a uniform draw
        # inside the search space so the caller never has to handle a None sample.
        best_x = np.random.uniform(bounds[:, 0], bounds[:, 1], n_params)

    return best_x


def fill_plane(parameter_values, integer_type_params, density=20):
    if len(integer_type_params) == 0:
        return []

    projected_parameter_values = [[]]

    for pos, parameter_value in enumerate(parameter_values):
        if pos not in integer_type_params:
            projections = [parameter_value]
        else:
            l = int(parameter_value)
            projections = []
            for i in range(density + 1):
                projections.append(l + (i / density))

        new_projected_parameter_values = []
        for projected_value in projections:
            for current_parameter_values in projected_parameter_values:
                new_projected_parameter_values.append(current_parameter_values + [projected_value])
        projected_parameter_values = new_projected_parameter_values

    return projected_parameter_values


def init_para_eval_store(
        config: util.Config,
        cv_obj: DataSAIL_cv,
        parameters: list[Parameter],
        raw_feature_matrix_store_id: ray.ObjectRef | None,
        best_first_scores: util.Scores,
        samples: SampleSpace,
    ):

    store = ray.put((cv_obj, parameters, raw_feature_matrix_store_id, best_first_scores, samples.feat_corr_matrix, samples.feature_names))

    config_ref_container = [ray.put(config)]

    return store, config_ref_container


def bayes_random_init(
    config: util.Config,
    parameters: list[Parameter],
    initial_cv_obj: DataSAIL_cv,
    best_scores: util.Scores,
    best_first_scores: util.Scores,
    n_pre_samples: int,
    distance_map,
    samples: SampleSpace | None =None,
    samples_store_id: ray.ObjectRef | None =None,
    raw_feature_matrix_store_id: ray.ObjectRef | None =None,
    fix_cat=True,
    debug=False,
    store_params = False
):
    param_names = [p.name for p in parameters]

    bounds = np.array([p.half_step_limits for p in parameters])

    initial_values = [p.getValue(config) for p in parameters]

    integer_type_params = set()
    fix_parameters_pos = []

    for parameter_number, parameter in enumerate(parameters):
        if parameter.param_type == "integer":
            integer_type_params.add(parameter_number)

    best_params = initial_values

    x_list = [np.array(initial_values)]
    y_list = [best_scores.objective_value(config)]

    n_params = len(param_names)

    new_optimimum = False

    para_eval_ret_ids = []

    if n_pre_samples is None:
        n_pre_samples = min([config.proc_n, (2*n_params) + 1])

    if config.multi_gpu > 1:
        n_pre_samples = max([n_pre_samples, config.multi_gpu])

    if debug:
        n_pre_samples = 2  # Just for testing

    config.logger.info(f"bayesian optimization: {param_names}, {n_pre_samples=}")
    config.logger.info("Current best scores:")
    best_scores.printOut(config=config)
    config.logger.info("Current best first scores:")
    best_first_scores.printOut(config=config)
    config.logger.info(f"Objective score: {best_scores.objective_value(config)}")

    if config.multi_gpu > 0:
        store = init_para_eval_store(config, initial_cv_obj, parameters, raw_feature_matrix_store_id, best_first_scores, samples)
        return x_list, y_list, bounds, n_params, best_scores, best_first_scores, best_params, initial_values, new_optimimum, len(fix_parameters_pos), param_names, integer_type_params, initial_cv_obj, store
    else:
        para_random_init = False


    t0 = time.time()

    # if not 'geometric_exponent' in param_names:
    if para_random_init:
        results = []
        randomized_parameters = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0]))

        # Always calculate one set of parameters unparalized to set the confusion maps (or other slice specific stuff that resets after each round of the HPO)
        init_params = randomized_parameters[0]
        for pos, para_value in enumerate(init_params):
            parameters[pos].setValue(config, para_value)

        if debug:
            config.logger.info(f"Init params: {init_params}")

        store, config_ref_container = init_para_eval_store(config, initial_cv_obj, parameters, raw_feature_matrix_store_id, best_first_scores, samples)

        config.logger.info(f"after store init, samples is None: {samples is None}")

        para_number = config.proc_n // config.multi_gpu

        current_params_id = 0
        remote_processes = []
        for proc_id in range(config.multi_gpu):
            com_queue = Queue()
            out_queue = Queue()
            com_queue.put((randomized_parameters[current_params_id]))
            current_params_id += 1
            proc_id = para_eval.remote(com_queue, out_queue, store, para_number, 1.0, proc_id, config_ref_container)
            remote_processes.append((com_queue, out_queue, proc_id))

        config.logger.info(f"Para random init started: # of packages: {len(para_eval_ret_ids)} # of subthreads: {para_number} {len(remote_processes)=}")

        dones = []
        for i in range(len(remote_processes)):
            dones.append(False)
        all_done = False
        while not all_done:
            for i, (com_queue, out_queue, proc_id) in enumerate(remote_processes):
                if dones[i]:
                    continue
                if not out_queue.empty():
                    (scores, params, first_scores) = out_queue.get(timeout=5)
                else:
                    continue
                results.append((scores, first_scores, params))

                if current_params_id >= len(randomized_parameters):
                    com_queue.put(None)
                    dones[i] = True
                else:
                    com_queue.put(randomized_parameters[current_params_id])
                    current_params_id += 1
            all_done = True
            for done in dones:
                if not done:
                    all_done = False
            if not all_done:
                time.sleep(0.5)

    else:
        store = None
        results = []
        try:
            cv_obj = initial_cv_obj
            init_scaler = MinMaxScaler().fit(np.array([bounds[:, 0], bounds[:, 1]]))
            pre_samples = [
                draw_random_sample(config, bounds, init_scaler, initial_values)[0]
                for _ in range(n_pre_samples)
            ]
            for params in pre_samples:
                for pos, para_value in enumerate(params):
                    parameters[pos].setValue(config, para_value)
                if config.multi_gpu > 0:
                    gpu_share = config.multi_gpu
                else:
                    gpu_share = None
                scores, first_scores = get_scores(
                    config,
                    cv_obj,
                    samples.feat_corr_matrix,
                    samples.feature_names,
                    raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                    get_first_scores=True,
                    cv_interuption=(0.95, best_first_scores),
                    gpu_share=gpu_share
                    )
                results.append((scores, first_scores, params))
                # projected_parameter_values = fill_plane(params, integer_type_params)
                # for projected_param in projected_parameter_values:
                #    results.append((scores, projected_param))
        except:
            [e, f, g] = sys.exc_info()
            g = traceback.format_exc()
            config.logger.error(f"ERROR in bayes_random_init: {n_pre_samples}, {bounds}\n{e}\n{f}\n{g}")
            sys.exit()

    t1 = time.time()
    config.logger.info(f"Time for BayesInit: {t1 - t0}")

    cat_param_map = {}

    for scores, first_scores, params in results:
        obj_sc = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)
        if obj_sc is None or obj_sc != obj_sc:
            config.logger.info(f"========= Warning: None or NaN objective score for: {param_names}, {params}")
            scores = util.Scores(zero=True)
            obj_sc = scores.objective_value(config)

        x_list.append(params)
        y_list.append(obj_sc)

        if config.verbosity >= 3:
            config.logger.info(f"Objective score: {obj_sc}")

        if util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty):
            best_scores = scores
            best_first_scores = first_scores
            best_params = params
            new_optimimum = True
            
            config.logger.info("===========Found new optimum:=======================\n")
            config.logger.info(f'{params}')
            scores.printOut(config=config)
            config.logger.info("====================================================")
            if store_params:
                config.saveHyperParameter("hyperparameters_endless_HPO.conf")
        elif config.verbosity >= 2:
            config.logger.info(
                f"No new optimun ({util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)}): {util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)}"
            )

        if fix_cat:
            for p_pos, parameter in enumerate(parameters):
                if not parameter.param_type == "categorical":
                    continue
                if parameter.name not in cat_param_map:
                    cat_param_map[parameter.name] = {}
                val = round(params[p_pos])
                if val not in cat_param_map[parameter.name]:
                    cat_param_map[parameter.name][val] = []
                cat_param_map[parameter.name][val].append(obj_sc)

    config.logger.info("Random init finished")
    
    if fix_cat:
        for p_pos, parameter in enumerate(parameters):
            if not parameter.param_type == "categorical":
                continue
            max_max_val = None
            max_median_val = None
            for val in cat_param_map[parameter.name]:
                max_sc = max(cat_param_map[parameter.name][val])
                median_sc = util.median(cat_param_map[parameter.name][val])
                if max_max_val is None:
                    max_max_val = val
                    max_median_val = val
                    max_max_sc = max_sc
                    max_median_sc = median_sc
                else:
                    if max_sc > max_max_sc:
                        max_max_val = val
                        max_max_sc = max_sc
                    if median_sc > max_median_sc:
                        max_median_val = val
                        max_median_sc = median_sc
                if max_max_val == max_median_val:
                    fix_parameters_pos.append(p_pos)
                    parameter.setValue(config, max_max_val)
                    config.logger.info(f"Fix categorical feature {param_names[p_pos]} to {config.getByString(parameter.name)}")

        for p_pos in reversed(fix_parameters_pos):
            del parameters[p_pos]
    return x_list, y_list, bounds, n_params, best_scores, best_first_scores, best_params, initial_values, new_optimimum, len(fix_parameters_pos), param_names, integer_type_params, initial_cv_obj, store


# Taken from https://github.com/thuijskens/bayesian-optimization
# Changed to match the specific problem
def bayesian_optimisation(
    n_iters: int,
    config: util.Config,
    parameters: list[Parameter],
    cv_obj: DataSAIL_cv,
    best_scores: util.Scores,
    best_first_scores: util.Scores,
    distance_map,
    samples: SampleSpace | None =None,
    samples_store_id: ray.ObjectRef | None =None,
    raw_feature_matrix_store_id: ray.ObjectRef | None =None,
    n_pre_samples: int =5,
    gp_params=None,
    random_search=False,
    alpha=1e-6,
    epsilon=1e-9,
    debug=False,
    store_params = False,
    get_bayes_tuple = False,
    bayes_tuple = None
):
    """bayesian_optimisation
    Uses Gaussian Processes to optimise the loss function `sample_loss`.
    Arguments:
    ----------
        n_iters: integer.
            Number of iterations to run the search algorithm.
        sample_loss: function.
            Function to be optimised.
        bounds: array-like, shape = [n_params, 2].
            Lower and upper bounds on the parameters of the function `sample_loss`.
        x0: array-like, shape = [n_pre_samples, n_params].
            Array of initial points to sample the loss function for. If None, randomly
            samples from the loss function.
        n_pre_samples: integer.
            If x0 is None, samples `n_pre_samples` initial points from the loss function.
        gp_params: dictionary.
            Dictionary of parameters to pass on to the underlying Gaussian Process.
        random_search: integer.
            Flag that indicates whether to perform random search or L-BFGS-B optimisation
            over the acquisition function.
        alpha: double.
            Numerical jitter added to the diagonal of the kernel matrix. The actual
            observation noise is learned from the data by the WhiteKernel component,
            so this only needs to keep the Cholesky decomposition well conditioned.
        epsilon: double.
            Precision tolerance for floats.
    """

    n_fixed_params = 1

    x_list: list[np.ndarray]

    # while n_fixed_params > 0:
    
    x_list, y_list, bounds, n_params, best_scores, best_first_scores, best_params, initial_values, new_optimimum, n_fixed_params, param_names, integer_type_params, cv_obj, stores = bayes_random_init(
        config,
        parameters,
        cv_obj,
        best_scores,
        best_first_scores,
        n_pre_samples,
        distance_map,
        samples=samples,
        samples_store_id=samples_store_id,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        fix_cat=False,
        debug=debug,
        store_params = store_params
    )

    if bayes_tuple is not None:
        x_list, y_list = bayes_tuple
    
    
    min_max_samples = [[], []]
    for bound in bounds:
        min_max_samples[0].append(bound[0])
        min_max_samples[1].append(bound[1])

    scaled_bounds = np.array([[0.0, 1.0]] * len(bounds))

    if config.verbosity >= 2:
        config.logger.info(f"In bayesian_optimization {bounds=} {len(x_list)=}")

    scaler = MinMaxScaler()
    scaler.fit(min_max_samples)

    xp: np.ndarray = np.array(x_list)
    yp: np.ndarray = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        # The objective is a noisy estimate (each evaluation retrains the forests with a
        # fresh random_state), so the GP has to be allowed to model that noise. A plain
        # Matern() with alpha=1e-6 forces the GP to interpolate every observation, which
        # collapses sigma to ~0 around known points, degenerates the expected improvement
        # and pushes the sampler into the duplicate/random fallback.
        #
        # - ConstantKernel gives the covariance a learnable amplitude.
        # - Matern gets one length scale per dimension (ARD) so that irrelevant
        #   hyperparameters can be shrunk out instead of sharing a single length scale.
        # - WhiteKernel learns the evaluation noise from the data itself, so there is no
        #   need to measure and hard-code the variance.
        kernel = gp.kernels.ConstantKernel(1.0, (1e-3, 1e3)) * gp.kernels.Matern(
            length_scale=np.ones(n_params), length_scale_bounds=(1e-2, 1e2), nu=2.5
        ) + gp.kernels.WhiteKernel(noise_level=1e-2, noise_level_bounds=(1e-8, 1e0))
        model = gp.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=True)

    scaled_xp: np.ndarray = scaler.transform(xp)

    if n_iters is None:
        n_iters = 2 ** (n_params + 1)
    if debug:
        n_iters = 4

    if config.multi_gpu > 1:
        n_iters = max([2**(n_params-2), 8])

    return_cv_obj = cv_obj

    count_dups = 0

    trace = HpoTrace(config, param_names, n_iters)

    if config.verbosity >= 1:
        config.logger.info(f"Number of bayesian optimization iterations: {n_iters=}")

    if config.multi_gpu < 1:

        for n in range(n_iters):
            if config.verbosity >= 2:
                tl0 = time.time()
            try:
                fit_gp(model, scaled_xp, yp)
            except:
                config.logger.info(f"{xp=}, {yp=}")
                config.logger.info(f"{x_list=}, {y_list=}")
                raise "None in Input"

            trace.log_kernel(model, yp)

            if config.verbosity >= 2:
                tl1 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 1: {tl1 - tl0}")

            # Sample next hyperparameter
            if random_search:
                x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
                ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
                next_sample: np.ndarray = x_random[np.argmax(ei), :]
            else:
                next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=scaled_bounds, n_restarts=100)

            if config.verbosity >= 2:
                tl2 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 2: {tl2 - tl1}")

            # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
            if np.any(np.sum(np.abs(next_sample - scaled_xp), axis = 1) <= epsilon):
                if config.verbosity >= 2:
                    config.logger.info(f"Sampled a duplicate: {next_sample} {bounds}")
                next_sample, draw_kind = draw_random_sample(config, bounds, scaler, best_params)
                count_dups += 1
                trace.record_dispatch(next_sample, f"duplicate-fallback/{draw_kind}")
            else:
                gp_mu, gp_sigma = model.predict([next_sample], return_std=True)
                next_sample = scaler.inverse_transform([next_sample])[0]
                trace.record_dispatch(next_sample, "gaussian-process", mu=float(gp_mu[0]), sigma=float(gp_sigma[0]))

            if config.verbosity >= 1:
                config.logger.info(f"Try out next sampled HP: {next_sample}")

            if count_dups == 4:
                config.logger.info("Break bayesian optimization, due to double dups")
                break

            if config.verbosity >= 2:
                tl3 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 3: {tl3 - tl2}")

            # Sample loss for new set of parameters
            for pos, para_value in enumerate(next_sample):
                parameters[pos].setValue(config, para_value)

            if config.verbosity >= 2:
                tl4 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 4: {tl4 - tl3}")

            scores, first_scores = get_scores(
                config,
                cv_obj,
                samples.feat_corr_matrix,
                samples.feature_names,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                debug=debug,
                get_first_scores=True,
                cv_interuption=(0.95, best_first_scores)
            )

            if config.verbosity >= 2:
                tl5 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 5: {tl5 - tl4}")

            # When the cross validation was interrupted after the first fold, first_scores
            # is the estimated objective value (a float) and scores only holds the first
            # fold's result. Such a partial evaluation may be used to inform the GP, but it
            # must never become the new optimum, otherwise best_first_scores turns into a
            # float and the next cv_interuption comparison raises an AttributeError.
            interrupted = isinstance(first_scores, float)

            if interrupted:
                cv_score = first_scores
            else:
                cv_score = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)

            if config.verbosity >= 2:
                tl6 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 6: {tl6 - tl5}")

            if config.verbosity >= 1:
                config.logger.info(f"Bayesian optimization, iteration: {n}")
                config.logger.info(f"Objective score: {cv_score}, unpenalized: {scores.objective_value(config)} {interrupted=}")

            trace.record_result(next_sample, cv_score, interrupted=interrupted, scores=scores)

            previous_best = util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)
            if isinstance(previous_best, bool):
                previous_best = None

            accept = (not interrupted) and util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty)
            if accept:
                allowed, gate_message = trace.accept_gate(cv_score, previous_best)
                if not allowed:
                    accept = False
                    config.logger.info(f"Rejected candidate: {gate_message}")
                elif gate_message is not None and config.verbosity >= 2:
                    config.logger.info(f"Accepting candidate: {gate_message}")

            if accept:
                best_scores = scores
                best_first_scores = first_scores
                best_params = next_sample
                new_optimimum = True
                config.logger.info("===========================\nFound new optimum\n===\n")
                trace.record_accept(cv_score, previous_best)
                config.logger.info(f"Accepted hyperparameters: {trace.describe(best_params)}")
                config.logParameter()
                scores.printOut(config=config)
                config.logger.info("===========================")
                return_cv_obj = cv_obj
                if store_params:
                    config.saveHyperParameter("hyperparameters_endless_HPO.conf")
            elif config.verbosity >= 3:
                config.logger.info(
                    f"No new optimun ({util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)}): {util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)}"
                )

            if cv_score is None or cv_score != cv_score:
                config.logger.info(f" === cv_score is None or Nan: {next_sample}")
                scores = util.Scores(zero=True)
                cv_score = scores.objective_value(config)

            if config.verbosity >= 2:
                tl7 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 7: {tl7 - tl6}")

            # Update lists
            x_list.append(next_sample)
            y_list.append(cv_score)

            # projected_parameter_values = fill_plane(next_sample, integer_type_params)
            # for projected_param in projected_parameter_values:
            #    x_list.append(projected_param)
            #    y_list.append(cv_score)

            # Update xp and yp
            xp = np.array(x_list)
            yp = np.array(y_list)
            scaled_xp = scaler.transform(xp)

            if config.verbosity >= 3:
                tl8 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 8: {tl8 - tl7}")

    else:
        remote_processes = []
        threads_per_gpu = config.threads_per_gpu
        number_of_procs = max([1,round(config.multi_gpu * threads_per_gpu)])
        threads_per_gpu = number_of_procs / config.multi_gpu
        
        para_number = min([config.proc_n , max([1,config.proc_n // (config.multi_gpu * threads_per_gpu)])])
        
        current_params_id = 0
        n_of_sent_hpo_sets = 0

        gpu_share = 1/threads_per_gpu
        remote_function = para_eval #.options(num_gpus = gpu_share)
        
        store, config_ref_container = stores

        #for i in range(config.multi_gpu):
        #    pg = ray.util.placement_group([{"CPU": config.proc_n // (config.multi_gpu), 'GPU':1}], name=f"pg_{i}")
        #    ray.get(pg.ready())

        for p in range(number_of_procs):
            next_sample, draw_kind = draw_random_sample(config, bounds, scaler, best_params)
            trace.record_dispatch(next_sample, f"init/{draw_kind}")

            com_queue = Queue()
            out_queue = Queue()
            com_queue.put(next_sample)
            n_of_sent_hpo_sets += 1
            
            #gpu_id = p//config.threads_per_gpu
            #pg = ray.util.get_placement_group(f"pg_{gpu_id}")

            proc_id = remote_function.options(
                    num_cpus=0.5,
                    num_gpus=0,
                    #scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                    #    placement_group=pg, placement_group_capture_child_tasks=True
                    #)
                ).remote(com_queue, out_queue, store, para_number, gpu_share, p, config_ref_container)
            
            #proc_id = remote_function.remote(com_queue, out_queue, store, para_number, gpu_share, p, config_ref_container)
            
            remote_processes.append((com_queue, out_queue, proc_id))

        if config.verbosity >= 1:
            config.logger.info(f'Started {len(remote_processes)} para_eval processes: {para_number=} {gpu_share=}')

        dones = []
        counts = []
        for i in range(len(remote_processes)):
            dones.append(False)
            counts.append(0)
        all_done = False
        
        append_counter = 0
        overall_timeout_start = time.time()
        timeout_start = None

        message_counter = 0
        count_var = 0

        while not all_done:
            for i, (com_queue, out_queue, proc_id) in enumerate(remote_processes):
                
                if not out_queue.empty():
                    (scores, next_sample, first_scores) = out_queue.get(timeout=0.5)
                    n_of_sent_hpo_sets -= 1
                else:
                    continue
                """
                elif current_params_id >= n_iters:
                    if not dones[i]:
                        com_queue.put(None)
                        dones[i] = True
                    continue
                """

                interrupted = isinstance(first_scores, float)

                if interrupted:
                    cv_score = first_scores
                else:
                    cv_score = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)

                if config.verbosity >= 1:
                    config.logger.info(f"Bayesian optimization, iteration: gpu_id: {i} {counts[i]} {current_params_id=} {n_of_sent_hpo_sets=}")
                    counts[i] += 1
                    config.logger.info(f"Objective score: {cv_score}, unpenalized: {scores.objective_value(config)} {interrupted=}")

                trace.record_result(next_sample, cv_score, interrupted=interrupted, scores=scores)

                previous_best = util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)
                if isinstance(previous_best, bool):
                    previous_best = None

                accept = (not interrupted) and util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty)
                if accept:
                    allowed, gate_message = trace.accept_gate(cv_score, previous_best)
                    if not allowed:
                        accept = False
                        config.logger.info(f"Rejected candidate: {gate_message}")
                    elif gate_message is not None and config.verbosity >= 2:
                        config.logger.info(f"Accepting candidate: {gate_message}")

                if accept:
                    best_scores = scores
                    best_first_scores = first_scores
                    best_params = next_sample
                    new_optimimum = True
                    config.logger.info("===========================\nFound new optimum\n===\n")
                    trace.record_accept(cv_score, previous_best)
                    config.logger.info(f"Accepted hyperparameters: {trace.describe(best_params)}")
                    for pos, para_value in enumerate(best_params):
                        parameters[pos].setValue(config, para_value)
                    config.logParameter()
                    scores.printOut(config=config)
                    config.logger.info("===========================")
                    return_cv_obj = cv_obj
                    if store_params:
                        config.saveHyperParameter("hyperparameters_endless_HPO.conf")
                elif config.verbosity >= 4:
                    config.logger.info(
                        f"No new optimun ({util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)}): {util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)}"
                    )

                if cv_score is None or cv_score != cv_score:
                    config.logger.info(f" === cv_score is None or Nan: {next_sample}")
                    scores = util.Scores(zero=True)
                    cv_score = scores.objective_value(config)
                else:
                    timeout_start = time.time()

                # Update lists
                x_list.append(next_sample)
                y_list.append(cv_score)
                append_counter += 1

                # Update xp and yp
                xp = np.array(x_list)
                yp = np.array(y_list)
                scaled_xp = scaler.transform(xp)

                if current_params_id >= n_iters:
                    com_queue.put(None)
                    dones[i] = True
                else:
                    if append_counter >= 2:
                        try:
                            fit_gp(model, scaled_xp, yp)
                        except:
                            config.logger.info(f"{xp=}, {yp=}")
                            config.logger.info(f"{x_list=}, {y_list=}")
                            raise "None in Input"

                        trace.log_kernel(model, yp)

                        # Sample next hyperparameter
                        if random_search:
                            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
                            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
                            next_sample: np.ndarray = x_random[np.argmax(ei), :]
                        else:
                            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=scaled_bounds, n_restarts=100)

                        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
                        if np.any(np.sum(np.abs(next_sample - scaled_xp), axis = 1) <= epsilon):
                            if config.verbosity >= 2:
                                config.logger.info(f"Sampled a duplicate: {next_sample} {bounds}")
                            next_sample, draw_kind = draw_random_sample(config, bounds, scaler, best_params)
                            count_dups += 1
                            trace.record_dispatch(next_sample, f"duplicate-fallback/{draw_kind}")
                        else:
                            gp_mu, gp_sigma = model.predict([next_sample], return_std=True)
                            next_sample = scaler.inverse_transform([next_sample])[0]
                            trace.record_dispatch(next_sample, "gaussian-process", mu=float(gp_mu[0]), sigma=float(gp_sigma[0]))
                        overall_timeout_start = time.time()
                        current_params_id += 1
                        append_counter = 0
                    else:
                        next_sample, draw_kind = draw_random_sample(config, bounds, scaler, best_params)
                        trace.record_dispatch(next_sample, f"fill/{draw_kind}")
                    com_queue.put(next_sample)
                    n_of_sent_hpo_sets += 1
                    
            all_done = True
            for done in dones:
                if not done:
                    all_done = False
            if n_of_sent_hpo_sets > 0 and not all_done:
                if timeout_start is not None:
                    timeout = time.time() - timeout_start
                    if timeout > 10*3600:
                        all_done = True
                    else:
                        all_done = False
                if overall_timeout_start is not None and not all_done:
                    overall_timeout = time.time() - overall_timeout_start
                    if overall_timeout > 2*36_000:
                        all_done = True
                    else:
                        all_done = False

            if not all_done:
                if message_counter < 10000:
                    message_counter += 1
                else:
                    message_counter = 0
                    count_var += 1
                    if timeout_start is not None:
                        timeout = time.time() - timeout_start
                    else:
                        timeout = None
                    if overall_timeout_start is not None and not all_done:
                        overall_timeout = time.time() - overall_timeout_start
                    else:
                        overall_timeout = None
                    config.logger.info(f"Not all done {count_var}: {timeout=} {timeout_start=} {overall_timeout=} {overall_timeout_start=} {dones=}")

                time.sleep(0.5)

        for com_queue, out_queue, proc_id in remote_processes:
            try:
                ray.cancel(proc_id)
            except (TypeError, ray.exceptions.RayError) as e:
                if config.verbosity >= 2:
                    config.logger.info(f"Could not cancel para_eval process: {e}")
                continue

        """
        for i in range(config.multi_gpu):
            pg = ray.util.get_placement_group(f"pg_{i}")
            
            ray.util.remove_placement_group(pg)
        """
            
    if new_optimimum:
        for pos, para_value in enumerate(best_params):
            parameters[pos].setValue(config, para_value)
            config.logger.info(f"===========Found new optimum by setting {param_names[pos]}, to {para_value} ==============")
    else:
        for pos, para_value in enumerate(initial_values):
            parameters[pos].setValue(config, para_value)
        config.logger.info("No new optimum in this subspace, reverted to the incumbent values")

    trace.summary(best_scores, new_optimimum)

    if get_bayes_tuple:
        return new_optimimum, best_scores, best_first_scores, return_cv_obj, x_list, y_list

    return new_optimimum, best_scores, best_first_scores, return_cv_obj


def max_to_n_of_features(limits, n_of_features):
    if limits[1] != "max":
        return limits
    else:
        return [limits[0], n_of_features]


def twoDim(
        parameter_1,
        parameter_2,
        best_scores,
        first_scores,
        config,
        score_matrix,
        cv_obj,
        distance_map,
        samples=None,
        samples_store_id=None,
        raw_feature_matrix_store_id=None,
        debug=False):
    cat_count = 0
    for p_type in [parameter_1.param_type, parameter_2.param_type]:
        if p_type == "categorical":
            cat_count += 1

    if cat_count == 2:
        return  # TODO when we have at least 2 categorical features

    if cat_count == 1:
        if parameter_1.param_type == "categorical":
            return bayesianAndCat(
                parameter_1,
                [parameter_2],
                best_scores,
                first_scores,
                config,
                score_matrix,
                cv_obj,
                distance_map,
                samples=samples,
                samples_store_id=samples_store_id,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                debug=debug)
        elif parameter_2.param_type == "categorical":
            return bayesianAndCat(
                parameter_2,
                [parameter_1],
                best_scores,
                first_scores,
                config,
                score_matrix,
                cv_obj,
                distance_map,
                samples=samples,
                samples_store_id=samples_store_id,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                debug=debug)

    return bayesian_optimisation(
        None,
        config,
        [parameter_1, parameter_2],
        cv_obj,
        best_scores,
        first_scores,
        distance_map,
        n_pre_samples=None,
        samples=samples,
        samples_store_id=samples_store_id,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        debug=debug
    )


def threeDim(
        parameter_1,
        parameter_2,
        parameter_3,
        best_scores,
        first_scores,
        config,
        score_matrix,
        cv_obj,
        distance_map,
        samples=None,
        samples_store_id=None,
        raw_feature_matrix_store_id=None,
        debug=False
        ):
    cat_count = 0
    for p_type in [parameter_1.param_type, parameter_2.param_type, parameter_3.param_type]:
        if p_type == "categorical":
            cat_count += 1

    if cat_count == 3:
        return cat3D(
            parameter_1,
            parameter_2,
            parameter_3,
            best_scores,
            first_scores,
            config,
            score_matrix,
            cv_obj,
            distance_map,
            samples=samples,
            samples_store_id=samples_store_id,
            raw_feature_matrix_store_id=raw_feature_matrix_store_id
        )

    if cat_count == 2:
        return  # TODO when we have at least 2 categorical features

    if cat_count == 1:
        if parameter_1.param_type == "categorical":
            return bayesianAndCat(
                parameter_1,
                [parameter_2, parameter_3],
                best_scores,
                first_scores,
                config,
                score_matrix,
                cv_obj,
                distance_map,
                samples=samples,
                samples_store_id=samples_store_id,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                debug=debug,
            )
        elif parameter_2.param_type == "categorical":
            return bayesianAndCat(
                parameter_2,
                [parameter_1, parameter_3],
                best_scores,
                first_scores,
                config,
                score_matrix,
                cv_obj,
                distance_map,
                samples=samples,
                samples_store_id=samples_store_id,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                debug=debug,
            )
        elif parameter_3.param_type == "categorical":
            return bayesianAndCat(
                parameter_3,
                [parameter_1, parameter_2],
                best_scores,
                first_scores,
                config,
                score_matrix,
                cv_obj,
                distance_map,
                samples=samples,
                samples_store_id=samples_store_id,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                debug=debug,
            )

    return bayesian_optimisation(
        None,
        config,
        [parameter_1, parameter_2, parameter_3],
        cv_obj,
        best_scores,
        first_scores,
        distance_map,
        n_pre_samples=None,
        samples=samples,
        samples_store_id=samples_store_id,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        debug=debug,
    )


def bayesianAndCat(
        parameter_1,
        parameters,
        best_scores,
        first_scores,
        config,
        score_matrix,
        cv_obj,
        distance_map,
        samples=None,
        debug=False,
        samples_store_id=None,
        raw_feature_matrix_store_id=None,
        ):
    config.logger.info(f"bayesian optimization and Cat {parameter_1.name}")

    new_optimimum = False
    best_parameter_value_1 = parameter_1.getValue(config)
    best_parameter_values = []
    for param in parameters:
        config.logger.info(param.name)
        best_parameter_values.append(param.getValue(config))

    for parameter_value_1 in parameter_1.possible_values:
        parameter_1.setValue(config, parameter_value_1)

        bay_optimimum, scores, fscores, cv_obj = bayesian_optimisation(
            None,
            config,
            parameters,
            cv_obj,
            best_scores,
            first_scores,
            distance_map,
            n_pre_samples=None,
            samples=samples,
            samples_store_id=samples_store_id,
            raw_feature_matrix_store_id=raw_feature_matrix_store_id,
            debug=debug,
        )

        if util.objective_function_criterium(config, scores, best_scores):
            best_scores = scores
            first_scores = fscores
            best_parameter_value_1 = parameter_1.getValue(config)
            best_parameter_values = []
            for param in parameters:
                best_parameter_values.append(param.getValue(config))
            new_optimimum = True

    parameter_1.setValue(config, best_parameter_value_1)
    for i, para_value in enumerate(best_parameter_values):
        parameters[i].setValue(config, para_value)

    return new_optimimum, best_scores, first_scores, cv_obj


def cat3D(parameter_1, parameter_2, parameter_3, best_scores, config, score_matrix, cv_obj, distance_map, samples=None, samples_store_id=None, raw_feature_matrix_store_id=None):
    # Todo when we get at least 3 categorical features
    return


def get_scores(
        config: util.Config,
        cv_obj: DataSAIL_cv | dict[int, ray.ObjectRef],
        feat_corr_matrix,
        feature_names,
        raw_feature_matrix_store_id=None,
        remote=True,
        para_number=None,
        debug=False,
        get_first_scores=False,
        cv_interuption=None,
        gpu_share = None,
        proc_id = 0,
        config_ref_container=None
        ):
    
    if config.verbosity >= 2:
        config.logger.info(f"Call of get_scores: {gpu_share=} {para_number=}")
    t0 = time.time()
    
    _, scores = trainForest.trainForest(
        config,
        cv_obj,
        feat_corr_matrix,
        feature_names,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        repeat=config.repeat_training,
        cv_repeat=config.cv_hpo,
        remote=remote,
        para_number=para_number,
        debug=debug,
        get_first_scores=get_first_scores,
        cv_interuption=cv_interuption,
        gpu_share=gpu_share,
        proc_id=proc_id,
        config_ref_container=config_ref_container
    )
    t1 = time.time()
    config.logger.info(f"Time for training forest in get_scores: {t1 - t0}")
    
    if get_first_scores:
        first_scores, scores = scores
        return scores, first_scores
    return scores


@ray.remote
def para_eval(com_queue: Queue, out_queue: Queue, store, para_number, gpu_share, proc_id, config_ref_container):
    (cv_obj, parameters, raw_feature_matrix_store_id, best_first_scores, feat_corr_matrix, feature_names) = store
    config = ray.get(config_ref_container[0])

    util.reset_logger_for_remotes(config)
    if config.verbosity >= 2:
        config.logger.info(f"Call of para_eval: {com_queue.empty()=} {proc_id=}")
      
    if config.verbosity >= 4:
        for name, size in sorted(((name, deep_get_size_of(value)) for name, value in locals().items()), key=lambda x: -x[1])[:10]:
            config.logger.info("In para_eval: {:>30}: {:>8}".format(name, sizeof_fmt(size)))


    not_done = True
    while not_done:
        if com_queue.empty():
            time.sleep(0.5)
            continue
        params = com_queue.get()
        if params is None:
            return

        for pos, para_value in enumerate(params):
            parameters[pos].setValue(config, para_value)
        scores, first_scores = get_scores(
            config,
            cv_obj,
            feat_corr_matrix,
            feature_names,
            raw_feature_matrix_store_id=raw_feature_matrix_store_id,
            remote=True,
            para_number=para_number,
            get_first_scores=True,
            cv_interuption=(0.95, best_first_scores),
            gpu_share=gpu_share, proc_id=proc_id,
            config_ref_container=config_ref_container
            )
        
        
        out_queue.put((scores, params, first_scores))
    return


def initConfParameters(config, parameters, thresh_only = False):
    if config.forest_type == "xgboost":
        return parameters
    parameters["confusion_rank_threshold"] = Parameter(
        "confusion_rank_threshold", "integer", half_step_limits=config.confusion_rank_threshold_bounds, transform_limits=(max_to_n_of_features, config.n_of_features)
    )
    if thresh_only:
        return parameters
    parameters["confusion_goodwill"] = Parameter("confusion_goodwill", "real", half_step_limits=config.confusion_goodwill_bounds)
    parameters["err_warping_exp"] = Parameter("err_warping_exp", "real", half_step_limits=config.err_warping_exp_bounds)
    parameters["confusion_normalization_exp"] = Parameter("confusion_normalization_exp", "real", half_step_limits=config.confusion_normalization_exp_bounds)

    return parameters


def initFSForestParameters(config, parameters):
    if config.forest_type == "xgboost":
        return parameters
    # parameters['fs_tree_depth'] = Parameter('fs_tree_depth','integer',half_step_limits = config.tree_depth_half_step)
    # parameters['fs_num_of_trees'] = Parameter('fs_num_of_trees','integer',half_step_limits = config.forest_size_half_step)
    # parameters['fs_min_impurity_decrease_exp'] = Parameter('fs_min_impurity_decrease_exp','real',half_step_limits = config.min_impurity_decrease_exp_half_step)
    parameters["fs_min_sample_split"] = Parameter("fs_min_sample_split", "integer", half_step_limits=config.min_sample_split_half_step)
    parameters["fs_tree_min_leaf_samples"] = Parameter("fs_tree_min_leaf_samples", "integer", half_step_limits=config.min_sample_leaf_half_step)
    parameters["fs_max_sample_parameter"] = Parameter("fs_max_sample_parameter", "real", half_step_limits=config.max_sample_half_step)

    return parameters


def initReguParameters(config, parameters):
    # parameters['reg_thresh_exp'] = Parameter('reg_thresh_exp','real',half_step_limits = config.reg_thresh_exp_half_step)
    if config.regression:
        parameters["reg_alpha_exp"] = Parameter("reg_alpha_exp", "real", half_step_limits=config.reg_alpha_exp_half_step)
    else:
        parameters["reg_c_exp"] = Parameter("reg_c_exp", "real", half_step_limits=config.reg_c_exp_half_step)
    return parameters


def initMeanCorrParameters(config, parameters):
    parameters["tvmb_rank_threshold"] = Parameter("tvmb_rank_threshold", "integer", half_step_limits=config.tvmb_rank_half_step, transform_limits=(max_to_n_of_features, config.n_of_features))
    # parameters['tvpmb_rank_threshold'] = Parameter('tvpmb_rank_threshold','integer',half_step_limits = config.tvpmb_rank_half_step, transform_limits = (max_to_n_of_features, config.n_of_features))
    parameters["p_val_thresh"] = Parameter("p_val_thresh", "real", half_step_limits=config.p_val_thresh_half_step, transform_limits=(max_to_n_of_features, config.n_of_features))
    return parameters


def initParameters(
        config: util.Config,
        do_feat_selection=True,
        do_forest_param=True,
        do_sample_weighting=True,
        split_fs_parameters=False):
    
    parameters = {}
    fs_parameters = {}
    fss_parameters = {}
    if config.hpo_do_feat_selection and do_feat_selection:
        if (
            config.feature_selection == "confusion"
            or config.feature_selection == "sequential_confusion"
            or config.feature_selection == "sequential_confusion_and_regu"
            or config.feature_selection == "confusion_and_regu"
        ):
            parameters['corr_thresh'] = Parameter('corr_thresh', 'real', half_step_limits=config.corr_thresh_bounds)
            fs_parameters = initConfParameters(config, fs_parameters)
            fss_parameters = initFSForestParameters(config, fss_parameters)

            if config.feature_selection == "sequential_confusion" or config.feature_selection == "sequential_confusion_and_regu":
                fs_parameters["sequential_confusion_rank_threshold"] = Parameter(
                    "sequential_confusion_rank_threshold", "integer", half_step_limits=config.sequential_confusion_rank_threshold_bounds, transform_limits=(max_to_n_of_features, config.n_of_features)
                )
            if config.feature_selection == "sequential_confusion_and_regu" or config.feature_selection == "confusion_and_regu":
                fs_parameters = initReguParameters(config, fs_parameters)

        elif config.feature_selection == "threeStaged":
            fs_parameters = initConfParameters(config, fs_parameters)
            fs_parameters = initMeanCorrParameters(config, fs_parameters)
            fs_parameters = initReguParameters(config, fs_parameters)
        elif config.feature_selection == "threeStaged_listranking":
            fs_parameters = initReguParameters(config, fs_parameters)
            fs_parameters["confusion_goodwill"] = Parameter("confusion_goodwill", "real", half_step_limits=config.confusion_goodwill_bounds)
            fs_parameters["err_warping_exp"] = Parameter("err_warping_exp", "real", half_step_limits=config.err_warping_exp_bounds)
            fs_parameters["confusion_normalization_exp"] = Parameter("confusion_normalization_exp", "real", half_step_limits=config.confusion_normalization_exp_bounds)
            fs_parameters["list_ranking_thresh"] = Parameter(
                "list_ranking_thresh", "integer", half_step_limits=config.list_ranking_thresh_bounds, transform_limits=(max_to_n_of_features, config.n_of_features)
            )
    else:
        #fs_parameters["list_ranking_thresh"] = Parameter(
        #        "list_ranking_thresh", "integer", half_step_limits=config.list_ranking_thresh_bounds, transform_limits=(max_to_n_of_features, config.n_of_features)
        #    )
        parameters['corr_thresh'] = Parameter('corr_thresh', 'real', half_step_limits=config.corr_thresh_bounds)
        fs_parameters = initConfParameters(config, fs_parameters, thresh_only=True)

    if not split_fs_parameters:
        parameters = fs_parameters

    if config.hpo_do_forest_param:
        if config.hpo_do_feat_selection and do_feat_selection:
            parameters["max_sample_parameter"] = Parameter("max_sample_parameter", "real", half_step_limits=config.max_sample_half_step)

        if config.forest_type == "random" or config.forest_type == "gradient_boost":
            parameters["num_of_trees"] = Parameter("num_of_trees", "integer", half_step_limits=config.forest_size_half_step)
            parameters["min_sample_split"] = Parameter("min_sample_split", "integer", half_step_limits=config.min_sample_split_half_step)
            parameters["max_feature_cont_parameter"] = Parameter("max_feature_cont_parameter", "real", half_step_limits=config.max_feature_cont_parameter_bounds)
            parameters["tree_min_leaf_samples"] = Parameter("tree_min_leaf_samples", "integer", half_step_limits=config.min_sample_leaf_half_step)
            parameters["ccp_alpha_exp"] = Parameter("ccp_alpha_exp", "real", half_step_limits=config.ccp_alpha_exp_half_step)
            parameters["min_impurity_decrease_exp"] = Parameter("min_impurity_decrease_exp", "real", half_step_limits=config.min_impurity_decrease_exp_half_step)
            parameters["tree_depth"] = Parameter("tree_depth", "integer", half_step_limits=config.tree_depth_half_step)

        if config.forest_type == "gradient_boost" or config.forest_type == "xgboost":
            # Learning rates are searched on a log scale. The old range went up to 2.0,
            # which is far outside anything useful for gradient boosting (values above
            # ~0.5 make the first trees overshoot and early stopping fires immediately),
            # and uniform sampling put 95% of the draws above 0.1.
            if config.hpo_do_feat_selection and do_feat_selection:
                parameters["learning_rate"] = Parameter("learning_rate", "real", half_step_limits=[0.001, 0.5], log_scale=True)
            parameters["learning_rate_1"] = Parameter("learning_rate_1", "real", half_step_limits=[0.001, 0.5], log_scale=True)

        if config.forest_type == "xgboost":
            if config.hpo_do_feat_selection and do_feat_selection:
                # Counts, weights and regularization terms are ratio scaled, so they are
                # searched in log space. log_offset=1 keeps the logarithm finite for the
                # terms whose lower bound is 0 and keeps 0 itself reachable.
                parameters["early_stopping"] = Parameter("early_stopping", "integer", half_step_limits=[1, 1000], log_scale=True)
                parameters["min_child_weight"] = Parameter("min_child_weight", "real", half_step_limits=[0., 100.], log_scale=True, log_offset=1.0)
                parameters["xgb_gamma"] = Parameter("xgb_gamma", "real", half_step_limits=[0.,10.], log_scale=True, log_offset=1.0)
                parameters["xgb_alpha"] = Parameter("xgb_alpha", "real", half_step_limits=[0.,5.], log_scale=True, log_offset=1.0)
                parameters["xgb_lambda"] = Parameter("xgb_lambda", "real", half_step_limits=[0.,5.], log_scale=True, log_offset=1.0)
                # colsample_* must stay in (0, 1]; xgboost rejects 0.
                parameters["colsample_bytree"] = Parameter("colsample_bytree", "real", half_step_limits=[0.01,1.])
                parameters["colsample_bylevel"] = Parameter("colsample_bylevel", "real", half_step_limits=[0.01,1.])
                parameters["colsample_bynode"] = Parameter("colsample_bynode", "real", half_step_limits=[0.01,1.])
                parameters["max_delta_step"] = Parameter("max_delta_step", "real", half_step_limits=[0.,50.], log_scale=True, log_offset=1.0)
                parameters["feat_impact_thresh"] = Parameter("feat_impact_thresh", "real", half_step_limits=[-0.01,0.01])
                parameters["tree_depth"] = Parameter("tree_depth", "integer", half_step_limits=[1,31])
                parameters["num_of_trees"] = Parameter("num_of_trees", "integer", half_step_limits=[10,10_000], log_scale=True)
                parameters["max_cat_to_onehot"] = Parameter("max_cat_to_onehot", "integer", half_step_limits=[1,500], log_scale=True)
                parameters["max_cat_threshold"] = Parameter("max_cat_threshold", "integer", half_step_limits=[1,500], log_scale=True)

            parameters["max_sample_parameter_1"] = Parameter("max_sample_parameter_1", "real", half_step_limits=config.max_sample_half_step)
            parameters["early_stopping_1"] = Parameter("early_stopping_1", "integer", half_step_limits=[1, 1000], log_scale=True)
            parameters["min_child_weight_1"] = Parameter("min_child_weight_1", "real", half_step_limits=[0., 500.], log_scale=True, log_offset=1.0)
            parameters["xgb_gamma_1"] = Parameter("xgb_gamma_1", "real", half_step_limits=[0.,10.], log_scale=True, log_offset=1.0)
            parameters["xgb_alpha_1"] = Parameter("xgb_alpha_1", "real", half_step_limits=[0.,5.], log_scale=True, log_offset=1.0)
            parameters["xgb_lambda_1"] = Parameter("xgb_lambda_1", "real", half_step_limits=[0.,5.], log_scale=True, log_offset=1.0)
            parameters["colsample_bytree_1"] = Parameter("colsample_bytree_1", "real", half_step_limits=[0.01,1.])
            parameters["colsample_bylevel_1"] = Parameter("colsample_bylevel_1", "real", half_step_limits=[0.01,1.])
            parameters["colsample_bynode_1"] = Parameter("colsample_bynode_1", "real", half_step_limits=[0.01,1.])
            parameters["max_delta_step_1"] = Parameter("max_delta_step_1", "real", half_step_limits=[0.,200.], log_scale=True, log_offset=1.0)
            parameters["tree_depth_1"] = Parameter("tree_depth_1", "integer", half_step_limits=[1,31])
            parameters["num_of_trees_1"] = Parameter("num_of_trees_1", "integer", half_step_limits=[10,10_000], log_scale=True)
            parameters["max_cat_to_onehot_1"] = Parameter("max_cat_to_onehot_1", "integer", half_step_limits=[1,500], log_scale=True)
            parameters["max_cat_threshold_1"] = Parameter("max_cat_threshold_1", "integer", half_step_limits=[1,500], log_scale=True)

        if not config.regression:
            parameters["criterion"] = Parameter("criterion", "categorical", possible_values=config.criteria, classification_specific=True)

    if config.hpo_do_sample_weighting:
        if config.regression:
            # if not config.geometric_weighting:
            #    parameters['sample_weight_parameter'] = Parameter('sample_weight_parameter','real',half_step_limits = config.sample_weight_parameter_half_step,
            #                                                        regression_specific = True)
            #    parameters['number_of_bins'] = Parameter('number_of_bins','integer',half_step_limits = config.number_of_bins_half_step,regression_specific = True)
            # else:
            # pass
            # parameters['geometric_exponent'] = Parameter('geometric_exponent', 'real', half_step_limits = config.geometric_exponent_bounds)
            pass
        else:
            parameters["class_weight"] = Parameter("class_weight", "categorical", possible_values=config.class_weights, classification_specific=True)

    if split_fs_parameters:
        return fss_parameters, fs_parameters, parameters
    return parameters


def remeasure_incumbent(
    config: util.Config,
    cv_obj: DataSAIL_cv,
    best_scores: util.Scores,
    first_scores: util.Scores,
    samples: SampleSpace | None = None,
    raw_feature_matrix_store_id: ray.ObjectRef | None = None,
    debug=False,
):
    """Re-evaluate the hyperparameter set currently held in `config` and replace
    `best_scores` with that fresh measurement.

    `best_scores` is otherwise a running maximum over noisy evaluations: every model
    training draws a fresh random_state, so the incumbent ends up sitting well above the
    true mean of its own configuration (winner's curse). A challenger then has to be
    better *and* get equally lucky to be accepted, which is why the optimization keeps
    producing scores that are close to, but never above, the recorded best.

    Overwriting the incumbent with an independent measurement makes the comparison
    unbiased again. The measurement uses the same protocol as the HPO candidates so that
    incumbent and challenger scores stay comparable.
    """
    if samples is None:
        config.logger.info("Skipping incumbent re-measurement: no sample space available")
        return best_scores, first_scores

    if config.multi_gpu > 0:
        gpu_share = config.multi_gpu
    else:
        gpu_share = None

    try:
        scores, new_first_scores = get_scores(
            config,
            cv_obj,
            samples.feat_corr_matrix,
            samples.feature_names,
            raw_feature_matrix_store_id=raw_feature_matrix_store_id,
            remote=(config.multi_gpu > 0),
            debug=debug,
            get_first_scores=True,
            gpu_share=gpu_share,
        )
    except Exception:
        [e, f, g] = sys.exc_info()
        g = traceback.format_exc()
        config.logger.error(f"ERROR while re-measuring the incumbent, keeping the old scores:\n{e}\n{f}\n{g}")
        return best_scores, first_scores

    obj_sc = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)
    if obj_sc is None or obj_sc != obj_sc:
        config.logger.info("Incumbent re-measurement returned None or NaN, keeping the old scores")
        return best_scores, first_scores

    old_obj_sc = util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)
    config.logger.info(
        f"=========== Re-measured the incumbent: {old_obj_sc} -> {obj_sc} ==========="
    )
    scores.printOut(config=config)

    return scores, new_first_scores


def threeDimHyperOptimization(
    config: util.Config,
    cv_obj: DataSAIL_cv,
    best_scores: util.Scores,
    first_scores: util.Scores,
    samples: SampleSpace | None =None,
    samples_store_id: ray.ObjectRef | None =None,
    raw_feature_matrix_store_id: ray.ObjectRef | None =None,
    distance_map=None,
    debug=False
):
    util.set_estimation_delta(config, first_scores, best_scores)
    random.seed()
    parameters: dict[str, Parameter]
    fss_parameters, fs_parameters, parameters = initParameters(config, split_fs_parameters=True)
    converged = False
    n = 1
    score_matrix = {}

    while not converged:
        converged = True
        #if n > 1:
        #    cv_obj.reset_confusion_maps()

        if n > 1:
            # The scores from round n-1 are a maximum over many noisy evaluations. Take a
            # fresh, independent measurement of the incumbent before challenging it again.
            best_scores, first_scores = remeasure_incumbent(
                config,
                cv_obj,
                best_scores,
                first_scores,
                samples=samples,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                debug=debug,
            )
            util.set_estimation_delta(config, first_scores, best_scores)

        if not debug:
            for param in [fss_parameters]:
                param_names = list(param.keys())
                random.shuffle(param_names)

                while len(param_names) > 2:
                    param_trio = param_names.pop(), param_names.pop(), param_names.pop()
                    new_opti, best_scores, first_scores, cv_obj = threeDim(
                        param[param_trio[0]],
                        param[param_trio[1]],
                        param[param_trio[2]],
                        best_scores,
                        first_scores,
                        config,
                        score_matrix,
                        cv_obj,
                        distance_map,
                        samples=samples,
                        samples_store_id=samples_store_id,
                        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                        debug=debug,
                    )
                    if new_opti:
                        converged = False

            if len(fs_parameters) > 1:
                new_opti, best_scores, first_scores, cv_obj = bayesian_optimisation(
                    None,
                    config,
                    list(fs_parameters.values()),
                    cv_obj,
                    best_scores,
                    first_scores,
                    distance_map,
                    samples=samples,
                    samples_store_id=samples_store_id,
                    raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                    n_pre_samples=None,
                    debug=debug,
                )
                if new_opti:
                    converged = False

        for param in [parameters]:
            param_names = list(param.keys())
            random.shuffle(param_names)

            SDTree = SubdimensionTree(
                param_names,
                parameters,
                config,
                cv_obj,
                best_scores,
                first_scores,
                samples,
                samples_store_id,
                raw_feature_matrix_store_id,
                distance_map = distance_map)

            tree_converged, best_scores, first_scores, cv_obj = SDTree.ascend()
            # Must not overwrite: the feature selection stages above may already have
            # found a new optimum, in which case the outer loop has to run again even if
            # the subdimension tree itself converged.
            converged = converged and tree_converged

        config.logger.info(f"Iteration: {n}")
        config.logParameter()
        config.saveHyperParameter(f"hyperparameters_ThreeDim_epoch_{n}.conf")
        if best_scores is not None:
            best_scores.printOut(config=config)
        n += 1

    cv_obj.reset_confusion_maps()
    return


def bayesianComplete(
    config,
    cv_obj: DataSAIL_cv,
    best_scores: util.Scores,
    samples=None,
    samples_store_id=None,
    raw_feature_matrix_store_id=None,
    distance_map=None,
    debug=False
    ):

    parameters = initParameters(config, do_feat_selection=False)

    new_optimimum, best_scores, cv_obj = bayesian_optimisation(
        9_999_999,
        config,
        [parameters[p] for p in parameters],
        cv_obj,
        best_scores,
        distance_map,
        samples=samples,
        samples_store_id=samples_store_id,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        n_pre_samples=100,
        debug=debug,
        store_params = True
        )

    config.logger.info("Bayesian optimization finished")
    config.logParameter()
    best_scores.printOut()

    return

LEAF_SIZE = 4
MAX_PARAMS = 24

class SubdimensionNode:
    def __init__(self, param_names: list[str], parent, tree):
        self.parent = parent
        self.tree = tree
        self.param_names = param_names
        if len(param_names) > LEAF_SIZE:
            h = len(param_names) // 2
            self.left_params = param_names[:h]
            self.right_params = param_names[h:]
            self.left_child = SubdimensionNode(self.left_params, self, tree)
            self.right_child = SubdimensionNode(self.right_params, self, tree)
            self.is_leaf = False
        else:
            self.is_leaf = True
        
    def optimize(self):
        if self.is_leaf:
            param_set = [self.tree.parameters[param] for param in self.param_names]
            new_opti, self.tree.best_scores, self.tree.first_scores, self.tree.cv_obj, x_list, y_list = bayesian_optimisation(
                None,
                self.tree.config,
                param_set,
                self.tree.cv_obj,
                self.tree.best_scores,
                self.tree.first_scores,
                self.tree.distance_map,
                samples=self.tree.samples,
                samples_store_id=self.tree.samples_store_id,
                raw_feature_matrix_store_id=self.tree.raw_feature_matrix_store_id,
                n_pre_samples=None,
                get_bayes_tuple=True
            )

            if new_opti:
                self.tree.converged = False

            return x_list, y_list

        else:
            static_right_param_values: np.ndarray = np.array([self.tree.parameters[param_name].getValue(self.tree.config) for param_name in self.right_params])
            left_x, left_y = self.left_child.optimize()
            static_left_param_values = np.array([self.tree.parameters[param_name].getValue(self.tree.config) for param_name in self.left_params])
            right_x, right_y = self.right_child.optimize()

            combined_x = []
            combined_y = []

            for pos, param_values in enumerate(left_x):
                completed_param_values = np.concatenate([param_values, static_right_param_values])
                combined_x.append(completed_param_values)
                combined_y.append(left_y[pos])

            for pos, param_values in enumerate(right_x):
                completed_param_values = np.concatenate([static_left_param_values, param_values])
                combined_x.append(completed_param_values)
                combined_y.append(right_y[pos])

            if len(self.param_names) > MAX_PARAMS:
                return combined_x, combined_y

            param_set = [self.tree.parameters[param] for param in self.param_names]
            new_opti, self.tree.best_scores, self.tree.first_scores, self.tree.cv_obj, x_list, y_list = bayesian_optimisation(
                None,
                self.tree.config,
                param_set,
                self.tree.cv_obj,
                self.tree.best_scores,
                self.tree.first_scores,
                self.tree.distance_map,
                samples=self.tree.samples,
                samples_store_id=self.tree.samples_store_id,
                raw_feature_matrix_store_id=self.tree.raw_feature_matrix_store_id,
                n_pre_samples=None,
                get_bayes_tuple=True,
                bayes_tuple = (combined_x, combined_y)
            )

            if new_opti:
                self.tree.converged = False

            return x_list, y_list


class SubdimensionTree:
    def __init__(
            self,
            param_names: list[str],
            parameters: dict[str, Parameter],
            config: util.Config,
            cv_obj: DataSAIL_cv,
            best_scores: util.Scores,
            first_scores: util.Scores,
            samples,
            samples_store_id,
            raw_feature_matrix_store_id,
            distance_map = None
            ):
        self.parameters = parameters
        self.config = config
        self.cv_obj = cv_obj
        self.best_scores = best_scores
        self.first_scores = first_scores
        self.samples = samples
        self.samples_store_id = samples_store_id
        self.raw_feature_matrix_store_id = raw_feature_matrix_store_id
        self.distance_map = distance_map
        random.shuffle(param_names)
        self.root = SubdimensionNode(param_names, None, self)
        self.converged = True
        
    def ascend(self):
        self.root.optimize()
        return self.converged, self.best_scores, self.first_scores, self.cv_obj