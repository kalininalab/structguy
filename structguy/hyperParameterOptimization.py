import math
import os
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

    def __init__(self, config, param_names, n_iters, parameters=None):
        self.config = config
        self.param_names = param_names
        self.parameters = parameters
        self.n_iters = n_iters
        self.confirmation = None      # (entry score, confirmed score, held) once confirmed
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
        self._degenerate_warned = False
        self.gp_history = []          # (predicted mu, predicted sigma, observed) of gp proposals
        self.last_gp_mu = None        # prediction for the most recent gp proposal
        self.gp_since_accept = 0      # gp results since the last accepted improvement
        self.gp_history_at_accept = 0 # len(gp_history) at the last accepted improvement
        self.stop_reason = None       # set once should_stop() decided to stop
        self.n_failed = 0             # evaluations with folds that produced no model
        self.identified_noise = []    # every noise level the gp could identify in this subspace
        self.length_scales = None     # last fitted Matern length scale per parameter

    @staticmethod
    def key(sample):
        return tuple(round(float(v), 12) for v in np.ravel(sample))

    def describe(self, sample):
        """Render a sample as `name=value` pairs.

        Samples live in search space, which for a log scaled parameter is log10 of the
        real value. Printing that raw is actively misleading (a logged
        `learning_rate_1=-1.50516` is really 0.03125), so show the real value and keep
        the search space number in brackets.
        """
        values = np.ravel(sample)
        if self.parameters is None:
            return ", ".join(f"{n}={float(v):.6g}" for n, v in zip(self.param_names, values))

        parts = []
        for parameter, raw in zip(self.parameters, values):
            raw = float(raw)
            if getattr(parameter, "log_scale", False):
                parts.append(f"{parameter.name}={parameter.from_search_space(raw):.6g} [log10 {raw:+.4f}]")
            else:
                parts.append(f"{parameter.name}={raw:.6g}")
        return ", ".join(parts)

    def record_dispatch(self, sample, source, mu=None, sigma=None):
        self.provenance[self.key(sample)] = (source, mu, sigma)
        self.n_dispatched += 1
        if mu is not None:
            self.last_gp_mu = mu
        if self.config.verbosity >= 2:
            pred = "" if mu is None else f" | GP predicted {mu:.6f} +- {sigma:.6f}"
            self.config.logger.info(f"HPO dispatch #{self.n_dispatched} [{source}]{pred}: {self.describe(sample)}")

    def record_result(self, sample, cv_score, interrupted=False, scores=None, failed=False):
        source, mu, sigma = self.provenance.pop(self.key(sample), ("unknown", None, None))

        # Folds that produced no model enter the score as zeros, so the score of such an
        # evaluation is an artifact of how many folds dropped out, not a value of the
        # objective. Keep it out of every statistic (noise, gp errors, early stop bias),
        # but let a failed gp proposal count toward the early stop patience, so a gp that
        # only proposes failing configurations still stops.
        if failed:
            self.n_failed += 1
            if mu is not None:
                self.gp_since_accept += 1
            if self.config.verbosity >= 1:
                n_folds = getattr(scores, "failed_folds", 0)
                self.config.logger.info(
                    f"HPO result [{source}] FAILED: {n_folds} fold(s) without a model, score={cv_score} "
                    f"is not used as a value of the objective"
                )
            return source
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
                self.gp_history.append((mu, sigma, cv_score))
                self.gp_since_accept += 1
        if self.config.verbosity >= 1:
            if mu is None:
                pred = ""
            elif usable:
                pred = f" | GP predicted {mu:.6f} +- {sigma:.6f} (error {abs(mu - cv_score):.6f})"
            else:
                pred = f" | GP predicted {mu:.6f} +- {sigma:.6f}"
            self.config.logger.info(f"HPO result [{source}] score={cv_score} {interrupted=}{pred}")

        self.check_degenerate_spread()
        return source

    # Below this, candidates that differ in their hyperparameters are producing scores
    # that agree to floating point noise. Chosen well under any real effect (the smallest
    # ever measured here was 7e-4) and well above GPU non-determinism (~1e-8).
    DEGENERATE_SPREAD = 1e-6

    def check_degenerate_spread(self):
        """Warn when the candidates stop depending on their hyperparameters at all.

        If a whole subspace returns the same score, the sampled values are not reaching
        the model. That happened for months because para_eval forwarded the *subspace
        entry* config reference to the training tasks instead of one holding the
        candidate's values, so every candidate trained an identical model and the search
        was measuring noise. It is silent in every other diagnostic, so check for it
        directly and say so early rather than after a subspace has burned hours.
        """
        if self._degenerate_warned or len(self.scores) < 8:
            return
        spread = max(self.scores) - min(self.scores)
        if spread > self.DEGENERATE_SPREAD:
            self._degenerate_warned = True      # real variation seen, stop checking
            return

        self._degenerate_warned = True
        self.config.logger.warning(
            f"=============== WARNING: hyperparameters appear to have no effect ===============\n"
            f"The first {len(self.scores)} candidates of this subspace span only {spread:.3g} "
            f"({min(self.scores):.10f} .. {max(self.scores):.10f}), which is floating point noise "
            f"rather than a response to the {len(self.param_names)} parameters being varied "
            f"({self.param_names}).\n"
            f"The sampled values are most likely not reaching the training tasks, so the search "
            f"is ranking identical models. Check that the configuration handed to the remote "
            f"trainers carries the candidate's values.\n"
            f"================================================================================="
        )

    def record_accept(self, cv_score, previous_best):
        self.gp_since_accept = 0
        self.gp_history_at_accept = len(self.gp_history)
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

        # The noise the gps identified in earlier subspaces of this run. The objective is the
        # same everywhere, so it carries over; without it the gate either had no estimate
        # (and accepted +0.00003) or fell back to the gp's prediction errors (0.0105, which
        # rejected real improvements of +0.0013) whenever this subspace's fit sat at its bound.
        memory = getattr(self.config, "hpo_noise_memory", None)
        if memory:
            return float(np.median(memory))

        residual_scale = self.gp_residual_scale()
        if residual_scale is not None:
            return residual_scale

        return self.robust_scatter()

    @staticmethod
    def robust_scale(values):
        """Scaled median absolute deviation, None if it is 0 (more than half identical)."""
        center = float(np.median(values))
        mad = float(np.median([abs(v - center) for v in values]))
        return 1.4826 * mad if mad > 0 else None

    def gp_residual_scale(self):
        """Robust spread of the gp's misses (predicted - observed) around their median.

        Used when the gp's own noise level is not identifiable. Unlike the scatter of the
        candidate scores it does not contain the real effect of the hyperparameters (an
        improving subspace inflated that scatter until the gate blocked every further
        improvement), and centering on the median removes the gp's systematic optimism.
        It still contains the surrogate's own error, so it errs on the high side.
        """
        if len(self.gp_history) < 6:
            return None
        return self.robust_scale([mu - y for mu, _, y in self.gp_history])

    def robust_scatter(self):
        """Robust scale (scaled median absolute deviation) of all candidate scores.

        Not the standard deviation: a single failed or partial evaluation (a score of 0.49
        among 0.63s) inflated it to 0.05 and with it the gate, so that no real improvement
        could be accepted any more. None with too few scores, or when more than half of
        them are identical (a plateau), where the scatter says nothing about the noise.
        """
        if len(self.scores) < 6:
            return None
        return self.robust_scale(self.scores)

    def stop_threshold(self):
        """The smallest improvement worth continuing for: the acceptance gate, with
        hpo_min_improvement as a floor. Returns (threshold, description)."""
        floor = float(getattr(self.config, "hpo_min_improvement", 0.001) or 0.0)
        factor = getattr(self.config, "hpo_noise_gate", 1.0)
        noise = self.noise_hint()
        if factor and factor > 0 and noise is not None and factor * noise > floor:
            return factor * noise, f"acceptance gate {factor:g} x noise {noise:.6f} [{self.noise_source()}]"
        return floor, "hpo_min_improvement floor"

    def should_stop(self, incumbent):
        """Early stopping of this subspace.

        Stops once, over the last hpo_stop_window gp proposals,
          (a) none was accepted or beats the incumbent by the threshold (stop_threshold),
              so the window only starts after the last accepted improvement, and
          (b) the gp's most recent prediction, minus the optimism it showed on those
              proposals (median of predicted - observed), does not either.
        Neither the raw prediction error nor the scatter of the scores is used: the gp
        is systematically optimistic about the points it selects, so a small error can
        hide a promise that never materializes, and a small scatter comes as much from
        sampling around the incumbent or on a plateau as from a flat objective.

        Once true it stays true, so the remaining in-flight evaluations wind down.
        """
        if self.stop_reason is not None:
            return True
        c = self.config
        if not getattr(c, "hpo_early_stop", True):
            return False
        window = int(getattr(c, "hpo_stop_window", 8) or 0)
        if window < 1 or self.gp_since_accept < window:
            return False
        if incumbent is None or isinstance(incumbent, bool) or incumbent != incumbent:
            return False

        threshold, threshold_source = self.stop_threshold()
        recent = self.gp_history[max(self.gp_history_at_accept, len(self.gp_history) - window):]
        if not recent:
            self.stop_reason = f"all of the last {window} gp proposals failed (folds without a model)"
            c.logger.info(f"=========== HPO early stop of this subspace: {self.stop_reason} ===========")
            return True
        residuals = [mu - y for mu, _, y in recent]
        # Median: a single failed evaluation is a huge miss and would dominate a mean
        bias = float(np.median(residuals))
        best_gain = max(y for _, _, y in recent) - incumbent
        next_mu = self.last_gp_mu if self.last_gp_mu is not None else max(mu for mu, _, _ in recent)
        outlook = next_mu - bias - incumbent

        if best_gain >= threshold or outlook >= threshold:
            return False

        residual_scale = self.robust_scale(residuals) or 0.0
        repeated = sum(1 for _, _, y in recent if sum(1 for s in self.scores if abs(s - y) <= 1e-12) > 1)
        self.stop_reason = (
            f"last {window} gp proposals ({len(recent)} valid): best gain {best_gain:+.6f}, gp optimism {bias:+.6f}, "
            f"bias corrected outlook {outlook:+.6f}, both below {threshold:.6f} ({threshold_source}); "
            f"robust residual spread {residual_scale:.6f}, {repeated}/{window} scores repeat an earlier one"
        )
        c.logger.info(f"=========== HPO early stop of this subspace: {self.stop_reason} ===========")
        return True

    def noise_source(self):
        if self.repeat_stds:
            return f"repeated trainings (n={len(self.repeat_stds)}, repeat={self.repeat})"
        if self.gp_noise is not None and self.gp_noise > 0:
            return "gaussian process WhiteKernel"
        memory = getattr(self.config, "hpo_noise_memory", None)
        if memory:
            return f"noise identified in {len(memory)} earlier subspaces of this run"
        if self.gp_residual_scale() is not None:
            return f"robust spread of the gp prediction errors (n={len(self.gp_history)})"
        if self.robust_scatter() is not None:
            return "robust scatter of all candidates (crude fallback, overestimates)"
        return "unavailable"

    def log_kernel(self, model, yp=None):
        try:
            kernel = model.kernel_
        except AttributeError:
            return
        if self.config.verbosity >= 2:
            self.config.logger.info(f"HPO fitted GP kernel: {kernel}")

        # Matern length scale per parameter (ConstantKernel * Matern + WhiteKernel), in
        # the [0, 1] scaled search space: short means the score reacts to the parameter
        try:
            length_scales = np.atleast_1d(kernel.k1.k2.length_scale)
            if len(length_scales) == len(self.param_names):
                self.length_scales = dict(zip(self.param_names, (float(v) for v in length_scales)))
        except AttributeError:
            pass

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
            self.identified_noise.append(self.gp_noise)

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
        if len(self.scores) >= 2:
            spread = max(self.scores) - min(self.scores)
            c.logger.info(f"Candidate score spread: {spread:.3g}")
            if spread <= self.DEGENERATE_SPREAD:
                c.logger.warning(
                    "  -> that is floating point noise: the hyperparameters did not reach the "
                    "model and this subspace ranked identical configurations"
                )
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
        if self.confirmation is not None:
            entry, confirmed, held = self.confirmation
            selected = max(self.scores) if self.scores else float("nan")
            c.logger.info(
                f"Confirmation of the winner: selected {selected:.6f} -> re-measured {confirmed:.6f} "
                f"(selection bias {selected - confirmed:+.6f}), subspace entry was {entry}"
            )
            c.logger.info(f"  -> the improvement {'held up' if held else 'did NOT hold up, parameters rolled back'}")
        elif new_optimimum:
            c.logger.info("Winner was NOT confirmed by a fresh measurement (hpo_confirm_optimum is off)")
        if self.n_failed:
            c.logger.info(f"Failed evaluations (folds without a model): {self.n_failed}")
        if self.stop_reason is not None:
            c.logger.info(
                f"Stopped early after {self.n_dispatched} of {self.n_iters} dispatches "
                f"({max(0, self.n_iters - self.n_dispatched)} saved): {self.stop_reason}"
            )
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
                    message = out_queue.get(timeout=5)
                    if message[0] == "started":
                        continue
                    _, scores, params, first_scores = message
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

    # Failed evaluations (folds without a model) get the worst valid score as gp target,
    # as in bayesian_optimisation, and are never adopted.
    valid_scores = [y for y in y_list if y is not None and y == y]
    for scores, first_scores, params in results:
        obj_sc = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)
        if getattr(scores, 'failed_folds', 0) == 0 and obj_sc is not None and obj_sc == obj_sc:
            valid_scores.append(obj_sc)

    for scores, first_scores, params in results:
        obj_sc = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)
        if obj_sc is None or obj_sc != obj_sc:
            config.logger.info(f"========= Warning: None or NaN objective score for: {param_names}, {params}")
            scores = util.Scores(zero=True)
            obj_sc = scores.objective_value(config)

        failed = getattr(scores, 'failed_folds', 0) > 0
        if failed:
            config.logger.info(f"Random init evaluation FAILED ({scores.failed_folds} fold(s) without a model), score={obj_sc}: {params}")
            if valid_scores:
                obj_sc = min(valid_scores)

        x_list.append(params)
        y_list.append(obj_sc)

        if config.verbosity >= 3:
            config.logger.info(f"Objective score: {obj_sc}")

        if not failed and util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty):
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
    bayes_tuple = None,
    relevance_log: dict | None = None
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

    # The scores this subspace starts from, kept so that a winner can be rolled back if
    # its confirmation measurement does not beat them.
    entry_best_scores = best_scores
    entry_best_first_scores = best_first_scores

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

    trace = HpoTrace(config, param_names, n_iters, parameters=parameters)

    def gp_target(raw_score):
        # A failed evaluation (folds without a model, entering as zeros) is not a value
        # of the objective. Modeling it as one put cliffs into the gp's landscape and
        # inflated its noise estimate, so the gp sees the worst valid score instead.
        valid = trace.scores if trace.scores else [y for y in y_list if y is not None and y == y]
        return min(valid) if valid else raw_score

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

            failed = getattr(scores, 'failed_folds', 0) > 0
            trace.record_result(next_sample, cv_score, interrupted=interrupted, scores=scores, failed=failed)

            previous_best = util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)
            if isinstance(previous_best, bool):
                previous_best = None

            accept = (not interrupted) and (not failed) and util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty)
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
            y_list.append(gp_target(cv_score) if failed else cv_score)

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

            if trace.should_stop(util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)):
                break

    else:
        remote_processes = []
        # --tpgpu is the number of models trained at the same time on one GPU, so there are
        # multi_gpu * threads_per_gpu model slots. One candidate trains a model per sub-slice of
        # every CV fold; as many candidates run in parallel as it takes to fill the slots.
        model_slots = config.multi_gpu * int(config.threads_per_gpu)
        if config.cv_counters is not None:
            n_cv_folds = len(config.cv_counters)
        elif config.cv_hpo_limiter is not None:
            n_cv_folds = min(config.cv_hpo_limiter, config.crossValidation_fold)
        else:
            n_cv_folds = config.crossValidation_fold
        models_per_candidate = max(1, n_cv_folds * (config.crossValidation_fold - 1))
        number_of_procs = max(1, math.ceil(model_slots / models_per_candidate))

        para_number = max(1, config.proc_n // number_of_procs)

        current_params_id = 0
        n_of_sent_hpo_sets = 0

        # The GPUs each candidate hands out to its CV folds
        gpu_share = config.multi_gpu / number_of_procs
        remote_function = para_eval #.options(num_gpus = gpu_share)
        
        store, config_ref_container = stores

        #for i in range(config.multi_gpu):
        #    pg = ray.util.placement_group([{"CPU": config.proc_n // (config.multi_gpu), 'GPU':1}], name=f"pg_{i}")
        #    ray.get(pg.ready())

        def launch_worker(p):
            com_queue = Queue()
            out_queue = Queue()
            proc_id = remote_function.options(
                    num_cpus=0.5,
                    num_gpus=0,
                ).remote(com_queue, out_queue, store, para_number, gpu_share, p, config_ref_container)
            return (com_queue, out_queue, proc_id)

        dispatched = []
        for p in range(number_of_procs):
            next_sample, draw_kind = draw_random_sample(config, bounds, scaler, best_params)
            trace.record_dispatch(next_sample, f"init/{draw_kind}")

            worker = launch_worker(p)
            worker[0].put(next_sample)
            n_of_sent_hpo_sets += 1

            remote_processes.append(worker)
            dispatched.append(next_sample)

        if config.verbosity >= 1:
            config.logger.info(f'Started {len(remote_processes)} para_eval processes for {model_slots} model slots ({config.threads_per_gpu} per GPU, {models_per_candidate} models per candidate): {para_number=} {gpu_share=}')

        dones = []
        counts = []
        for i in range(len(remote_processes)):
            dones.append(False)
            counts.append(0)
        # The candidate every worker is evaluating (None: no candidate)
        in_flight = list(dispatched)
        all_done = False
        # Lost workers: starts seen per worker (a second one is a restart), candidates lost
        # with a dead or restarted worker (scored as failed evaluations), replacements made
        worker_starts = [0] * len(remote_processes)
        lost_candidates = {}
        n_worker_replacements = 0
        max_worker_replacements = int(getattr(config, "hpo_max_worker_restarts", 3))
        
        append_counter = 0
        overall_timeout_start = time.time()
        timeout_start = None

        message_counter = 0
        count_var = 0

        while not all_done:
            # A para_eval task only ends by itself after its None stop signal. Ending earlier means it died
            # (exception, killed): without this check the loop waited for its answer until the timeouts.
            for i, lost in check_dead_para_eval_workers(config, remote_processes, dones, in_flight):
                if n_worker_replacements >= max_worker_replacements:
                    config.logger.error(f"para_eval worker {i} is not replaced, hpo_max_worker_restarts={max_worker_replacements} reached; the subspace continues without it")
                    dones[i] = True
                    in_flight[i] = None
                    continue
                n_worker_replacements += 1
                for old_queue in remote_processes[i][:2]:
                    try:
                        old_queue.shutdown()
                    except Exception:
                        pass
                remote_processes[i] = launch_worker(i)
                worker_starts[i] = 0
                config.logger.error(f"para_eval worker {i} replaced ({n_worker_replacements}/{max_worker_replacements}); its candidate is scored as a failed evaluation")
                if lost is not None:
                    lost_candidates[i] = lost
                else:
                    next_sample, draw_kind = draw_random_sample(config, bounds, scaler, best_params)
                    trace.record_dispatch(next_sample, f"restart/{draw_kind}")
                    remote_processes[i][0].put(next_sample)
                    in_flight[i] = next_sample
            n_of_sent_hpo_sets = sum(sample is not None for sample in in_flight)
            if all(dones):
                break

            for i, (com_queue, out_queue, proc_id) in enumerate(remote_processes):
                if dones[i] and i not in lost_candidates:
                    continue

                if i in lost_candidates:
                    # The worker was lost with this candidate: judge it as a failed evaluation
                    next_sample = lost_candidates.pop(i)
                    scores = util.Scores(zero=True)
                    first_scores = util.Scores(zero=True)
                elif not out_queue.empty():
                    message = out_queue.get(timeout=0.5)
                    if message[0] == "started":
                        _, worker_pid, queue_was_empty = message
                        worker_starts[i] += 1
                        if worker_starts[i] == 1:
                            continue
                        # Ray re-executed the task despite max_retries=0 (e.g. after an out of
                        # memory kill). If the queue was empty, the dead attempt had taken the
                        # candidate in flight and the new one would wait for work forever.
                        if not queue_was_empty or in_flight[i] is None:
                            config.logger.error(f"para_eval worker {i} restarted (pid {worker_pid}), its candidate was still queued and is evaluated by the new attempt")
                            continue
                        if n_worker_replacements >= max_worker_replacements:
                            config.logger.error(f"para_eval worker {i} restarted again (pid {worker_pid}), hpo_max_worker_restarts={max_worker_replacements} reached; stopping it")
                            com_queue.put(None)
                            dones[i] = True
                            in_flight[i] = None
                            continue
                        n_worker_replacements += 1
                        config.logger.error(f"para_eval worker {i} restarted (pid {worker_pid}) and lost its candidate, which is scored as a failed evaluation")
                        next_sample = in_flight[i]
                        scores = util.Scores(zero=True)
                        first_scores = util.Scores(zero=True)
                    else:
                        _, scores, next_sample, first_scores = message
                else:
                    continue
                n_of_sent_hpo_sets -= 1
                in_flight[i] = None
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

                failed = getattr(scores, 'failed_folds', 0) > 0
                trace.record_result(next_sample, cv_score, interrupted=interrupted, scores=scores, failed=failed)

                previous_best = util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)
                if isinstance(previous_best, bool):
                    previous_best = None

                accept = (not interrupted) and (not failed) and util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty)
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
                y_list.append(gp_target(cv_score) if failed else cv_score)
                append_counter += 1

                # Update xp and yp
                xp = np.array(x_list)
                yp = np.array(y_list)
                scaled_xp = scaler.transform(xp)

                # Early stopping: the worker is released like at the end of the budget,
                # the evaluations still in flight finish and are judged as usual.
                stop = trace.should_stop(util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty))

                if current_params_id >= n_iters or stop:
                    com_queue.put(None)
                    dones[i] = True
                else:
                    # Refit and propose after every result when one worker runs (with two, every
                    # second evaluation used to be a random fill draw), after 2 results when
                    # several run, and only once the gp has n_params + 2 points; before that the
                    # local / uniform draws below are the warm-up.
                    gp_ready = append_counter >= min(2, len(remote_processes)) and len(y_list) >= n_params + 2
                    if gp_ready:
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
                    in_flight[i] = next_sample

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

    # Under common random numbers an evaluation is deterministic given the seed base, so
    # re-running the winner's hyperparameters would reproduce its score and confirm
    # nothing. What has to be checked there is whether the improvement survives *new*
    # seeds, and that is what the re-measurement at the start of the next round does.
    confirm = (
        new_optimimum
        and getattr(config, "hpo_confirm_optimum", True)
        and not getattr(config, "common_random_numbers", True)
    )
    if new_optimimum and not confirm and getattr(config, "common_random_numbers", True):
        config.logger.info(
            "Skipping the confirmation measurement: with common random numbers it would "
            "repeat the same seeds and reproduce the same score. The improvement is "
            "tested against fresh seeds at the start of the next round instead."
        )

    # All para_eval workers have been stopped above, so the driver can train on the GPUs
    # itself without competing with them.
    if confirm:
        new_optimimum, best_scores, best_first_scores = confirm_new_optimum(
            config,
            cv_obj,
            parameters,
            initial_values,
            entry_best_scores,
            entry_best_first_scores,
            best_scores,
            best_first_scores,
            trace,
            samples=samples,
            raw_feature_matrix_store_id=raw_feature_matrix_store_id,
            debug=debug,
        )

    trace.summary(best_scores, new_optimimum)

    # This subspace's identified noise, for the gates of the following subspaces
    noise_memory = getattr(config, "hpo_noise_memory", None)
    if noise_memory is not None and trace.identified_noise:
        noise_memory.append(float(np.median(trace.identified_noise)))

    # Relevance of the parameters as the gp saw them, for the weighting of the tree
    if relevance_log is not None and trace.length_scales:
        for name, length_scale in trace.length_scales.items():
            relevance_log.setdefault(name, []).append(length_scale)

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


def check_dead_para_eval_workers(config, remote_processes, dones, in_flight):
    """Find para_eval workers whose task ended without having been stopped, and log why.

    Returns [(worker index, candidate it was evaluating or None)]; the caller decides
    whether to replace them.
    """
    dead = []
    running = [(i, proc_id) for i, (_, _, proc_id) in enumerate(remote_processes) if not dones[i]]
    if not running:
        return dead
    ready, _ = ray.wait([proc_id for _, proc_id in running], num_returns=len(running), timeout=0)
    ready = set(ready)
    for i, proc_id in running:
        if proc_id not in ready:
            continue
        try:
            ray.get(proc_id)
            reason = 'it returned without a stop signal'
        except Exception as e:
            reason = f'it failed:\n{e}'
        candidate = None if in_flight[i] is None else [float(v) for v in in_flight[i]]
        config.logger.error(f'para_eval worker {i} ended (candidate in evaluation: {candidate}): {reason}')
        dead.append((i, in_flight[i]))
    return dead


# max_retries=0: a retried para_eval found its candidate already taken by the dead attempt
# and waited on an empty queue, while the driver waited for its answer (10 h lost on the HPC)
@ray.remote(max_retries=0)
def para_eval(com_queue: Queue, out_queue: Queue, store, para_number, gpu_share, proc_id, config_ref_container):
    # Announced before anything else, so the driver can tell a restarted worker from a new
    # one. Whether the queue was empty decides if a restart lost the candidate in flight.
    queue_was_empty = com_queue.empty()
    out_queue.put(("started", os.getpid(), queue_was_empty))

    (cv_obj, parameters, raw_feature_matrix_store_id, best_first_scores, feat_corr_matrix, feature_names) = store
    config = ray.get(config_ref_container[0])

    util.reset_logger_for_remotes(config)
    if config.verbosity >= 2:
        config.logger.info(f"Call of para_eval: com_queue.empty()={queue_was_empty} {proc_id=}")
      
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

        # The candidate's hyperparameters have just been written into this worker's own
        # `config`, but the training does not happen here: trainForest hands the work to
        # further remote tasks, and both trainRegressionForestWrapper and
        # double_booster_remote rebuild their configuration with
        # ray.get(config_ref_container[0]).
        #
        # Forwarding the container we were handed therefore gives them the configuration
        # as it was at the *start of the subspace* and silently discards every sampled
        # value, so every candidate trains an identical model and the whole subspace
        # returns the same score. The only hyperparameter that used to survive was
        # corr_thresh, because filterCorrelatedFeats() is evaluated by trainForest itself
        # and its result is shipped separately.
        #
        # Publish the mutated configuration and hand the workers that instead. The
        # reference is rebound on every iteration, so the previous one is released and
        # the object store does not grow.
        candidate_config_ref = [ray.put(config)]

        # An exception here would end this worker without an answer, and the HPO driver would wait for it until
        # its timeouts (hours). A failed candidate gets a zero score instead and the error goes to the main log.
        try:
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
                config_ref_container=candidate_config_ref
                )
        except Exception:
            config.logger.error(f"para_eval {proc_id}: evaluation of the candidate {[float(v) for v in params]} failed, scored as zero:\n{traceback.format_exc()}")
            scores = util.Scores(zero=True)
            first_scores = util.Scores(zero=True)

        out_queue.put(("result", scores, params, first_scores))
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

    if config.hpo_do_feat_selection and do_feat_selection:
        # Evaluated by trainForest itself (featureSelection.filterLocalFeatures), like corr_thresh
        parameters["local_features_N"] = Parameter("local_features_N", "integer", half_step_limits=[0, 20])

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
                if getattr(config, 'feat_impact_filter', 'z') == 'raw':
                    # Lower bound 0: negative thresholds removed every feature of some sub slices,
                    # zeroing whole folds (94 such events in the goldstandard_strict run)
                    # In units of the impact metric: protein-wise Spearman changes, or absolute errors of the target
                    impact_upper = 0.05 if getattr(config, 'feat_impact_metric', 'protein_spearman') == 'protein_spearman' else 0.01
                    parameters["feat_impact_thresh"] = Parameter("feat_impact_thresh", "real", half_step_limits=[0.,impact_upper])
                else:
                    # In standard errors of the impacts: +3 removes only clearly harmful features, below 0 also the
                    # unused ones, -3 keeps only clearly helpful ones (trainForest.double_booster). The raw threshold
                    # searched [0, 0.05] against impacts of ~0.002, so 90% of its range removed nothing.
                    parameters["feat_impact_z"] = Parameter("feat_impact_z", "real", half_step_limits=[-3., 3.])
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


def measure_current_config(
    config: util.Config,
    cv_obj: DataSAIL_cv,
    samples: SampleSpace | None = None,
    raw_feature_matrix_store_id: ray.ObjectRef | None = None,
    debug=False,
    label="configuration",
):
    """Evaluate the hyperparameters currently held in `config`, once, fresh.

    Uses the same protocol as the HPO candidates so the resulting score is directly
    comparable to theirs. Returns (scores, first_scores), or (None, None) if the
    measurement could not be made.

    Callers must only invoke this when no para_eval workers are in flight, otherwise the
    driver's own training competes with them for the GPUs.
    """
    if samples is None:
        config.logger.info(f"Skipping measurement of the {label}: no sample space available")
        return None, None

    if config.multi_gpu > 0:
        gpu_share = config.multi_gpu
    else:
        gpu_share = None

    try:
        scores, first_scores = get_scores(
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
        config.logger.error(f"ERROR while measuring the {label}:\n{e}\n{f}\n{g}")
        return None, None

    obj_sc = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)
    if obj_sc is None or obj_sc != obj_sc:
        config.logger.info(f"Measurement of the {label} returned None or NaN")
        return None, None

    return scores, first_scores


def random_start_parameters(config: util.Config) -> list[Parameter]:
    """The parameters the selected HPO mode will search, so the random start covers
    exactly that space."""
    if config.hyperOptimization == "bayesianComplete":
        return list(initParameters(config, do_feat_selection=False).values())

    fss_parameters, fs_parameters, parameters = initParameters(config, split_fs_parameters=True)
    return list({**fss_parameters, **fs_parameters, **parameters}.values())


def random_start(
    config: util.Config,
    cv_obj: DataSAIL_cv,
    parameters: list[Parameter],
    samples: SampleSpace | None = None,
    raw_feature_matrix_store_id: ray.ObjectRef | None = None,
    n_rounds: int = 20,
    debug=False,
):
    """Replace the initial training on the given hyperparameters with `n_rounds` random
    hyperparameter sets, and leave the best of them in `config` as the starting point
    of the HPO.

    Every round is a full measurement (no cv interruption), so the scores are comparable
    to each other and to the ones the HPO produces afterwards. Returns
    (best_scores, best_first_scores), or (None, None) if no round produced a usable score.
    """
    if debug:
        n_rounds = 2  # Just for testing

    bounds = np.array([p.half_step_limits for p in parameters])
    param_names = [p.name for p in parameters]

    config.logger.info(f"=========== Random start: {n_rounds} random hyperparameter sets over {param_names} ===========")

    best_scores = None
    best_first_scores = None
    best_params = None
    # Not config.snapshotHyperParameter(): config.hyperparameters does not list all of the
    # searched parameters, so the raw values are kept here instead.
    given_values = {p.name: config.getByString(p.name) for p in parameters}

    t0 = time.time()
    for n in range(n_rounds):
        params = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
        for pos, para_value in enumerate(params):
            parameters[pos].setValue(config, para_value)

        if config.verbosity >= 1:
            config.logger.info(f"Random start round {n + 1}/{n_rounds}: {params}")

        scores, first_scores = measure_current_config(
            config, cv_obj, samples=samples,
            raw_feature_matrix_store_id=raw_feature_matrix_store_id,
            debug=debug, label=f"random start round {n + 1}",
        )

        if scores is None:
            continue

        obj_sc = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)
        if getattr(scores, 'failed_folds', 0) > 0:
            config.logger.info(f"Random start round {n + 1} FAILED: {scores.failed_folds} fold(s) without a model, score={obj_sc}")
            continue
        if util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty):
            best_scores = scores
            best_first_scores = first_scores
            best_params = params
            config.logger.info("===========Found new optimum:=======================\n")
            config.logger.info(f'{params}')
            scores.printOut(config=config)
            config.logger.info("====================================================")
        elif config.verbosity >= 2:
            config.logger.info(
                f"No new optimun ({util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)}): {obj_sc}"
            )

    config.logger.info(f"Time for random start: {time.time() - t0}")

    if best_params is None:
        config.logger.warning("Random start produced no usable score, falling back to the given hyperparameters")
        for name, value in given_values.items():
            setattr(config, name, value)
        return None, None

    for pos, para_value in enumerate(best_params):
        parameters[pos].setValue(config, para_value)
    config.logger.info("=========== Random start finished, starting the HPO from: ===========")
    for p in parameters:
        config.logger.info(f"{p.name}: {config.getByString(p.name)}")
    best_scores.printOut(config=config)

    return best_scores, best_first_scores


def confirm_new_optimum(
    config: util.Config,
    cv_obj: DataSAIL_cv,
    parameters: list[Parameter],
    initial_values,
    entry_best_scores: util.Scores,
    entry_best_first_scores: util.Scores,
    best_scores: util.Scores,
    best_first_scores: util.Scores,
    trace,
    samples: SampleSpace | None = None,
    raw_feature_matrix_store_id: ray.ObjectRef | None = None,
    debug=False,
):
    """Re-measure the winner of a subspace before letting it stand.

    The score that won was chosen *because* it was the highest of many noisy draws, so it
    is biased upwards; that bias is what makes the incumbent unbeatable afterwards. An
    independent measurement of the same configuration carries no such selection, so it
    replaces the winning score outright rather than being averaged into it.

    If the fresh measurement no longer beats the score this subspace started from, the
    improvement was luck and the parameters are rolled back.

    Expects `config` to already hold the winning values. Returns
    (new_optimimum, best_scores, best_first_scores).
    """
    confirm_scores, confirm_first_scores = measure_current_config(
        config, cv_obj, samples=samples,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        debug=debug, label="candidate optimum",
    )

    if confirm_scores is None:
        config.logger.info("Could not confirm the new optimum, keeping it unverified")
        return True, best_scores, best_first_scores

    confirmed = util.get_objective_score(config, confirm_scores, feature_penalty=config.feature_penalty)
    selected = util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)
    entry = util.get_objective_score(config, entry_best_scores, feature_penalty=config.feature_penalty)
    if isinstance(entry, bool):
        entry = None

    config.logger.info(
        f"=========== Confirmation of the subspace winner: selected {selected} -> re-measured {confirmed} "
        f"(subspace started from {entry}) ==========="
    )

    if entry is not None and confirmed <= entry:
        config.logger.info(
            f"The improvement did not hold up ({confirmed:.6f} <= {entry:.6f}); rolling the "
            f"parameters of this subspace back to the incumbent values"
        )
        for pos, para_value in enumerate(initial_values):
            parameters[pos].setValue(config, para_value)
        trace.confirmation = (entry, confirmed, False)
        return False, entry_best_scores, entry_best_first_scores

    trace.confirmation = (entry, confirmed, True)
    confirm_scores.printOut(config=config)
    return True, confirm_scores, confirm_first_scores


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
    scores, new_first_scores = measure_current_config(
        config, cv_obj, samples=samples,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        debug=debug, label="incumbent",
    )

    if scores is None:
        config.logger.info("Keeping the old incumbent scores")
        return best_scores, first_scores

    obj_sc = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)
    old_obj_sc = util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)
    config.logger.info(
        f"=========== Re-measured the incumbent: {old_obj_sc} -> {obj_sc} ==========="
    )
    scores.printOut(config=config)

    return scores, new_first_scores


def reconcile_with_archive(config: util.Config, archive, best_scores, first_scores):
    """Keep the best configuration whose score was measured rather than selected.

    A round can end on a configuration that is worse than the one it started from: every
    subspace only ever compares against the current incumbent, so a sequence of accepts
    that individually looked fine can still drift downhill, and nothing used to notice.
    The archive holds the best configuration seen so far together with an honest
    measurement of it, and this restores that configuration whenever the freshly measured
    incumbent has fallen behind it.

    `archive` is (hyperparameter snapshot, scores, first_scores). Returns the updated
    (archive, best_scores, first_scores).
    """
    current = util.get_objective_score(config, best_scores, feature_penalty=config.feature_penalty)
    archived = util.get_objective_score(config, archive[1], feature_penalty=config.feature_penalty)

    if not isinstance(current, float) or current != current:
        config.logger.info("Archive check skipped: the current incumbent has no usable score")
        return archive, best_scores, first_scores

    if not isinstance(archived, float) or archived != archived:
        return (config.snapshotHyperParameter(), best_scores, first_scores), best_scores, first_scores

    if current >= archived:
        config.logger.info(
            f"Archive check: current incumbent {current:.6f} >= best archived {archived:.6f}, archiving it"
        )
        return (config.snapshotHyperParameter(), best_scores, first_scores), best_scores, first_scores

    config.logger.info(
        f"=========== Archive check: current incumbent {current:.6f} has fallen behind the best "
        f"measured configuration {archived:.6f}; restoring that configuration ==========="
    )
    config.restoreHyperParameter(archive[0])
    config.logParameter()
    return archive, archive[1], archive[2]


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

    # A search range can have been narrowed since the incumbent was found (e.g.
    # feat_impact_thresh, now >= 0). Move such values onto the range; the incumbent is
    # re-measured at the start of the round, so its score reflects the change.
    for param_set in (fss_parameters, fs_parameters, parameters):
        for parameter in param_set.values():
            if parameter.param_type == "categorical":
                continue
            value = parameter.getValue(config)
            if value is None:
                continue
            low, high = parameter.half_step_limits
            if value < low or value > high:
                clipped = min(max(value, low), high)
                config.logger.info(f"Incumbent {parameter.name} is outside its search range [{low}, {high}] (search space value {value}); clipped to {clipped}")
                parameter.setValue(config, clipped)

    # GP length scales of every subspace optimization of this run, the evidence the
    # weighted tree blends into its prior weights
    relevance_log = {}
    # Noise level identified per subspace, the acceptance gate's fallback (HpoTrace.noise_hint)
    config.hpo_noise_memory = []

    # The starting configuration and its measured score. Nothing may leave the
    # optimization in a state worse than this without it being noticed.
    archive = (config.snapshotHyperParameter(), best_scores, first_scores)

    crn = getattr(config, "common_random_numbers", True)

    while not converged:
        converged = True
        #if n > 1:
        #    cv_obj.reset_confusion_maps()

        if crn:
            # Every evaluation in this round trains on the same seeds, so differences
            # between candidates are differences between hyperparameters rather than
            # between random draws. The set changes each round, which is what turns the
            # re-measurement below into a real test: a configuration that only won
            # because of the seeds it was found on will not survive new ones.
            config.crn_seed_base = n
            config.logger.info(f"Round {n}: common random number seed base set to {n}")

        if n > 1 or crn:
            # The scores from the previous round are a maximum over many evaluations, and
            # under common random numbers they were taken on the previous round's seeds.
            # Re-measure the incumbent so it is both unbiased and directly comparable to
            # the candidates this round will produce.
            best_scores, first_scores = remeasure_incumbent(
                config,
                cv_obj,
                best_scores,
                first_scores,
                samples=samples,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                debug=debug,
            )
            archive, best_scores, first_scores = reconcile_with_archive(
                config, archive, best_scores, first_scores
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

            if getattr(config, "hpo_weighted_tree", True):
                SDTree = WeightedSubdimensionTree(
                    param_names,
                    parameters,
                    config,
                    cv_obj,
                    best_scores,
                    first_scores,
                    samples,
                    samples_store_id,
                    raw_feature_matrix_store_id,
                    distance_map=distance_map,
                    relevance_log=relevance_log)
            else:
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

    # The loop exits on the configuration the last round happened to end on, whose score
    # is once more a selected maximum. Measure it, and hand back whichever of it and the
    # archive is actually better, so the optimization never returns something worse than
    # the configuration it was started from.
    best_scores, first_scores = remeasure_incumbent(
        config,
        cv_obj,
        best_scores,
        first_scores,
        samples=samples,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        debug=debug,
    )
    archive, best_scores, first_scores = reconcile_with_archive(config, archive, best_scores, first_scores)

    config.logger.info("=============== Hyperparameter optimization finished ===============")
    config.logParameter()
    if best_scores is not None:
        best_scores.printOut(config=config)
    config.saveHyperParameter()

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

# Smallest subspace the tree will optimize. Was 4, which produced leaves that the
# gaussian process could not learn anything in: a leaf gets n_iters=8, so about 17
# evaluations, and an ARD kernel over 4 parameters already has 6 hyperparameters of its
# own to fit. Measured on the goldstandard_strict run, the surrogate's prediction error
# was 3.2x the evaluation noise in the 4 dimensional leaves against 1.1x in the 8 and 16
# dimensional nodes, where the error sits at the noise floor and cannot get better. The
# leaves cost about 9 h per round and produced one accepted improvement out of twelve, so
# they are not worth running.
LEAF_SIZE = 8
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


# Prior importance of the tree parameters, by base name (the second booster's copies carry
# the suffix _1). From the tunability study of Probst, Boulesteix & Bischl (JMLR 2019,
# 38 OpenML datasets: eta clearly first, then min_child_weight, subsample, nrounds,
# lambda, alpha, then max_depth and colsample) and the fANOVA ranking of SigOpt Mulch
# (2023: eta > max_depth > max_delta_step > alpha > num_boost_round > gamma > lambda >
# subsample > min_child_weight). corr_thresh and feat_impact_thresh decide which features
# the final booster sees, so they are weighted like the learning rate.
PARAMETER_PRIOR_WEIGHTS = {
    "learning_rate": 1.0,
    "corr_thresh": 1.0,
    "feat_impact_thresh": 1.0,
    "feat_impact_z": 1.0,
    "local_features_N": 1.0,
    "tree_depth": 0.6,
    "num_of_trees": 0.6,
    "early_stopping": 0.6,
    "min_child_weight": 0.6,
    "max_sample_parameter": 0.6,
    "xgb_alpha": 0.4,
    "xgb_lambda": 0.4,
    "xgb_gamma": 0.4,
    "max_delta_step": 0.4,
    "colsample_bytree": 0.25,
    "colsample_bylevel": 0.25,
    "colsample_bynode": 0.25,
    # Only have an effect with categorical features
    "max_cat_to_onehot": 0.1,
    "max_cat_threshold": 0.1,
}
DEFAULT_PRIOR_WEIGHT = 0.4
FEATURE_SELECTION_PARAMETERS = {"corr_thresh", "feat_impact_thresh", "feat_impact_z", "local_features_N"}
SECOND_BOOSTER_SUFFIX = "_1"
# Parameters that trade off against each other and are optimized together in at least one
# leaf: a smaller learning rate needs more trees and a later early stop.
COUPLED_PARAMETERS = [("learning_rate", "num_of_trees", "early_stopping")]


def parameter_base_name(name: str) -> str:
    if name.endswith(SECOND_BOOSTER_SUFFIX):
        return name[:-len(SECOND_BOOSTER_SUFFIX)]
    return name


def parameter_prior_weight(name: str, config: util.Config) -> float:
    """Prior importance of a parameter: its tier, times hpo_first_booster_factor for the
    first (feature selection) booster's parameters. config.hpo_param_weights overrides."""
    overrides = getattr(config, "hpo_param_weights", None) or {}
    if name in overrides:
        return float(overrides[name])
    base = parameter_base_name(name)
    weight = PARAMETER_PRIOR_WEIGHTS.get(base, DEFAULT_PRIOR_WEIGHT)
    is_booster_parameter = base in PARAMETER_PRIOR_WEIGHTS and base not in FEATURE_SELECTION_PARAMETERS
    if is_booster_parameter and not name.endswith(SECOND_BOOSTER_SUFFIX):
        weight *= float(getattr(config, "hpo_first_booster_factor", 0.5))
    return weight


def relevance_from_length_scales(length_scales: list[float]) -> float:
    """How strongly the score reacted to a parameter, from the gp's ARD length scales in
    the [0, 1] scaled search space, one per fit: each fit gives 1 for length scales up to
    1 (the score changes within the range), down to 0.05 at the upper bound of 100 (no
    effect), and the fits are averaged on that scale. Not the median length scale: with
    two fits that is their mean, where one fit at the bound (100) outweighs any evidence."""
    per_fit = [min(1.0, max(0.05, 1.0 / max(float(ls), 1e-9))) for ls in length_scales]
    return float(sum(per_fit) / len(per_fit))


def parameter_weights(param_names: list[str], config: util.Config, relevance_log: dict | None = None) -> dict[str, dict]:
    """Prior weight blended with the relevance the gps measured.

    w = (1 - a) * prior + a * relevance, with a = n / (n + hpo_relevance_prior_strength)
    and n the number of fits the parameter took part in, so the measurements take over as
    they accumulate. Returns {name: {prior, relevance, n, weight}}.
    """
    strength = float(getattr(config, "hpo_relevance_prior_strength", 4))
    table = {}
    for name in param_names:
        prior = parameter_prior_weight(name, config)
        scales = (relevance_log or {}).get(name, [])
        if scales:
            relevance = relevance_from_length_scales(scales)
            alpha = len(scales) / (len(scales) + strength)
            weight = (1.0 - alpha) * prior + alpha * relevance
        else:
            relevance = None
            weight = prior
        table[name] = {"prior": prior, "relevance": relevance, "n": len(scales), "weight": weight}
    return table


def build_weighted_leaves(weights: dict[str, float], config: util.Config, rng=random) -> list[list[str]]:
    """Pack the parameters into leaves by weight.

    Every parameter gets max(1, round(hpo_max_copies * w / w_max)) copies, so important
    parameters are optimized in several leaves. The first copies of each COUPLED_PARAMETERS
    group (per booster) go in as one unit. Units are packed first fit decreasing (ties
    broken at random, so every round groups differently) into leaves whose summed weight
    stays within hpo_leaf_budget and whose size stays within hpo_max_leaf_size, never
    twice the same parameter in a leaf. Important parameters therefore end up in small,
    focused leaves and unimportant ones in large ones. Leaves below hpo_min_leaf_size are
    dissolved into the lightest leaves that can take their parameters.
    """
    budget = float(getattr(config, "hpo_leaf_budget", 2.5))
    max_copies = int(getattr(config, "hpo_max_copies", 3))
    min_size = int(getattr(config, "hpo_min_leaf_size", 3))
    max_size = int(getattr(config, "hpo_max_leaf_size", LEAF_SIZE))

    w_max = max(weights.values())
    copies = {p: max(1, round(max_copies * w / w_max)) for p, w in weights.items()}

    units: list[list[str]] = []
    for suffix in ("", SECOND_BOOSTER_SUFFIX):
        for group in COUPLED_PARAMETERS:
            names = [g + suffix for g in group]
            if all(n in copies for n in names):
                units.append(names)
                for n in names:
                    copies[n] -= 1
    for p, n_copies in copies.items():
        units.extend([[p]] * n_copies)

    def weight_of(names):
        return sum(weights[p] for p in names)

    units.sort(key=lambda unit: (-weight_of(unit), rng.random()))

    leaves: list[list[str]] = []
    for unit in units:
        for leaf in leaves:
            if (not set(unit) & set(leaf)
                    and weight_of(leaf) + weight_of(unit) <= budget + 1e-9
                    and len(leaf) + len(unit) <= max_size):
                leaf.extend(unit)
                break
        else:
            leaves.append(list(unit))

    for leaf in [l for l in leaves if len(l) < min_size]:
        if len(leaves) == 1:
            break
        leaves.remove(leaf)
        if config.verbosity >= 2:
            config.logger.info(f"Dissolving leaf {leaf}: below hpo_min_leaf_size, its parameters may push other leaves over the budget")
        for p in leaf:
            candidates = [l for l in leaves if p not in l and len(l) < max_size]
            if not candidates:
                candidates = [l for l in leaves if p not in l]
            if candidates:
                min(candidates, key=weight_of).append(p)
            # else: p is already in every remaining leaf, dropping this copy loses nothing

    return leaves


class WeightedSubdimensionTree:
    """Subspace tree built from parameter weights (see build_weighted_leaves).

    The leaves are optimized one after the other. Adjacent leaves (in packing order, so the
    heavy ones pair up) form parents that are optimized over the union of their parameters,
    started from the histories of their children, as long as that union has at most
    MAX_PARAMS parameters (hpo_tree_parent_levels = 0 disables the parents).
    """

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
            distance_map=None,
            relevance_log: dict | None = None,
            rng=random,
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
        self.relevance_log = relevance_log
        self.converged = True

        self.weight_table = parameter_weights(param_names, config, relevance_log)
        weights = {name: row["weight"] for name, row in self.weight_table.items()}
        self.leaves = build_weighted_leaves(weights, config, rng=rng)
        self.log_tree(weights)

    def log_tree(self, weights):
        c = self.config
        counts = {}
        for leaf in self.leaves:
            for p in leaf:
                counts[p] = counts.get(p, 0) + 1
        c.logger.info("=========== Weighted subspace tree ===========")
        c.logger.info(f"{'parameter':<24} {'prior':>6} {'relev.':>6} {'fits':>4} {'weight':>6} {'leaves':>6}")
        for name, row in sorted(self.weight_table.items(), key=lambda kv: -kv[1]["weight"]):
            relevance = "-" if row["relevance"] is None else f"{row['relevance']:.2f}"
            c.logger.info(f"{name:<24} {row['prior']:6.2f} {relevance:>6} {row['n']:4d} {row['weight']:6.2f} {counts.get(name, 0):6d}")
        for i, leaf in enumerate(self.leaves):
            c.logger.info(f"Leaf {i} (weight {sum(weights[p] for p in leaf):.2f}, {len(leaf)} parameters): {leaf}")
        c.logger.info("==============================================")

    def optimize_subspace(self, names: list[str], bayes_tuple=None):
        new_opti, self.best_scores, self.first_scores, self.cv_obj, x_list, y_list = bayesian_optimisation(
            None,
            self.config,
            [self.parameters[name] for name in names],
            self.cv_obj,
            self.best_scores,
            self.first_scores,
            self.distance_map,
            samples=self.samples,
            samples_store_id=self.samples_store_id,
            raw_feature_matrix_store_id=self.raw_feature_matrix_store_id,
            n_pre_samples=None,
            get_bayes_tuple=True,
            bayes_tuple=bayes_tuple,
            relevance_log=self.relevance_log,
        )
        if new_opti:
            self.converged = False
        return x_list, y_list

    def ascend(self):
        parent_levels = int(getattr(self.config, "hpo_tree_parent_levels", 1))
        groups = [self.leaves[i:i + 2] for i in range(0, len(self.leaves), 2)] if parent_levels > 0 else [[leaf] for leaf in self.leaves]

        for group in groups:
            parent_names = list(dict.fromkeys(p for leaf in group for p in leaf))
            combined_x, combined_y = [], []
            for leaf in group:
                # The parameters outside the leaf keep their values while it runs, so its
                # history extends to the parent's space with this snapshot
                snapshot = {name: self.parameters[name].getValue(self.config) for name in parent_names}
                x_list, y_list = self.optimize_subspace(leaf)
                positions = {name: pos for pos, name in enumerate(leaf)}
                for x, y in zip(x_list, y_list):
                    combined_x.append(np.array([x[positions[name]] if name in positions else snapshot[name] for name in parent_names], dtype=float))
                    combined_y.append(y)

            if len(group) > 1 and len(parent_names) <= MAX_PARAMS:
                self.config.logger.info(f"Optimizing the parent of {len(group)} leaves over {len(parent_names)} parameters")
                self.optimize_subspace(parent_names, bayes_tuple=(combined_x, combined_y))

        return self.converged, self.best_scores, self.first_scores, self.cv_obj
