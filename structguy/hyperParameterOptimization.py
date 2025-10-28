import random
import time
import sys
import traceback
import ray
import math
import numpy as np
import sklearn.gaussian_process as gp
from scipy.stats import norm
from scipy.optimize import minimize
from sklearn.preprocessing import MinMaxScaler

from structguy import util, trainForest
from structguy.sampleSpace import DataSAIL_cv, CrossValidationSlice
from structman.base_utils.base_utils import pack, unpack

from ray.util.queue import Queue


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
        expected_improvement[sigma == 0.0] == 0.0

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
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):
        x0 = starting_point.reshape(1, -1)[0]
        # config.logger.info(f'In sample_next_hyperparameter, x0: {x0}, bounds: {bounds}')
        res = minimize(fun=acquisition_func, x0=x0, bounds=bounds, method="L-BFGS-B", args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

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


def bayes_random_init(
    config,
    parameters,
    initial_cv_obj: DataSAIL_cv,
    best_scores,
    best_first_scores,
    n_pre_samples,
    distance_map,
    slice_slices: dict[int, list[CrossValidationSlice]],
    samples=None,
    samples_store_id=None,
    fix_cat=True,
    debug=False,
    force_confusion=False,
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

    if config.multi_gpu > 1:
        #para_random_init = True
        store = ray.put((config, pack(initial_cv_obj), parameters, samples_store_id, pack(slice_slices), force_confusion, best_first_scores))
        return x_list, y_list, bounds, n_params, best_scores, best_first_scores, best_params, initial_values, new_optimimum, len(fix_parameters_pos), param_names, integer_type_params, initial_cv_obj, slice_slices, store
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

        store = ray.put((config, pack(initial_cv_obj), parameters, samples_store_id, pack(slice_slices), force_confusion, best_first_scores))

        config.logger.info(f"after store init, samples is None: {samples is None}")

        para_number = config.proc_n // config.multi_gpu

        current_params_id = 0
        remote_processes = []
        for gpu_id in range(config.multi_gpu):
            com_queue = Queue()
            out_queue = Queue()
            com_queue.put((randomized_parameters[current_params_id]))
            current_params_id += 1
            proc_id = para_eval.remote(com_queue, out_queue, store, para_number, gpu_id)
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
            for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
                for pos, para_value in enumerate(params):
                    parameters[pos].setValue(config, para_value)
                scores, cv_obj, slice_slices, first_scores = get_scores(
                    config,
                    cv_obj,
                    distance_map,
                    slice_slices,
                    samples=samples,
                    samples_store_id=samples_store_id,
                    force_confusion=force_confusion,
                    get_first_scores=True,
                    cv_interuption=(0.95, best_first_scores)
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
    return x_list, y_list, bounds, n_params, best_scores, best_first_scores, best_params, initial_values, new_optimimum, len(fix_parameters_pos), param_names, integer_type_params, initial_cv_obj, slice_slices, store


# Taken from https://github.com/thuijskens/bayesian-optimization
# Changed to match the specific problem
def bayesian_optimisation(
    n_iters,
    config,
    parameters,
    cv_obj: DataSAIL_cv,
    best_scores,
    best_first_scores,
    distance_map,
    slice_slices: dict[int, list[CrossValidationSlice]],
    samples=None,
    samples_store_id=None,
    n_pre_samples=5,
    gp_params=None,
    random_search=False,
    alpha=1e-6,
    epsilon=1e-9,
    debug=False,
    force_confusion=False,
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
            Variance of the error term of the GP.
        epsilon: double.
            Precision tolerance for floats.
    """

    n_fixed_params = 1

    x_list: list[np.ndarray]

    # while n_fixed_params > 0:
    
    x_list, y_list, bounds, n_params, best_scores, best_first_scores, best_params, initial_values, new_optimimum, n_fixed_params, param_names, integer_type_params, cv_obj, slice_slices, store = bayes_random_init(
        config,
        parameters,
        cv_obj,
        best_scores,
        best_first_scores,
        n_pre_samples,
        distance_map,
        slice_slices,
        samples=samples,
        samples_store_id=samples_store_id,
        fix_cat=False,
        debug=debug,
        force_confusion=force_confusion,
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
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel, alpha=alpha, n_restarts_optimizer=10, normalize_y=True)

    scaled_xp: np.ndarray = scaler.transform(xp)

    if n_iters is None:
        n_iters = 2 ** (n_params + 1)
    if debug:
        n_iters = 4

    if config.multi_gpu > 1:
        n_iters = max([n_params*2, 2])

    return_cv_obj = cv_obj
    return_slice_slices = slice_slices

    count_dups = 0

    if config.verbosity >= 1:
        config.logger.info(f"Number of bayesian optimization iterations: {n_iters}")

    if config.multi_gpu < 1:

        for n in range(n_iters):
            if config.verbosity >= 2:
                tl0 = time.time()
            try:
                model.fit(scaled_xp, yp)
            except:
                config.logger.info(f"{xp=}, {yp=}")
                config.logger.info(f"{x_list=}, {y_list=}")
                raise "None in Input"

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
                next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
                count_dups += 1
            else:
                next_sample = scaler.inverse_transform([next_sample])[0]

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

            scores, cv_obj, slice_slices, first_scores = get_scores(
                config,
                cv_obj,
                distance_map,
                slice_slices,
                samples=samples,
                samples_store_id=samples_store_id,
                force_confusion=force_confusion,
                debug=debug,
                get_first_scores=True,
                cv_interuption=(0.95, best_first_scores)
            )

            if config.verbosity >= 2:
                tl5 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 5: {tl5 - tl4}")

            cv_score = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)

            if config.verbosity >= 2:
                tl6 = time.time()
                config.logger.info(f"Bayesian optimisation loop part 6: {tl6 - tl5}")

            if config.verbosity >= 1:
                config.logger.info(f"Bayesian optimization, iteration: {n}")
                config.logger.info(f"Objective score: {cv_score}, unpenalized: {scores.objective_value(config)}")

            if util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty):
                best_scores = scores
                best_first_scores = first_scores
                best_params = next_sample
                new_optimimum = True
                config.logger.info("===========================\nFound new optimum\n===\n")
                config.logParameter()
                scores.printOut(config=config)
                config.logger.info("===========================")
                return_cv_obj = cv_obj
                return_slice_slices = slice_slices
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
        threads_per_gpu = 4
        para_number = max([1,config.proc_n // (config.multi_gpu * threads_per_gpu)])
        remote_function = para_eval.options(num_gpus = 1/threads_per_gpu)
        current_params_id = 0
        n_of_sent_hpo_sets = 0
        for gpu_id in range(config.multi_gpu):
            for _ in range(threads_per_gpu):
                next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])

                com_queue = Queue()
                out_queue = Queue()
                com_queue.put((next_sample))
                n_of_sent_hpo_sets += 1
                
                proc_id = remote_function.remote(com_queue, out_queue, store, para_number, gpu_id)
                remote_processes.append((com_queue, out_queue, proc_id))

        dones = []
        counts = []
        for i in range(len(remote_processes)):
            dones.append(False)
            counts.append(0)
        all_done = False
        
        append_counter = 0

        while not all_done:
            for i, (com_queue, out_queue, proc_id) in enumerate(remote_processes):
 
                if not out_queue.empty():
                    (scores, next_sample, first_scores) = out_queue.get(timeout=5)
                    n_of_sent_hpo_sets -= 1
                elif current_params_id >= n_iters:
                    if not dones[i]:
                        com_queue.put(None)
                        dones[i] = True
                    continue
                else:
                    continue

                if isinstance(first_scores, float):
                    cv_score = first_scores
                else:
                    cv_score = util.get_objective_score(config, scores, feature_penalty=config.feature_penalty)

                if config.verbosity >= 1:
                    config.logger.info(f"Bayesian optimization, iteration: gpu_id: {i} {counts[i]} {current_params_id=}")
                    counts[i] += 1
                    config.logger.info(f"Objective score: {cv_score}, unpenalized: {scores.objective_value(config)} {isinstance(first_scores, float)}")

                
                if (not isinstance(first_scores, float)) and util.objective_function_criterium(config, scores, best_scores, feature_penalty=config.feature_penalty):
                    best_scores = scores
                    best_first_scores = first_scores
                    best_params = next_sample
                    new_optimimum = True
                    config.logger.info("===========================\nFound new optimum\n===\n")
                    for pos, para_value in enumerate(best_params):
                        parameters[pos].setValue(config, para_value)
                    config.logParameter()
                    scores.printOut(config=config)
                    config.logger.info("===========================")
                    return_cv_obj = cv_obj
                    return_slice_slices = slice_slices
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
                            model.fit(scaled_xp, yp)
                        except:
                            config.logger.info(f"{xp=}, {yp=}")
                            config.logger.info(f"{x_list=}, {y_list=}")
                            raise "None in Input"

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
                            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
                            count_dups += 1
                        else:
                            next_sample = scaler.inverse_transform([next_sample])[0]
                        current_params_id += 1
                        append_counter = 0
                    else:
                        next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
                    com_queue.put(next_sample)
                    n_of_sent_hpo_sets += 1
                    
            all_done = True
            for done in dones:
                if not done:
                    all_done = False
            if n_of_sent_hpo_sets > 0:
                all_done = False
            if not all_done:
                time.sleep(0.5)

    if new_optimimum:
        for pos, para_value in enumerate(best_params):
            parameters[pos].setValue(config, para_value)
            config.logger.info(f"===========Found new optimum by setting {param_names[pos]}, to {para_value} ==============")
    else:
        for pos, para_value in enumerate(initial_values):
            parameters[pos].setValue(config, para_value)
    if get_bayes_tuple:
        return new_optimimum, best_scores, best_first_scores, return_cv_obj, return_slice_slices, x_list, y_list

    return new_optimimum, best_scores, best_first_scores, return_cv_obj, return_slice_slices


def max_to_n_of_features(limits, n_of_features):
    if limits[1] != "max":
        return limits
    else:
        return [limits[0], n_of_features]


class Parameter:
    def __init__(self, name, param_type, half_step_limits=None, possible_values=None, regression_specific=False, classification_specific=False, transform_limits=None):
        self.name = name
        self.half_step_limits = half_step_limits
        self.possible_values = possible_values
        self.param_type = param_type
        self.regression_specific = regression_specific
        self.classification_specific = classification_specific

        if self.half_step_limits is None:
            self.half_step_limits = [0, (len(self.possible_values) - 1)]
        elif transform_limits is not None:
            transform_function, additional_args = transform_limits
            self.half_step_limits = transform_function(half_step_limits, additional_args)

    def setValue(self, config, val):
        if self.param_type == "categorical" and not isinstance(val, str):
            val = self.possible_values[round(val)]
        config.setByString(self.name, val)

    def getValue(self, config):
        if self.param_type == "categorical":
            category = config.getByString(self.name)
            for pos, cat in enumerate(self.possible_values):
                if cat == category:
                    val = pos
        else:
            val = config.getByString(self.name)
        return val

def twoDim(
        parameter_1,
        parameter_2,
        best_scores,
        first_scores,
        config,
        score_matrix,
        cv_obj,
        distance_map,
        slice_slices, 
        samples=None,
        samples_store_id=None,
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
                slice_slices,
                samples=samples,
                samples_store_id=samples_store_id,
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
                slice_slices,
                samples=samples,
                samples_store_id=samples_store_id,
                debug=debug)

    return bayesian_optimisation(
        None,
        config,
        [parameter_1, parameter_2],
        cv_obj,
        best_scores,
        first_scores,
        distance_map,
        slice_slices,
        n_pre_samples=None,
        samples=samples,
        samples_store_id=samples_store_id,
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
        slice_slices,
        samples=None,
        samples_store_id=None,
        force_confusion=False,
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
            slice_slices,
            samples=samples,
            samples_store_id=samples_store_id,
            force_confusion=force_confusion,
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
                slice_slices,
                samples=samples,
                samples_store_id=samples_store_id,
                force_confusion=force_confusion,
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
                slice_slices,
                samples=samples,
                samples_store_id=samples_store_id,
                force_confusion=force_confusion,
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
                slice_slices,
                samples=samples,
                samples_store_id=samples_store_id,
                force_confusion=force_confusion,
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
        slice_slices,
        n_pre_samples=None,
        samples=samples,
        samples_store_id=samples_store_id,
        force_confusion=force_confusion,
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
        slice_slices,
        samples=None,
        debug=False,
        samples_store_id=None,
        force_confusion=False):
    config.logger.info(f"bayesian optimization and Cat {parameter_1.name}")

    new_optimimum = False
    best_parameter_value_1 = parameter_1.getValue(config)
    best_parameter_values = []
    for param in parameters:
        config.logger.info(param.name)
        best_parameter_values.append(param.getValue(config))

    for parameter_value_1 in parameter_1.possible_values:
        parameter_1.setValue(config, parameter_value_1)

        bay_optimimum, scores, fscores, cv_obj, slice_slices = bayesian_optimisation(
            None,
            config,
            parameters,
            cv_obj,
            best_scores,
            first_scores,
            distance_map,
            slice_slices,
            n_pre_samples=None,
            samples=samples,
            samples_store_id=samples_store_id,
            force_confusion=force_confusion,
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

    return new_optimimum, best_scores, first_scores, cv_obj, slice_slices


def cat3D(parameter_1, parameter_2, parameter_3, best_scores, config, score_matrix, cv_obj, distance_map, samples=None, samples_store_id=None):
    # Todo when we get at least 3 categorical features
    return


def get_scores(
        config,
        cv_obj,
        distance_map,
        slice_slices,
        samples=None,
        samples_store_id=None,
        remote=True,
        para_number=None,
        debug=False,
        force_confusion=False,
        get_first_scores=False,
        cv_interuption=None,
        gpu_id = None
        ):
    
    t0 = time.time()
    if samples is None:
        samples = unpack(ray.get(samples_store_id))
    _, scores, cv_obj, slice_slices = trainForest.trainForest(
        config,
        cv_obj,
        samples=samples,
        samples_store_id=samples_store_id,
        slice_slices=slice_slices,
        distance_map=distance_map,
        repeat=config.repeat_training,
        cv_repeat=config.cv_hpo,
        remote=remote,
        para_number=para_number,
        debug=debug,
        force_confusion=force_confusion,
        get_first_scores=get_first_scores,
        cv_interuption=cv_interuption,
        gpu_id = gpu_id
    )
    t1 = time.time()
    config.logger.info(f"Time for training forest in get_scores: {t1 - t0}")
    
    if get_first_scores:
        first_scores, scores = scores
        return scores, cv_obj, slice_slices, first_scores
    return scores, cv_obj, slice_slices


@ray.remote(max_calls=1, num_gpus = 1)
def para_eval(com_queue: Queue, out_queue: Queue, store, para_number, gpu_id):
    (config, packed_cv_obj, parameters, samples_store_id, packed_slice_slices, force_confusion, best_first_scores) = store
    cv_obj = unpack(packed_cv_obj)
    slice_slices = unpack(packed_slice_slices)

    gpu_id = ray.get_runtime_context().get_accelerator_ids()["GPU"][0]

    not_done = True
    while not_done:
        if com_queue.empty():
            time.sleep(0.5)
            continue
        try:
            params = com_queue.get(timeout=180)
        except:
            return
        if params is None:
            return

        for pos, para_value in enumerate(params):
            parameters[pos].setValue(config, para_value)
        scores, cv_obj, slice_slices, first_scores = get_scores(
            config,
            cv_obj,
            None,
            slice_slices,
            samples_store_id=samples_store_id,
            remote=False,
            para_number=para_number,
            force_confusion=force_confusion,
            get_first_scores=True,
            cv_interuption=(0.95, best_first_scores),
            gpu_id = gpu_id)
        
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
            parameters["learning_rate"] = Parameter("learning_rate", "real", half_step_limits=[0.0, 2.0])
            parameters["learning_rate_1"] = Parameter("learning_rate_1", "real", half_step_limits=[0.0, 2.0])
        if config.forest_type == "xgboost":
            parameters["early_stopping"] = Parameter("early_stopping", "integer", half_step_limits=[1, 1000])
            parameters["min_child_weight"] = Parameter("min_child_weight", "real", half_step_limits=[0., 100.])
            parameters["xgb_gamma"] = Parameter("xgb_gamma", "real", half_step_limits=[0.,10.])
            parameters["xgb_alpha"] = Parameter("xgb_alpha", "real", half_step_limits=[0.,5.])
            parameters["xgb_lambda"] = Parameter("xgb_lambda", "real", half_step_limits=[0.,5.])
            parameters["colsample_bytree"] = Parameter("colsample_bytree", "real", half_step_limits=[0.,1.])
            parameters["colsample_bylevel"] = Parameter("colsample_bylevel", "real", half_step_limits=[0.,1.])
            parameters["colsample_bynode"] = Parameter("colsample_bynode", "real", half_step_limits=[0.,1.])
            parameters["max_delta_step"] = Parameter("max_delta_step", "real", half_step_limits=[0.,50.])
            parameters["feat_impact_thresh"] = Parameter("feat_impact_thresh", "real", half_step_limits=[-0.01,0.01])
            parameters["tree_depth"] = Parameter("tree_depth", "integer", half_step_limits=[1,31])
            parameters["num_of_trees"] = Parameter("num_of_trees", "integer", half_step_limits=[10,10_000])
            parameters["max_cat_to_onehot"] = Parameter("max_cat_to_onehot", "integer", half_step_limits=[1,500])
            parameters["max_cat_threshold"] = Parameter("max_cat_threshold", "integer", half_step_limits=[1,100])

            parameters["early_stopping_1"] = Parameter("early_stopping_1", "integer", half_step_limits=[1, 1000])
            parameters["min_child_weight_1"] = Parameter("min_child_weight_1", "real", half_step_limits=[0., 100.])
            parameters["xgb_gamma_1"] = Parameter("xgb_gamma_1", "real", half_step_limits=[0.,10.])
            parameters["xgb_alpha_1"] = Parameter("xgb_alpha_1", "real", half_step_limits=[0.,5.])
            parameters["xgb_lambda_1"] = Parameter("xgb_lambda_1", "real", half_step_limits=[0.,5.])
            parameters["colsample_bytree_1"] = Parameter("colsample_bytree_1", "real", half_step_limits=[0.,1.])
            parameters["colsample_bylevel_1"] = Parameter("colsample_bylevel_1", "real", half_step_limits=[0.,1.])
            parameters["colsample_bynode_1"] = Parameter("colsample_bynode_1", "real", half_step_limits=[0.,1.])
            parameters["max_delta_step_1"] = Parameter("max_delta_step_1", "real", half_step_limits=[0.,50.])
            parameters["tree_depth_1"] = Parameter("tree_depth_1", "integer_1", half_step_limits=[1,31])
            parameters["num_of_trees_1"] = Parameter("num_of_trees_1", "integer_1", half_step_limits=[10,10_000])
            parameters["max_cat_to_onehot_1"] = Parameter("max_cat_to_onehot_1", "integer", half_step_limits=[1,500])
            parameters["max_cat_threshold_1"] = Parameter("max_cat_threshold_1", "integer", half_step_limits=[1,100])

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


def threeDimHyperOptimization(
    config: util.Config,
    cv_obj: DataSAIL_cv,
    best_scores: util.Scores,
    first_scores: util.Scores,
    slice_slices: dict[int, list[CrossValidationSlice]],
    samples=None,
    samples_store_id=None,
    distance_map=None,
    debug=False
):
    util.set_estimation_delta(config, first_scores, best_scores)
    random.seed()
    fss_parameters, fs_parameters, parameters = initParameters(config, split_fs_parameters=True)
    converged = False
    n = 1
    score_matrix = {}

    while not converged:
        converged = True
        if n > 1:
            cv_obj.reset_confusion_maps()

        if not debug:
            for param in [fss_parameters]:
                param_names = list(param.keys())
                random.shuffle(param_names)

                while len(param_names) > 2:
                    param_trio = param_names.pop(), param_names.pop(), param_names.pop()
                    new_opti, best_scores, first_scores, cv_obj, slice_slices = threeDim(
                        param[param_trio[0]],
                        param[param_trio[1]],
                        param[param_trio[2]],
                        best_scores,
                        first_scores,
                        config,
                        score_matrix,
                        cv_obj,
                        distance_map,
                        slice_slices,
                        samples=samples,
                        samples_store_id=samples_store_id,
                        force_confusion=True,
                        debug=debug,
                    )
                    if new_opti:
                        converged = False

            cv_obj.reset_confusion_maps()

            if len(fs_parameters) > 1:
                new_opti, best_scores, first_scores, cv_obj, slice_slices = bayesian_optimisation(
                    None,
                    config,
                    list(fs_parameters.values()),
                    cv_obj,
                    best_scores,
                    first_scores,
                    distance_map,
                    slice_slices,
                    samples=samples,
                    samples_store_id=samples_store_id,
                    n_pre_samples=None,
                    debug=debug,
                )
                if new_opti:
                    converged = False

        for param in [parameters]:
            param_names = list(param.keys())
            random.shuffle(param_names)

            if "confusion_rank_threshold" in fs_parameters:
                while len(param_names) > 2:
                    param_set = [
                        parameters[param_names.pop()],
                        parameters[param_names.pop()],
                        parameters[param_names.pop()],
                        fs_parameters["confusion_rank_threshold"]
                    ]
                    new_opti, best_scores, first_scores, cv_obj, slice_slices = bayesian_optimisation(
                        None,
                        config,
                        param_set,
                        score_matrix,
                        cv_obj,
                        best_scores,
                        first_scores,
                        distance_map,
                        slice_slices,
                        samples=samples,
                        samples_store_id=samples_store_id,
                        n_pre_samples=None,
                        debug=debug,
                    )
                    if new_opti:
                        converged = False

                while len(param_names) > 1:
                    param_trio = param_names.pop(), param_names.pop()  # , 'confusion_rank_threshold'#param_names.pop()
                    new_opti, best_scores, first_scores, cv_obj, slice_slices = threeDim(
                        param[param_trio[0]],
                        param[param_trio[1]],
                        fs_parameters["confusion_rank_threshold"],
                        best_scores,
                        first_scores,
                        config,
                        score_matrix,
                        cv_obj,
                        distance_map,
                        slice_slices,
                        samples=samples,
                        samples_store_id=samples_store_id,
                        debug=debug,
                    )
                    if new_opti:
                        converged = False

                while len(param_names) == 1:
                    new_opti, best_scores, first_scores, cv_obj, slice_slices = twoDim(
                        param[param_names.pop()],
                        fs_parameters["confusion_rank_threshold"],
                        best_scores,
                        first_scores,
                        config,
                        score_matrix,
                        cv_obj,
                        distance_map,
                        slice_slices,
                        samples=samples,
                        samples_store_id=samples_store_id,
                        debug=debug,
                    )
                    if new_opti:
                        converged = False
            else:
                """
                while len(param_names) > 4:
                    param_set = [
                        parameters[param_names.pop()],
                        parameters[param_names.pop()],
                        parameters[param_names.pop()],
                        parameters[param_names.pop()],
                        parameters[param_names.pop()]
                    ]
                    new_opti, best_scores, first_scores, cv_obj, slice_slices = bayesian_optimisation(
                        None,
                        config,
                        param_set,
                        cv_obj,
                        best_scores,
                        first_scores,
                        distance_map,
                        slice_slices,
                        samples=samples,
                        samples_store_id=samples_store_id,
                        n_pre_samples=None,
                        debug=debug,
                    )
                    if new_opti:
                        converged = False

                param_set = [parameters[param] for param in param_names]
                new_opti, best_scores, first_scores, cv_obj, slice_slices = bayesian_optimisation(
                    None,
                    config,
                    param_set,
                    cv_obj,
                    best_scores,
                    first_scores,
                    distance_map,
                    slice_slices,
                    samples=samples,
                    samples_store_id=samples_store_id,
                    n_pre_samples=None,
                    debug=debug,
                )
                if new_opti:
                    converged = False
                """
                SDTree = SubdimensionTree(
                    param_names,
                    parameters,
                    config,
                    cv_obj,
                    best_scores,
                    first_scores,
                    slice_slices,
                    samples,
                    samples_store_id,
                    distance_map = distance_map)

                converged, best_scores, first_scores, cv_obj, slice_slices = SDTree.ascend()

        config.logger.info(f"Iteration: {n}")
        config.logParameter()
        config.saveHyperParameter(f"hyperparameters_ThreeDim_epoch_{n}.conf")
        if best_scores is not None:
            best_scores.printOut(config=config)
        n += 1
        # slice_slices = None

    cv_obj.reset_confusion_maps()
    return


def bayesianComplete(
    config,
    cv_obj: DataSAIL_cv,
    best_scores: util.Scores,
    slice_slices: dict[int, list[CrossValidationSlice]],
    samples=None,
    samples_store_id=None,
    distance_map=None,
    debug=False
    ):

    parameters = initParameters(config, do_feat_selection=False)

    new_optimimum, best_scores, cv_obj, slice_slices = bayesian_optimisation(
        9_999_999,
        config,
        [parameters[p] for p in parameters],
        cv_obj,
        best_scores,
        distance_map,
        slice_slices,
        samples=samples,
        samples_store_id=samples_store_id,
        n_pre_samples=100,
        debug=debug,
        store_params = True
        )

    config.logger.info("Bayesian optimization finished")
    config.logParameter()
    best_scores.printOut()

    return

class SubdimensionNode:
    def __init__(self, param_names: list[str], parent, tree):
        self.parent = parent
        self.tree = tree
        self.param_names = param_names
        if len(param_names) > 2:
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
            new_opti, self.tree.best_scores, self.tree.first_scores, self.tree.cv_obj, self.tree.slice_slices, x_list, y_list = bayesian_optimisation(
                None,
                self.tree.config,
                param_set,
                self.tree.cv_obj,
                self.tree.best_scores,
                self.tree.first_scores,
                self.tree.distance_map,
                self.tree.slice_slices,
                samples=self.tree.samples,
                samples_store_id=self.tree.samples_store_id,
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

            if len(self.param_names) > 8:
                return combined_x, combined_y

            param_set = [self.tree.parameters[param] for param in self.param_names]
            new_opti, self.tree.best_scores, self.tree.first_scores, self.tree.cv_obj, self.tree.slice_slices, x_list, y_list = bayesian_optimisation(
                None,
                self.tree.config,
                param_set,
                self.tree.cv_obj,
                self.tree.best_scores,
                self.tree.first_scores,
                self.tree.distance_map,
                self.tree.slice_slices,
                samples=self.tree.samples,
                samples_store_id=self.tree.samples_store_id,
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
            slice_slices: dict[int, list[CrossValidationSlice]],
            samples,
            samples_store_id,
            distance_map = None
            ):
        self.parameters = parameters
        self.config = config
        self.cv_obj = cv_obj
        self.best_scores = best_scores
        self.first_scores = first_scores
        self.slice_slices = slice_slices
        self.samples = samples
        self.samples_store_id = samples_store_id
        self.distance_map = distance_map
        random.shuffle(param_names)
        self.root = SubdimensionNode(param_names, None, self)
        self.converged = True
        
    def ascend(self):
        self.root.optimize()
        return self.converged, self.best_scores, self.first_scores, self.cv_obj, self.slice_slices