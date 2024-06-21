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
from structguy.sampleSpace import DataSAIL_cv
from structman.base_utils.base_utils import pack, unpack

#Taken from https://github.com/thuijskens/bayesian-optimization
def expected_improvement(x, gaussian_process, evaluated_loss, greater_is_better=False, n_params=1):
    """ expected_improvement
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
    with np.errstate(divide='ignore'):
        Z = scaling_factor * (mu - loss_optimum) / sigma
        expected_improvement = scaling_factor * (mu - loss_optimum) * norm.cdf(Z) + sigma * norm.pdf(Z)
        expected_improvement[sigma == 0.0] == 0.0

    return -1 * expected_improvement

#Taken from https://github.com/thuijskens/bayesian-optimization
def sample_next_hyperparameter(acquisition_func, gaussian_process, evaluated_loss, greater_is_better=False,
                               bounds=(0, 10), n_restarts=25):
    """ sample_next_hyperparameter
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
    best_x = None
    best_acquisition_value = 1
    n_params = bounds.shape[0]

    for starting_point in np.random.uniform(bounds[:, 0], bounds[:, 1], size=(n_restarts, n_params)):

        x0 = starting_point.reshape(1, -1)[0]
        #print(f'In sample_next_hyperparameter, x0: {x0}, bounds: {bounds}')
        res = minimize(fun=acquisition_func,
                       x0=x0,
                       bounds=bounds,
                       method='L-BFGS-B',
                       args=(gaussian_process, evaluated_loss, greater_is_better, n_params))

        if res.fun < best_acquisition_value:
            best_acquisition_value = res.fun
            best_x = res.x

    return best_x

def fill_plane(parameter_values, integer_type_params, density = 20):
    if len(integer_type_params) == 0:
        return []

    projected_parameter_values = [[]]

    for pos, parameter_value in enumerate(parameter_values):
        if pos not in integer_type_params:
            projections = [parameter_value]
        else:
            l = int(parameter_value)
            projections = []
            for i in range(density+1):
                projections.append(l + (i/density))

        new_projected_parameter_values = []
        for projected_value in projections:
            for current_parameter_values in projected_parameter_values:
                new_projected_parameter_values.append(current_parameter_values + [projected_value])
        projected_parameter_values = new_projected_parameter_values

    return projected_parameter_values

def bayes_random_init(config, parameters, score_matrix, initial_cv_obj, best_scores, n_pre_samples, distance_map, slice_slices, samples = None, samples_store_id = None, fix_cat = True, debug = False, force_confusion = False):
    param_names = [p.name for p in parameters]

    bounds = np.array([p.half_step_limits for p in parameters])

    initial_values = [p.getValue(config) for p in parameters]

    integer_type_params = set()

    for parameter_number, parameter in enumerate(parameters):
        if parameter.param_type == 'integer':
            integer_type_params.add(parameter_number)

    best_params = initial_values

    x_list = [np.array(initial_values)]
    y_list = [best_scores.objective_value(config)]

    n_params = len(param_names)

    new_optimimum = False

    para_eval_ret_ids = []

    if n_pre_samples is None:
        n_pre_samples = min([config.proc_n, 2**n_params])

    #n_pre_samples = 1 #Just for testing

    sample_size_threshold = config.gigs_of_ram * 300
    n_of_samples = len(initial_cv_obj.slices[0].train_targets) + len(initial_cv_obj.slices[0].test_targets)

    if n_of_samples < sample_size_threshold:
        para_random_init = True
    else:
        para_random_init = False

    print('bayesian optimization:',param_names,n_pre_samples,para_random_init)
    print('Current best scores:')
    best_scores.printOut()
    print(f'Objective score: {best_scores.objective_value(config)}')

    t0 = time.time()

    #if not 'geometric_exponent' in param_names:
    if para_random_init:
        randomized_parameters = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0]))
        max_packages = max([2, 8 * (sample_size_threshold//n_of_samples)])
        packagesize = math.ceil((n_pre_samples-1) / max_packages)

        #Always calculate one set of parameters unparalized to set the confusion maps (or other slice specific stuff that resets after each round of the HPO)
        init_params = randomized_parameters[0]
        for pos,para_value in enumerate(init_params):
            parameters[pos].setValue(config, para_value)

        if debug:
            print(f'Init params: {init_params}')

        scores, cv_obj, slice_slices = get_scores(config, score_matrix, initial_cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, debug = debug, force_confusion = force_confusion)
        results = [(scores, init_params, cv_obj)]

        store = ray.put((config, pack(cv_obj.serialize()), parameters, samples_store_id, slice_slices, force_confusion))

        print(f'after store init, samples is None: {samples is None}, max_packages: {max_packages}, package_size: {packagesize}')

        para_number = config.proc_n//max_packages

        # Get n_pre_samples amount of random points
        try:
            package = []
            for params in randomized_parameters[1:]:
                package.append(params)
                if len(package) == packagesize:
                    para_eval_ret_ids.append(para_eval.remote(package, store, para_number))
                    package = []
            if len(package) > 0:
                para_eval_ret_ids.append(para_eval.remote(package, store, para_number))
        except:
            [e, f, g] = sys.exc_info()
            g = traceback.format_exc()
            print(f'ERROR in bayes_random_init: {n_pre_samples}, {bounds}\n{e}\n{f}\n{g}')
            sys.exit()

        print(f'Para random init started: # of packages: {len(para_eval_ret_ids)} # of subthreads: {para_number}')

        para_results_package = ray.get(para_eval_ret_ids)
        for para_results in para_results_package:
            for scores, params, packed_cv_obj in para_results:
                results.append((scores, params, packed_cv_obj))
    else:
        results = []
        try:
            cv_obj = initial_cv_obj
            for params in np.random.uniform(bounds[:, 0], bounds[:, 1], (n_pre_samples, bounds.shape[0])):
                for pos,para_value in enumerate(params):
                    parameters[pos].setValue(config, para_value)
                scores, cv_obj, slice_slices = get_scores(config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, force_confusion = force_confusion)
                results.append((scores, params, cv_obj))
                #projected_parameter_values = fill_plane(params, integer_type_params)
                #for projected_param in projected_parameter_values:
                #    results.append((scores, projected_param))
        except:
            [e, f, g] = sys.exc_info()
            g = traceback.format_exc()
            print(f'ERROR in bayes_random_init: {n_pre_samples}, {bounds}\n{e}\n{f}\n{g}')
            sys.exit()

    t1 = time.time()
    print(f'Time for BayesInit: {t1 - t0}')

    cat_param_map = {}

    return_cv_obj = initial_cv_obj
    for scores, params, cv_obj in results:
        obj_sc = scores.objective_value(config)
        if obj_sc is None or obj_sc != obj_sc:
            print('========= Warning: None or NaN objective score for:', param_names, params)
            scores = util.Scores(zero=True)
            obj_sc = scores.objective_value(config)

        x_list.append(params)
        y_list.append(obj_sc)

        #print('==DEBUG OUT==')
        #print(params)
        #print('Best score:',best_scores.objective_value(config))
        #print('Score:',obj_sc)
        #print('=============')

        if util.objective_function_criterium(config, scores, best_scores, feature_penalty = config.feature_penalty):
            best_scores = scores
            best_params = params
            new_optimimum = True
            if para_random_init:
                try:
                    cv_obj_list = unpack(cv_obj)
                    cv_obj = DataSAIL_cv(as_list = cv_obj_list)
                except:
                    #The first obj is not packed
                    pass
            return_cv_obj = cv_obj
            print('===========Found new optimum:=======================\n')
            print(params)
            scores.printOut()
            print('====================================================')
        elif config.verbosity >= 3:
            print(f'No new optimun ({util.get_objective_score(config, best_scores)}): {util.get_objective_score(config, scores)}')


        if fix_cat:
            for p_pos,parameter in enumerate(parameters):
                if not parameter.param_type == 'categorical':
                    continue
                if not parameter.name in cat_param_map:
                    cat_param_map[parameter.name] = {}
                val = round(params[p_pos])
                if not val in cat_param_map[parameter.name]:
                    cat_param_map[parameter.name][val] = []
                cat_param_map[parameter.name][val].append(obj_sc)

    print('Random init finished')
    fix_parameters_pos = []
    if fix_cat:

        for p_pos,parameter in enumerate(parameters):
            if not parameter.param_type == 'categorical':
                continue
            max_max_val = None
            max_median_val = None
            for val in cat_param_map[parameter.name]:
                max_sc = max(cat_param_map[parameter.name][val])
                median_sc = util.median(cat_param_map[parameter.name][val])
                if max_max_val == None:
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
                    print('Fix categorical feature',param_names[p_pos],'to',config.getByString(parameter.name))

        for p_pos in reversed(fix_parameters_pos):
            del parameters[p_pos]
    return x_list, y_list, bounds, n_params, best_scores, best_params, initial_values, new_optimimum, len(fix_parameters_pos), param_names, integer_type_params, return_cv_obj, slice_slices

# Taken from https://github.com/thuijskens/bayesian-optimization
# Changed to match the specific problem
def bayesian_optimisation(n_iters, config, parameters, score_matrix, cv_obj, best_scores, distance_map, slice_slices, samples = None, samples_store_id = None, n_pre_samples=5,
                          gp_params=None, random_search=False, alpha=1e-5, epsilon=1e-7, debug = False, force_confusion = False):
    """ bayesian_optimisation
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

    #while n_fixed_params > 0:
    x_list, y_list, bounds, n_params, best_scores, best_params, initial_values, new_optimimum, n_fixed_params, param_names, integer_type_params, cv_obj, slice_slices = bayes_random_init(config, parameters, score_matrix, cv_obj, best_scores, n_pre_samples, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, fix_cat = False, debug = debug, force_confusion = force_confusion)

    min_max_samples = [[],[]]
    for bound in bounds:
        min_max_samples[0].append(bound[0])
        min_max_samples[1].append(bound[1])

    scaled_bounds = np.array([[0.,1.]]*len(bounds))

    if config.verbosity >= 2:
        print(f'In bayesian_optimization, scaled_bounds: {scaled_bounds}')

    scaler = MinMaxScaler()
    scaler.fit(min_max_samples)

    xp = np.array(x_list)
    yp = np.array(y_list)

    # Create the GP
    if gp_params is not None:
        model = gp.GaussianProcessRegressor(**gp_params)
    else:
        kernel = gp.kernels.Matern()
        model = gp.GaussianProcessRegressor(kernel=kernel,
                                            alpha=alpha,
                                            n_restarts_optimizer=10,
                                            normalize_y=True)

    scaled_xp = scaler.transform(xp)

    if n_iters is None:
        n_iters = 2**(n_params+1)

    return_cv_obj = cv_obj
    return_slice_slices = slice_slices

    for n in range(n_iters):
        if config.verbosity >= 2:
            tl0 = time.time()
        try:
            model.fit(scaled_xp, yp)
        except:
            print(xp,yp)
            print(x_list,y_list)
            raise 'None in Input'

        if config.verbosity >= 2:
            tl1 = time.time()
            print(f'Bayesian optimisation loop part 1: {tl1-tl0}')

        # Sample next hyperparameter
        if random_search:
            x_random = np.random.uniform(bounds[:, 0], bounds[:, 1], size=(random_search, n_params))
            ei = -1 * expected_improvement(x_random, model, yp, greater_is_better=True, n_params=n_params)
            next_sample = x_random[np.argmax(ei), :]
        else:
            next_sample = sample_next_hyperparameter(expected_improvement, model, yp, greater_is_better=True, bounds=scaled_bounds, n_restarts=100)

        if config.verbosity >= 2:
            tl2 = time.time()
            print(f'Bayesian optimisation loop part 2: {tl2-tl1}')

        # Duplicates will break the GP. In case of a duplicate, we will randomly sample a next query point.
        if np.any(np.abs(next_sample - xp) <= epsilon):
            next_sample = np.random.uniform(bounds[:, 0], bounds[:, 1], bounds.shape[0])
        else:
            next_sample = scaler.inverse_transform([next_sample])[0]

        if config.verbosity >= 2:
            tl3 = time.time()
            print(f'Bayesian optimisation loop part 3: {tl3-tl2}')

        # Sample loss for new set of parameters
        for pos,para_value in enumerate(next_sample):
            parameters[pos].setValue(config, para_value)

        if config.verbosity >= 2:
            tl4 = time.time()
            print(f'Bayesian optimisation loop part 4: {tl4-tl3}')

        scores, cv_obj, slice_slices = get_scores(config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, force_confusion = force_confusion)

        if config.verbosity >= 2:
            tl5 = time.time()
            print(f'Bayesian optimisation loop part 5: {tl5-tl4}')

        cv_score = scores.objective_value(config)

        if config.verbosity >= 2:
            tl6 = time.time()
            print(f'Bayesian optimisation loop part 6: {tl6-tl5}')

        print('Bayesian optimization, iteration:',n)
        #print('===========================\nBayesian optimization, iteration:',n,'\n===\n')
        #config.printParameter()
        #scores.printOut()
        #print('===========================')

        #print('==DEBUG OUT==')
        #config.printParameter()
        #print('Best score:',best_scores.objective_value(config))
        #print('Score:',cv_score)
        #print('=============')

        if util.objective_function_criterium(config, scores, best_scores, feature_penalty = config.feature_penalty):
            best_scores = scores
            best_params = next_sample
            new_optimimum = True
            print('===========================\nFound new optimum\n===\n')
            config.printParameter()
            scores.printOut()
            print('===========================')
            return_cv_obj = cv_obj
            return_slice_slices = slice_slices
        elif config.verbosity >= 3:
            print(f'No new optimun ({util.get_objective_score(config, best_scores)}): {util.get_objective_score(config, scores)}')

        if cv_score is None or cv_score != cv_score:
            print(' === cv_score is None or Nan:',next_sample)
            scores = util.Scores(zero=True)
            cv_score = scores.objective_value(config)

        if config.verbosity >= 2:
            tl7 = time.time()
            print(f'Bayesian optimisation loop part 7: {tl7-tl6}')

        # Update lists
        x_list.append(next_sample)
        y_list.append(cv_score)

        #projected_parameter_values = fill_plane(next_sample, integer_type_params)
        #for projected_param in projected_parameter_values:
        #    x_list.append(projected_param)
        #    y_list.append(cv_score)

        # Update xp and yp
        xp = np.array(x_list)
        yp = np.array(y_list)
        scaled_xp = scaler.transform(xp)

        if config.verbosity >= 3:
            tl8 = time.time()
            print(f'Bayesian optimisation loop part 8: {tl8-tl7}')

    if new_optimimum:
        for pos, para_value in enumerate(best_params):
            parameters[pos].setValue(config, para_value)
            print('===========Found new optimum by setting',param_names[pos],'to',para_value,'==============')
    else:
        for pos, para_value in enumerate(initial_values):
            parameters[pos].setValue(config, para_value)

    return new_optimimum, best_scores, return_cv_obj, return_slice_slices

def max_to_n_of_features(limits, n_of_features):
    if limits[1] != 'max':
        return limits
    else:
        return [limits[0], n_of_features]

class Parameter:
    def __init__(self,name,param_type,half_step_limits = None,possible_values = None,regression_specific = False, classification_specific = False, transform_limits = None):
        self.name = name
        self.half_step_limits = half_step_limits
        self.possible_values = possible_values
        self.param_type = param_type
        self.regression_specific = regression_specific
        self.classification_specific = classification_specific

        if self.half_step_limits is None:
            self.half_step_limits = [0,(len(self.possible_values) -1)]
        elif transform_limits is not None:
            transform_function, additional_args = transform_limits
            self.half_step_limits = transform_function(half_step_limits, additional_args)

    def setValue(self, config, val):
        if self.param_type == 'categorical' and not isinstance(val,str):
            val = self.possible_values[round(val)]
        config.setByString(self.name,val)

    def getValue(self, config):
        if self.param_type == 'categorical':
            category = config.getByString(self.name)
            for pos,cat in enumerate(self.possible_values):
                if cat == category:
                    val = pos
        else:
            val = config.getByString(self.name)
        return val

def parametersInScoreMatrix(config, score_matrix):
    score_tuple = config.getScoreTuple()
    if score_tuple in score_matrix:
        return True
    else:
        return False

def getFromScoreMatrix(config, score_matrix):
    score_tuple = config.getScoreTuple()
    return score_matrix[score_tuple]

def addToScoreMatrix(scores_obj, config, score_matrix):
    score_tuple = config.getScoreTuple()
    score_matrix[score_tuple] = scores_obj
    return

def twoDim(parameter_1, parameter_2, best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = None, samples_store_id = None):
    cat_count = 0
    for p_type in [parameter_1.param_type, parameter_2.param_type]:
        if p_type == 'categorical':
            cat_count += 1

    if cat_count == 2:
        return #TODO when we have at least 2 categorical features

    if cat_count == 1:
        if parameter_1.param_type == 'categorical':
            return bayesianAndCat(parameter_1, [parameter_2], best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id)
        elif parameter_2.param_type == 'categorical':
            return bayesianAndCat(parameter_2, [parameter_1], best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id)

    return bayesian_optimisation(None, config, [parameter_1, parameter_2], score_matrix, cv_obj, best_scores, distance_map, slice_slices, n_pre_samples = None, samples = samples, samples_store_id = samples_store_id)

def threeDim(parameter_1, parameter_2, parameter_3, best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = None, samples_store_id = None, force_confusion = False):
    cat_count = 0
    for p_type in [parameter_1.param_type, parameter_2.param_type, parameter_3.param_type]:
        if p_type == 'categorical':
            cat_count += 1

    if cat_count == 3:
        return cat3D(parameter_1, parameter_2, parameter_3, best_scores,config,score_matrix,cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, force_confusion = force_confusion)

    if cat_count == 2:
        return #TODO when we have at least 2 categorical features

    if cat_count == 1:
        if parameter_1.param_type == 'categorical':
            return bayesianAndCat(parameter_1, [parameter_2, parameter_3], best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, force_confusion = force_confusion)
        elif parameter_2.param_type == 'categorical':
            return bayesianAndCat(parameter_2, [parameter_1, parameter_3], best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, force_confusion = force_confusion)
        elif parameter_3.param_type == 'categorical':
            return bayesianAndCat(parameter_3, [parameter_1, parameter_2], best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, force_confusion = force_confusion)

    return bayesian_optimisation(None, config, [parameter_1, parameter_2, parameter_3], score_matrix, cv_obj, best_scores, distance_map, slice_slices, n_pre_samples = None, samples = samples, samples_store_id = samples_store_id, force_confusion = force_confusion)

def bayesianAndCat(parameter_1, parameters, best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = None, samples_store_id = None, force_confusion = False):
    print('bayesian optimization and Cat', parameter_1.name)

    new_optimimum = False
    best_parameter_value_1 = parameter_1.getValue(config)
    best_parameter_values = []
    for param in parameters:
        print(param.name)
        best_parameter_values.append(param.getValue(config))

    for parameter_value_1 in parameter_1.possible_values:
        parameter_1.setValue(config,parameter_value_1)

        bay_optimimum, scores, cv_obj, slice_slices = bayesian_optimisation(None, config, parameters, score_matrix, cv_obj, best_scores, distance_map, slice_slices, n_pre_samples = None, samples = samples, samples_store_id = samples_store_id, force_confusion = force_confusion)

        if util.objective_function_criterium(config, scores, best_scores):
            best_scores = scores
            best_parameter_value_1 = parameter_1.getValue(config)
            best_parameter_values = []
            for param in parameters:
                best_parameter_values.append(param.getValue(config))
            new_optimimum = True

    parameter_1.setValue(config,best_parameter_value_1)
    for i,para_value in enumerate(best_parameter_values):
        parameters[i].setValue(config,para_value)

    return new_optimimum, best_scores, cv_obj, slice_slices

def cat3D(parameter_1, parameter_2, parameter_3, best_scores, config, score_matrix, cv_obj, distance_map, samples = None, samples_store_id = None):
    #Todo when we get at least 3 categorical features
    return


def get_scores(config, score_matrix, cv_obj, distance_map, slice_slices, samples = None, samples_store_id = None, remote = True, para_number = None, debug = False, force_confusion = False):
    if (score_matrix is not None) and parametersInScoreMatrix(config,score_matrix):
        scores = getFromScoreMatrix(config,score_matrix)
        print('Parameter set already known, skip scores calculation')
    else:
        t0 = time.time()
        _, scores, cv_obj, slice_slices = trainForest.trainForest(config, cv_obj, samples = samples, samples_store_id = samples_store_id, slice_slices = slice_slices, distance_map = distance_map, repeat = config.repeat_training,cv_repeat = config.cv_hpo, remote = remote, para_number = para_number, debug = debug, force_confusion = force_confusion)
        t1 = time.time()
        print(f'Time for training forest in get_scores: {t1-t0}')
        if score_matrix is not None:
            addToScoreMatrix(scores, config, score_matrix)
    return scores, cv_obj, slice_slices

@ray.remote(max_calls = 1)
def para_eval(package, store, para_number):
    (config, packed_cv_obj, parameters, samples_store_id, slice_slices, force_confusion) = store
    cv_obj_list = unpack(packed_cv_obj)
    cv_obj = DataSAIL_cv(as_list = cv_obj_list)
    returns = []
    for params in package:
        for pos,para_value in enumerate(params):
            parameters[pos].setValue(config, para_value)
        scores, cv_obj, slice_slices = get_scores(config, None, cv_obj, None, slice_slices, samples_store_id = samples_store_id, remote = False, para_number = para_number, force_confusion = force_confusion)
        packed_cv_obj = pack(cv_obj.serialize())
        returns.append((scores, params, packed_cv_obj))
    return returns

def initConfParameters(config, parameters):
    parameters['confusion_rank_threshold'] = Parameter('confusion_rank_threshold', 'integer', half_step_limits = config.confusion_rank_threshold_bounds, transform_limits = (max_to_n_of_features, config.n_of_features))
    parameters['confusion_goodwill'] = Parameter('confusion_goodwill', 'real', half_step_limits = config.confusion_goodwill_bounds)
    parameters['err_warping_exp'] = Parameter('err_warping_exp', 'real', half_step_limits = config.err_warping_exp_bounds)
    parameters['confusion_normalization_exp'] = Parameter('confusion_normalization_exp', 'real', half_step_limits= config.confusion_normalization_exp_bounds)

    return parameters

def initFSForestParameters(config, parameters):
    parameters['fs_tree_depth'] = Parameter('fs_tree_depth','integer',half_step_limits = config.tree_depth_half_step)
    parameters['fs_num_of_trees'] = Parameter('fs_num_of_trees','integer',half_step_limits = config.forest_size_half_step)
    parameters['fs_min_impurity_decrease_exp'] = Parameter('fs_min_impurity_decrease_exp','real',half_step_limits = config.min_impurity_decrease_exp_half_step)
    parameters['fs_min_sample_split'] = Parameter('fs_min_sample_split','integer',half_step_limits = config.min_sample_split_half_step)
    parameters['fs_tree_min_leaf_samples'] = Parameter('fs_tree_min_leaf_samples','integer',half_step_limits = config.min_sample_leaf_half_step)
    parameters['fs_max_sample_parameter'] = Parameter('fs_max_sample_parameter','real',half_step_limits = config.max_sample_half_step)

    return parameters

def initReguParameters(config, parameters):
    #parameters['reg_thresh_exp'] = Parameter('reg_thresh_exp','real',half_step_limits = config.reg_thresh_exp_half_step)
    if config.regression:
        parameters['reg_alpha_exp'] = Parameter('reg_alpha_exp','real',half_step_limits = config.reg_alpha_exp_half_step)
    else:
        parameters['reg_c_exp'] = Parameter('reg_c_exp','real',half_step_limits = config.reg_c_exp_half_step)
    return parameters

def initMeanCorrParameters(config, parameters):
    parameters['tvmb_rank_threshold'] = Parameter('tvmb_rank_threshold','integer',half_step_limits = config.tvmb_rank_half_step, transform_limits = (max_to_n_of_features, config.n_of_features))
    #parameters['tvpmb_rank_threshold'] = Parameter('tvpmb_rank_threshold','integer',half_step_limits = config.tvpmb_rank_half_step, transform_limits = (max_to_n_of_features, config.n_of_features))
    parameters['p_val_thresh'] = Parameter('p_val_thresh','real',half_step_limits = config.p_val_thresh_half_step, transform_limits = (max_to_n_of_features, config.n_of_features))
    return parameters

def initParameters(config, do_feat_selection = True, do_forest_param = True, do_sample_weighting = True, split_fs_parameters = False):
    parameters = {}
    fs_parameters = {}
    fss_parameters = {}
    if config.hpo_do_feat_selection:
        if config.feature_selection == 'meanCorrelation':
            fs_parameters = initMeanCorrParameters(config, fs_parameters)
        if config.feature_selection == 'regularization':
            fs_parameters = initReguParameters(config, fs_parameters)

        if config.feature_selection == 'double':
            fs_parameters = initMeanCorrParameters(config, fs_parameters)
            fs_parameters = initReguParameters(config, fs_parameters)

        elif config.feature_selection == 'confusion' or config.feature_selection == 'sequential_confusion' or config.feature_selection == 'sequential_confusion_and_regu' or config.feature_selection == 'confusion_and_regu':
            fs_parameters = initConfParameters(config, fs_parameters)
            fss_parameters = initFSForestParameters(config, fss_parameters)

            if config.feature_selection == 'sequential_confusion' or config.feature_selection == 'sequential_confusion_and_regu':
                fs_parameters['sequential_confusion_rank_threshold'] = Parameter('sequential_confusion_rank_threshold', 'integer', half_step_limits = config.sequential_confusion_rank_threshold_bounds, transform_limits = (max_to_n_of_features, config.n_of_features))
            if config.feature_selection == 'sequential_confusion_and_regu' or config.feature_selection == 'confusion_and_regu':
                fs_parameters = initReguParameters(config, fs_parameters)

        elif config.feature_selection == 'threeStaged':
            fs_parameters = initConfParameters(config, fs_parameters)
            fs_parameters = initMeanCorrParameters(config, fs_parameters)
            fs_parameters = initReguParameters(config, fs_parameters)
        elif config.feature_selection == 'threeStaged_listranking':
            fs_parameters = initReguParameters(config, fs_parameters)
            fs_parameters['confusion_goodwill'] = Parameter('confusion_goodwill', 'real', half_step_limits = config.confusion_goodwill_bounds)
            fs_parameters['err_warping_exp'] = Parameter('err_warping_exp', 'real', half_step_limits = config.err_warping_exp_bounds)
            fs_parameters['confusion_normalization_exp'] = Parameter('confusion_normalization_exp', 'real', half_step_limits= config.confusion_normalization_exp_bounds)
            fs_parameters['list_ranking_thresh'] = Parameter('list_ranking_thresh', 'integer', half_step_limits = config.list_ranking_thresh_bounds, transform_limits = (max_to_n_of_features, config.n_of_features))


    if not split_fs_parameters:
        parameters = fs_parameters

    if config.hpo_do_forest_param:
        
        parameters['tree_depth'] = Parameter('tree_depth','integer',half_step_limits = config.tree_depth_half_step)
        parameters['num_of_trees'] = Parameter('num_of_trees','integer',half_step_limits = config.forest_size_half_step)
        parameters['min_impurity_decrease_exp'] = Parameter('min_impurity_decrease_exp','real',half_step_limits = config.min_impurity_decrease_exp_half_step)
        
        if config.forest_type == 'random' or 'gradient_boost':
            parameters['min_sample_split'] = Parameter('min_sample_split','integer',half_step_limits = config.min_sample_split_half_step)
            parameters['max_feature_cont_parameter'] = Parameter('max_feature_cont_parameter','real',half_step_limits = config.max_feature_cont_parameter_bounds)
            parameters['tree_min_leaf_samples'] = Parameter('tree_min_leaf_samples','integer',half_step_limits = config.min_sample_leaf_half_step)
            parameters['ccp_alpha_exp'] = Parameter('ccp_alpha_exp','real',half_step_limits = config.ccp_alpha_exp_half_step)
        if config.forest_type == 'random':
            parameters['max_sample_parameter'] = Parameter('max_sample_parameter','real',half_step_limits = config.max_sample_half_step)
        elif config.forest_type == 'gradient_boost' or config.forest_type == 'xgboost':
            parameters['learning_rate'] = Parameter('learning_rate', 'real', half_step_limits = [0.0, 2.0])
        
        if not config.regression:
            parameters['criterion'] = Parameter('criterion','categorical',possible_values = config.criteria,classification_specific = True)

    if config.hpo_do_sample_weighting:
        if config.regression:
            #if not config.geometric_weighting:
            #    parameters['sample_weight_parameter'] = Parameter('sample_weight_parameter','real',half_step_limits = config.sample_weight_parameter_half_step,
            #                                                        regression_specific = True)
            #    parameters['number_of_bins'] = Parameter('number_of_bins','integer',half_step_limits = config.number_of_bins_half_step,regression_specific = True)
            #else:
                #pass
            #parameters['geometric_exponent'] = Parameter('geometric_exponent', 'real', half_step_limits = config.geometric_exponent_bounds)
            pass
        else:
            parameters['class_weight'] = Parameter('class_weight','categorical',possible_values = config.class_weights,classification_specific = True)
    
    if split_fs_parameters:
        return fss_parameters, fs_parameters, parameters
    return parameters


def threeDimHyperOptimization(config, cv_obj, best_scores, slice_slices, samples = None, samples_store_id = None, distance_map = None, debug = False):
    fss_parameters, fs_parameters, parameters = initParameters(config, split_fs_parameters = True)
    converged = False
    n = 1
    score_matrix = {}

    while not converged:
        converged = True
        if n > 1:
            cv_obj.reset_confusion_maps()

        for param in [fss_parameters]:
            param_names = list(param.keys())
            random.shuffle(param_names)

            while len(param_names) > 2:
                param_trio = param_names.pop(), param_names.pop(), param_names.pop()
                new_opti, best_scores, cv_obj, slice_slices = threeDim(param[param_trio[0]], param[param_trio[1]], param[param_trio[2]], best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, force_confusion = True)
                if new_opti:
                    converged = False

        new_opti, best_scores, cv_obj, slice_slices = bayesian_optimisation(None, config, list(fs_parameters.values()), score_matrix, cv_obj, best_scores, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id, n_pre_samples = None, debug = debug)
        if new_opti:
            converged = False
            
        for param in [parameters]:
            param_names = list(param.keys())
            random.shuffle(param_names)

            while len(param_names) > 1:
                param_trio = param_names.pop(), param_names.pop()#, 'confusion_rank_threshold'#param_names.pop()
                new_opti, best_scores, cv_obj, slice_slices = threeDim(param[param_trio[0]], param[param_trio[1]], fs_parameters['confusion_rank_threshold'], best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id)
                if new_opti:
                    converged = False

            while len(param_names) == 1:
                new_opti, best_scores, cv_obj, slice_slices = twoDim(param_names.pop(), fs_parameters['confusion_rank_threshold'], best_scores, config, score_matrix, cv_obj, distance_map, slice_slices, samples = samples, samples_store_id = samples_store_id)
                if new_opti:
                    converged = False


        print('Iteration: ',n)
        config.printParameter()
        config.saveHyperParameter(f"hyperparameters_ThreeDim_epoch_{n}.conf")
        if best_scores is not None:
            best_scores.printOut()
        n += 1
        slice_slices = None
 
    cv_obj.reset_confusion_maps()
    return

def bayesianComplete(config, cv_obj, best_scores, samples, distance_map = None):
    parameters = initParameters(config)

    score_matrix = {}
    
    new_optimimum, best_scores = bayesian_optimisation(200, config, [parameters[p] for p in parameters], score_matrix, cv_obj, best_scores, distance_map, samples, n_pre_samples = None)

    print('Bayesian optimization finished')
    config.printParameter()
    best_scores.printOut()

    return

