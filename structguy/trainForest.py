from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

import time
import ray
from scipy import stats
import xgboost as xgb

from structguy import featureSelection, util
from structman.base_utils.base_utils import pack, unpack

def makeBinaryClassifier(data,thresh,flip_sign=False):
    binary = []
    for value in data:
        if not flip_sign:
            if value < thresh:
                binary.append(0)
            else:
                binary.append(1)
        else:
            if value > thresh:
                binary.append(0)
            else:
                binary.append(1)
    return binary

@ray.remote(max_calls = 1)
def trainClassificationForest(config, samples, cv_slice, print_out=True,skip_scoring = False, remote = False, cv_counter = None):
    depth = config.tree_depth
    min_sample_split = config.min_sample_split
    proc = config.proc_n
    leaf_samples = config.tree_min_leaf_samples
    n_of_trees = int(config.num_of_trees)
    max_leaf_nodes = config.max_leaf_nodes
    class_weight = config.class_weight
    max_features = config.max_feature_parameter
    bootstrap = config.bootstrap_parameter
    min_impurity_decrease = 10**(-config.min_impurity_decrease_exp)
    oob_score = config.oob_score
    ccp_alpha = 10**(-config.ccp_alpha_exp)
    max_sample_parameter = config.max_sample_parameter
    criterion = config.criterion

    zero_scores = util.Scores(zero=True, n_of_features = len(cv_slice.feature_names))

    if remote:
        zero_return = zero_scores, None, cv_counter
    else:
        zero_return = None,zero_scores, cv_counter

    if depth < 1:
        return zero_return
    if leaf_samples < 1:
        return zero_return
    if min_sample_split < 2:
        return zero_return
    if n_of_trees < 1:
        return zero_return
    if min_impurity_decrease < 0.0 or min_impurity_decrease > 1.0:
        return zero_return
    if ccp_alpha < 0.0:
        return zero_return
    if max_sample_parameter <= 0.0 or max_sample_parameter > 1.0:
        return zero_return

    slice_updated = featureSelection.select_features(config, cv_slice, samples, print_out = print_out)

    if slice_updated is None:
        return zero_return

    if print_out:
        config.printParameter()

    forest = RandomForestClassifier(
        n_estimators=n_of_trees,
        max_depth = depth,
        min_samples_leaf = leaf_samples,
        max_features=max_features,
        n_jobs=proc,
        min_samples_split=min_sample_split,
        max_leaf_nodes=max_leaf_nodes,
        class_weight = class_weight,
        bootstrap = bootstrap,
        criterion = criterion,
        max_samples = max_sample_parameter)

    if print_out:
        print('Fit classification forest, size of feat_matrix:',
              len(samples.raw_feature_matrix),'size of first feature vector:',len(samples.raw_feature_matrix[0]))

    forest.fit(cv_slice.train_feature_matrix,cv_slice.train_targets)

    if skip_scoring:
        return forest, None, cv_counter

    y_pred = forest.predict(cv_slice.test_feature_matrix)

    acc = accuracy_score(cv_slice.test_targets,y_pred)
    int_targets = cv_slice.classToInt(cv_slice.test_targets)
    int_preds = cv_slice.classToInt(y_pred)
    roc = roc_auc_score(int_targets,int_preds)

    f1 = f1_score(int_targets,int_preds)

    precision = precision_score(int_targets,int_preds)
    recall = recall_score(int_targets,int_preds)

    mcc = matthews_corrcoef(int_targets,int_preds)

    scores_obj = util.Scores(acc = acc,roc = roc,precision = precision,recall = recall,f1 = f1,mcc = mcc, n_of_features = len(cv_slice.feature_names))

    if print_out:
        scores_obj.printOut()

    if not slice_updated:
        cv_slice = None

    if remote:
        return scores_obj, pack(cv_slice), cv_counter
    return forest,scores_obj, cv_counter

def test_for_constant_array(a):
    if len(a) <= 1:
        return True
    first_value = a[0]
    for second_value in a[1:]:
        if first_value != second_value:
            return False
    return True


@ray.remote(max_calls = 1)
def trainRegressionForestWrapper(config, cv_slice, samples_store_id = None, slice_slices = None, distance_map = None, print_out=True, skip_scoring = False, cv_counter = None, debug = False, overwrite_proc_n = None, skip_feature_selection = False, force_confusion = False):
    return trainRegressionForest(config, unpack(cv_slice), samples_store_id = samples_store_id[0], slice_slices= slice_slices, distance_map = distance_map,print_out=print_out,skip_scoring = skip_scoring, cv_counter = cv_counter, debug = debug, remote = True, skip_feature_selection = skip_feature_selection, overwrite_proc_n = overwrite_proc_n, force_confusion = force_confusion)

def return_zero(zero_return, remote, cv_slice):
    if remote:
        del cv_slice
    return zero_return

def trainRegressionForest(config, cv_slice, samples = None, samples_store_id = None, slice_slices = None, distance_map = None, print_out=True,skip_scoring = False, cv_counter = None, remote = False, overwrite_proc_n = None, debug = False, skip_feature_selection = False, force_confusion = False):
    times = []
    t0 = time.time()

    times_collection = [('trainRegressionForest Part ', times)]

    depth = config.tree_depth
    min_sample_split = config.min_sample_split
    if overwrite_proc_n is None:
        proc = config.proc_n
    else:
        proc = int(overwrite_proc_n)
    leaf_samples = config.tree_min_leaf_samples
    n_of_trees = int(config.num_of_trees)
    rel_max_leaf_node_pruning = config.rel_max_leaf_node_pruning

    max_feature_parameter = config.max_feature_parameter
    if config.max_feature_cont_parameter is not None:
        max_feature_parameter = config.max_feature_cont_parameter

    if config.min_impurity_decrease_exp < config.maximal_exp:
        min_impurity_decrease = 10**(-config.min_impurity_decrease_exp)
    else:
        min_impurity_decrease = 0.

    if config.fs_min_impurity_decrease_exp < config.maximal_exp:
        fs_min_impurity_decrease = 10**-(config.fs_min_impurity_decrease_exp)
    else:
        fs_min_impurity_decrease = 0.

    if config.ccp_alpha_exp < config.maximal_exp:
        ccp_alpha = 10**-(config.ccp_alpha_exp)
    else:
        ccp_alpha = 0.

    if config.fs_ccp_alpha_exp < config.maximal_exp:
        fs_ccp_alpha = 10**-(config.fs_cpp_alpha_exp)
    else:
        fs_ccp_alpha = 0.

    max_sample_parameter = config.max_sample_parameter
    criterion = config.criterion
    number_of_bins = config.number_of_bins

    zero_scores = util.Scores(zero=True, n_of_features = len(cv_slice.feature_names))

    if remote:
        zero_return = zero_scores, None, cv_counter, times_collection, None
    else:
        zero_return = None, zero_scores, cv_counter, None, times_collection, slice_slices

    if depth < 1:
        return return_zero(zero_return, remote, cv_slice)
    if leaf_samples < 1:
        return return_zero(zero_return, remote, cv_slice)
    if min_sample_split < 2:
        return return_zero(zero_return, remote, cv_slice)
    if n_of_trees < 1:
        return return_zero(zero_return, remote, cv_slice)
    if min_impurity_decrease < 0.0 or min_impurity_decrease > 1.0:
        return return_zero(zero_return, remote, cv_slice)
    if ccp_alpha < 0.0:
        return return_zero(zero_return, remote, cv_slice)
    
    if max_sample_parameter <= 0.0 or max_sample_parameter > 1.0:
        return return_zero(zero_return, remote, cv_slice)
    
    if skip_feature_selection and (config.fs_max_sample_parameter <= 0.0 or config.fs_max_sample_parameter > 1.0):
        return return_zero(zero_return, remote, cv_slice)
    
    if isinstance(max_feature_parameter,float):
       if max_feature_parameter <= 0.0 or max_feature_parameter > 1.0:
           return return_zero(zero_return, remote, cv_slice)
    if number_of_bins < 1:
        return return_zero(zero_return, remote, cv_slice)

    t1 = time.time()
    if config.verbosity >= 2:
        print(f'Train regression forest part 1: {t1-t0}, Threads: {proc}, Feature selection: {not skip_feature_selection}')
    times.append(('1', t1-t0))

    if not skip_feature_selection:
        slice_updated, filtered_features, feat_select_times, slice_slices = featureSelection.select_features(config, cv_slice, slice_slices, samples_store_id = samples_store_id, samples = samples, print_out = print_out, debug = debug, overwrite_proc_n = proc, force_confusion = force_confusion)
    else:
        slice_updated = False
        feat_select_times = []
        slice_slices = None

    times_collection.append((f'select_features ({slice_updated}) Part ', feat_select_times))


    if slice_updated is None:
        print('ERROR =============== Feature selection failed:',cv_counter)
        config.printParameter()
        print('==============================================')
        return return_zero(zero_return, remote, cv_slice)

    if len(cv_slice.feature_names) == 0:
        print('Warning =============== Feature vector has len 0',cv_counter)
        #config.printParameter()
        #print('==============================================')
        return return_zero(zero_return, remote, cv_slice)

    #print('=============== Feature vector has len ',len(cv_slice.train_feature_matrix[0]),cv_counter)
    #config.printParameter()
    #print('==============================================')

    if max_sample_parameter == 1.0:
        max_sample_parameter = None

    if rel_max_leaf_node_pruning != None:
        max_leaf_nodes = int((2**(depth))/rel_max_leaf_node_pruning)

    if print_out:
        config.printParameter()

    t2 = time.time()
    if config.verbosity >= 3:
        print(f'Train regression forest part 2: {t2-t1}, {proc}')
    times.append(('2', t2-t1))

    if config.forest_type == 'gradient_boost' and not skip_feature_selection:
        forest = GradientBoostingRegressor(
            n_estimators = n_of_trees,
            max_depth = depth,
            min_samples_leaf = leaf_samples,
            max_features = max_feature_parameter,
            min_samples_split = min_sample_split,
            ccp_alpha = ccp_alpha,
            min_impurity_decrease = min_impurity_decrease,
            learning_rate = config.learning_rate,
            #no_iter_no_change = config.early_stopping,
            subsample = max_sample_parameter
        )
    elif config.forest_type == 'xgboost' and not skip_feature_selection:
        import xgboost as xgb
        packed_slice_slice = slice_slices[0]
        try:
            slice_slice = unpack(packed_slice_slice)
        except:
            slice_slice = packed_slice_slice
        slice_slice.filterFeatures(filtered_features)
        if debug:
            slice_slice.printBalance(config)
        if samples is None:
            samples = ray.get(samples_store_id)
        protwise_test_data_tuples = slice_slice.get_prot_wise_test_data_tuples(samples)
        es_list = []
        data_tuple_list = []
        for n, prot_id in enumerate(protwise_test_data_tuples):
            es = xgb.callback.EarlyStopping(
                rounds = config.early_stopping,
                min_delta=1e-3,
                save_best=True,
                maximize=True,
                data_name=f"validation_{n}"
            )
            es_list.append(es)
            data_tuple_list.append(protwise_test_data_tuples[prot_id])
        forest = xgb.XGBRegressor(
            n_jobs = proc,
            n_estimators = n_of_trees,
            max_depth = depth,
            gamma = min_impurity_decrease,
            learning_rate = config.learning_rate,
            min_child_weight = config.min_child_weight,
            early_stopping_rounds = config.early_stopping,
            subsample = max_sample_parameter,
            verbosity = 0,
            callbacks = es_list,
            eval_metric = util.rho_eval_for_xgboost
        )
    elif skip_feature_selection:
        forest = RandomForestRegressor(
                    n_estimators = config.fs_num_of_trees,
                    max_depth = config.fs_tree_depth,
                    min_samples_leaf = config.fs_tree_min_leaf_samples,
                    n_jobs = proc,
                    min_samples_split = config.fs_min_sample_split,
                    ccp_alpha = fs_ccp_alpha,
                    min_impurity_decrease = fs_min_impurity_decrease,
                    max_samples = config.fs_max_sample_parameter,
                    criterion = criterion)
    else:
        forest = RandomForestRegressor(
                    n_estimators=n_of_trees,
                    max_depth = depth,
                    min_samples_leaf = leaf_samples,
                    max_features=max_feature_parameter,
                    n_jobs=proc,
                    min_samples_split=min_sample_split,
                    ccp_alpha=ccp_alpha,
                    min_impurity_decrease=min_impurity_decrease,
                    max_samples = max_sample_parameter,
                    criterion = criterion)

    t3 = time.time()
    if config.verbosity >= 3:    
        print(f'Train regression forest part 3: {t3-t2}')
    times.append(('3', t3-t2))

    t4 = time.time()
    if config.verbosity >= 3:
        print(f'Train regression forest part 4: {t4-t3}')
    times.append(('4', t4-t3))

    if config.verbosity >= 5:
        util.sanity_check_value_list(cv_slice.train_targets, label_vector = cv_slice.train_sample_ids, datastructure_name = f'Train target of {cv_slice.name}')

    t5 = time.time()
    if config.verbosity >= 3:    
        print(f'Train regression forest part 5: {t5-t4}, # of features: {len(cv_slice.feature_names)}')
    times.append(('5', t5-t4))

    if print_out or config.verbosity >= 3:
        print(f'Train regression {config.forest_type} forest, call of fit with # of features: {len(cv_slice.feature_names)}, skip feature selection {skip_feature_selection}, slice update {slice_updated}, skip scoring {skip_scoring}')
    
    if samples is None:
        samples = ray.get(samples_store_id)

    weights_updated = False
    if config.forest_type == 'xgboost' and not skip_feature_selection:
        if config.weighting == 'geometric':
            slice_slice.calcSampleWeights(config, distance_map)
        elif config.weighting == 'subsample_distance':
            weights_updated = slice_slice.calcSubsampleDistanceWeights(config, para_number = proc)
        train_feature_matrix = slice_slice.get_train_feature_matrix(samples)
        forest.fit(train_feature_matrix, slice_slice.train_targets, eval_set = data_tuple_list, sample_weight=slice_slice.train_class_weight_vector)
        slice_slice.filterFeatures([])
    else:
        if config.weighting == 'geometric':
            cv_slice.calcSampleWeights(config, distance_map)
        elif config.weighting == 'subsample_distance':
            weights_updated = cv_slice.calcSubsampleDistanceWeights(config, para_number = proc)
        train_feature_matrix = cv_slice.get_train_feature_matrix(samples)
        forest.fit(train_feature_matrix, cv_slice.train_targets, sample_weight=cv_slice.train_class_weight_vector)

    slice_updated = slice_updated or weights_updated
    t6 = time.time()
    if config.verbosity >= 3:
        print(f'Train regression forest part 6: {t6-t5}')
    times.append(('6', t6-t5))

    if skip_scoring:

        return forest, None, cv_counter, cv_slice, times_collection, slice_slices

    if debug:
        cv_slice.printBalance(config)

    test_feature_matrix = cv_slice.get_test_feature_matrix(samples)
    y_pred = forest.predict(test_feature_matrix)

    t7 = time.time()
    if config.verbosity >= 3:    
        print(f'Train regression forest part 7: {t7-t6}')
    times.append(('7', t7-t6))

    if test_for_constant_array(y_pred):
        if print_out:
            zero_scores.printOut()
        return return_zero(zero_return, remote, cv_slice)

    if config.verbosity >= 5:
        util.sanity_check_value_list(cv_slice.test_class_weight_vector, label_vector = cv_slice.test_sample_ids, datastructure_name = f'Test weight vector of {cv_slice.name}')
        util.sanity_check_value_list(cv_slice.test_targets, label_vector = cv_slice.test_sample_ids, datastructure_name = f'Test target of {cv_slice.name}')
        util.sanity_check_value_list(y_pred, label_vector = cv_slice.test_sample_ids, datastructure_name = f'Predictions of {cv_slice.name}')

    weighted_r2 = r2_score(cv_slice.test_targets,y_pred,sample_weight = cv_slice.test_class_weight_vector)
    weighted_mse = mean_squared_error(cv_slice.test_targets,y_pred,sample_weight = cv_slice.test_class_weight_vector)

    r2 = r2_score(cv_slice.test_targets,y_pred)
    mse = mean_squared_error(cv_slice.test_targets,y_pred)

    #observed_value_threshold = config.binary_thresh

    tv_median = util.median(cv_slice.test_targets)

    test_targets_b = makeBinaryClassifier(cv_slice.test_targets,tv_median)

    #middle_value = (max(y_pred) + min(y_pred))/2.
    y_pred_median = util.median(y_pred)

    y_pred_b = makeBinaryClassifier(y_pred,y_pred_median)
    mcc = matthews_corrcoef(test_targets_b,y_pred_b)

    exp_score = mcc/(1.+mse)
    #mcc = exp_score #JUST FOR TESTING

    corr,p_value = stats.spearmanr(cv_slice.test_targets,y_pred)
    pearson,pearson_p = stats.pearsonr(cv_slice.test_targets,y_pred)

    prot_wise_spearmans, mean_spearman = util.calc_protein_wise_corr(cv_slice.test_targets, y_pred, cv_slice.test_sample_ids, stats.spearmanr)
    prot_wise_pearsons, mean_pearson = util.calc_protein_wise_corr(cv_slice.test_targets, y_pred, cv_slice.test_sample_ids, stats.pearsonr)

    scores_obj = util.Scores(mse = mse,r2 = r2,corr = corr,mcc = mcc,pearson_r=pearson, wmse = weighted_mse, wr2 = weighted_r2, n_of_features = len(cv_slice.feature_names), mean_spearman = mean_spearman, mean_pearson = mean_pearson)

    if print_out or debug:
        scores_obj.printOut()
        print(f'Prot-wise Spearmans correlations:\n{prot_wise_spearmans}\n')
        print(f'Prot-wise Pearsons correlations:\n{prot_wise_pearsons}\n')

    t8 = time.time()
    if config.verbosity >= 3:
        print(f'Train regression forest part 8: {t8-t7}')
    times.append(('8', t8-t7))

    if remote:
        return scores_obj, pack(cv_slice), cv_counter, times_collection, slice_slices
    return forest, scores_obj, cv_counter, cv_slice, times_collection, slice_slices

def trainForest(config, cross_val_object, samples_store_id = None, samples = None, slice_slices = None, distance_map = None, repeat = 1, print_out = False, cv_repeat = False, skip_scoring=False, remote = True, debug = False, para_number = None, skip_feature_selection = False, force_confusion = False):
    #if cv_repeat is False, the cross_val_object is a cross validation slice object instead
    zero_scores_obj = util.Scores(zero=True)
    if para_number == 1:
        remote = False

    if config.suppress_remote_forests or debug:
        remote = False
        para_number = None

    if not cv_repeat:
        if len(cross_val_object.feature_names) < 1:
            print(f'Call of trainForest without features: {cross_val_object.name}')
            return None, zero_scores_obj, cross_val_object, slice_slices

    if config.verbosity >= 2 or debug:
        print(f'Call of trainForest: repeat {repeat}, cv_repeat {cv_repeat}, remote {remote}, para_number {para_number}, skip_feature_selection {skip_feature_selection}, debug: {debug}')

    t0 = time.time()

    scores_list = []
    worst_scores = None
    worst_forest = None
    forest = None
    if not cv_repeat:
        for i in range(0,repeat): #This can be used to ensure the robustness of the current parameter configuration
            if print_out:
                print('Training with #of features:',len(cross_val_object.feature_names),'and #of samples:',len(cross_val_object.train_targets))

            if config.regression:
                forest, scores_obj, cv_counter, cv_slice, reg_forest_times, _slice_slices = trainRegressionForest(config, cross_val_object, samples_store_id = samples_store_id, samples = samples, slice_slices = slice_slices, distance_map = distance_map, print_out = print_out,skip_scoring = skip_scoring, debug = debug, skip_feature_selection = skip_feature_selection, overwrite_proc_n = para_number, force_confusion = force_confusion)
            else:
                forest, scores_obj, cv_counter, cv_slice = trainClassificationForest(config,cross_val_object,print_out = print_out,skip_scoring = skip_scoring, skip_feature_selection = skip_feature_selection)
            if repeat > 1:
                if i == 0:
                    worst_scores = scores_obj
                    worst_forest = forest
                elif util.objective_function_criterium(config,worst_scores,scores_obj): #if worst_scores ar better than scores_obj
                    worst_scores = scores_obj
                    worst_forest = forest
            if config.optimize_mean:
                scores_list.append(scores_obj)
        if repeat > 1:
            scores_obj = worst_scores
            forest = worst_forest
            if config.optimize_mean:
                scores_obj = util.mean_scores(scores_list)

        if slice_slices is None:
            slice_slices = _slice_slices
        elif _slice_slices is not None:
            for slice_number, slice_slice in enumerate(_slice_slices):
                if slice_slice is not None:
                    slice_slices[slice_number] = slice_slice
        ret_slice_slices = slice_slices

    else: #This can be used to perform a hyperparameter optimization on the whole dataset
        slice_result_ids = []

        if config.cv_hpo_limiter is not None:
            if config.cv_counters is not None:
                cv_counters = config.cv_counters
            else:
                cv_counters = list(cross_val_object.slices.keys())[0:config.cv_hpo_limiter]
                config.cv_counters = cv_counters
                if config.verbosity >= 1:
                    print(f'\nCross validation in HPO limited to: {cv_counters}\n')
        else:
            cv_counters = cross_val_object.slices.keys()

        if remote:
            if para_number is None:
                para_number = config.proc_n // len(cv_counters)
            else:
                para_number = para_number // len(cv_counters)
            cv_slice_stores = {}
        else:
            cv_slice_stores = None

        for cv_counter in cv_counters:
            cv_slice = cross_val_object.slices[cv_counter]
            if remote:
                t01 = time.time()
                packed_cv_slice = pack(cv_slice)
                cv_slice_stores[cv_counter] = packed_cv_slice
                t02 = time.time()
                if config.verbosity >= 2:
                    print(f'Time for packing cv_slice {cv_counter} in trainForest: {t02-t01} {slice_slices is None} {para_number}')

            if slice_slices is not None:
                s_slice_slices = slice_slices[cv_counter]
            else:
                s_slice_slices = None

            if remote:
                if config.regression:
                    if samples_store_id is None:
                        samples_store_id = ray.put(samples)
                    slice_result_ids.append(trainRegressionForestWrapper.remote(config, packed_cv_slice, samples_store_id = [samples_store_id], slice_slices = s_slice_slices, distance_map = distance_map, print_out = print_out, cv_counter = cv_counter, debug = debug, skip_feature_selection = skip_feature_selection, overwrite_proc_n = para_number,skip_scoring = skip_scoring, force_confusion = force_confusion))
                else:
                    slice_result_ids.append(trainClassificationForestWrapper.remote(config, packed_cv_slice, print_out = print_out, cv_counter = cv_counter,skip_scoring = skip_scoring, skip_feature_selection = skip_feature_selection))
            else:
                if config.regression:
                    slice_result_ids.append(trainRegressionForest(config, cv_slice, samples = samples, samples_store_id = samples_store_id, slice_slices = s_slice_slices, distance_map = distance_map, print_out = print_out, cv_counter = cv_counter, overwrite_proc_n = para_number, debug = debug, skip_feature_selection = skip_feature_selection,skip_scoring = skip_scoring, force_confusion = force_confusion))
                else:
                    slice_result_ids.append(trainClassificationForest(config,cv_slice,print_out = print_out, cv_counter = cv_counter,skip_scoring = skip_scoring, skip_feature_selection = skip_feature_selection))

        if remote:
            results = ray.get(slice_result_ids)
        else:
            results = slice_result_ids

        ret_slice_slices = slice_slices
        if ret_slice_slices is None:
            ret_slice_slices = {}

        for res in results:
            if remote:
                scores_obj, packed_cv_slice, cv_counter, reg_forest_times, _slice_slices = res
                cv_slice = unpack(packed_cv_slice)
            else:
                forest, scores_obj, cv_counter, cv_slice, reg_forest_times, _slice_slices = res

            if config.verbosity >= 3:
                for precursor, times_list in reg_forest_times:
                    for part_id, time_in_sec in times_list:
                        print(f'{precursor}{part_id}: {time_in_sec}')

            if cv_slice is not None:
                del cross_val_object.slices[cv_counter]
                cross_val_object.slices[cv_counter] = cv_slice
                if remote:
                    cv_slice_stores = None

            if cv_counter not in ret_slice_slices:
                ret_slice_slices[cv_counter] = _slice_slices
            elif ret_slice_slices[cv_counter] is None:
                ret_slice_slices[cv_counter] = _slice_slices
            elif _slice_slices is not None:
                ret_slice_slices[cv_counter] = _slice_slices

            if scores_obj is None:
                raise 'Scores must not be None here'

            if config.optimize_mean:
                scores_list.append(scores_obj)
            else:
                if worst_scores is None:
                    worst_scores = scores_obj
                elif util.objective_function_criterium(config,worst_scores,scores_obj): #if worst_scores ar better than scores_obj
                    worst_scores = scores_obj
        del results
        
        if config.optimize_mean:
            scores_obj = util.mean_scores(scores_list)
        else:
            scores_obj = worst_scores
            forest = worst_forest

    t1 = time.time()
    if config.verbosity >= 2:
        print(f'Time for trainForest: {t1-t0}')

    return forest, scores_obj, cross_val_object, ret_slice_slices
