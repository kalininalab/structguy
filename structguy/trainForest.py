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
import sys
import os
import traceback
import ray
import contextlib
from scipy import stats
import xgboost as xgb
from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback


from filelock import FileLock
from structguy import featureSelection, util
from structman.base_utils.base_utils import pack, unpack, add_to_times, print_times, aggregate_times
from structguy.support_classes import CrossValidationSlice
from structguy.sampleSpace import DataSAIL_cv
import numpy
from ray.util.queue import Queue
from ray.train import RunConfig
import shap
from numba import njit

def makeBinaryClassifier(data, thresh, flip_sign=False):
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

def test_for_constant_array(a):
    if len(a) <= 1:
        return True
    first_value = a[0]
    for second_value in a[1:]:
        if first_value != second_value:
            return False
    return True


@ray.remote
def para_prediction(forest_dump_file, feat_matrix):
    forest = util.loadModel(forest_dump_file)[0]
    pred = forest.predict(feat_matrix)
    return pred

@ray.remote(max_calls=1, num_gpus=1)
def trainRegressionForestWrapper(
    config,
    cv_slice,
    samples_store_id=None,
    slice_slices=None,
    distance_map=None,
    print_out=True,
    skip_scoring=False,
    cv_counter=None,
    debug=False,
    overwrite_proc_n=None,
    skip_feature_selection=False,
    force_confusion=False,
    score_train=True,
    
):
    gpu_id = ray.get_runtime_context().get_accelerator_ids()["GPU"][0]
    return trainRegressionForest(
        config,
        unpack(cv_slice),
        samples_store_id=samples_store_id[0],
        slice_slices=slice_slices,
        distance_map=distance_map,
        print_out=print_out,
        skip_scoring=skip_scoring,
        cv_counter=cv_counter,
        debug=debug,
        remote=True,
        skip_feature_selection=skip_feature_selection,
        overwrite_proc_n=overwrite_proc_n,
        force_confusion=force_confusion,
        score_train=score_train,
        gpu_id=gpu_id
    )


@njit
def shap_internal_loop(
        N_feat_names: int,
        explanation: numpy.ndarray,
        prediction_vector: numpy.ndarray,
        target_vector: numpy.ndarray
        ) -> numpy.ndarray:
    acc_feat_impacts: numpy.ndarray = numpy.zeros(N_feat_names)
    for sample_pos, shap_values in enumerate(explanation):
        pred: float = prediction_vector[sample_pos]
        true_val: float = target_vector[sample_pos]
        loss: float = abs(true_val - pred)
        for feat_pos in range(N_feat_names):
            shap_val: float = shap_values[feat_pos]
            pert_pred: float = pred - shap_val
            feature_impact: float = loss - abs(true_val - pert_pred)
            acc_feat_impacts[feat_pos] += feature_impact

    return acc_feat_impacts

def shap_analysis(config, forest, d_feat_vecs: xgb.DMatrix, feature_names, prediction_vector, target_vector):
    if config.verbosity >= 2:
        config.logger.info(f'Call of shap_analysis: {config.gpu_mode=}')
    
    times = []
    ta = time.time()
    
    explanation = forest.predict(d_feat_vecs, pred_contribs=True)

    """
    done = False
    n = 1
    while not done:
        try:
            if n > 1:
                h = len(feat_vecs) // 2
                d_feat_vecs_1 = xgb.DMatrix(feat_vecs[:h], feature_names = feature_names)
                d_feat_vecs_2 = xgb.DMatrix(feat_vecs[h:], feature_names = feature_names)
                explanation_1 = forest.predict(d_feat_vecs_1, pred_contribs=True)
                explanation_2 = forest.predict(d_feat_vecs_2, pred_contribs=True)

                explanation = numpy.concatenate(explanation_1, explanation_2)
            else:
                explanation = forest.predict(d_feat_vecs, pred_contribs=True)
            done = True
        except xgb.core.XGBoostError:
            done = False
            
            n+=1
            if n == 4:
                return None, times
            config.logger.info(f'Catched XGBoost Error, try again {n}')
            time.sleep(n**2)
    """
    ta = add_to_times(times, ta)

    acc_feat_impacts = shap_internal_loop(
        len(feature_names),
        explanation,
        numpy.array(prediction_vector),
        numpy.array(target_vector)
        )
    
    ta = add_to_times(times, ta)

    acc_feat_impacts = sorted(zip(feature_names, [x/d_feat_vecs.num_row() for x in acc_feat_impacts]), key=lambda x:x[1], reverse=True)
    ta = add_to_times(times, ta)
    return acc_feat_impacts, times


def cut_and_predict(feat_matrix, model):
    h = len(feat_matrix)//2
    pred_1 = model.predict(feat_matrix[:h])
    pred_2 = model.predict(feat_matrix[h:])

    concatted = numpy.concatenate(pred_1, pred_2)
    return concatted


def return_zero(zero_return, remote, cv_slice):
    if remote:
        del cv_slice
    return zero_return


def rho_eval_for_xgboost_cb(predt: numpy.ndarray, dtest: xgb.DMatrix) -> tuple[str, float]:
    if isinstance(dtest, xgb.DMatrix):
        y = dtest.get_label()
    else:
        y = dtest
    corr, _ = stats.spearmanr(predt, y)
    return 'irho', (1.0-corr)

def xgb_train_wrapper(
        config: util.Config,
        dtrain: xgb.DMatrix,
        dtest_feature_matrix: xgb.DMatrix,
        second_round = False,
        ):
        
    es_list = []
    evals: list[tuple[xgb.DMatrix, str]] = []
    eval_label = 'eval'
    

    if not second_round:
        es = xgb.callback.EarlyStopping(
            rounds=config.early_stopping,
            min_delta=1e-4,
            save_best=True,
            maximize=False,
            data_name='eval',
            metric_name='irho',
        )
        es_list.append(es)
        evals.append((dtest_feature_matrix, eval_label))
        xgb_params = {
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": config.tree_depth,
            "reg_alpha": config.xgb_alpha,
            "reg_lambda": config.xgb_lambda,
            "colsample_bytree": config.colsample_bytree,
            "max_delta_step": config.max_delta_step,
            "gamma": config.xgb_gamma,
            "learning_rate": config.learning_rate,
            "min_child_weight": config.min_child_weight,
            "early_stopping_rounds": config.early_stopping,
            "subsample": config.max_sample_parameter,
            "callbacks": es_list,
            #"eval_metric": ['irho'],
            "disable_default_eval_metric": True,
            "max_cat_to_onehot": int(config.max_cat_to_onehot),
            "max_cat_threshold": int(config.max_cat_threshold)
            }
        forest = xgb.train(xgb_params, dtrain, num_boost_round=int(config.num_of_trees), early_stopping_rounds= config.early_stopping, evals=evals, maximize=False, custom_metric=rho_eval_for_xgboost_cb, callbacks=es_list)
    else:
        es = xgb.callback.EarlyStopping(
            rounds=config.early_stopping_1,
            min_delta=1e-4,
            save_best=True,
            maximize=False,
            data_name='eval',
            metric_name='irho',
        )
        es_list.append(es)
        evals.append((dtest_feature_matrix, eval_label))
        xgb_params = {
            "tree_method": "hist",
            "device": "cuda",
            "max_depth": int(config.tree_depth_1),
            "reg_alpha": config.xgb_alpha_1,
            "reg_lambda": config.xgb_lambda_1,
            "colsample_bytree": config.colsample_bytree_1,
            "max_delta_step": config.max_delta_step_1,
            "gamma": config.xgb_gamma_1,
            "learning_rate": config.learning_rate_1,
            "min_child_weight": config.min_child_weight_1,
            "early_stopping_rounds": int(config.early_stopping_1),
            "subsample": config.max_sample_parameter_1,
            "callbacks": es_list,
            #"eval_metric": ['irho'],
            "disable_default_eval_metric": True,
            "max_cat_to_onehot": int(config.max_cat_to_onehot_1),
            "max_cat_threshold": int(config.max_cat_threshold_1)
            }
        forest = xgb.train(xgb_params, dtrain, num_boost_round=int(config.num_of_trees_1), early_stopping_rounds=int(config.early_stopping_1), evals=evals, maximize=False, custom_metric=rho_eval_for_xgboost_cb, callbacks=es_list)

    return forest

def trainRegressionForest(
    config: util.Config,
    cv_slice: CrossValidationSlice,
    samples=None,
    samples_store_id=None,
    slice_slices=None,
    distance_map=None,
    print_out=True,
    skip_scoring=False,
    score_train=True,
    cv_counter=None,
    remote=False,
    overwrite_proc_n=None,
    debug=False,
    skip_feature_selection=False,
    force_confusion=False,
    gpu_id=None
):
    times = []
    t_start = ta = time.time()


    depth = config.tree_depth
    min_sample_split = config.min_sample_split
    if overwrite_proc_n is None:
        proc = config.proc_n
    else:
        proc = int(overwrite_proc_n)
    leaf_samples = config.tree_min_leaf_samples
    n_of_trees = int(config.num_of_trees)

    max_feature_parameter = config.max_feature_parameter
    if config.max_feature_cont_parameter is not None:
        max_feature_parameter = config.max_feature_cont_parameter

    if config.min_impurity_decrease_exp < config.maximal_exp:
        min_impurity_decrease = 10 ** (-config.min_impurity_decrease_exp)
    else:
        min_impurity_decrease = 0.0

    if config.fs_min_impurity_decrease_exp < config.maximal_exp:
        fs_min_impurity_decrease = 10**-(config.fs_min_impurity_decrease_exp)
    else:
        fs_min_impurity_decrease = 0.0

    if config.ccp_alpha_exp < config.maximal_exp:
        ccp_alpha = 10**-(config.ccp_alpha_exp)
    else:
        ccp_alpha = 0.0

    if config.fs_ccp_alpha_exp < config.maximal_exp:
        fs_ccp_alpha = 10**-(config.fs_cpp_alpha_exp)
    else:
        fs_ccp_alpha = 0.0

    max_sample_parameter = config.max_sample_parameter
    criterion = config.criterion
    number_of_bins = config.number_of_bins

    zero_scores = util.Scores(zero=True, n_of_features=len(cv_slice.feature_names))

    if remote:
        zero_return = zero_scores, None, cv_counter, times, None
    else:
        zero_return = (
            None,
            zero_scores,
            cv_counter,
            None,
            times,
            slice_slices,
        )

    if depth < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {depth=}")
        return return_zero(zero_return, remote, cv_slice)
    if leaf_samples < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {leaf_samples=}")
        return return_zero(zero_return, remote, cv_slice)
    if min_sample_split < 2:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {min_sample_split=}")
        return return_zero(zero_return, remote, cv_slice)
    if n_of_trees < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {n_of_trees=}")
        return return_zero(zero_return, remote, cv_slice)
    if min_impurity_decrease < 0.0 or min_impurity_decrease > 1.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {min_impurity_decrease=}")
        return return_zero(zero_return, remote, cv_slice)
    if ccp_alpha < 0.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {ccp_alpha=}")
        return return_zero(zero_return, remote, cv_slice)

    if max_sample_parameter <= 0.0 or max_sample_parameter > 1.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {max_sample_parameter=}")
        return return_zero(zero_return, remote, cv_slice)

    if skip_feature_selection and (config.fs_max_sample_parameter <= 0.0 or config.fs_max_sample_parameter > 1.0):
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {config.fs_max_sample_parameter=}")
        return return_zero(zero_return, remote, cv_slice)

    if isinstance(max_feature_parameter, float):
        if max_feature_parameter <= 0.0 or max_feature_parameter > 1.0:
            if config.verbosity >= 3:
                config.logger.info(f"Return Zero: {max_feature_parameter=}")
            return return_zero(zero_return, remote, cv_slice)
    if number_of_bins < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {number_of_bins=}")
        return return_zero(zero_return, remote, cv_slice)

    ta = add_to_times(times, ta) #0
    if config.verbosity >= 2:
        config.logger.info(f"Train regression forest part 1, Threads: {proc} {gpu_id=}, Feature selection: {not skip_feature_selection} {samples is None=} {config.forest_type=} {config.gpu_mode=} {config.multi_gpu=}")

    if not skip_feature_selection and not config.random_split:
        get_loss_map = False
        slice_updated, filtered_features, feat_select_times, slice_slices, loss_map = featureSelection.select_features(
            config,
            cv_slice,
            slice_slices,
            samples_store_id=samples_store_id,
            samples=samples,
            print_out=print_out,
            debug=debug,
            overwrite_proc_n=proc,
            force_confusion=force_confusion,
            get_loss_map=get_loss_map,
            remote=remote
        )
        if not config.random_split and get_loss_map:
            cv_slice.update_train_weight_vector(loss_map)
    else:
        slice_updated = False
        feat_select_times = []
        slice_slices = None
        loss_map = None
        filtered_features = []

    if config.auto_weighting:
        cv_slice.check_auto_weights()

    times.append(feat_select_times) #1

    if slice_updated is None:
        config.logger.error(f"ERROR =============== Feature selection failed: {cv_counter}")
        config.logParameter()
        config.logger.error("==============================================")
        return return_zero(zero_return, remote, cv_slice)

    if len(cv_slice.feature_names) == 0:
        config.logger.warning(f"Warning =============== Feature vector has len 0 {cv_counter}")
        return return_zero(zero_return, remote, cv_slice)

    if max_sample_parameter == 1.0:
        max_sample_parameter = None

    if print_out:
        config.logParameter()

    ta = add_to_times(times, ta) #2
    if config.verbosity >= 2:
        config.logger.info(f"Train regression forest part 2, {proc=} {samples is None=} {score_train=} {len(filtered_features)=}")
    
    if config.forest_type == "xgboost" and not skip_feature_selection:
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
            samples = unpack(ray.get(samples_store_id))

        #protwise_test_data_tuples = slice_slice.get_prot_wise_test_data_tuples(samples)

    elif skip_feature_selection:
        forest = RandomForestRegressor(
            n_estimators=config.fs_num_of_trees,
            max_depth=config.fs_tree_depth,
            min_samples_leaf=config.fs_tree_min_leaf_samples,
            n_jobs=config.proc_n,
            min_samples_split=config.fs_min_sample_split,
            ccp_alpha=fs_ccp_alpha,
            min_impurity_decrease=fs_min_impurity_decrease,
            max_samples=config.fs_max_sample_parameter,
            criterion=criterion,
        )
    
    ta = add_to_times(times, ta) #3

    if config.verbosity >= 5:
        util.sanity_check_value_list(
            cv_slice.train_targets,
            label_vector=cv_slice.train_sample_ids,
            datastructure_name=f"Train target of {cv_slice.name}",
        )

    ta = add_to_times(times, ta) #4
    if config.verbosity >= 3:
        config.logger.info(f"Train regression forest part 3, {len(cv_slice.feature_names)=} {samples is None=}")

    if print_out or config.verbosity >= 3:
        config.logger.info(
            f"Train regression {config.forest_type} forest, call of fit with # of features: {len(cv_slice.feature_names)}, skip feature selection {skip_feature_selection}, slice update {slice_updated}, skip scoring {skip_scoring}"
        )

    if samples is None:
        samples = unpack(ray.get(samples_store_id))

    ta = add_to_times(times, ta) #5

    weights_updated = False

    
    if config.forest_type == "xgboost" and not skip_feature_selection:
        if config.weighting == "geometric":
            slice_slice.calcSampleWeights(config, distance_map)
        elif config.weighting == "subsample_distance":
            weights_updated = slice_slice.calcSubsampleDistanceWeights(config, para_number=proc)

        dtrain = slice_slice.get_dtrain(samples, sub_sampling=config.sub_sample_factor)  
        
        dtest_feature_matrix = slice_slice.get_dtest(samples)

        ta = add_to_times(times, ta) #6

        forest = xgb_train_wrapper(config, dtrain, dtest_feature_matrix)
        if forest is None:
            return return_zero(zero_return, remote, cv_slice)


    slice_updated = slice_updated or weights_updated
    ta = add_to_times(times, ta) #7

    if config.forest_type == 'xgboost' and not skip_feature_selection:

        ta = add_to_times(times, ta) #8

        y_pred = forest.predict(dtest_feature_matrix)

        ta = add_to_times(times, ta) #9

        acc_feat_impacts, shap_times = shap_analysis(config, forest, dtest_feature_matrix, slice_slice.feature_names, y_pred, slice_slice.test_targets)
        times.append(shap_times)

        ta = add_to_times(times, ta) #10
        if acc_feat_impacts is None:
            return return_zero(zero_return, remote, cv_slice)

        config.logger.info(f'{acc_feat_impacts[:5]=}\n{acc_feat_impacts[-5:]=}')
        feats_to_remove = filtered_features[:]
        for feat_name, feat_impact in acc_feat_impacts:
            if feat_impact >= config.feat_impact_thresh:
                feats_to_remove.append(feat_name)

        config.logger.info(f'{len(feats_to_remove)=} {len(filtered_features)=} {len(cv_slice.feature_names) + len(filtered_features)=}')

        if len(feats_to_remove) >= (len(cv_slice.feature_names)+ len(filtered_features)):
            return return_zero(zero_return, remote, cv_slice)
        

        if len(feats_to_remove) > len(filtered_features):
            cv_slice.filterFeatures(feats_to_remove)
            slice_slice.filterFeatures(feats_to_remove)      
            ta = add_to_times(times, ta) #11

            dtrain = slice_slice.get_dtrain(samples, sub_sampling=config.sub_sample_factor)

            dtest_feature_matrix = slice_slice.get_dtest(samples)

            forest = xgb_train_wrapper(config, dtrain, dtest_feature_matrix, second_round=True)
            if forest is None:
                return return_zero(zero_return, remote, cv_slice)


            slice_slice.filterFeatures([])
        else:
            cv_slice.filterFeatures(filtered_features)
            slice_slice.filterFeatures([])

        ta = add_to_times(times, ta) #12

    if skip_scoring:
        return forest, None, cv_counter, cv_slice, times, slice_slices

    if debug:
        cv_slice.printBalance(config)

    dtest_feature_matrix = cv_slice.get_dtest(samples)

    try:
        y_pred = forest.predict(dtest_feature_matrix)
    except ValueError:
        [e, f, g] = sys.exc_info()
        g = traceback.format_exc()
        config.logger.info(f'Catched error {config.forest_type=} {skip_feature_selection=}: {e}\n{f}\n{g}\n')
        return return_zero(zero_return, remote, cv_slice)
    if score_train:
        dtrain_feature_matrix = cv_slice.get_dtrain(samples, sub_sampling=config.sub_sample_factor)
        done = False
        n = 1
        while not done:
            try:
                if n > 1:
                    x_pred = cut_and_predict(dtrain_feature_matrix, forest)
                else:
                    x_pred = forest.predict(dtrain_feature_matrix)
                done = True
            except ValueError:
                if n < 4:
                    time.sleep(n**2)
                    done = False
                    n += 1
                else:
                    [e, f, g] = sys.exc_info()
                    g = traceback.format_exc()
                    config.logger.info(f'Catched error {config.forest_type=} {skip_feature_selection=}: {e}\n{f}\n{g}\n')
                    return return_zero(zero_return, remote, cv_slice)

    ta = add_to_times(times, ta) #8/13

    if test_for_constant_array(y_pred):
        if print_out:
            zero_scores.printOut()
        if config.verbosity >= 3:
            config.logger.info("Return Zero: test_for_constant_array was True")
        return return_zero(zero_return, remote, cv_slice)

    if config.verbosity >= 5:
        util.sanity_check_value_list(
            cv_slice.test_class_weight_vector,
            label_vector=cv_slice.test_sample_ids,
            datastructure_name=f"Test weight vector of {cv_slice.name}",
        )
        util.sanity_check_value_list(
            cv_slice.test_targets,
            label_vector=cv_slice.test_sample_ids,
            datastructure_name=f"Test target of {cv_slice.name}",
        )
        util.sanity_check_value_list(
            y_pred,
            label_vector=cv_slice.test_sample_ids,
            datastructure_name=f"Predictions of {cv_slice.name}",
        )

    t_end  = time.time()
    t_complete = t_end-t_start

    scores_obj = calc_scores_obj(cv_slice.test_targets, y_pred, cv_slice.test_sample_ids, cv_slice.test_class_weight_vector, cv_slice.feature_names, runtime_penalty = t_complete)
    if score_train:
        train_scores_obj = calc_scores_obj(cv_slice.train_targets, x_pred, cv_slice.train_sample_ids, cv_slice.train_class_weight_vector, cv_slice.feature_names, runtime_penalty = t_complete)
        scores_obj.train_scores = train_scores_obj
        if print_out:
            config.logger.info('Train scores:')
            train_scores_obj.printOut()
    
    if print_out or debug:
        scores_obj.printOut()

    ta = add_to_times(times, ta) #9/14

    if remote:
        return scores_obj, pack(cv_slice), cv_counter, times, slice_slices
    return forest, scores_obj, cv_counter, cv_slice, times, slice_slices


def calc_scores_obj(target_vector, prediction_vector, sample_id_vector, weight_vector, feature_names, runtime_penalty = 0.):
    weighted_r2 = r2_score(target_vector, prediction_vector, sample_weight=weight_vector)
    weighted_mse = mean_squared_error(target_vector, prediction_vector, sample_weight=weight_vector)

    r2 = r2_score(target_vector, prediction_vector)
    mse = mean_squared_error(target_vector, prediction_vector)

    # observed_value_threshold = config.binary_thresh

    tv_median = util.median(target_vector)

    test_targets_b = makeBinaryClassifier(target_vector, tv_median)

    # middle_value = (max(prediction_vector) + min(prediction_vector))/2.
    prediction_vector_median = util.median(prediction_vector)

    prediction_vector_b = makeBinaryClassifier(prediction_vector, prediction_vector_median)
    mcc = matthews_corrcoef(test_targets_b, prediction_vector_b)

    #exp_score = mcc / (1.0 + mse)
    # mcc = exp_score #JUST FOR TESTING

    corr, p_value = stats.spearmanr(target_vector, prediction_vector)
    pearson, pearson_p = stats.pearsonr(target_vector, prediction_vector)

    prot_wise_spearmans, mean_spearman, _ = util.calc_protein_wise_corr(target_vector, prediction_vector, sample_id_vector, stats.spearmanr)
    prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(target_vector, prediction_vector, sample_id_vector, stats.pearsonr)

    scores_obj = util.Scores(
        mse=mse,
        r2=r2,
        corr=corr,
        mcc=mcc,
        pearson_r=pearson,
        wmse=weighted_mse,
        wr2=weighted_r2,
        n_of_features=len(feature_names),
        mean_spearman=mean_spearman,
        mean_pearson=mean_pearson,
        runtime_penalty=runtime_penalty
    )
    return scores_obj

def trainForest(
    config: util.Config,
    cross_val_object: CrossValidationSlice | DataSAIL_cv,
    samples_store_id=None,
    samples=None,
    slice_slices: None | list[CrossValidationSlice] | dict[int, list[CrossValidationSlice]] = None,
    distance_map=None,
    repeat=1,
    print_out=False,
    cv_repeat=False,
    skip_scoring=False,
    remote=False,
    debug=False,
    para_number=None,
    skip_feature_selection=False,
    force_confusion=False,
    score_train=True,
    get_first_scores=False,
    cv_interuption=None,
    gpu_id = None,
) -> tuple[RandomForestRegressor | None, util.Scores, CrossValidationSlice | DataSAIL_cv, None | list[CrossValidationSlice] | dict[int, list[CrossValidationSlice]]]:
    # if cv_repeat is False, the cross_val_object is a cross validation slice object instead
    zero_scores_obj = util.Scores(zero=True)
    # if para_number == 1:

    if config.suppress_remote_forests or debug or config.gpu_mode:
        para_number = None

    if not cv_repeat:
        if len(cross_val_object.feature_names) < 1:
            config.logger.info(f"Call of trainForest without features: {cross_val_object.name}")
            return None, zero_scores_obj, cross_val_object, slice_slices

    if config.verbosity >= 2 or debug:
        config.logger.info(f"Call of trainForest: {repeat=}, {cv_repeat=}, {remote=}, {para_number=}, {skip_feature_selection=}, {debug=}")

    t0 = time.time()

    scores_list = []
    worst_scores = None
    worst_forest = None
    forest = None
    total_times = []
    if not cv_repeat:
        for i in range(0, repeat):  # This can be used to ensure the robustness of the current parameter configuration
            if print_out:
                config.logger.info(f"Training with #of features: {len(cross_val_object.feature_names)} and #of samples: {len(cross_val_object.sub_sampled_train_targets)}")

            (
                forest,
                scores_obj,
                cv_counter,
                cv_slice,
                reg_forest_times,
                _slice_slices,
            ) = trainRegressionForest(
                config,
                cross_val_object,
                samples_store_id=samples_store_id,
                samples=samples,
                slice_slices=slice_slices,
                distance_map=distance_map,
                print_out=print_out,
                skip_scoring=skip_scoring,
                debug=debug,
                skip_feature_selection=skip_feature_selection,
                overwrite_proc_n=para_number,
                force_confusion=force_confusion,
                score_train=score_train,
                gpu_id=gpu_id
            )
            total_times = aggregate_times(total_times, reg_forest_times)

            if repeat > 1:
                if i == 0:
                    worst_scores = scores_obj
                    worst_forest = forest
                elif util.objective_function_criterium(config, worst_scores, scores_obj):  # if worst_scores ar better than scores_obj
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
            slice_slices: list[CrossValidationSlice] = _slice_slices
        elif _slice_slices is not None:
            for slice_number, slice_slice in enumerate(_slice_slices):
                if slice_slice is not None:
                    slice_slices[slice_number] = slice_slice
        ret_slice_slices = slice_slices

    else:  # This can be used to perform a hyperparameter optimization on the whole dataset
        slice_result_ids = []

        if config.cv_hpo_limiter is not None:
            if config.cv_counters is not None:
                cv_counters = config.cv_counters
            else:
                cv_counters = list(cross_val_object.slices.keys())[0 : config.cv_hpo_limiter]
                config.cv_counters = cv_counters
                if config.verbosity >= 1:
                    config.logger.info(f"\nCross validation in HPO limited to: {cv_counters}\n")
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

        for cv_id, cv_counter in enumerate(cv_counters):
            cv_slice: CrossValidationSlice = cross_val_object.slices[cv_counter]
            if remote:
                t01 = time.time()
                packed_cv_slice: bytes = pack(cv_slice)
                cv_slice_stores[cv_counter] = packed_cv_slice
                t02 = time.time()
                if config.verbosity >= 2:
                    config.logger.info(f"Time for packing cv_slice {cv_counter=} in trainForest: {t02 - t01} {slice_slices is None=} {para_number=} {(samples_store_id is None)=}")

            if slice_slices is not None and cv_counter in slice_slices:
                s_slice_slices = slice_slices[cv_counter]
            else:
                s_slice_slices = None

            if remote:
                if samples_store_id is None:
                    samples_store_id = ray.put(pack(samples))
                slice_result_ids.append(
                    trainRegressionForestWrapper.remote(
                        config,
                        packed_cv_slice,
                        samples_store_id=[samples_store_id],
                        slice_slices=s_slice_slices,
                        distance_map=distance_map,
                        print_out=print_out,
                        cv_counter=cv_counter,
                        debug=debug,
                        skip_feature_selection=skip_feature_selection,
                        overwrite_proc_n=para_number,
                        skip_scoring=skip_scoring,
                        force_confusion=force_confusion,
                        score_train=score_train,
                    )
                )
                
            else:
                slice_result_ids.append(
                    trainRegressionForest(
                        config,
                        cv_slice,
                        samples=samples,
                        samples_store_id=samples_store_id,
                        slice_slices=s_slice_slices,
                        distance_map=distance_map,
                        print_out=print_out,
                        cv_counter=cv_counter,
                        overwrite_proc_n=para_number,
                        debug=debug,
                        skip_feature_selection=skip_feature_selection,
                        skip_scoring=skip_scoring,
                        force_confusion=force_confusion,
                        score_train=score_train,
                        gpu_id=gpu_id
                    )
                )
                if cv_interuption is not None and len(slice_result_ids) == 1:
                    (
                        forest,
                        scores_obj,
                        cv_counter,
                        cv_slice,
                        reg_forest_times,
                        _slice_slices,
                    ) = slice_result_ids[0]
                    margin, best_first_scores = cv_interuption
                    if not util.objective_function_criterium(config, scores_obj, best_first_scores, feature_penalty=config.feature_penalty, margin=margin):
                        return forest, (scores_obj, scores_obj), cross_val_object, slice_slices
                

        if remote:
            results = ray.get(slice_result_ids)
        else:
            results = slice_result_ids

        ret_slice_slices: dict[int, list[CrossValidationSlice]] = slice_slices
        if ret_slice_slices is None:
            ret_slice_slices = {}

        for res in results:
            if remote:
                (
                    scores_obj,
                    packed_cv_slice,
                    cv_counter,
                    reg_forest_times,
                    _slice_slices,
                ) = res
                cv_slice = unpack(packed_cv_slice)
            else:
                (
                    forest,
                    scores_obj,
                    cv_counter,
                    cv_slice,
                    reg_forest_times,
                    _slice_slices,
                ) = res

            total_times = aggregate_times(total_times, reg_forest_times)

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
                raise "Scores must not be None here"

            if config.optimize_mean:
                scores_list.append(scores_obj)
            else:
                if worst_scores is None:
                    worst_scores = scores_obj
                elif util.objective_function_criterium(config, worst_scores, scores_obj):  # if worst_scores ar better than scores_obj
                    worst_scores = scores_obj
        del results

        if config.optimize_mean:
            first_scores = scores_list[0]
            scores_obj = util.mean_scores(scores_list)
            if get_first_scores:
                scores_obj = (first_scores, scores_obj)
        else:
            scores_obj = worst_scores
            forest = worst_forest

    if config.verbosity >= 2:
        print_times(total_times, label = 'Train forest')

    t1 = time.time()
    if config.verbosity >= 2:
        config.logger.info(f"Time for trainForest: {t1 - t0}")

    return forest, scores_obj, cross_val_object, ret_slice_slices
