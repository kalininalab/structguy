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

import cupy as cp
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.mr import PoolMemoryResource, CudaAsyncMemoryResource, set_current_device_resource

from filelock import FileLock, Timeout
from structguy import featureSelection, util
from structman.base_utils.base_utils import pack, unpack, add_to_times, print_times, aggregate_times
from structguy.support_classes import CrossValidationSlice
from structguy.sampleSpace import DataSAIL_cv, SampleSpace
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

@ray.remote
def trainRegressionForestWrapper(
    store: tuple[util.Config, list[list[tuple]], list[str], ray.ObjectRef, dict | None],
    cv_slice: CrossValidationSlice,
    print_out=True,
    skip_scoring=False,
    cv_counter=None,
    debug=False,
    overwrite_proc_n=None,
    skip_feature_selection=False,
    score_train=True,
    gpu_share=None,
    proc_id=0
):
    config, feats_to_filter, samples_store_id, raw_feature_matrix_store_id, distance_map = store
    util.reset_logger_for_remotes(config)
    return trainRegressionForest(
        config,
        cv_slice,
        feats_to_filter,
        samples_store_id=samples_store_id,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        distance_map=distance_map,
        print_out=print_out,
        skip_scoring=skip_scoring,
        cv_counter=cv_counter,
        debug=debug,
        remote=True,
        skip_feature_selection=skip_feature_selection,
        overwrite_proc_n=overwrite_proc_n,
        score_train=score_train,
        sub_gpu_share=gpu_share,
        proc_id=proc_id
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

def ext_shap_analysis(
        feature_names: list[str],
        booster: xgb.Booster,
        sliced_test_emd_matrices: list[xgb.ExtMemQuantileDMatrix]):

    times = []
    ta = time.time()

    acc_feat_impacts = None

    n_samples = 0

    for dmatrix in sliced_test_emd_matrices:
        n_samples += dmatrix.num_row()
        explanation = booster.predict(dmatrix, pred_contribs=True)

        pred_vector = booster.predict(dmatrix)

        acc_feat_impacts_slice = shap_internal_loop(
            len(feature_names),
            explanation,
            pred_vector,
            dmatrix.get_label()
            )
        
        if acc_feat_impacts is None:
            acc_feat_impacts = acc_feat_impacts_slice
        else:
            acc_feat_impacts += acc_feat_impacts_slice

    ta = add_to_times(times, ta)

    acc_feat_impacts = sorted(zip(feature_names, [x/n_samples for x in acc_feat_impacts]), key=lambda x:x[1], reverse=True)
    ta = add_to_times(times, ta)
    return acc_feat_impacts, times

def shap_analysis(config, forest, d_feat_vecs: xgb.DMatrix, feature_names, prediction_vector, target_vector):
    if config.verbosity >= 3:
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



@njit
def jit_spear(arr1: numpy.ndarray, arr2: numpy.ndarray) -> float:
    rarr1: numpy.ndarray = arr1.argsort().argsort()
    rarr2: numpy.ndarray = arr2.argsort().argsort()

    corr: float = numpy.corrcoef(rarr1, rarr2)[0][1]
    return corr

@njit 
def get_pred_true_tuples(predt: numpy.ndarray, true_labels: numpy.ndarray, code_vec: numpy.ndarray):
    a: int = max(code_vec) + 1
    b: int = len(code_vec)
    pred_lists: numpy.ndarray = numpy.empty((a,b,))
    pred_lists[:] = numpy.nan
    true_lists: numpy.ndarray = numpy.empty((a,b,))
    true_lists[:] = numpy.nan
    for pos, code in enumerate(code_vec):
        lab: float = true_labels[pos]
        pred: float = predt[pos]
        true_lists[code][pos] = lab
        pred_lists[code][pos] = pred
    return true_lists, pred_lists

@njit
def jit_mean_spear(predt: numpy.ndarray, true_labels: numpy.ndarray, code_vec: numpy.ndarray) -> float:
    pred_true_tuples: list[tuple[list[float], list[float]]] = []
    for pos, code in enumerate(code_vec):
        if len(pred_true_tuples) == code:
            pred_true_tuples.append(([],[]))
        pred_true_tuples[code][0].append(true_labels[pos])
        pred_true_tuples[code][1].append(predt[pos])
    
    corrs: list[float] = []
    trues: list[float]
    preds: list[float]
    for trues, preds in pred_true_tuples:
        sorted_trues: list[float] = sorted(trues)
        sorted_preds: list[float] = sorted(preds)
        x:float
        rarr1: list[int] = []
        for x in trues:
            index: int = sorted_trues.index(x)
            rarr1.append(index)
        rarr2: list[int] = [sorted_preds.index(x) for x in preds]

        corr: float = numpy.corrcoef(numpy.array(rarr1), numpy.array(rarr2))[0][1]
        #corr: float = jit_spear(numpy.array(trues), numpy.array(preds))
        corrs.append(corr)
    if len(corrs) == 0:
        mean_corr: float = 0.
    else:
        mean_corr: float = sum(corrs)/len(corrs)
    return mean_corr

def rho_eval_for_xgboost_cb(predt: numpy.ndarray, dtest: xgb.DMatrix) -> tuple[str, float]:
    if isinstance(dtest, xgb.DMatrix):
        y = dtest.get_label()

        true_lists, pred_lists = get_pred_true_tuples(predt, y, dtest.encoded_prot_vec)
        
        corrs = []
        for index, trues in enumerate(true_lists):
            #print(f'before {len(trues)=}')
            trues = trues[~numpy.isnan(trues)]
            #print(f'after {len(trues)=}')
            preds = pred_lists[index]
            preds = preds[~numpy.isnan(preds)]
            corr, _ = stats.spearmanr(trues, preds)
            corrs.append(corr)
        if len(corrs) == 0:
            mean_corr = 0.
        else:
            mean_corr = sum(corrs)/len(corrs)

        #mean_corr = jit_mean_spear(predt, y, dtest.encoded_prot_vec)
            
        return 'irho', (1.0-mean_corr)

    else:
        y = dtest
        corr, _ = stats.spearmanr(predt, y)
        return 'irho', (1.0-corr)

def booster_list_process_and_predict(booster_list, cv_slice: CrossValidationSlice, samples):
    y_preds = []
    for booster, feat_names in booster_list:
        cv_slice.set_to_features(feat_names)
        test_feature_matrix = cv_slice.get_dtest(samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix)
        y_preds.append(booster.predict(test_feature_matrix))
    y_pred = numpy.mean(y_preds, axis=0)
    return y_pred

def booster_list_predict(booster_list, feat_mats):
    y_preds = []
    for booster_index, (booster, _) in enumerate(booster_list):
        y_preds.append(booster.predict(feat_mats[booster_index]))
    y_pred = numpy.mean(y_preds, axis=0)
    return y_pred

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
            "colsample_bylevel": config.colsample_bylevel,
            "colsample_bynode": config.colsample_bynode,
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
            "max_cat_threshold": int(config.max_cat_threshold),
            'random_state' : int(time.time())
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
            "colsample_bylevel": config.colsample_bylevel_1,
            "colsample_bynode": config.colsample_bynode_1,
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
            "max_cat_threshold": int(config.max_cat_threshold_1),
            'random_state' : int(time.time())
            }
        forest = xgb.train(xgb_params, dtrain, num_boost_round=int(config.num_of_trees_1), early_stopping_rounds=int(config.early_stopping_1), evals=evals, maximize=False, custom_metric=rho_eval_for_xgboost_cb, callbacks=es_list)

    return forest

def setup_memory_resources(config: util.Config, sub_share: float):
    gmem = util.get_gpu_memory()[0]
    init_pool = 1024*1024*int(gmem*sub_share*0.05)
    max_pool = 1024*1024*int(gmem*sub_share*0.8)

    if config.verbosity >= 4:
        config.logger.info(f'Setup memory resources: {init_pool=} {max_pool=}')
        
    # It's important to use RMM for GPU-based external memory to improve performance.
    # If XGBoost is not built with RMM support, a warning will be raised.
    # We use the pool memory resource here for simplicity, you can also try the
    # `ArenaMemoryResource` for improved memory fragmentation handling.

    amr = CudaAsyncMemoryResource(initial_pool_size=init_pool, release_threshold = 2*init_pool)
    
    mr = PoolMemoryResource(amr, initial_pool_size=init_pool, maximum_pool_size=max_pool)
    set_current_device_resource(mr)
    # Set the allocator for cupy as well.
    cp.cuda.set_allocator(rmm_cupy_allocator)

def retrieve_dmatrix(
        dump_precursor: str,
        config: util.Config,
        raw_feature_matrix_store_id: ray.ObjectRef,
        cv_slice: CrossValidationSlice,
        sub_share: float,
        get_sliced_test_matrices: bool=False,
        only_test: bool =False
        ):

    times = []
    ta = time.time()
    feat_pos_dict, features = ray.get(raw_feature_matrix_store_id)
    ta = add_to_times(times, ta)
    try:
        setup_memory_resources(config, sub_share)
        dtrain, t_file_paths = cv_slice.get_extmem_dtrain(dump_precursor, config, feat_pos_dict, features, sub_share)
        ta = add_to_times(times, ta)
        dtest_feature_matrix, te_file_paths, sliced_test_matrices = cv_slice.get_extmem_dtest(dump_precursor, config, feat_pos_dict, features, sub_share, dtrain, get_sliced_test_matrices = get_sliced_test_matrices)
        ta = add_to_times(times, ta)
        del feat_pos_dict
        del features
        ta = add_to_times(times, ta)
    except (MemoryError, RuntimeError, xgb.core.XGBoostError) as err:
        raise err
        
    if get_sliced_test_matrices:
        return dtrain, dtest_feature_matrix, sliced_test_matrices, t_file_paths, te_file_paths, times
    return dtrain, dtest_feature_matrix, t_file_paths, te_file_paths, times

@ray.remote
def double_booster_remote(packed_slice_slice, store, proc_id: str, sub_share: float):
    config: util.Config
    config, filtered_features, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, retain_model = store
    util.reset_logger_for_remotes(config)
    times = []
    ta = time.time()

    if isinstance(packed_slice_slice, CrossValidationSlice):
        slice_slice = packed_slice_slice
    else:
        slice_slice = unpack(packed_slice_slice)
    ta = add_to_times(times, ta) #0
    slice_slice.filterFeatures(filtered_features)
    ta = add_to_times(times, ta) #1

    lock_file = f'remote_proc_{proc_id.split('_')[0]}.lock'

    dump_precursor = f'{config.tmp_folder}/ext_mem_data_{proc_id}'

    if config.verbosity >= 3:
        config.logger.info(f'Call of double_booster_remote: {lock_file=} {dump_precursor=} {retain_model=}')

    dtrain, dtest_feature_matrix, sliced_test_emd_matrices, t_file_paths, te_file_paths, ret_times = retrieve_dmatrix(
        dump_precursor,
        config,
        raw_feature_matrix_store_id,
        slice_slice,
        sub_share,
        get_sliced_test_matrices = True
        )
    times.append(ret_times)
    ta = add_to_times(times, ta) #2

    if config.verbosity >= 3:
        config.logger.info(f'Reached after first data retrieval in double_booster_remote {proc_id}')


    booster = xgb_train_wrapper(config, dtrain, dtest_feature_matrix)
    ta = add_to_times(times, ta) #3

    if config.verbosity >= 3:
        config.logger.info(f'Reached after first training in double_booster_remote {proc_id}')

    if booster is None:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 1', logger=config.logger)
        del dtrain
        del dtest_feature_matrix
        return None
    
    acc_feat_impacts, shap_times = ext_shap_analysis(slice_slice.feature_names, booster, sliced_test_emd_matrices)

    """
    y_pred = booster.predict(dtest_feature_matrix)
    ta = add_to_times(times, ta) #4

    acc_feat_impacts, shap_times = shap_analysis(config, booster, dtest_feature_matrix, slice_slice.feature_names, y_pred, slice_slice.test_targets)
    
    """
    times.append(shap_times) #5
    ta = add_to_times(times, ta) #6
    
    del booster

    if config.verbosity >= 3:
        config.logger.info(f'Reached after shap analysis in double_booster_remote {proc_id}')

    if acc_feat_impacts is None:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 2', logger=config.logger)
        del dtrain
        del dtest_feature_matrix
        return None
    
    feats_to_remove = filtered_features[:]
    for feat_name, feat_impact in acc_feat_impacts:
        if feat_impact >= config.feat_impact_thresh:
            feats_to_remove.append(feat_name)

    if len(feats_to_remove) >= (len(cv_slice.feature_names)+ len(filtered_features)):
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 3', logger=config.logger)
        del dtrain
        del dtest_feature_matrix
        return None
    
    slice_slice.filterFeatures(feats_to_remove)
    ta = add_to_times(times, ta) #7

    dtrain, dtest_feature_matrix, t_file_paths, te_file_paths, ret_times = retrieve_dmatrix(
        dump_precursor,
        config,
        raw_feature_matrix_store_id,
        slice_slice,
        sub_share
        )
    times.append(ret_times)
    ta = add_to_times(times, ta) #8

    booster_2 = xgb_train_wrapper(config, dtrain, dtest_feature_matrix, second_round=True)
    ta = add_to_times(times, ta) #9

    if config.verbosity >= 3:
        config.logger.info(f'Reached after second training in double_booster_remote {proc_id}')

    if booster_2 is None:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 4', logger=config.logger)
        del dtrain
        del dtest_feature_matrix
        return None
    
    if skip_scoring:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 5', logger=config.logger)
        del dtrain
        del dtest_feature_matrix
        return booster_2, slice_slice.feature_names[:]

    cv_slice.filterFeatures(feats_to_remove)
    ta = add_to_times(times, ta) #10

    if config.verbosity >= 3:
        config.logger.info(f'Reached after cv_slice feat filter in double_booster_remote {proc_id}')

    if score_train:
        dtrain, dtest_feature_matrix, t_file_paths, te_file_paths, ret_times = retrieve_dmatrix(
        dump_precursor,
        config,
        raw_feature_matrix_store_id,
        cv_slice,
        sub_share
        )
        times.append(ret_times)
        y_pred = booster_2.predict(dtest_feature_matrix)
        x_pred = booster_2.predict(dtrain)

    else:
        _, dtest_feature_matrix, _, te_file_paths, ret_times = retrieve_dmatrix(
        dump_precursor,
        config,
        raw_feature_matrix_store_id,
        cv_slice,
        sub_share,
        only_test = True
        )
        times.append(ret_times)
        y_pred = booster_2.predict(dtest_feature_matrix)
        x_pred = None

    
    ta = add_to_times(times, ta) #11

    if config.verbosity >= 3:
        config.logger.info(f'Reached the end of double_booster_remote {proc_id}')

    if config.verbosity >= 4:
        print_times(times, label = 'double booster 6', logger=config.logger)

    del dtrain
    del dtest_feature_matrix

    if retain_model:
        del booster_2
        return y_pred, x_pred
    else:
        return booster_2, slice_slice.feature_names[:], y_pred, x_pred



def trainRegressionForest(
    config: util.Config,
    cv_slice: CrossValidationSlice,
    feats_to_filter: list[str],
    samples: SampleSpace | None =None,
    samples_store_id: ray.ObjectRef | None =None,
    raw_feature_matrix_store_id: ray.ObjectRef | None =None,
    distance_map=None,
    print_out=True,
    skip_scoring=False,
    score_train=True,
    cv_counter=None,
    remote=False,
    overwrite_proc_n=None,
    debug=False,
    skip_feature_selection=False,
    sub_gpu_share=None,
    proc_id=0
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
            times
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
    if config.verbosity >= 3:
        config.logger.info(f"Train regression forest part 1, Threads: {proc}, Feature selection: {not skip_feature_selection} {samples is None=} {config.forest_type=} {config.gpu_mode=} {config.multi_gpu=}")

    if config.auto_weighting:
        cv_slice.check_auto_weights()

    if len(cv_slice.feature_names) == 0:
        config.logger.warning(f"Warning =============== Feature vector has len 0 {cv_counter}")
        return return_zero(zero_return, remote, cv_slice)

    if max_sample_parameter == 1.0:
        max_sample_parameter = None

    if print_out:
        config.logParameter()

    ta = add_to_times(times, ta) #2
    if config.verbosity >= 3:
        config.logger.info(f"Train regression forest part 2, {proc=} {samples is None=} {score_train=} {len(feats_to_filter)=}")
    
    if not config.forest_type == "xgboost" and not skip_feature_selection:
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
            f"Train regression {config.forest_type} forest, call of fit with # of features: {len(cv_slice.feature_names)}, skip feature selection {skip_feature_selection}, skip scoring {skip_scoring}"
        )

    ta = add_to_times(times, ta) #5
    
    if config.forest_type == "xgboost" and not skip_feature_selection:
        booster_list = []
        if sub_gpu_share is not None:
            sub_share = sub_gpu_share/len(cv_slice.slice_slices)
            if sub_share < 1.0 and sub_share > 0.5:
                sub_share = 0.5
            if config.verbosity >= 3:
                config.logger.info(f'call of double_booster_remote: {sub_share=}')
            remote_function = double_booster_remote.options(num_gpus = sub_share)
            remote_proc_ids = []
            
            store = ray.put((config, feats_to_filter, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, remote))
            for nested_proc_id, packed_slice_slice in enumerate(cv_slice.slice_slices):
                remote_proc_ids.append(remote_function.remote(packed_slice_slice, store, f'{proc_id}_{nested_proc_id}', sub_share))

            done = False
            y_preds = []
            x_preds = []
            booster_2_list = []

            while not done:
                ready, not_ready = ray.wait(remote_proc_ids, timeout = 1)

                if len(ready) > 0:
                    if config.verbosity >= 3:
                        config.logger.info(f'Double booster returned {len(ready)=}')
                    results = ray.get(ready)
                    
                    for res in results:
                        if res is None:
                            if config.verbosity >= 1:
                                config.logger.info('double_booster_remote returned None')
                            return return_zero(zero_return, remote, cv_slice)
                        if skip_scoring:
                            booster_2_list.append(res)
                        elif not remote:
                            booster, sl_sl_feat_names, y_pred, x_pred = res
                            booster_2_list.append((booster, sl_sl_feat_names))
                            y_preds.append(y_pred)
                            x_preds.append(x_pred)
                        else:
                            y_pred, x_pred = res
                            y_preds.append(y_pred)
                            x_preds.append(x_pred)

                    for proc in ready:
                        ray.cancel(proc, force=True)

                remote_proc_ids = not_ready
                if len(remote_proc_ids) == 0:
                    done = True

            if not skip_scoring:
                y_pred = numpy.mean(y_preds, axis=0)
                if score_train:
                    x_pred = numpy.mean(x_preds, axis=0)
        else:
            if samples is None:
                samples = unpack(ray.get(samples_store_id))
            for stored_slice_slice in cv_slice.slice_slices:
                slice_slice = unpack(ray.get(stored_slice_slice))
                dtrain = slice_slice.get_dtrain(samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix, sub_sampling=config.sub_sample_factor)
                
                dtest_feature_matrix = slice_slice.get_dtest(samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix, dtrain)

                ta = add_to_times(times, ta) #6

                booster = xgb_train_wrapper(config, dtrain, dtest_feature_matrix)
                if booster is None:
                    return return_zero(zero_return, remote, cv_slice)
                booster_list.append((booster, dtest_feature_matrix))

    ta = add_to_times(times, ta) #7

    if config.forest_type == 'xgboost' and not skip_feature_selection:
        
        test_feat_mats = []
        train_feat_mats = []
        if sub_gpu_share is None:
            booster_2_list = []
            for booster_index, (booster, dtest_feature_matrix) in enumerate(booster_list):
                slice_slice = unpack(ray.get(cv_slice.slice_slices[booster_index]))
                ta = add_to_times(times, ta) #8

                y_pred = booster.predict(dtest_feature_matrix)

                ta = add_to_times(times, ta) #9

                acc_feat_impacts, shap_times = shap_analysis(config, booster, dtest_feature_matrix, slice_slice.feature_names, y_pred, slice_slice.test_targets)
                times.append(shap_times)

                ta = add_to_times(times, ta) #10
                if acc_feat_impacts is None:
                    return return_zero(zero_return, remote, cv_slice)

                config.logger.info(f'{acc_feat_impacts[:5]=}\n{acc_feat_impacts[-5:]=}')
                feats_to_remove = feats_to_filter[:]
                for feat_name, feat_impact in acc_feat_impacts:
                    if feat_impact >= config.feat_impact_thresh:
                        feats_to_remove.append(feat_name)

                config.logger.info(f'{len(feats_to_remove)=} {len(feats_to_filter)=} {len(cv_slice.feature_names) + len(feats_to_filter)=}')

                if len(feats_to_remove) >= (len(cv_slice.feature_names)+ len(feats_to_filter)):
                    return return_zero(zero_return, remote, cv_slice)
                

                #if len(feats_to_remove) > len(feats_to_filter):
                cv_slice.filterFeatures(feats_to_remove)
                slice_slice.filterFeatures(feats_to_remove)

                if not skip_scoring:
                    test_feat_mats.append(cv_slice.get_dtest(samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix))
                    train_feat_mats.append(cv_slice.get_dtrain(samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix, sub_sampling=config.sub_sample_factor))

                ta = add_to_times(times, ta) #11

                dtrain = slice_slice.get_dtrain(samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix, sub_sampling=config.sub_sample_factor)

                dtest_feature_matrix = slice_slice.get_dtest(samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix)

                booster_2 = xgb_train_wrapper(config, dtrain, dtest_feature_matrix, second_round=True)
                if booster_2 is None:
                    return return_zero(zero_return, remote, cv_slice)

                booster_2_list.append((booster_2, slice_slice.feature_names[:]))
                

        ta = add_to_times(times, ta) #12
        if skip_scoring:
            return booster_2_list, None, cv_counter, cv_slice, times

    elif skip_scoring:
        return forest, None, cv_counter, cv_slice, times

    if debug:
        cv_slice.printBalance(config)

    if sub_gpu_share is None:
        try:
            y_pred = booster_list_predict(booster_2_list, test_feat_mats)
        except ValueError:
            [e, f, g] = sys.exc_info()
            g = traceback.format_exc()
            config.logger.info(f'Catched error {config.forest_type=} {skip_feature_selection=}: {e}\n{f}\n{g}\n')
            return return_zero(zero_return, remote, cv_slice)
        if score_train:
            x_pred = booster_list_predict(booster_2_list, train_feat_mats)
                

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
        return scores_obj, cv_slice, cv_counter, times
    return booster_2_list, scores_obj, cv_counter, cv_slice, times


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
    cross_val_object: CrossValidationSlice | DataSAIL_cv | dict[int, ray.ObjectRef],
    feat_corr_matrix,
    feature_names,
    samples_store_id: ray.ObjectRef | None =None,
    raw_feature_matrix_store_id: ray.ObjectRef | None=None,
    samples: SampleSpace | None =None,
    distance_map=None,
    repeat=1,
    print_out=False,
    cv_repeat=False,
    skip_scoring=False,
    remote=False,
    debug=False,
    para_number=None,
    skip_feature_selection=False,
    score_train=True,
    get_first_scores=False,
    cv_interuption=None,
    gpu_share=None,
    proc_id=0
) -> tuple[RandomForestRegressor | None, util.Scores, CrossValidationSlice | DataSAIL_cv, None | list[CrossValidationSlice] | dict[int, list[CrossValidationSlice]]]:
    # if cv_repeat is False, the cross_val_object is a cross validation slice object instead
    zero_scores_obj = util.Scores(zero=True)
    # if para_number == 1:

    if config.suppress_remote_forests or debug or config.gpu_mode:
        para_number = None

    if not cv_repeat:
        if len(cross_val_object.feature_names) < 1:
            config.logger.info(f"Call of trainForest without features: {cross_val_object.name}")
            return None, zero_scores_obj, cross_val_object

    if config.verbosity >= 3 or debug:
        config.logger.info(f"Call of trainForest: {repeat=}, {cv_repeat=}, {remote=}, {para_number=}, {skip_feature_selection=}, {debug=}, {samples is None=}")

    t0 = time.time()

    #identify the highly correlated features
    feats_to_filter = featureSelection.filterCorrelatedFeats(config, feat_corr_matrix, feature_names)

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
                booster_list,
                scores_obj,
                cv_counter,
                cv_slice,
                reg_forest_times,
            ) = trainRegressionForest(
                config,
                cross_val_object,
                feats_to_filter,
                samples_store_id=samples_store_id,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                samples=samples,
                distance_map=distance_map,
                print_out=print_out,
                skip_scoring=skip_scoring,
                debug=debug,
                skip_feature_selection=skip_feature_selection,
                overwrite_proc_n=para_number,
                score_train=score_train,
                sub_gpu_share=gpu_share,
                proc_id=proc_id
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
            #forest = worst_forest
            if config.optimize_mean:
                scores_obj = util.mean_scores(scores_list)

    else:  # This can be used to perform a hyperparameter optimization on the whole dataset
        slice_result_ids = []

        if config.cv_hpo_limiter is not None:
            if config.cv_counters is not None:
                cv_counters = config.cv_counters
            else:
                if remote:
                    cv_counters = list(cross_val_object.keys())[0 : config.cv_hpo_limiter]
                else:
                    cv_counters = list(cross_val_object.slices.keys())[0 : config.cv_hpo_limiter]
                config.cv_counters = cv_counters
                if config.verbosity >= 1:
                    config.logger.info(f"\nCross validation in HPO limited to: {cv_counters}\n")
        else:
            if not isinstance(cross_val_object, DataSAIL_cv):
                cv_counters = cross_val_object.keys()
            else:
                cv_counters = cross_val_object.slices.keys()

        if remote:
            if para_number is None:
                para_number = config.proc_n // len(cv_counters)
            else:
                para_number = para_number // len(cv_counters)

            quota = gpu_share/len(cv_counters)
            remote_wrapper_function = trainRegressionForestWrapper

            remote_store = ray.put((config, feats_to_filter, samples_store_id, raw_feature_matrix_store_id, distance_map))

        cv_repeat_scores = []
        cv_repeat_first_scores = []

        for i in range(0, repeat):

            for cv_id, cv_counter in enumerate(cv_counters):
                if cv_id == 0:
                    first_cv_counter = cv_counter
                
                if remote:
                    if isinstance(cross_val_object, DataSAIL_cv):
                        cv_slice = cross_val_object.slices[cv_counter]
                    else:
                        cv_slice: CrossValidationSlice = cross_val_object[cv_counter]
                    if samples_store_id is None:
                        samples_store_id = ray.put(pack(samples))

                    slice_result_ids.append(
                        remote_wrapper_function.remote(
                            remote_store,
                            cv_slice,
                            print_out=print_out,
                            cv_counter=cv_counter,
                            debug=debug,
                            skip_feature_selection=skip_feature_selection,
                            overwrite_proc_n=para_number,
                            skip_scoring=skip_scoring,
                            score_train=score_train,
                            gpu_share=quota,
                            proc_id = f'{proc_id}_{cv_id}'
                        )
                    )
                    
                else:
                    cv_slice: CrossValidationSlice = cross_val_object.slices[cv_counter]

                    slice_result_ids.append(
                        trainRegressionForest(
                            config,
                            cv_slice,
                            feats_to_filter,
                            samples=samples,
                            samples_store_id=samples_store_id,
                            raw_feature_matrix_store_id=raw_feature_matrix_store_id,
                            distance_map=distance_map,
                            print_out=print_out,
                            cv_counter=cv_counter,
                            overwrite_proc_n=para_number,
                            debug=debug,
                            skip_feature_selection=skip_feature_selection,
                            skip_scoring=skip_scoring,
                            score_train=score_train,
                            sub_gpu_share=gpu_share,
                            proc_id = proc_id
                        )
                    )
                    if cv_interuption is not None and len(slice_result_ids) == 1:
                        (
                            forest,
                            scores_obj,
                            cv_counter,
                            cv_slice,
                            reg_forest_times
                        ) = slice_result_ids[0]
                        margin, best_first_scores = cv_interuption
                        if not util.objective_function_criterium(config, scores_obj, best_first_scores, feature_penalty=config.feature_penalty, margin=margin):
                            obj_first_score = util.get_objective_score(config, scores_obj, feature_penalty = config.feature_penalty)
                            estimated_obj_score = obj_first_score + config.estimation_delta
                            return forest, (estimated_obj_score, scores_obj), cross_val_object
                    

            if remote:
                results = ray.get(slice_result_ids)
            else:
                results = slice_result_ids

            for res in results:
                if remote:
                    (
                        scores_obj,
                        cv_slice,
                        cv_counter,
                        reg_forest_times
                    ) = res
                    booster_list = None
                else:
                    (
                        booster_list,
                        scores_obj,
                        cv_counter,
                        cv_slice,
                        reg_forest_times
                    ) = res

                total_times = aggregate_times(total_times, reg_forest_times)

                if cv_slice is not None:
                    del cross_val_object.slices[cv_counter]
                    cross_val_object.slices[cv_counter] = cv_slice

                if scores_obj is None:
                    raise "Scores must not be None here"

                if cv_counter == first_cv_counter:
                    first_scores = scores_obj

                if config.optimize_mean:
                    scores_list.append(scores_obj)
                else:
                    if worst_scores is None:
                        worst_scores = scores_obj
                    elif util.objective_function_criterium(config, worst_scores, scores_obj):  # if worst_scores ar better than scores_obj
                        worst_scores = scores_obj
            del results

            if config.optimize_mean:
                scores_obj = util.mean_scores(scores_list)
                
            else:
                scores_obj = worst_scores
                forest = worst_forest

            if repeat > 1:
                cv_repeat_scores.append(scores_obj)
                cv_repeat_first_scores.append(first_scores)
                scores_list = []

        if repeat > 1:
            scores_obj = util.mean_scores(cv_repeat_scores)
            first_scores = util.mean_scores(cv_repeat_first_scores)

        if get_first_scores:
            scores_obj = (first_scores, scores_obj)

    if config.verbosity >= 3:
        print_times(total_times, label = 'Train forest', logger=config.logger)

    t1 = time.time()
    if config.verbosity >= 2:
        config.logger.info(f"Time for trainForest: {t1 - t0}")

    return booster_list, scores_obj, cross_val_object