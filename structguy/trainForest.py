#from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
#from sklearn.ensemble import RandomForestClassifier
#from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
#from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
#from sklearn.metrics import roc_auc_score
#from sklearn.metrics import precision_score
#from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

#from memory_profiler import profile

import time
import sys
import os
import signal
import traceback
import ray
#import contextlib
from scipy import stats
import xgboost as xgb
#from ray.train.xgboost import XGBoostTrainer, RayTrainReportCallback

import cupy as cp
import cuda.bindings.driver as driver
import cuda.bindings.runtime as cudart
from cupy.cuda import MemoryAsyncPool

import rmm
from rmm.allocators.cupy import rmm_cupy_allocator
from rmm.allocators.numba import RMMNumbaManager
from rmm.mr import PoolMemoryResource, CudaAsyncMemoryResource, set_current_device_resource, set_per_device_resource, CudaMemoryResource, ArenaMemoryResource

#from filelock import FileLock, Timeout
from structguy import featureSelection, util
from structman.base_utils.base_utils import pack, unpack, add_to_times, print_times, aggregate_times
from structman.lib.sdsc.sdsc_utils import deep_get_size_of, sizeof_fmt
from structguy.support_classes import CrossValidationSlice
from structguy.sampleSpace import DataSAIL_cv, SampleSpace
import numpy
#from ray.util.queue import Queue
#from ray.train import RunConfig
#import shap
import numba
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
    store: tuple[list[list[tuple]], list[str], ray.ObjectRef, dict | None],
    cv_slice: CrossValidationSlice,
    config_ref_container: list[ray.ObjectRef],
    subslice_refs: list[ray.ObjectRef],
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
    feats_to_filter, raw_feature_matrix_store_id = store
    config: util.Config = ray.get(config_ref_container[0])
    util.reset_logger_for_remotes(config)

    if config.verbosity >= 4:
        config.logger.info(f'Call of trainRegressionForestWrapper {type(cv_slice)=}')

    return trainRegressionForest(
        config,
        cv_slice,
        feats_to_filter,
        raw_feature_matrix_store_id=raw_feature_matrix_store_id,
        print_out=print_out,
        skip_scoring=skip_scoring,
        cv_counter=cv_counter,
        debug=debug,
        remote=True,
        skip_feature_selection=skip_feature_selection,
        overwrite_proc_n=overwrite_proc_n,
        score_train=score_train,
        sub_gpu_share=gpu_share,
        proc_id=proc_id,
        config_ref_container=config_ref_container,
        subslice_refs = subslice_refs
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

#@profile
def ext_shap_analysis(
        feature_names: list[str],
        booster: xgb.Booster,
        data_refs: list[tuple[ray.ObjectRef, ray.ObjectRef, ray.ObjectRef]]):

    times = []
    ta = time.time()

    acc_feat_impacts = None
    feat_names, cat_vec, feat_id_vec = ray.get(data_refs[0][2])

    n_samples: int = 0

    for X_ref, y_ref, _ in data_refs:

        X: numpy.ndarray = ray.get(X_ref)
        X = X[:, feat_id_vec]
        n_samples += len(X)

        dmatrix: xgb.DMatrix = xgb.DMatrix(X, feature_names=feat_names, feature_types=cat_vec, enable_categorical=True)

        explanation = booster.predict(dmatrix, pred_contribs=True)

        pred_vector = booster.predict(dmatrix)

        acc_feat_impacts_slice = shap_internal_loop(
            len(feature_names),
            explanation,
            pred_vector,
            ray.get(y_ref)
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

def rho_eval_for_xgboost_cb(predt: cp.ndarray, dtest: xgb.DMatrix) -> tuple[str, float]:
    # Ensure inputs are on GPU
    # Note: If predt is numpy, cp.asarray(predt) is fast but does a copy.
    y = cp.asarray(dtest.get_label())
    p = cp.asarray(predt)
    
    # 1. Handle NaNs globally (Vectorized)
    mask = ~cp.isnan(y) & ~cp.isnan(p)
    y = y[mask]
    p = p[mask]

    # 2. Get IDs for grouping (encoded_prot_vec)
    # Assuming dtest.encoded_prot_vec is already a cupy array or can be converted
    ids = cp.asarray(dtest.encoded_prot_vec)[mask]

    if ids.size == 0:
        return 'irho', 1.0

    # 3. Vectorized Spearman Rank (The Magic Part)
    # We sort by (ID, Value) to rank within groups efficiently
    def get_ranks(val, group_ids):
        # Sort by group, then by value
        idx = cp.lexsort(cp.stack([val, group_ids]))
        group_ids_sorted = group_ids[idx]
        
        # Identify group boundaries
        change_mask = cp.empty(group_ids_sorted.size, dtype=cp.bool_)
        change_mask[0] = True
        change_mask[1:] = group_ids_sorted[1:] != group_ids_sorted[:-1]
        
        # Calculate ranks within groups using a cumulative count
        # This is a common pattern to avoid Python loops
        all_ranks = cp.arange(len(val))
        group_starts = cp.where(change_mask)[0]
        # Subtract the start index of each group from the global rank
        ranks = all_ranks - cp.take(group_starts, cp.searchsorted(group_starts, all_ranks, side='right') - 1)
        
        # Invert the sort to original order
        return ranks[cp.argsort(idx)]

    y_ranks = get_ranks(y, ids)
    p_ranks = get_ranks(p, ids)

    # 4. Vectorized Pearson on the Ranks (Mean Correlation)
    # Instead of a loop, we calculate the covariance for all groups at once
    def grouped_pearson(r1, r2, group_ids):
        # Implementation of mean correlation across groups using cupy.add.at or groupby logic
        # For simplicity, if groups are balanced, you can reshape. 
        # If unbalanced, a CuPy-based groupby-mean is needed.
        # Alternatively, a simple global correlation is much faster:
        return cp.corrcoef(r1, r2)[0, 1]

    mean_corr = grouped_pearson(y_ranks, p_ranks, ids)
    
    return 'irho', float(1.0 - mean_corr)


"""
def rho_eval_for_xgboost_cb(predt: numpy.ndarray, dtest: xgb.DMatrix) -> tuple[str, float]:
    if isinstance(dtest, xgb.DMatrix):
        print(f'{type(predt)=}')
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
"""
        
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

#@profile
def xgb_train_wrapper(
        config: util.Config,
        dtrain: xgb.DMatrix,
        dtest_feature_matrix: xgb.DMatrix,
        second_round = False,
        ):
        
    es_list = []
    evals: list[tuple[xgb.DMatrix, str]] = []
    eval_label = 'eval'
    
    #if config.use_external_memory_qdm:
    if True:
        if config.setup_cuda_mem:
            mem_context = xgb.config_context(use_cuda_async_pool=True)
        else:
            mem_context = xgb.config_context(use_rmm=True)
    else:
        mem_context = xgb.config_context()

    with mem_context:
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
                'max_bin' : 512,
                'sampling_method': 'gradient_based',
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
                'max_bin' : 512,
                'sampling_method': 'gradient_based',
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

def setup_memory_resources(config: util.Config, sub_share: float, cuda_setup=True):
    if cuda_setup:
        setup_cuda_memory(config, sub_share)
    else:
        setup_rmm_memory(config, sub_share)

def setup_cuda_memory(config: util.Config, sub_share: float):
    # Get the default memory pool and configure the release threshold
    status, dft_pool = cudart.cudaDeviceGetDefaultMemPool(0)
    # Set the release threshold to 90% of total device memory
    status, free, total = cudart.cudaMemGetInfo()

    pool_size = int(total * 0.9 * sub_share)

    if config.verbosity >= 4:
        config.logger.info(f'Setup cuda memory resources: {pool_size=}')

    v = driver.cuuint64_t(pool_size)

    cudart.cudaMemPoolSetAttribute(
        dft_pool,
        cudart.cudaMemPoolAttr.cudaMemPoolAttrReleaseThreshold,
        v,
    )
    # Set the allocator for cupy as well.
    cp.cuda.set_allocator(MemoryAsyncPool().malloc)

#@profile
def setup_rmm_memory(config: util.Config, sub_share: float):
    gmem = util.get_gpu_memory()[0]
    init_pool = 1024*1024*int(gmem*sub_share*0.7)
    max_pool = 1024*1024*int(gmem*sub_share*0.99)

    if config.verbosity >= 4:
        config.logger.info(f'Setup memory resources: {init_pool=} {max_pool=} {config.multi_gpu=}')
        
    # It's important to use RMM for GPU-based external memory to improve performance.
    # If XGBoost is not built with RMM support, a warning will be raised.
    # We use the pool memory resource here for simplicity, you can also try the
    # `ArenaMemoryResource` for improved memory fragmentation handling.

    #amr = CudaAsyncMemoryResource(initial_pool_size=init_pool, release_threshold = 2*init_pool)
    
    #mr = PoolMemoryResource(amr, initial_pool_size=init_pool, maximum_pool_size=max_pool)
    
    """
    if config.multi_gpu > 1:
        for device_id in range(config.multi_gpu):
            with cp.cuda.Device(device_id):
                mr = CudaMemoryResource()
                #mr = ArenaMemoryResource(mr, arena_size=max_pool)
                mr = PoolMemoryResource(mr, initial_pool_size=init_pool, maximum_pool_size=max_pool)
                set_per_device_resource(device_id, mr)
                cp.cuda.set_allocator(rmm_cupy_allocator)
    """
    #else:

    mr = CudaAsyncMemoryResource()
    #mr = CudaMemoryResource()
    #mr = ArenaMemoryResource(mr, arena_size=max_pool)
    mr = PoolMemoryResource(mr, initial_pool_size=init_pool, maximum_pool_size=max_pool)
    set_current_device_resource(mr)

    # Set the allocator for cupy as well.
    #cp.cuda.set_allocator(rmm.allocators.cupy.rmm_cupy_allocator)
    cp.cuda.set_allocator(rmm_cupy_allocator)
    #numba.cuda.set_memory_manager(rmm.allocators.numba.RMMNumbaManager)
    numba.cuda.set_memory_manager(RMMNumbaManager)


#@profile
def retrieve_dmatrix(
        config: util.Config,
        raw_feature_matrix_store_id: ray.ObjectRef,
        cv_slice: CrossValidationSlice,
        ext_mem = False
        ):

    times = []
    ta = time.time()
    feat_pos_dict, features = ray.get(raw_feature_matrix_store_id)
    ta = add_to_times(times, ta)
    if ext_mem:
        try:
            dtrain, t_file_paths = cv_slice.get_extmem_dtrain(config, feat_pos_dict, features)
            ta = add_to_times(times, ta)
            dtest_feature_matrix, test_data_refs = cv_slice.get_extmem_dtest(config, feat_pos_dict, features, dtrain)
            ta = add_to_times(times, ta)
            del feat_pos_dict
            del features
            ta = add_to_times(times, ta)
        except (MemoryError, RuntimeError, xgb.core.XGBoostError) as err:
            raise err
        
    else:
        dtrain = cv_slice.dmat_from_disc(config, feat_pos_dict, features)
        t_file_paths = None
        ta = add_to_times(times, ta)

        dtest_feature_matrix, test_data_refs = cv_slice.dtest_from_disc(config, feat_pos_dict, features, dtrain)

        ta = add_to_times(times, ta)
        del feat_pos_dict
        del features

    return dtrain, dtest_feature_matrix, t_file_paths, test_data_refs, times

@ray.remote(max_retries=0)
#@profile
def double_booster_remote(packed_slice_slice, store, proc_id: str, sub_share: float, config_ref_container: list[ray.ObjectRef]):
    config: util.Config
    filtered_features, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, retain_model = store
    config = ray.get(config_ref_container[0])
    cv_slice = unpack(cv_slice)
    util.reset_logger_for_remotes(config)
    if config.verbosity >= 4:
        config.logger.info(f'Call of double_booster_remote: {type(packed_slice_slice)=}')

    return double_booster(packed_slice_slice, proc_id, sub_share, config, filtered_features, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, retain_model)

#@profile
def double_booster(packed_slice_slice, proc_id, sub_share, config, filtered_features, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, retain_model):
    times = []
    ta = time.time()

    p_id = os.getpid()

    if isinstance(packed_slice_slice, CrossValidationSlice):
        slice_slice = packed_slice_slice
    else:
        slice_slice = unpack(packed_slice_slice)
        del packed_slice_slice

    ta = add_to_times(times, ta) #0
    slice_slice.filterFeatures(filtered_features)
    ta = add_to_times(times, ta) #1

    if config.verbosity >= 4:
        config.logger.info(f'Call of double_booster: {retain_model=}')
    if config.verbosity >= 5:
        slice_slice.log_attr_sizes(config.logger, label = f'slice_slice {proc_id} ')
        cv_slice.log_attr_sizes(config.logger, label = f'cv slice {proc_id} ')
        config.logger.info(f'{p_id=} {ray.get_runtime_context().get()=}')

    #if config.use_external_memory_qdm:
    setup_memory_resources(config, sub_share, cuda_setup=config.setup_cuda_mem)

    if config.verbosity >= 3:
        config.logger.info(f'Reached after memory setup in double_booster_remote {proc_id} {config.setup_cuda_mem=}')

    dtrain, dtest_feature_matrix, _ , test_data_refs, ret_times = retrieve_dmatrix(
        config,
        raw_feature_matrix_store_id,
        slice_slice,
        ext_mem=config.use_external_memory_qdm
        )
    times.append(ret_times) #2
    ta = add_to_times(times, ta) #3

    if config.verbosity >= 3:
        config.logger.info(f'Reached after first data retrieval in double_booster_remote {proc_id}')


    booster = xgb_train_wrapper(config, dtrain, dtest_feature_matrix)
    del dtrain
    del dtest_feature_matrix
    ta = add_to_times(times, ta) #4

    if config.verbosity >= 3:
        config.logger.info(f'Reached after first training in double_booster_remote {proc_id}')

    if booster is None:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 1', logger=config.logger)

        return p_id
    
    acc_feat_impacts, shap_times = ext_shap_analysis(slice_slice.feature_names, booster, test_data_refs)
    
    if config.verbosity >= 4:
        for name, size in sorted(((name, deep_get_size_of(value)) for name, value in locals().items()), key=lambda x: -x[1])[:10]:
            config.logger.info("In dbr: {:>30}: {:>8}".format(name, sizeof_fmt(size)))

        for name, size in sorted(((name, deep_get_size_of(value)) for name, value in globals().items()), key=lambda x: -x[1])[:10]:
            config.logger.info("Globals in dbr: {:>30}: {:>8}".format(name, sizeof_fmt(size)))

        config.logger.info(f'{config.feat_impact_thresh=} {acc_feat_impacts=}')

        if proc_id == '0_0_0':
            util.dump_ray_logs_snapshot(f'{config.outfolder}/ray_dump_snapshot_0.log')

    ta = add_to_times(times, ta) #5

    """
    y_pred = booster.predict(dtest_feature_matrix)
    

    acc_feat_impacts, shap_times = shap_analysis(config, booster, dtest_feature_matrix, slice_slice.feature_names, y_pred, slice_slice.test_targets)
    
    """
    times.append(shap_times) #6
    ta = add_to_times(times, ta) #7
    
    del booster

    if config.verbosity >= 3:
        config.logger.info(f'Reached after shap analysis in double_booster_remote {proc_id}')

    if acc_feat_impacts is None:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 2', logger=config.logger)
        
        return p_id
    
    feats_to_remove = filtered_features[:]
    for feat_name, feat_impact in acc_feat_impacts:
        if feat_impact >= config.feat_impact_thresh:
            feats_to_remove.append(feat_name)

    if len(feats_to_remove) >= (len(cv_slice.feature_names)+ len(filtered_features)):
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 3', logger=config.logger)
        return p_id
    
    slice_slice.filterFeatures(feats_to_remove)
    ta = add_to_times(times, ta) #8

    dtrain, dtest_feature_matrix, _, _, ret_times = retrieve_dmatrix(
        config,
        raw_feature_matrix_store_id,
        slice_slice,
        ext_mem=config.use_external_memory_qdm
        )
    times.append(ret_times) #9
    ta = add_to_times(times, ta) #10

    if config.verbosity >= 4:
        config.logger.info('Reached after second data retrieval')
        if proc_id == '0_0_0':
            util.dump_ray_logs_snapshot(f'{config.outfolder}/ray_dump_snapshot_1.log')

    booster_2 = xgb_train_wrapper(config, dtrain, dtest_feature_matrix, second_round=True)
    ta = add_to_times(times, ta) #11

    if config.verbosity >= 3:
        config.logger.info(f'Reached after second training in double_booster_remote {proc_id}')

    if booster_2 is None:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 4', logger=config.logger)
        del dtrain
        del dtest_feature_matrix
        return p_id
    
    if skip_scoring:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 5', logger=config.logger)
        del dtrain
        del dtest_feature_matrix
        return booster_2, slice_slice.feature_names[:], p_id

    cv_slice.filterFeatures(feats_to_remove)
    ta = add_to_times(times, ta) #12

    if config.verbosity >= 3:
        config.logger.info(f'Reached after cv_slice feat filter in double_booster_remote {proc_id} {cv_slice.train_slice_ids=} {score_train=}')

    if score_train:
        dtrain, dtest_feature_matrix, _, _, ret_times = retrieve_dmatrix(
        config,
        raw_feature_matrix_store_id,
        cv_slice,
        ext_mem=config.use_external_memory_qdm
        )
        times.append(ret_times) #13

        x_pred = booster_2.predict(dtrain)
        x_true = dtrain.get_label()
        x_prot_vec = cp.asnumpy(dtrain.encoded_prot_vec)

        del dtrain

        if config.verbosity >= 5:
            train_scores_obj = calc_scores_obj(x_true, x_pred, x_prot_vec, cv_slice.train_class_weight_vector, cv_slice.feature_names)
            config.logger.info('Train scores:')
            train_scores_obj.printOut(config = config)

    else:
        del dtrain
        _, dtest_feature_matrix, _, _, ret_times = retrieve_dmatrix(
        config,
        raw_feature_matrix_store_id,
        cv_slice,
        ext_mem=config.use_external_memory_qdm
        )
        times.append(ret_times) #13
        
        x_pred = None
        x_true = None
        x_prot_vec = None

    y_pred = booster_2.predict(dtest_feature_matrix)
    y_true = dtest_feature_matrix.get_label()
    y_prot_vec = cp.asnumpy(dtest_feature_matrix.encoded_prot_vec)

    ta = add_to_times(times, ta) #14

    if config.verbosity >= 3:
        config.logger.info(f'Reached the end of double_booster_remote {proc_id} {type(y_pred)=} {type(y_true)=} {type(y_prot_vec)=} {type(x_pred)=} {type(x_true)=} {type(x_prot_vec)=}')

    if config.verbosity >= 4:
        print_times(times, label = 'double booster 6', logger=config.logger)

    del dtest_feature_matrix

    if retain_model:
        del booster_2
        return y_pred, y_true, y_prot_vec, x_pred, x_true, x_prot_vec, p_id
    else:
        return booster_2, slice_slice.feature_names[:], y_pred, y_true, y_prot_vec, x_pred, x_true, x_prot_vec, p_id


#@profile
def trainRegressionForest(
    config: util.Config,
    packed_cv_slice: bytes,
    feats_to_filter: list[str],
    raw_feature_matrix_store_id: ray.ObjectRef | None =None,
    print_out=True,
    skip_scoring=False,
    score_train=True,
    cv_counter=None,
    remote=False,
    overwrite_proc_n=None,
    debug=False,
    skip_feature_selection=False,
    sub_gpu_share=None,
    proc_id=0,
    config_ref_container=None,
    subslice_refs=None
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

    if config.verbosity >= 4:
        config.logger.info(f'Call of trainRegressionForest {type(packed_cv_slice)=} {type(subslice_refs)=}')

    if isinstance(packed_cv_slice, CrossValidationSlice): 
        cv_slice: CrossValidationSlice = packed_cv_slice
    else:
        cv_slice: CrossValidationSlice = unpack(packed_cv_slice)
        del packed_cv_slice

    if subslice_refs is not None:
        cv_slice.slice_slices = subslice_refs

    if config.verbosity >= 4:
        config.logger.info(f'After unpacking in trainRegressionForest {type(cv_slice)=} {cv_slice.slice_slices=}')

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
        return zero_return
    if leaf_samples < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {leaf_samples=}")
        return zero_return
    if min_sample_split < 2:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {min_sample_split=}")
        return zero_return
    if n_of_trees < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {n_of_trees=}")
        return zero_return
    if min_impurity_decrease < 0.0 or min_impurity_decrease > 1.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {min_impurity_decrease=}")
        return zero_return
    if ccp_alpha < 0.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {ccp_alpha=}")
        return zero_return

    if max_sample_parameter <= 0.0 or max_sample_parameter > 1.0:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {max_sample_parameter=}")
        return zero_return

    if skip_feature_selection and (config.fs_max_sample_parameter <= 0.0 or config.fs_max_sample_parameter > 1.0):
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {config.fs_max_sample_parameter=}")
        return zero_return

    if isinstance(max_feature_parameter, float):
        if max_feature_parameter <= 0.0 or max_feature_parameter > 1.0:
            if config.verbosity >= 3:
                config.logger.info(f"Return Zero: {max_feature_parameter=}")
            return zero_return
    if number_of_bins < 1:
        if config.verbosity >= 3:
            config.logger.info(f"Return Zero: {number_of_bins=}")
        return zero_return

    ta = add_to_times(times, ta) #0
    if config.verbosity >= 3:
        config.logger.info(f"Train regression forest part 1, Threads: {proc}, Feature selection: {not skip_feature_selection} {config.forest_type=} {config.gpu_mode=} {config.multi_gpu=} {config.auto_weighting=}")


    if config.auto_weighting:
        cv_slice.check_auto_weights()

    if len(cv_slice.feature_names) == 0:
        config.logger.warning(f"Warning =============== Feature vector has len 0 {cv_counter}")
        return zero_return

    if max_sample_parameter == 1.0:
        max_sample_parameter = None

    if print_out:
        config.logParameter()

    ta = add_to_times(times, ta) #2
    if config.verbosity >= 3:
        config.logger.info(f"Train regression forest part 2, {proc=} {score_train=} {len(feats_to_filter)=}")
    
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
        config.logger.info(f"Train regression forest part 3, {len(cv_slice.feature_names)=}")

    if print_out or config.verbosity >= 3:
        config.logger.info(
            f"Train regression {config.forest_type} forest, call of fit with # of features: {len(cv_slice.feature_names)}, skip feature selection {skip_feature_selection}, skip scoring {skip_scoring} {sub_gpu_share=}"
        )

    if config.verbosity >= 4:
        for name, size in sorted(((name, deep_get_size_of(value)) for name, value in locals().items()), key=lambda x: -x[1])[:10]:
            config.logger.info("{:>30}: {:>8}".format(name, sizeof_fmt(size)))

        cv_slice.log_attr_sizes(config.logger, label = 'CV slice attributes: ')
        config.logger.info('Config attributes:')
        config.log_attr_sizes()

    ta = add_to_times(times, ta) #5
    
    if config.suppress_remote_forests:
        sub_gpu_share = None

    if config.forest_type == "xgboost" and not skip_feature_selection:
        booster_list = []
        if sub_gpu_share is not None:
            sub_share = sub_gpu_share/len(cv_slice.slice_slices)
            if sub_share < 1.0 and sub_share > 0.5:
                sub_share = 0.5
            if config.verbosity >= 3:
                config.logger.info(f'call of double_booster_remotes: {sub_share=} {len(cv_slice.slice_slices)=}')

            if isinstance(proc_id, int):
                gpu_id = proc_id//config.threads_per_gpu
            else:
                gpu_id = int(proc_id.split('_')[0])//config.threads_per_gpu

            try:
                pg = ray.util.get_placement_group(f"pg_{gpu_id}")
                if config.verbosity >= 4:
                    config.logger.info(f'Setting placement group for double booster: pg_{gpu_id} {proc_id=}')
                grouped = True
            except ValueError as e:
                grouped = False
                if config.verbosity >= 4:
                    config.logger.info(f'Could not find a placement group {proc_id=}: {e=}')

            if grouped:
                remote_function = double_booster_remote.options(
                    num_cpus=1,
                    num_gpus=sub_share,
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg , placement_group_capture_child_tasks=True
                        )
                    )
                
                if config.verbosity >= 4:
                    config.logger.info(f'double booster remote configured: {proc_id=}')
                
            else:
                remote_function = double_booster_remote.options(num_gpus=sub_share)
                
            remote_proc_ids = []
            if config_ref_container is None:
                config_ref_container = [ray.put(config)]
            store = ray.put((feats_to_filter, pack(cv_slice), skip_scoring, score_train, raw_feature_matrix_store_id, remote))
            for nested_proc_id, packed_slice_slice in enumerate(cv_slice.slice_slices):

                remote_proc_ids.append(remote_function.remote(packed_slice_slice, store, f'{proc_id}_{nested_proc_id}', sub_share, config_ref_container))

            done = False
            y_preds = []
            x_preds = []
            booster_2_list = []

            ray_dumped = False

            while not done:
                ready, not_ready = ray.wait(remote_proc_ids, timeout = 1)

                if len(ready) > 0:
                    if config.verbosity >= 3:
                        if not ray_dumped:
                            ray_dumped = True
                            util.dump_ray_logs_snapshot(f'{config.outfolder}/ray_dump_snapshot_2.log')
                        config.logger.info(f'Double booster returned {ready=}')
                    try:
                        results = ray.get(ready, timeout=60)
                    except ray.exceptions.GetTimeoutError:
                        remote_proc_ids = not_ready
                        if len(remote_proc_ids) == 0:
                            done = True
                        util.dump_ray_logs_snapshot(f'{config.outfolder}/ray_dump_snapshot_2.log')
                        config.logger.info('double_booster_remote ray.get timed out')
                        continue
                    except ray.exceptions.WorkerCrashedError:
                        remote_proc_ids = not_ready
                        if len(remote_proc_ids) == 0:
                            done = True
                        util.dump_ray_logs_snapshot(f'{config.outfolder}/ray_dump_snapshot_2.log')
                        config.logger.info('double_booster_remote ray.get crashed')
                        continue

                    if config.verbosity >= 4:
                        config.logger.info(f'Double booster returned with {type(results[0])=}')
                    for res in results:
                        if isinstance(res, int):
                            if config.verbosity >= 1:
                                config.logger.info('double_booster_remote returned None')

                            os.kill(res, signal.SIGTERM)
                            return zero_return
                        if skip_scoring:
                            booster, sl_sl_feat_names, db_p_id = res
                            booster_2_list.append((booster, sl_sl_feat_names))
                        elif not remote:
                            booster, sl_sl_feat_names, y_pred, y_true, y_prot_vec, x_pred, x_true, x_prot_vec, db_p_id = res
                            booster_2_list.append((booster, sl_sl_feat_names))
                            y_preds.append(y_pred)
                            x_preds.append(x_pred)
                        else:
                            y_pred, y_true, y_prot_vec, x_pred, x_true, x_prot_vec, db_p_id = res
                            y_preds.append(y_pred)
                            x_preds.append(x_pred)

                        os.kill(db_p_id, signal.SIGTERM)

                remote_proc_ids = not_ready
                if len(remote_proc_ids) == 0:
                    done = True

            #for packed_slice_slice in cv_slice.slice_slices:
            #    ray._private.internal_api.free(packed_slice_slice)

        else:
            y_preds = []
            x_preds = []
            booster_2_list = []
            for nested_proc_id, packed_slice_slice in enumerate(cv_slice.slice_slices):
                res = double_booster(packed_slice_slice, f'{proc_id}_{nested_proc_id}', 1.0, config, feats_to_filter, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, remote)

                if isinstance(res, int):
                    if config.verbosity >= 1:
                        config.logger.info('double_booster_remote returned None')
                    return zero_return
                
                if skip_scoring:
                    booster, sl_sl_feat_names, db_p_id = res
                    booster_2_list.append((booster, sl_sl_feat_names))
                elif not remote:
                    booster, sl_sl_feat_names, y_pred, y_true, y_prot_vec, x_pred, x_true, x_prot_vec, db_p_id = res
                    booster_2_list.append((booster, sl_sl_feat_names))
                    y_preds.append(y_pred)
                    x_preds.append(x_pred)
                else:
                    y_pred, y_true, y_prot_vec, x_pred, x_true, x_prot_vec, db_p_id = res
                    y_preds.append(y_pred)
                    x_preds.append(x_pred)

        if not skip_scoring:
            y_pred = numpy.mean(y_preds, axis=0)
            cv_slice.test_targets = y_true
            cv_slice.test_prot_vec = y_prot_vec
            if score_train:
                x_pred = numpy.mean(x_preds, axis=0)
                cv_slice.train_targets = x_true

    ta = add_to_times(times, ta) #7

    if config.forest_type == 'xgboost' and not skip_feature_selection:
        
        test_feat_mats = []
        train_feat_mats = []
 
        ta = add_to_times(times, ta) #12
        if skip_scoring:
            return booster_2_list, None, cv_counter, times

    elif skip_scoring:
        return forest, None, cv_counter, times

    if debug:
        cv_slice.printBalance(config)

    if sub_gpu_share is None:
        try:
            y_pred = booster_list_predict(booster_2_list, test_feat_mats)
        except ValueError:
            [e, f, g] = sys.exc_info()
            g = traceback.format_exc()
            config.logger.info(f'Catched error {config.forest_type=} {skip_feature_selection=}: {e}\n{f}\n{g}\n')
            return zero_return
        if score_train:
            x_pred = booster_list_predict(booster_2_list, train_feat_mats)
                

    ta = add_to_times(times, ta) #8/13

    if test_for_constant_array(y_pred):
        if print_out:
            zero_scores.printOut(config = config)
        if config.verbosity >= 3:
            config.logger.info("Return Zero: test_for_constant_array was True")
        return zero_return

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

    scores_obj = calc_scores_obj(cv_slice.test_targets, y_pred, cv_slice.test_prot_vec, cv_slice.test_class_weight_vector, cv_slice.feature_names, runtime_penalty = t_complete)
    if score_train:
        train_scores_obj = calc_scores_obj(cv_slice.train_targets, x_pred, x_prot_vec, cv_slice.train_class_weight_vector, cv_slice.feature_names, runtime_penalty = t_complete)
        scores_obj.train_scores = train_scores_obj
        if print_out:
            config.logger.info('Train scores:')
            train_scores_obj.printOut(config = config)
    
    if print_out or debug:
        scores_obj.printOut(config = config)

    ta = add_to_times(times, ta) #9/14

    if remote:
        return scores_obj, cv_counter, times
    return booster_2_list, scores_obj, cv_counter, times


def calc_scores_obj(target_vector, prediction_vector, prot_id_vec, weight_vector, feature_names, runtime_penalty = 0.):
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

    prot_wise_spearmans, mean_spearman, _ = util.calc_protein_wise_corr(target_vector, prediction_vector, prot_id_vec, stats.spearmanr)
    prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(target_vector, prediction_vector, prot_id_vec, stats.pearsonr)

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
        #runtime_penalty=runtime_penalty
    )
    return scores_obj

def trainForest(
    config: util.Config,
    cross_val_object: CrossValidationSlice | DataSAIL_cv | dict[int, ray.ObjectRef],
    feat_corr_matrix,
    feature_names,
    raw_feature_matrix_store_id: ray.ObjectRef | None=None,
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
    proc_id=0,
    config_ref_container: list[ray.ObjectRef] | None =None
) -> tuple[None, util.Scores, CrossValidationSlice | DataSAIL_cv, None | list[CrossValidationSlice] | dict[int, list[CrossValidationSlice]]]:
    # if cv_repeat is False, the cross_val_object is a cross validation slice object instead
    zero_scores_obj = util.Scores(zero=True)
    # if para_number == 1:

    if config.suppress_remote_forests or debug or config.gpu_mode:
        para_number = None

    if config.suppress_remote_forests:
        remote = False

    if not cv_repeat:
        if len(cross_val_object.feature_names) < 1:
            config.logger.info(f"Call of trainForest without features: {cross_val_object.name}")
            return None, zero_scores_obj, cross_val_object

    if config.verbosity >= 3 or debug:
        config.logger.info(f"Call of trainForest: {repeat=}, {cv_repeat=}, {remote=}, {para_number=}, {skip_feature_selection=}, {debug=}")

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
                reg_forest_times,
            ) = trainRegressionForest(
                config,
                cross_val_object,
                feats_to_filter,
                raw_feature_matrix_store_id=raw_feature_matrix_store_id,
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

            if config_ref_container is None:
                config_ref_container = [ray.put(config)]
            remote_store = ray.put((feats_to_filter, raw_feature_matrix_store_id))

            if config.verbosity >= 4:
                for name, size in sorted(((name, deep_get_size_of(value)) for name, value in locals().items()), key=lambda x: -x[1])[:10]:
                    config.logger.info("In trainForest: {:>30}: {:>8}".format(name, sizeof_fmt(size)))


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

                    if config.verbosity >= 4:
                        config.logger.info(f'Sending trainRegressionForestWrapper: {type(cv_slice)=}')

                    
                    try:
                        pg = ray.util.get_placement_group(f"pg_{cv_id}")
                        remote_wrapper_function.options(
                            num_cpus=1,
                            num_gpus=0,
                            scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                                placement_group=pg, placement_group_capture_child_tasks=True
                            )
                        )
                        if config.verbosity >= 4:
                            config.logger.info(f'Setting placement group before trainRegressionForestWrapper pg_{proc_id}')
                    except ValueError:
                       
                        remote_wrapper_function.options(
                            num_cpus=1,
                            num_gpus=0,
                            )
                        

                        
                    slice_result_ids.append(
                        remote_wrapper_function.remote(
                            remote_store,
                            cv_slice,
                            config_ref_container,
                            cross_val_object.subslice_refs[cv_counter],
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
                            raw_feature_matrix_store_id=raw_feature_matrix_store_id,
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
                        cv_counter,
                        reg_forest_times
                    ) = res
                    booster_list = None
                else:
                    (
                        booster_list,
                        scores_obj,
                        cv_counter,
                        reg_forest_times
                    ) = res

                total_times = aggregate_times(total_times, reg_forest_times)

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

    return booster_list, scores_obj