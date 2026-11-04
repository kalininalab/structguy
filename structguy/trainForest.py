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

import gc
import math
import time
import sys
import os
import hashlib
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
from structman.base_utils.base_utils import pack, unpack, add_to_times, print_times, aggregate_times, deep_get_size_of, sizeof_fmt
from structguy.support_classes import CrossValidationSlice, extmem_batch_bytes, select_columns_on_device
from structguy.sampleSpace import DataSAIL_cv
from structguy.feature_correlation import average_ranks, standardized
from structguy.feature_interactions import ext_shap_interactions, interaction_pairs, write_interaction_file, interaction_folder
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

# max_retries=0: ray's memory monitor kills retriable tasks first. A killed wrapper takes its
# double_booster_remote children with it and was retried from scratch, so the whole CV fold
# restarted on every kill (2026-11-02: the same initial model 12 times in 8 hours).
@ray.remote(max_retries=0)
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
    proc_id=0,
    repeat_index=0
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
        subslice_refs = subslice_refs,
        repeat_index=repeat_index
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

FEAT_IMPACT_CHUNK = 256


def protein_spearman_impacts(contributions: numpy.ndarray, pred: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """Change of the Spearman correlation of one protein when each feature's SHAP contribution is taken out.

    contributions: (n_samples, n_feats). Returns rho(y, pred - contribution_f) - rho(y, pred) per feature:
    positive means the ranking of this protein would be better without the feature.
    """
    n_feats = contributions.shape[1]
    z_y = standardized(average_ranks(y.reshape(1, -1).astype(numpy.float64)))[0].astype(numpy.float64)
    z_pred = standardized(average_ranks(pred.reshape(1, -1).astype(numpy.float64)))[0].astype(numpy.float64)
    rho_with = 0. if numpy.isnan(z_pred).any() else float(z_pred @ z_y)
    impacts = numpy.zeros(n_feats, dtype=numpy.float64)
    for start in range(0, n_feats, FEAT_IMPACT_CHUNK):
        chunk = contributions[:, start:start + FEAT_IMPACT_CHUNK].T.astype(numpy.float64)
        used = numpy.any(chunk != 0, axis=1)
        if not used.any():
            continue
        z_without = standardized(average_ranks(pred[None, :] - chunk[used])).astype(numpy.float64)
        # A prediction made constant by taking a feature out ranks nothing
        rho_without = numpy.nan_to_num(z_without @ z_y, nan=0.)
        impacts[start + numpy.flatnonzero(used)] = rho_without - rho_with
    return impacts


def abs_error_impacts(contributions: numpy.ndarray, pred: numpy.ndarray, y: numpy.ndarray) -> numpy.ndarray:
    """Mean decrease of the absolute error of one protein when each feature's SHAP contribution is taken out."""
    loss = numpy.abs(y - pred)[:, None]
    impacts = numpy.zeros(contributions.shape[1], dtype=numpy.float64)
    for start in range(0, contributions.shape[1], FEAT_IMPACT_CHUNK):
        chunk = contributions[:, start:start + FEAT_IMPACT_CHUNK].astype(numpy.float64)
        impacts[start:start + chunk.shape[1]] = (loss - numpy.abs(y[:, None] - (pred[:, None] - chunk))).mean(axis=0)
    return impacts


def impact_z_scores(mean: numpy.ndarray, total_sq: numpy.ndarray, n_proteins: int) -> numpy.ndarray:
    """How many standard errors each mean impact is away from 0.

    The standard error comes from the spread of the per-protein impacts. Features the booster never used have a mean
    of exactly 0 and get z = 0. With less than 2 proteins there is no spread across proteins; the impacts are then
    scaled by the robust spread (scaled MAD) of the non-zero impacts across the features instead.
    """
    z = numpy.zeros_like(mean)
    nonzero = mean != 0
    if n_proteins >= 2:
        var = numpy.maximum(total_sq / n_proteins - mean * mean, 0.) * n_proteins / (n_proteins - 1)
        se = numpy.sqrt(var / n_proteins)
        usable = nonzero & (se > 0)
        z[usable] = mean[usable] / se[usable]
    elif nonzero.any():
        values = mean[nonzero]
        scale = 1.4826 * float(numpy.median(numpy.abs(values - numpy.median(values))))
        if scale > 0:
            z[nonzero] = values / scale
    return z


def protein_wise_feature_impacts(metric: str, contributions, pred, y, prot_vec) -> tuple[numpy.ndarray, numpy.ndarray, int]:
    """Feature impacts averaged over the proteins (the unit of the out-of-distribution split), each protein weighted equally.

    Returns the mean impacts, their z-scores (impact_z_scores) and the number of proteins that contributed.
    """
    n_feats = contributions.shape[1]
    total = numpy.zeros(n_feats, dtype=numpy.float64)
    total_sq = numpy.zeros(n_feats, dtype=numpy.float64)
    n_proteins = 0
    order = numpy.argsort(prot_vec, kind='stable')
    sorted_prots = prot_vec[order]
    boundaries = numpy.flatnonzero(sorted_prots[1:] != sorted_prots[:-1]) + 1
    for rows in numpy.split(order, boundaries):
        y_q = y[rows]
        if metric == 'protein_spearman':
            # A protein with less than 2 samples or a constant target has no ranking
            if len(rows) < 2 or numpy.all(y_q == y_q[0]):
                continue
            impacts = protein_spearman_impacts(contributions[rows], pred[rows], y_q)
        else:
            impacts = abs_error_impacts(contributions[rows], pred[rows], y_q)
        total += impacts
        total_sq += impacts * impacts
        n_proteins += 1
    if n_proteins > 0:
        total /= n_proteins
    return total, impact_z_scores(total, total_sq, n_proteins), n_proteins


#@profile
def chunk_dmatrices(X: numpy.ndarray, feat_names, cat_vec, feat_id_vec, batch_bytes: int | None = None, gpu_column_select: bool = True):
    """DMatrices of the selected features of a stored chunk, in row batches of at most batch_bytes of selected data.

    With gpu_column_select the columns are selected on the GPU (select_columns_on_device), on the host they took
    ~100 s per 10 GB chunk. Without batch_bytes the chunk is one batch.
    """
    if batch_bytes is None:
        rows_per_batch = max(1, X.shape[0])
    else:
        rows_per_batch = max(1, batch_bytes // max(1, len(feat_id_vec) * 4))
    for r0 in range(0, X.shape[0], rows_per_batch):
        r1 = min(r0 + rows_per_batch, X.shape[0])
        if gpu_column_select:
            X_sel = select_columns_on_device(X[r0:r1], feat_id_vec)
        else:
            X_sel = X[r0:r1][:, feat_id_vec]
        yield xgb.DMatrix(X_sel, feature_names=feat_names, feature_types=cat_vec, enable_categorical=True)
        del X_sel


def ext_shap_analysis(
        feature_names: list[str],
        booster: xgb.Booster,
        data_refs: list[tuple[ray.ObjectRef, ray.ObjectRef, ray.ObjectRef]],
        prot_vec=None,
        metric: str = 'protein_spearman',
        batch_bytes: int | None = None,
        gpu_column_select: bool = True):
    """Impact of every feature on the held-out samples: how much better the predictions get without its SHAP contribution.

    metric 'protein_spearman': change of the Spearman correlation within each protein, averaged over the proteins
    (the objective of the early stopping and the HPO). 'abs_error': decrease of the absolute error, averaged per
    protein. Positive impacts mark harmful features. Returns ([(feat_name, impact, z)] sorted by z descending, where
    z is the impact in standard errors across the proteins (0 for features the booster never used), number of
    proteins, times).
    """

    times = []
    ta = time.time()

    feat_names, cat_vec, feat_id_vec = ray.get(data_refs[0][2])

    contributions, preds, targets = [], [], []
    for X_ref, y_ref, _ in data_refs:

        X: numpy.ndarray = ray.get(X_ref)
        for dmatrix in chunk_dmatrices(X, feat_names, cat_vec, feat_id_vec, batch_bytes, gpu_column_select):
            # The last column is the bias
            contributions.append(booster.predict(dmatrix, pred_contribs=True)[:, :len(feature_names)])
            preds.append(booster.predict(dmatrix))
            del dmatrix
        targets.append(numpy.asarray(ray.get(y_ref), dtype=numpy.float64))

    contributions = numpy.concatenate(contributions, axis=0)
    pred = numpy.concatenate(preds).astype(numpy.float64)
    y = numpy.concatenate(targets)
    ta = add_to_times(times, ta)

    if prot_vec is None or len(prot_vec) != len(y):
        # Without protein ids, all samples count as one protein
        prot_vec = numpy.zeros(len(y), dtype=numpy.int64)
    impacts, z_scores, n_proteins = protein_wise_feature_impacts(metric, contributions, pred, y, numpy.asarray(prot_vec))

    acc_feat_impacts = sorted(zip(feature_names, impacts.tolist(), z_scores.tolist()), key=lambda x: x[2], reverse=True)
    ta = add_to_times(times, ta)
    return acc_feat_impacts, n_proteins, times

def predict_from_refs(
        booster: xgb.Booster,
        data_refs: list[tuple[ray.ObjectRef, ray.ObjectRef, ray.ObjectRef]],
        batch_bytes: int | None = None,
        gpu_column_select: bool = True) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Predictions and labels for the stored chunks in data_refs, in their order.

    A plain DMatrix per chunk instead of an ExtMemQuantileDMatrix over all of them: predicting
    needs no quantile sketch, building one for the whole cv_slice took ~8 minutes per sub-slice.
    The raw values also predict exactly, the quantised ones of a DMatrix without ref=dtrain did not.
    """
    feat_names, cat_vec, feat_id_vec = ray.get(data_refs[0][2])

    preds, labels = [], []
    for X_ref, y_ref, _ in data_refs:
        X: numpy.ndarray = ray.get(X_ref)
        if X.shape[0] == 0:
            continue
        for dmatrix in chunk_dmatrices(X, feat_names, cat_vec, feat_id_vec, batch_bytes, gpu_column_select):
            preds.append(booster.predict(dmatrix))
            del dmatrix
        labels.append(numpy.asarray(ray.get(y_ref)))

    return numpy.concatenate(preds), numpy.concatenate(labels)

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

def booster_list_predict(
        booster_list: list[tuple[xgb.Booster, any]],
        feat_mats: list[xgb.DMatrix],
        n_row_slice = 50_000,
        get_std_vec = False,
        logger = None
        ):
    """Mean prediction of the boosters.

    With get_std_vec, additionally the std of the SHAP feature contributions per sample is
    returned. That needs exact TreeSHAP (pred_contribs), which is orders of magnitude slower
    than a plain prediction on deep trees, so only request it when it is actually needed.
    """
    y_preds = []
    y_stds = []
    t_start = time.time()
    n_boosters = len(booster_list)
    for booster_index, (booster, _) in enumerate(booster_list):
        feat_mat = feat_mats[booster_index]
        total_rows = feat_mat.num_row()
        t_booster = time.time()

        pred_parts = []
        std_parts = []
        for i in range(0, total_rows, n_row_slice):
            end = min(i + n_row_slice, total_rows)
            dslice = feat_mat.slice(numpy.arange(i, end))

            pred_parts.append(booster.predict(dslice))

            if get_std_vec:
                contribs = booster.predict(dslice, pred_contribs=True)
                # The last column is the bias term
                std_parts.append(numpy.std(contribs[:, :-1], axis=1))

            if logger is not None:
                done = booster_index * total_rows + end
                total = n_boosters * total_rows
                elapsed = time.time() - t_start
                eta = elapsed * (total - done) / done
                logger.info(f'Prediction: booster {booster_index + 1}/{n_boosters}, rows {end}/{total_rows}, elapsed {elapsed:.1f} s, ETA {eta:.1f} s')

        if logger is not None:
            logger.info(f'Prediction: booster {booster_index + 1}/{n_boosters} done in {time.time() - t_booster:.1f} s')

        y_preds.append(numpy.concatenate(pred_parts) if pred_parts else numpy.array([]))
        if get_std_vec:
            y_stds.append(numpy.concatenate(std_parts) if std_parts else numpy.array([]))
    y_pred = numpy.mean(y_preds, axis=0)
    if get_std_vec:
        y_std = numpy.mean(y_stds, axis=0)
        return y_pred, y_std
    else:
        return y_pred

def seed_component(value) -> int:
    """Turn one position identifier into an integer usable in the seed arithmetic.

    Slice ids are only set when a cross validation actually builds the slices. The final
    model is trained on a slice whose test_slice_id was never assigned and is therefore
    None; the same holds for any slice constructed outside of a cross validation. Those
    positions are unique anyway (there is exactly one of them), so mapping them to a
    constant keeps the seed deterministic instead of crashing.
    """
    if value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        # Named slices (strings) still have to map to a stable number.
        return int(hashlib.blake2b(str(value).encode(), digest_size=4).hexdigest(), 16)


#@profile
def derive_crn_seed(config: util.Config, repeat_index, cv_slice_id, subslice_id, second_round):
    """Deterministic training seed for one booster, identical across candidates.

    Seeding from the wall clock means two hyperparameter sets are never trained on the
    same random draws, so the difference between their scores carries both the effect of
    the hyperparameters and the effect of the seeds. When the real effect is small (on
    goldstandard_strict the largest measured effect of any hyperparameter was about 1.4x
    the evaluation noise) the seed term dominates and the comparison is mostly noise.

    Deriving the seed from the position in the cross validation instead makes every
    candidate see the same draws, so the seed term cancels in the comparison. The
    variance of a *difference* falls from 2*sigma^2 to 2*sigma^2*(1-rho), where rho is
    how strongly the two runs' noise is now correlated.

    The seed deliberately does not depend on the hyperparameters, on the worker that
    happens to run the evaluation, or on the time; it depends only on which repeat, which
    fold, which sub-booster and which of the two training rounds this is. `crn_seed_base`
    shifts the whole set, which is how an independent re-measurement gets fresh draws.
    """
    if not getattr(config, "common_random_numbers", True):
        return int(time.time())

    base = int(getattr(config, "crn_seed_base", 0) or 0)
    seed = (
        base * 7_919
        + seed_component(repeat_index) * 1_000_003
        + seed_component(cv_slice_id) * 10_007
        + seed_component(subslice_id) * 101
        + (1 if second_round else 0)
    )
    return seed % (2 ** 31 - 1)


def xgb_train_wrapper(
        config: util.Config,
        dtrain: xgb.DMatrix,
        dtest_feature_matrix: xgb.DMatrix,
        second_round = False,
        seed = None,
        ) -> xgb.Booster:

    if seed is None:
        seed = int(time.time())

    eval_label = 'eval'
    
    #if config.use_external_memory_qdm:
    if True:
        if config.setup_cuda_mem:
            mem_context = xgb.config_context(use_cuda_async_pool=True)
        else:
            mem_context = xgb.config_context(use_rmm=True)
    else:
        mem_context = xgb.config_context()

    if not config.use_external_memory_qdm:
        max_bin = 512
    else:
        max_bin = config.extMem_max_bin

    grow_policy = {}

    def train(params_override: dict) -> xgb.Booster:
        # EarlyStopping keeps state, so every call gets fresh callbacks
        es_list = []
        evals: list[tuple[xgb.DMatrix, str]] = []
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
                'max_bin' : max_bin,
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
                'random_state' : seed
                }
            xgb_params.update(gpu_tree_params(config, dtrain, max_bin, xgb_params['max_depth']))
            xgb_params.update(params_override)
            grow_policy['used'] = xgb_params.get('grow_policy', 'depthwise')
            grow_policy['cached_nodes'] = xgb_params['max_cached_hist_node']
            return xgb.train(xgb_params, dtrain, num_boost_round=int(config.num_of_trees), early_stopping_rounds= config.early_stopping, evals=evals, maximize=False, custom_metric=rho_eval_for_xgboost_cb, callbacks=es_list)
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
                'max_bin' : max_bin,
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
                'random_state' : seed
                }
            xgb_params.update(gpu_tree_params(config, dtrain, max_bin, xgb_params['max_depth']))
            xgb_params.update(params_override)
            grow_policy['used'] = xgb_params.get('grow_policy', 'depthwise')
            grow_policy['cached_nodes'] = xgb_params['max_cached_hist_node']
            return xgb.train(xgb_params, dtrain, num_boost_round=int(config.num_of_trees_1), early_stopping_rounds=int(config.early_stopping_1), evals=evals, maximize=False, custom_metric=rho_eval_for_xgboost_cb, callbacks=es_list)

    with mem_context:
        # The histogram estimate of gpu_tree_params is a model, the pool can still run out (fragmentation,
        # the sampled page of gradient based sampling). Lossguide grows one node at a time and needs far less,
        # a smaller histogram cache less again.
        override = {}
        while True:
            oom_message = None
            try:
                booster = train(override)
                break
            except xgb.core.XGBoostError as err:
                if 'out_of_memory' not in str(err) or override.get('max_cached_hist_node') is not None:
                    raise
                oom_message = str(err).splitlines()[0][:300]
            # Retried outside the except block: its traceback still references the failed booster and its GPU memory
            if grow_policy.get('used') != 'lossguide':
                override = {'grow_policy': 'lossguide', 'max_leaves': 0}
            else:
                override = {'grow_policy': 'lossguide', 'max_leaves': 0,
                            'max_cached_hist_node': max(8, grow_policy['cached_nodes'] // 4)}
            depth = config.tree_depth_1 if second_round else config.tree_depth
            config.logger.warning(
                f'GPU out of memory in xgb.train ({second_round=}, max_depth={depth}, grow_policy={grow_policy.get("used")}, '
                f'max_cached_hist_node={grow_policy.get("cached_nodes")}), retrying with {override}: {oom_message}'
            )
            gc.collect()

    return booster

HIST_BYTES_PER_BIN = 16  # one gradient pair of the GPU histogram


def gpu_node_batch() -> int:
    """Candidate nodes of one tree level that xgboost's GPU depthwise growth expands in one batch.

    kMaxNodeBatchSize in src/tree/updater_gpu_hist.cuh: 1024 since xgboost 3.1 ([EM] Optimization for deep
    trees, #11387), a fixed 32 before.
    """
    try:
        major, minor = (int(part) for part in xgb.__version__.split('.')[:2])
    except ValueError:
        return 1024
    return 1024 if (major, minor) >= (3, 1) else 32


def gpu_pool_size(config: util.Config) -> float:
    pool_bytes = getattr(config, 'gpu_pool_bytes', None)
    if not pool_bytes:
        pool_bytes = util.get_gpu_memory()[0] * 1024 * 1024 * config.vram_limit
    return pool_bytes


def gpu_tree_params(config: util.Config, dtrain, max_bin: int, max_depth) -> dict:
    """xgboost parameters that keep the histogram memory of the GPU hist method within the pool.

    Depthwise growth allocates the histograms of up to gpu_node_batch() nodes of a level and of their siblings in
    one go (the level batch). Below the max_cached_hist_node cap the cache buffer grows to cache + level batch, above
    it a separate overflow buffer holds a whole level batch, neither shrinks, and the old buffer lives on while it is
    copied: about cache + 2.5 level batches at the worst. For deep trees on many features that exceeds the pool;
    lossguide growth expands one node at a time instead. Without a leaf limit (max_leaves=0) and with the same
    max_depth, the splits follow the same gain and depth rules.
    """
    bytes_per_node = histogram_bytes_per_node(config, dtrain, max_bin)
    params = {'max_cached_hist_node': histogram_cache_nodes(config, bytes_per_node)}
    pool_bytes = gpu_pool_size(config)
    depth = max(1, int(max_depth))
    node_batch = gpu_node_batch()
    level_batch = 2 * min(node_batch, 2 ** min(depth - 1, 30)) * bytes_per_node
    hist_peak = params['max_cached_hist_node'] * bytes_per_node + 2.5 * level_batch
    budget = pool_bytes * getattr(config, 'hist_level_vram_share', 0.25)
    if hist_peak > budget:
        params['grow_policy'] = 'lossguide'
        params['max_leaves'] = 0
    if config.verbosity >= 1:
        config.logger.info(
            f"xgboost grow_policy={params.get('grow_policy', 'depthwise')}: max_depth={depth}, {node_batch} nodes per batch, "
            f"worst-case depthwise histogram batch {level_batch / 2**30:.2f} GiB, histogram peak {hist_peak / 2**30:.2f} GiB, "
            f"budget {budget / 2**30:.2f} GiB of a {pool_bytes / 2**30:.2f} GiB pool"
        )
    return params


def histogram_bytes_per_node(config: util.Config, dtrain, max_bin: int) -> int:
    """Bytes of one node histogram of the GPU hist method: one gradient pair per bin of the training cuts.

    xgboost sizes it by the total bins of the cuts (cuts.TotalBins() in updater_gpu_hist.cu), not by features x
    max_bin: a categorical feature has one bin per category, whatever max_bin is. On the 2026-11-02 run that made
    the histograms about 2.7x bigger than features x max_bin. Read from the cuts once per DMatrix, features x max_bin
    is the fallback when they are not available.
    """
    estimate = max(1, dtrain.num_col()) * max_bin
    n_bins = getattr(dtrain, 'structguy_total_bins', None)
    if n_bins is None:
        try:
            indptr, _ = dtrain.get_quantile_cut()
            n_bins = max(1, int(indptr[-1]))
        except Exception as err:
            config.logger.warning(f'Could not read the bins of the training cuts, assuming {dtrain.num_col()} features x {max_bin} bins: {err}')
            n_bins = estimate
        try:
            dtrain.structguy_total_bins = n_bins
        except AttributeError:
            pass
        if config.verbosity >= 3:
            config.logger.info(
                f'Training cuts: {n_bins} bins for {dtrain.num_col()} features ({n_bins / max(1, dtrain.num_col()):.1f} per feature), '
                f'{n_bins / estimate:.2f}x the {dtrain.num_col()} features x {max_bin} bins'
            )
    return n_bins * HIST_BYTES_PER_BIN


def histogram_cache_nodes(config: util.Config, bytes_per_node: int) -> int:
    """Value of xgboost's max_cached_hist_node that keeps the histogram allocations within a share of the GPU pool.

    The GPU hist method caches one histogram (bins gradient pairs) per tree node. Its buffer grows by
    doubling, and the cap is only checked before growing: a cache just below the cap still doubles. So the largest
    single allocation is up to twice the cap (and the old buffer lives on during the copy). hist_cache_vram_share is
    that largest allocation as share of the pool, the cap is half of it.
    """
    pool_bytes = gpu_pool_size(config)
    largest_allocation = pool_bytes * getattr(config, 'hist_cache_vram_share', 0.1)
    cache_cap = largest_allocation / 2
    n_nodes = max(8, int(cache_cap // bytes_per_node))
    if config.verbosity >= 3:
        config.logger.info(
            f'max_cached_hist_node={n_nodes}: '
            f'{bytes_per_node / 2**20:.1f} MiB per node histogram, cache cap {n_nodes * bytes_per_node / 2**30:.2f} GiB, '
            f'largest histogram allocation about {2 * n_nodes * bytes_per_node / 2**30:.2f} GiB of a {pool_bytes / 2**30:.2f} GiB pool'
        )
    return n_nodes


def usable_vram_share(config: util.Config, total_mib: float, sub_share: float) -> float:
    """MiB this process may claim on its GPU, after reserving room for CUDA contexts.

    `sub_share` is the fraction of a GPU this process was scheduled for, so roughly
    1/sub_share processes are co-resident on the card. Each of them needs its own CUDA
    context (a few hundred MiB that no pool allocator accounts for) plus slack for
    fragmentation. Sizing the pools against the raw card total makes the shares add up to
    100% of the device and the last process to start fails to allocate; that is what
    limits how far the workers can be fanned out. Subtracting the contexts up front makes
    the shares add up to something that actually fits.
    """
    share = min(max(float(sub_share), 1e-6), 1.0)
    n_co_resident = 1.0 / share
    reserve = float(getattr(config, "vram_context_reserve", 1024.0)) * n_co_resident
    # Never hand back less than 10% of the card, otherwise a badly configured reserve
    # would starve the process completely.
    usable = max(total_mib * 0.1, total_mib - reserve)

    if config.verbosity >= 3:
        config.logger.info(
            f"VRAM budget: {total_mib=:.0f} MiB {sub_share=:.4f} {n_co_resident=:.1f} "
            f"reserved={reserve:.0f} MiB usable={usable:.0f} MiB "
            f"-> this process may use {usable * share:.0f} MiB"
        )

    return usable


def setup_memory_resources(config: util.Config, sub_share: float, cuda_setup=True):
    if cuda_setup:
        setup_cuda_memory(config, sub_share)
    else:
        setup_rmm_memory(config, sub_share)

def setup_cuda_memory(config: util.Config, sub_share: float):
    # Get the default memory pool and configure the release threshold
    status, dft_pool = cudart.cudaDeviceGetDefaultMemPool(0)
    # Set the release threshold to 90% of this process' share of the device, after
    # reserving room for the CUDA contexts of the other processes on the same GPU.
    status, free, total = cudart.cudaMemGetInfo()

    usable = usable_vram_share(config, total / (1024.0 * 1024.0), sub_share) * 1024.0 * 1024.0
    pool_size = int(usable * 0.9 * sub_share * config.vram_limit)

    if config.verbosity >= 4:
        config.logger.info(f'Setup cuda memory resources: {pool_size=}')

    config.gpu_pool_bytes = pool_size

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
    usable = usable_vram_share(config, gmem, sub_share)
    # 0.99 of the share left no slack at all: with the pools of all co-resident processes
    # summing to the whole card there was nothing left for their CUDA contexts or for
    # allocator fragmentation. 0.95 of the reserved-adjusted share leaves both.
    init_pool = 1024*1024*int(usable*sub_share*0.7*config.vram_limit)
    max_pool = 1024*1024*int(usable*sub_share*0.95*config.vram_limit)
    config.gpu_pool_bytes = max_pool

    if config.verbosity >= 4:
        config.logger.info(f'Setup memory resources: {sub_share=} {init_pool=:_} {max_pool=:_} {config.multi_gpu=}')
        
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

# Retriable, so that the memory monitor picks a single sub-slice over its wrapper. The result is
# a deterministic function of the inputs (seeds from derive_crn_seed), a retry is safe.
@ray.remote(max_retries=2, max_calls=1)
#@profile
def double_booster_remote(packed_slice_slice, store, proc_id: str, sub_share: float, config_ref_container: list[ray.ObjectRef], subslice_index: int = 0):
    config: util.Config
    filtered_features, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, retain_model, repeat_index = store
    config = ray.get(config_ref_container[0])
    cv_slice = unpack(cv_slice)
    util.reset_logger_for_remotes(config)
    if config.verbosity >= 4:
        config.logger.info(f'Call of double_booster_remote: {type(packed_slice_slice)=}')

    return double_booster(packed_slice_slice, proc_id, sub_share, config, filtered_features, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, retain_model, repeat_index, subslice_index)

def select_impact_removals(config, acc_feat_impacts, impact_metric):
    """The features the SHAP impact filter removes from a sub slice, and a description for the log.

    acc_feat_impacts is [(feat_name, impact, z)] as returned by ext_shap_analysis, sorted by z descending.
    """
    if getattr(config, 'feat_impact_filter', 'z') == 'raw':
        # Strictly greater: a feature that the first booster never used has an impact of exactly 0 and stays
        impact_hits = [name for name, impact, _ in acc_feat_impacts if impact > config.feat_impact_thresh]
        filter_description = f'{len(impact_hits)} features with {impact_metric} impact > {config.feat_impact_thresh=}'
    else:
        # Remove every feature whose impact lies more than feat_impact_z standard errors above 0. Above 0 only
        # harmful features go (the higher, the more certain they must be); below 0 also the features the booster
        # never used (z = 0) and helpful ones that are not certain enough, down to keeping only reliably helpful ones.
        theta = float(getattr(config, 'feat_impact_z', 2.0))
        impact_hits = [name for name, _, z in acc_feat_impacts if z > theta]
        min_keep = int(getattr(config, 'feat_impact_min_keep', 10))
        if len(acc_feat_impacts) - len(impact_hits) < min_keep:
            # The most helpful features (lowest z, the list is sorted by z descending) always stay
            protected = {name for name, _, _ in acc_feat_impacts[::-1][:min_keep]}
            impact_hits = [name for name in impact_hits if name not in protected]
        z_of = {name: z for name, _, z in acc_feat_impacts}
        n_harmful = sum(z_of[name] > 0 for name in impact_hits)
        n_unused = sum(z_of[name] == 0 for name in impact_hits)
        filter_description = (
            f'z > feat_impact_z={theta:+.2f}: {n_harmful} harmful, {n_unused} unused, '
            f'{len(impact_hits) - n_harmful - n_unused} helpful but uncertain; at least {min_keep} kept'
        )
    return impact_hits, filter_description


#@profile
def double_booster(packed_slice_slice, proc_id, sub_share, config, filtered_features, cv_slice, skip_scoring, score_train, raw_feature_matrix_store_id, retain_model, repeat_index=0, subslice_index=0):
    times = []
    ta = time.time()

    p_id = os.getpid()

    if isinstance(packed_slice_slice, CrossValidationSlice):
        slice_slice = packed_slice_slice
    else:
        slice_slice = unpack(packed_slice_slice)
        del packed_slice_slice

    ta = add_to_times(times, ta) #0
    n_feats_before = len(slice_slice.feature_names)
    slice_slice.filterFeatures(filtered_features)
    if config.verbosity >= 2:
        config.logger.info(
            f'Correlation filter in {proc_id}: removed {n_feats_before - len(slice_slice.feature_names)} of {n_feats_before} features '
            f'of this sub slice, {len(slice_slice.feature_names)} left for the first booster '
            f'({len(filtered_features)} features flagged by {config.corr_thresh=}, the rest of them were already absent in this sub slice)'
        )
    if len(slice_slice.feature_names) == 0:
        if config.verbosity >= 2:
            config.logger.info(f'No features left in {proc_id} after the correlation filter, no model trained')
        return p_id
    ta = add_to_times(times, ta) #1

    # Seed from the position in the cross validation, not from the clock, so that every
    # candidate trains on the same draws and the comparison between two hyperparameter
    # sets is not swamped by the difference between their seeds. Uses the slice ids
    # rather than proc_id, because which worker picks up an evaluation varies between
    # candidates and would reintroduce exactly the variation this removes.
    # Only a cross validation assigns test_slice_id; the full slice of the final model and
    # its sub-slices leave it at None. Falling back to the position in slice_slices keeps
    # the sub-boosters of one forest on different seeds, and is just as reproducible,
    # because that order is fixed when the slices are built, not by the scheduling.
    cv_slice_id = getattr(cv_slice, 'test_slice_id', None)
    subslice_id = getattr(slice_slice, 'test_slice_id', None)
    if subslice_id is None:
        subslice_id = subslice_index
    seed_first = derive_crn_seed(config, repeat_index, cv_slice_id, subslice_id, False)
    seed_second = derive_crn_seed(config, repeat_index, cv_slice_id, subslice_id, True)

    if config.verbosity >= 4:
        config.logger.info(
            f'Call of double_booster: {retain_model=} {repeat_index=} {cv_slice_id=} '
            f'{subslice_id=} {seed_first=} {seed_second=}'
        )
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
        config.logger.info(f'Reached after first data retrieval in double_booster_remote {proc_id} {type(dtrain)=}')


    booster = xgb_train_wrapper(config, dtrain, dtest_feature_matrix, seed=seed_first)
    # The proteins of the held-out samples, in the order of test_data_refs, for the protein-wise feature impacts
    test_prot_vec = getattr(dtest_feature_matrix, 'encoded_prot_vec', None)
    if test_prot_vec is not None:
        test_prot_vec = cp.asnumpy(test_prot_vec) if isinstance(test_prot_vec, cp.ndarray) else numpy.asarray(test_prot_vec)
    del dtrain
    del dtest_feature_matrix
    ta = add_to_times(times, ta) #4

    if config.verbosity >= 3:
        config.logger.info(f'Reached after first training in double_booster_remote {proc_id}')

    if booster is None:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 1', logger=config.logger)

        return p_id
    
    impact_metric = getattr(config, 'feat_impact_metric', 'protein_spearman')
    acc_feat_impacts, n_impact_proteins, shap_times = ext_shap_analysis(
        slice_slice.feature_names, booster, test_data_refs, prot_vec=test_prot_vec, metric=impact_metric,
        batch_bytes=extmem_batch_bytes(config), gpu_column_select=getattr(config, 'gpu_column_select', True))
    if config.verbosity >= 3 and acc_feat_impacts:
        def format_impacts(entries):
            return ', '.join(f'{name}: {impact:+.4g} (z {z:+.2f})' for name, impact, z in entries)
        config.logger.info(
            f'Feature impacts ({impact_metric}, {n_impact_proteins} held-out proteins) in {proc_id}, '
            f'most harmful: {format_impacts(acc_feat_impacts[:10])}; most helpful: {format_impacts(acc_feat_impacts[::-1][:10])}'
        )

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

    if getattr(config, 'log_feature_interactions', False):
        # Interaction discovery is diagnostic: a failure here must not end the training
        try:
            t_inter = time.time()
            inter_names, inter_strength, inter_main, inter_proteins = ext_shap_interactions(
                slice_slice.feature_names, booster, test_data_refs, test_prot_vec, config, seed=seed_first)
            pairs = interaction_pairs(inter_names, inter_strength, inter_main, inter_proteins)
            path = f'{interaction_folder(config.outfolder)}/cv{cv_slice_id}_sub{subslice_id}_rep{repeat_index}_{proc_id}.tsv'
            write_interaction_file(path, pairs)
            if config.verbosity >= 2:
                config.logger.info(f'Feature interactions of {proc_id}: {len(pairs)} pairs of {len(inter_names)} features on {inter_proteins} proteins in {time.time() - t_inter:.1f} s -> {path}')
        except Exception:
            config.logger.error(f'Feature interaction analysis failed in {proc_id}:\n{traceback.format_exc()}')

    del booster

    if config.verbosity >= 3:
        config.logger.info(f'Reached after shap analysis in double_booster_remote {proc_id}')

    if acc_feat_impacts is None:
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 2', logger=config.logger)
        
        return p_id
    
    impact_hits, filter_description = select_impact_removals(config, acc_feat_impacts, impact_metric)
    feats_to_remove = filtered_features[:] + impact_hits

    if len(feats_to_remove) >= (len(cv_slice.feature_names)+ len(filtered_features)):
        if config.verbosity >= 4:
            print_times(times, label = 'double booster 3', logger=config.logger)
        return p_id
    
    n_feats_before = len(slice_slice.feature_names)
    slice_slice.filterFeatures(feats_to_remove)
    if config.verbosity >= 2:
        config.logger.info(
            f'SHAP feature impact filter in {proc_id}: removed {n_feats_before - len(slice_slice.feature_names)} '
            f'of {n_feats_before} features, {len(slice_slice.feature_names)} left for the second booster '
            f'({filter_description})'
        )
    # The check above compares against the parent slice and counts names that this sub
    # slice may not have, so it does not catch every case where nothing is left.
    if len(slice_slice.feature_names) == 0:
        if config.verbosity >= 2:
            config.logger.info(f'No features left in {proc_id} after the feature impact filter ({filter_description}), no model trained')
        return p_id
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

    booster_2: xgb.Booster = xgb_train_wrapper(config, dtrain, dtest_feature_matrix, second_round=True, seed=seed_second)
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

    n_feats_before = len(cv_slice.feature_names)
    cv_slice.filterFeatures(feats_to_remove)
    if config.verbosity >= 3:
        config.logger.info(
            f'Applied the correlation and feature impact filters of {proc_id} to the parent slice {cv_slice.name}: '
            f'{n_feats_before} -> {len(cv_slice.feature_names)} active features'
        )
    ta = add_to_times(times, ta) #12

    if config.verbosity >= 3:
        config.logger.info(f'Reached after cv_slice feat filter in double_booster_remote {proc_id} {cv_slice.train_slice_ids=} {score_train=}')

    del dtrain
    del dtest_feature_matrix

    # Scored on the parent slice, with the filters of this sub-slice applied above
    feat_pos_dict, features = ray.get(raw_feature_matrix_store_id)
    if score_train:
        train_refs, x_prot_vec = cv_slice.prepare_ext_mem_qdmatrix(config, feat_pos_dict, features)
        x_pred, x_true = predict_from_refs(booster_2, train_refs, extmem_batch_bytes(config), getattr(config, 'gpu_column_select', True))
        x_prot_vec = numpy.asarray(x_prot_vec)

        if config.verbosity >= 5:
            train_scores_obj = calc_scores_obj(x_true, x_pred, x_prot_vec, cv_slice.train_class_weight_vector, cv_slice.feature_names)
            config.logger.info('Train scores:')
            train_scores_obj.printOut(config = config)

    else:
        x_pred = None
        x_true = None
        x_prot_vec = None

    test_refs, y_prot_vec = cv_slice.prepare_test_ext_mem_qdmatrix(config, feat_pos_dict, features)
    y_pred, y_true = predict_from_refs(booster_2, test_refs, extmem_batch_bytes(config), getattr(config, 'gpu_column_select', True))
    y_prot_vec = numpy.asarray(y_prot_vec)
    del feat_pos_dict
    del features

    ta = add_to_times(times, ta) #14

    if config.verbosity >= 3:
        config.logger.info(f'Reached the end of double_booster_remote {proc_id} {type(y_pred)=} {type(y_true)=} {type(y_prot_vec)=} {type(x_pred)=} {type(x_true)=} {type(x_prot_vec)=}')

    if config.verbosity >= 4:
        print_times(times, label = 'double booster 6', logger=config.logger)

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
    subslice_refs=None,
    repeat_index=0
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

    # Must have the same shape as the regular returns at the end of this function
    if remote:
        zero_return = zero_scores, cv_counter, times
    else:
        zero_return = (
            None,
            zero_scores,
            cv_counter,
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
            f"Train regression {config.forest_type} forest, call of fit with # of features: {len(cv_slice.feature_names)} "
            f"(before the {len(feats_to_filter)} correlation filtered and the per sub slice filters), "
            f"skip feature selection {skip_feature_selection}, skip scoring {skip_scoring} {sub_gpu_share=}"
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
        
        if sub_gpu_share is not None:
            # Every sub-booster requests 1/threads_per_gpu of a GPU (--tpgpu), so ray trains at
            # most threads_per_gpu of them on one card at a time, whatever share of the GPUs the
            # caller was handed. The callers above request no GPU themselves.
            models_per_gpu = int(config.threads_per_gpu)
            sub_share = 1.0 / models_per_gpu
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

            # num_cpus bounds the host side work of xgboost (building the DMatrices) in these tasks.
            # Half of the cores are shared out over the GPUs, the rest stays for the wrappers and
            # HPO workers, which request CPUs themselves.
            gpu_task_cpus = getattr(config, 'gpu_task_cpus', None)
            if not gpu_task_cpus:
                gpu_task_cpus = max(1, int(ray.cluster_resources().get('CPU', 1)) // (2 * max(1, config.multi_gpu) * models_per_gpu))
            # Ray only sets OMP_NUM_THREADS to num_cpus when it is unset, and HTCondor sets it to the
            # whole slot (64): 8 tasks with 64 OpenMP threads each on 48 CPUs. Every task gets a
            # fresh worker anyway (max_calls=1), so a runtime env of its own costs nothing.
            omp_env = {'env_vars': {'OMP_NUM_THREADS': str(gpu_task_cpus)}}

            if grouped:
                remote_function = double_booster_remote.options(
                    num_cpus=gpu_task_cpus,
                    num_gpus=sub_share,
                    runtime_env=omp_env,
                    scheduling_strategy=ray.util.scheduling_strategies.PlacementGroupSchedulingStrategy(
                        placement_group=pg , placement_group_capture_child_tasks=True
                        )
                    )
                
                if config.verbosity >= 4:
                    config.logger.info(f'double booster remote configured: {proc_id=}')
                
            else:
                remote_function = double_booster_remote.options(num_cpus=gpu_task_cpus, num_gpus=sub_share, runtime_env=omp_env)

            if config_ref_container is None:
                config_ref_container = [ray.put(config)]
            store = ray.put((feats_to_filter, pack(cv_slice), skip_scoring, score_train, raw_feature_matrix_store_id, remote, repeat_index))

            # slice_by_slice keeps as many sub-slices in flight as this CV fold has model slots:
            # its share of the GPUs (sub_gpu_share) times the models per GPU. One at a time left
            # half of the cards idle with 8 GPUs and 4 folds.
            # With queued HPO candidates every sub-slice is submitted at once instead: the GPU
            # resources already limit how many train, and ray grants tasks of the same shape in the
            # order they were submitted. So this candidate's sub-slices run before those of the
            # candidate queued behind it, which only fills the GPUs this one leaves idle. Held back
            # here, the last sub-slices would queue behind the next candidate's.
            n_sub_slices = len(cv_slice.slice_slices)
            if config.slice_by_slice and not getattr(config, 'hpo_queued_candidates', 0):
                max_in_flight = min(n_sub_slices, max(1, math.ceil(sub_gpu_share * models_per_gpu)))
            else:
                max_in_flight = n_sub_slices
            if config.verbosity >= 3:
                config.logger.info(f'Double booster dispatch of {proc_id}: {n_sub_slices} sub-slices, {max_in_flight} at a time, {gpu_task_cpus} CPUs each')

            remote_proc_ids = []
            next_slice = 0

            def submit_next_sub_slices():
                nonlocal next_slice
                while next_slice < n_sub_slices and len(remote_proc_ids) < max_in_flight:
                    remote_proc_ids.append(remote_function.remote(
                        cv_slice.slice_slices[next_slice], store, f'{proc_id}_{next_slice}', sub_share, config_ref_container, next_slice))
                    next_slice += 1

            submit_next_sub_slices()

            y_preds = []
            x_preds = []
            booster_2_list: list[tuple[xgb.Booster, list[str]]] = []

            ray_dumped = False

            while len(remote_proc_ids) > 0:
                # The refs that are ready leave remote_proc_ids here. Keeping them in the list
                # handed the same result back on the next wait, and it was counted again.
                ready, remote_proc_ids = ray.wait(remote_proc_ids, timeout = 1)

                if len(ready) > 0:
                    if config.verbosity >= 3:
                        if not ray_dumped:
                            ray_dumped = True
                            util.dump_ray_logs_snapshot(f'{config.outfolder}/ray_dump_snapshot_2.log')
                        config.logger.info(f'Double booster returned {ready=}')
                    try:
                        results = ray.get(ready, timeout=60)
                    except ray.exceptions.GetTimeoutError:
                        util.dump_ray_logs_snapshot(f'{config.outfolder}/ray_dump_snapshot_2.log')
                        config.logger.info('double_booster_remote ray.get timed out')
                        submit_next_sub_slices()
                        continue
                    except ray.exceptions.WorkerCrashedError:
                        util.dump_ray_logs_snapshot(f'{config.outfolder}/ray_dump_snapshot_2.log')
                        config.logger.info('double_booster_remote ray.get crashed')
                        submit_next_sub_slices()
                        continue
                    except ray.exceptions.RayError as err:
                        # E.g. killed by the memory monitor more often than max_retries allows.
                        # Raised on, it would end the whole run; score this fold as failed instead.
                        config.logger.error(f'double_booster_remote of {proc_id} failed, the fold is scored as failed: {type(err).__name__}: {err}')
                        for ref in remote_proc_ids:
                            ray.cancel(ref, force=True)
                        return zero_return

                    if config.verbosity >= 4:
                        config.logger.info(f'Double booster returned with {type(results[0])=}')
                    for res in results:
                        if isinstance(res, int):
                            if config.verbosity >= 1:
                                config.logger.info('double_booster_remote returned None')
                            for ref in remote_proc_ids:
                                ray.cancel(ref, force=True)
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

                    submit_next_sub_slices()

            #for packed_slice_slice in cv_slice.slice_slices:
            #    ray._private.internal_api.free(packed_slice_slice)

        else:
            y_preds = []
            x_preds = []
            booster_2_list: list[tuple[xgb.Booster, list[str]]] = []
            # double_booster() calls cv_slice.filterFeatures() in place. Handing it the
            # shared cv_slice would accumulate the feature removals of every sub-booster,
            # so later sub-boosters would predict on a test matrix that no longer matches
            # the features their model was trained on. The remote branch above avoids this
            # by giving every worker its own unpacked copy; do the same here.
            packed_cv_slice = pack(cv_slice)
            for nested_proc_id, packed_slice_slice in enumerate(cv_slice.slice_slices):
                cv_slice_copy = unpack(packed_cv_slice)
                res = double_booster(packed_slice_slice, f'{proc_id}_{nested_proc_id}', 1.0, config, feats_to_filter, cv_slice_copy, skip_scoring, score_train, raw_feature_matrix_store_id, remote, repeat_index, nested_proc_id)

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

def gather_fold_results(config: util.Config, result_ids: list, cv_counters: list) -> list:
    """Results of the remote CV folds, in the order they finish. Once a fold fails, the others are cancelled.

    A candidate with a failed fold is judged as failed as a whole (failed_folds > 0 in the HPO), so the remaining
    folds would only keep GPUs busy, for hours on big datasets. Every cancelled fold is returned as a failed one.
    """
    counter_of = dict(zip(result_ids, cv_counters))
    pending = list(result_ids)
    results = []
    while pending:
        ready, pending = ray.wait(pending, num_returns=1)
        res = ray.get(ready[0])
        results.append(res)
        scores_obj, cv_counter = res[0], res[1]
        if getattr(scores_obj, 'failed_folds', 0) > 0 and pending:
            config.logger.error(f'CV fold {cv_counter} failed, the {len(pending)} remaining fold(s) are cancelled and count as failed')
            for ref in pending:
                try:
                    ray.cancel(ref, force=True, recursive=True)
                except (TypeError, ValueError, ray.exceptions.RayError) as err:
                    config.logger.warning(f'Could not cancel CV fold {counter_of[ref]}: {err}')
                results.append((util.Scores(zero=True), counter_of[ref], None))
            pending = []
    return results


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
) -> tuple[None | list[tuple[xgb.Booster, list[str]]], util.Scores, CrossValidationSlice | DataSAIL_cv, None | list[CrossValidationSlice] | dict[int, list[CrossValidationSlice]]]:
    # if cv_repeat is False, the cross_val_object is a cross validation slice object instead
    zero_scores_obj = util.Scores(zero=True)
    # if para_number == 1:

    if config.suppress_remote_forests or debug or config.gpu_mode:
        para_number = None

    if config.suppress_remote_forests:
        remote = False

    if not cv_repeat and len(cross_val_object.feature_names) < 1:
        config.logger.info(f"Call of trainForest without features: {cross_val_object.name}")
        return None, zero_scores_obj, cross_val_object

    if config.verbosity >= 3 or debug:
        config.logger.info(f"Call of trainForest: {repeat=}, {cv_repeat=}, {remote=}, {para_number=}, {skip_feature_selection=}, {debug=}")

    t0 = time.time()

    #remove the features of the farther interacting positions, then identify the highly correlated features among the rest
    local_feats_to_filter = featureSelection.filterLocalFeatures(config, feature_names)
    feats_to_filter = featureSelection.filterCorrelatedFeats(config, feat_corr_matrix, feature_names, excluded=local_feats_to_filter)

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
                proc_id=proc_id,
                repeat_index=i
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
            # Both of these have to be reset per repeat. Keeping slice_result_ids across
            # repeats makes ray.get() return the cached results of the previous repeats
            # again, so repeat i would average i*len(cv_counters) results and weight the
            # early repeats far more heavily than the late ones. Keeping worst_scores
            # across repeats turns it into the worst score over all repeats so far.
            slice_result_ids = []
            worst_scores = None
            scores_list = []

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
                            proc_id = f'{proc_id}_{cv_id}',
                            repeat_index = i
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
                            proc_id = proc_id,
                            repeat_index = i
                        )
                    )
                    if cv_interuption is not None and len(slice_result_ids) == 1:
                        (
                            forest,
                            scores_obj,
                            cv_counter,
                            reg_forest_times
                        ) = slice_result_ids[0]
                        margin, best_first_scores = cv_interuption
                        if not util.objective_function_criterium(config, scores_obj, best_first_scores, feature_penalty=config.feature_penalty, margin=margin):
                            obj_first_score = util.get_objective_score(config, scores_obj, feature_penalty = config.feature_penalty)
                            estimated_obj_score = obj_first_score + config.estimation_delta
                            return forest, (estimated_obj_score, scores_obj), cross_val_object
                    

            if remote:
                results = gather_fold_results(config, slice_result_ids, list(cv_counters))
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
                    booster_list: list[tuple[xgb.Booster, list[str]]]
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