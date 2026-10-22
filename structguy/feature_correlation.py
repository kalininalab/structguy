"""Pairwise feature correlations (pairwise-complete Spearman) of a feature matrix with missing values.

The features are grouped by their pattern of missing values. For two groups, every pair of their features shares
the same rows with values on both sides, so the correlations of all these pairs are one matrix product of the
standardized ranks over these rows. The work is split so that every feature is ranked at most once for every set
of shared rows; a group whose own rows are the shared rows reuses its precomputed standardized ranks.
"""
import heapq

import numpy as np
import ray

from structguy.util import log_timing

RANK_CHUNK = 256
# Features per work item along a group that needs no ranking again, and rows per step of ranking again (memory)
SPLIT_CHUNK = 512
RERANK_CHUNK = 512
JOBS_PER_PROCESS = 4


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Average ranks (ties get the mean of their positions, like scipy.stats.rankdata) along the rows of a 2D array without NaN."""
    n_rows, n = values.shape
    if n == 0:
        return np.empty((n_rows, 0), dtype=np.float64)
    order = np.argsort(values, axis=1)
    sorted_values = np.take_along_axis(values, order, axis=1)
    positions = np.broadcast_to(np.arange(n), (n_rows, n))
    starts_tie = np.ones((n_rows, n), dtype=bool)
    starts_tie[:, 1:] = sorted_values[:, 1:] != sorted_values[:, :-1]
    ends_tie = np.ones((n_rows, n), dtype=bool)
    ends_tie[:, :-1] = starts_tie[:, 1:]
    first = np.maximum.accumulate(np.where(starts_tie, positions, 0), axis=1)
    last = np.minimum.accumulate(np.where(ends_tie, positions, n - 1)[:, ::-1], axis=1)[:, ::-1]
    ranks = np.empty((n_rows, n), dtype=np.float64)
    np.put_along_axis(ranks, order, (first + last) / 2.0 + 1.0, axis=1)
    return ranks


def standardized(ranks: np.ndarray) -> np.ndarray:
    """Centers every row and scales it to unit norm, so the dot product of two rows is their Pearson correlation.

    Rows of a constant feature, and all rows if there are less than 2 columns, become NaN.
    """
    n_rows, n_cols = ranks.shape
    if n_cols < 2:
        return np.full((n_rows, n_cols), np.nan, dtype=np.float32)
    centered = ranks - ranks.mean(axis=1, keepdims=True)
    norm = np.sqrt((centered * centered).sum(axis=1, keepdims=True))
    with np.errstate(invalid='ignore', divide='ignore'):
        z = centered / norm
    z[norm[:, 0] == 0] = np.nan
    return z.astype(np.float32)


def own_mask_ranks(feat_matrix: np.ndarray, feat_idx: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ranks of the features over the rows of their mask (float32, exact up to 2**23) and their standardized form."""
    ranks = average_ranks(feat_matrix[np.ix_(feat_idx, np.flatnonzero(mask))])
    return ranks.astype(np.float32), standardized(ranks)


@ray.remote
def remote_own_mask_ranks(feat_matrix, feat_idx, mask):
    return own_mask_ranks(feat_matrix, feat_idx, mask)


def standardized_over(ranks: np.ndarray, z_own: np.ndarray, keep: np.ndarray) -> np.ndarray:
    """Standardized ranks over the columns keep of the own rows; ranking the ranks again gives the ranks of the values."""
    if keep.all():
        return z_own
    out = np.empty((ranks.shape[0], int(keep.sum())), dtype=np.float32)
    for start in range(0, ranks.shape[0], RERANK_CHUNK):
        out[start:start + RERANK_CHUNK] = standardized(average_ranks(ranks[start:start + RERANK_CHUNK][:, keep]))
    return out


def work_items(groups, n_total, n_tv):
    """(group_i, group_j, rows of i, rows of j, estimated cost) for all group pairs i <= j.

    A side whose own rows are the shared rows needs no ranking again, so the other side can be split into chunks
    without repeating work; otherwise the pair stays one item.
    """
    items = []
    for gi, (feat_i, mask_i) in enumerate(groups):
        for gj in range(gi, len(groups)):
            feat_j, mask_j = groups[gj]
            both = mask_i & mask_j
            n_both = int(both.sum())
            i_own = n_both == int(mask_i.sum())
            j_own = n_both == int(mask_j.sum())
            n_i, n_j = len(feat_i), len(feat_j)
            if j_own and n_i > SPLIT_CHUNK:
                slices = [(slice(s, s + SPLIT_CHUNK), slice(0, n_j)) for s in range(0, n_i, SPLIT_CHUNK)]
            elif i_own and n_j > SPLIT_CHUNK:
                slices = [(slice(0, n_i), slice(s, s + SPLIT_CHUNK)) for s in range(0, n_j, SPLIT_CHUNK)]
            else:
                slices = [(slice(0, n_i), slice(0, n_j))]
            for i_slice, j_slice in slices:
                len_i = len(range(*i_slice.indices(n_i)))
                len_j = len(range(*j_slice.indices(n_j)))
                rank_cost = (0 if i_own else len_i * n_both * 20) + (0 if j_own else len_j * n_both * 20)
                cost = rank_cost + len_i * len_j * n_both // 10 + 1
                items.append((gi, gj, i_slice, j_slice, cost))
    return items


def correlation_items(items, groups, rank_arrays, z_arrays, n_total, n_tv):
    """Correlation blocks of the work items: [(group_i, group_j, rows of i, rows of j, block, cov_both)]."""
    results = []
    for gi, gj, i_slice, j_slice, _ in items:
        mask_i, mask_j = groups[gi][1], groups[gj][1]
        both = mask_i & mask_j
        n_both = int(both.sum())
        n_both_none = n_tv - int((mask_i | mask_j).sum())
        denominator = n_total - n_both_none
        cov_both = n_both / denominator if denominator > 0 else 0.
        ranks_i, z_i = rank_arrays[gi][i_slice], z_arrays[gi][i_slice]
        ranks_j, z_j = rank_arrays[gj][j_slice], z_arrays[gj][j_slice]
        if n_both < 2:
            block = np.full((ranks_i.shape[0], ranks_j.shape[0]), np.nan, dtype=np.float32)
        else:
            block = standardized_over(ranks_i, z_i, both[mask_i]) @ standardized_over(ranks_j, z_j, both[mask_j]).T
        results.append((gi, gj, i_slice, j_slice, block, cov_both))
    return results


@ray.remote
def remote_correlation_items(items, bundle, n_total, n_tv):
    groups, rank_arrays, z_arrays = bundle
    return correlation_items(items, groups, rank_arrays, z_arrays, n_total, n_tv)


def balance(items, n_jobs):
    """Distributes the items over n_jobs lists of similar total cost, largest items first."""
    jobs = [[] for _ in range(n_jobs)]
    heap = [(0, k) for k in range(n_jobs)]
    for item in sorted(items, key=lambda item: -item[-1]):
        load, k = heapq.heappop(heap)
        jobs[k].append(item)
        heapq.heappush(heap, (load + item[-1], k))
    return [job for job in jobs if job]


def group_by_missing_values(valid: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """[(feature indices, mask of rows with values)] for every distinct pattern of missing values, largest first."""
    packed = np.packbits(valid, axis=1)
    groups: dict[bytes, list[int]] = {}
    for feat_nr in range(valid.shape[0]):
        groups.setdefault(packed[feat_nr].tobytes(), []).append(feat_nr)
    ordered = sorted(groups.values(), key=len, reverse=True)
    return [(np.array(feat_idx, dtype=np.int64), valid[feat_idx[0]].copy()) for feat_idx in ordered]


def compute_feature_correlations(feat_matrix: np.ndarray, tv_vector: np.ndarray, config):
    """feat_matrix: (n_feats, n_samples) float64 with NaN for missing values; tv_vector: (n_samples,) with NaN for missing.

    Returns the float32 array (n_feats, n_feats, 6) of (corr, cov_a, cov_b, cov_both, tv_corr_a, tv_corr_b) and the
    (min, max, mean, median) of every feature, all computed on the samples with a target value.
    """
    n_feats, n_total = feat_matrix.shape
    tv_valid = ~np.isnan(tv_vector)
    if not tv_valid.all():
        feat_matrix = np.ascontiguousarray(feat_matrix[:, tv_valid])
    tv_values = tv_vector[tv_valid]
    n_tv = len(tv_values)

    with log_timing(config, 'feature correlations: grouping by missing values and per feature statistics'):
        valid = ~np.isnan(feat_matrix)
        n_valid = valid.sum(axis=1)
        covs = n_valid / n_total
        groups = group_by_missing_values(valid)
        del valid
        feat_stats: list = [None] * n_feats
        for feat_idx, mask in groups:
            if not mask.any():
                for feat_nr in feat_idx:
                    feat_stats[feat_nr] = (None, None, None, None)
                continue
            values = feat_matrix[np.ix_(feat_idx, np.flatnonzero(mask))]
            for feat_nr, stat in zip(feat_idx, zip(values.min(axis=1).tolist(), values.max(axis=1).tolist(),
                                                   values.mean(axis=1).tolist(), np.median(values, axis=1).tolist())):
                feat_stats[feat_nr] = stat
        if config.verbosity >= 1:
            config.logger.info(f'feature correlations: {n_feats} features in {len(groups)} groups of missing values, largest: {[len(f) for f, _ in groups[:10]]}')

    use_ray = ray.is_initialized()
    with log_timing(config, 'feature correlations: ranks'):
        rank_jobs = [(g, start) for g, (feat_idx, _) in enumerate(groups) for start in range(0, len(feat_idx), RANK_CHUNK)]
        if use_ray:
            matrix_ref = ray.put(feat_matrix)
            rank_chunks = ray.get([remote_own_mask_ranks.remote(matrix_ref, groups[g][0][start:start + RANK_CHUNK], groups[g][1]) for g, start in rank_jobs])
            del matrix_ref
        else:
            rank_chunks = [own_mask_ranks(feat_matrix, groups[g][0][start:start + RANK_CHUNK], groups[g][1]) for g, start in rank_jobs]
        rank_parts = [[] for _ in groups]
        z_parts = [[] for _ in groups]
        for (g, _), (ranks, z) in zip(rank_jobs, rank_chunks):
            rank_parts[g].append(ranks)
            z_parts[g].append(z)
        del rank_chunks
        rank_arrays = [np.concatenate(parts, axis=0) for parts in rank_parts]
        z_arrays = [np.concatenate(parts, axis=0) for parts in z_parts]
        del rank_parts, z_parts

    tv_corrs = np.full(n_feats, np.nan, dtype=np.float64)
    for (feat_idx, mask), z in zip(groups, z_arrays):
        if mask.sum() >= 2:
            tv_z = standardized(average_ranks(tv_values[mask].reshape(1, -1)))[0]
            tv_corrs[feat_idx] = z.astype(np.float64) @ tv_z.astype(np.float64)

    with log_timing(config, 'feature correlations: correlation blocks'):
        items = work_items(groups, n_total, n_tv)
        n_jobs = max(1, JOBS_PER_PROCESS * config.proc_n) if use_ray else 1
        jobs = balance(items, n_jobs)
        if config.verbosity >= 1:
            config.logger.info(f'feature correlations: {len(items)} work items in {len(jobs)} jobs')
        if use_ray:
            # One put for all arrays (out-of-band, zero-copy in the workers): hundreds of puts in a row made
            # ray 2.43 abort with "Tried to create an owned object that already exists"
            bundle_ref = ray.put((groups, rank_arrays, z_arrays))
            results = ray.get([remote_correlation_items.remote(job, bundle_ref, n_total, n_tv) for job in jobs])
            del bundle_ref
        else:
            results = [correlation_items(job, groups, rank_arrays, z_arrays, n_total, n_tv) for job in jobs]
        del rank_arrays, z_arrays

    with log_timing(config, 'feature correlations: assembly'):
        corr = np.empty((n_feats, n_feats), dtype=np.float32)
        cov_both = np.empty((n_feats, n_feats), dtype=np.float32)
        for job_results in results:
            for gi, gj, i_slice, j_slice, block, cov in job_results:
                rows = groups[gi][0][i_slice]
                cols = groups[gj][0][j_slice]
                corr[np.ix_(rows, cols)] = block
                corr[np.ix_(cols, rows)] = block.T
                cov_both[np.ix_(rows, cols)] = cov
                cov_both[np.ix_(cols, rows)] = cov
        del results

        diagonal = np.arange(n_feats)
        corr[diagonal, diagonal] = 1.0
        denominator = n_total - (n_tv - n_valid)
        cov_both[diagonal, diagonal] = np.where(denominator > 0, n_valid / np.maximum(denominator, 1), 0.)

        feat_corr_matrix = np.empty((n_feats, n_feats, 6), dtype=np.float32)
        feat_corr_matrix[:, :, 0] = corr
        del corr
        feat_corr_matrix[:, :, 1] = covs[:, None]
        feat_corr_matrix[:, :, 2] = covs[None, :]
        feat_corr_matrix[:, :, 3] = cov_both
        del cov_both
        feat_corr_matrix[:, :, 4] = tv_corrs[:, None]
        feat_corr_matrix[:, :, 5] = tv_corrs[None, :]

    return feat_corr_matrix, feat_stats
