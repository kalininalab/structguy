"""Pairwise feature interactions from SHAP interaction values, and combination features built from feature pairs.

Discovery: for the held-out proteins of a sub slice, the mean absolute SHAP interaction value of every feature
pair, averaged per protein, is written to one file per booster; aggregate_feature_interactions ranks the pairs by
how consistently they interact across the boosters.

Combination features have self-describing names (comb||<op>||<feature a>||<feature b>), so a model trained with
them can rebuild them at prediction time from its feature list.
"""
import glob
import os

import numpy as np

COMBINATION_PREFIX = 'comb||'
COMBINATION_OPS = ('product', 'ratio', 'difference', 'sum', 'min', 'max')
# Upper bound of one pred_interactions output, (rows, F+1, F+1) float32
INTERACTION_BATCH_BYTES = 512 * 2**20
TOP_PAIRS_FOR_STABILITY = 100


# ---------------------------------------------------------------------------------------------------------------
# Discovery

def sample_rows_per_protein(prot_vec: np.ndarray, max_rows: int, rng: np.random.Generator) -> list[np.ndarray]:
    """Row indices per protein, at most an equal share of max_rows each (all rows of the smaller proteins)."""
    proteins = np.unique(prot_vec)
    if len(proteins) == 0:
        return []
    share = max(1, max_rows // len(proteins))
    rows_per_protein = []
    for protein in proteins:
        rows = np.flatnonzero(prot_vec == protein)
        if len(rows) > share:
            rows = np.sort(rng.choice(rows, size=share, replace=False))
        rows_per_protein.append(rows)
    return rows_per_protein


def protein_interaction_sums(predict_interactions, X_protein: np.ndarray, n_feats: int) -> tuple[np.ndarray, np.ndarray]:
    """Sums over the rows of one protein: |interaction| (n_feats x n_feats, diagonal zero) and |main SHAP value| (n_feats).

    predict_interactions(X) returns the SHAP interaction values (rows, n_feats + 1, n_feats + 1); the last row and
    column are the bias.
    """
    batch = max(1, INTERACTION_BATCH_BYTES // (4 * (n_feats + 1) ** 2))
    pair_sum = np.zeros((n_feats, n_feats), dtype=np.float64)
    main_sum = np.zeros(n_feats, dtype=np.float64)
    for start in range(0, len(X_protein), batch):
        values = predict_interactions(X_protein[start:start + batch])[:, :n_feats, :n_feats]
        main_sum += np.abs(values.sum(axis=2)).sum(axis=0)
        pair_sum += np.abs(values).sum(axis=0)
    np.fill_diagonal(pair_sum, 0.)
    return pair_sum, main_sum


def protein_mean_interactions(predict_interactions, X: np.ndarray, rows_per_protein: list[np.ndarray], n_feats: int):
    """Mean |interaction| per pair and mean |SHAP value| per feature, averaged over the rows of each protein and then
    over the proteins with equal weight. The pair strength is symmetric: |Φ_ij| + |Φ_ji|."""
    pair_total = np.zeros((n_feats, n_feats), dtype=np.float64)
    main_total = np.zeros(n_feats, dtype=np.float64)
    n_proteins = 0
    for rows in rows_per_protein:
        if len(rows) == 0:
            continue
        pair_sum, main_sum = protein_interaction_sums(predict_interactions, X[rows], n_feats)
        pair_total += pair_sum / len(rows)
        main_total += main_sum / len(rows)
        n_proteins += 1
    if n_proteins > 0:
        pair_total /= n_proteins
        main_total /= n_proteins
    return pair_total + pair_total.T, main_total, n_proteins


def ext_shap_interactions(feature_names, booster, data_refs, prot_vec, config, seed: int = 0):
    """Interaction strengths on the held-out samples of a booster: (names, strength matrix, main effects, n_proteins),
    restricted to the interaction_top_features features with the largest main effect."""
    import ray
    import xgboost as xgb

    feat_names, cat_vec, feat_id_vec = ray.get(data_refs[0][2])
    X = np.concatenate([ray.get(X_ref)[:, feat_id_vec] for X_ref, _, _ in data_refs], axis=0)
    if prot_vec is None or len(prot_vec) != len(X):
        prot_vec = np.zeros(len(X), dtype=np.int64)

    rng = np.random.default_rng(seed)
    rows_per_protein = sample_rows_per_protein(np.asarray(prot_vec), getattr(config, 'interaction_max_rows', 4000), rng)

    def predict_interactions(X_part):
        dmatrix = xgb.DMatrix(X_part, feature_names=feat_names, feature_types=cat_vec, enable_categorical=True)
        return booster.predict(dmatrix, pred_interactions=True)

    n_feats = len(feature_names)
    strength, main, n_proteins = protein_mean_interactions(predict_interactions, X, rows_per_protein, n_feats)

    top = np.argsort(-main)[:getattr(config, 'interaction_top_features', 200)]
    top = np.sort(top)
    return [feature_names[i] for i in top], strength[np.ix_(top, top)], main[top], n_proteins


def interaction_pairs(names, strength, main, n_proteins) -> list[tuple]:
    """(feat_a, feat_b, strength, relative strength, n_proteins) for every pair with an interaction, strongest first."""
    rows = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if strength[i, j] <= 0:
                continue
            main_sum = main[i] + main[j]
            relative = strength[i, j] / main_sum if main_sum > 0 else float('nan')
            rows.append((names[i], names[j], float(strength[i, j]), float(relative), n_proteins))
    rows.sort(key=lambda row: -row[2])
    return rows


def write_interaction_file(path, pairs):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('feat_a\tfeat_b\tstrength\trelative_strength\tn_proteins\n')
        for feat_a, feat_b, strength, relative, n_proteins in pairs:
            f.write(f'{feat_a}\t{feat_b}\t{strength}\t{relative}\t{n_proteins}\n')


def read_interaction_file(path) -> list[tuple]:
    pairs = []
    with open(path) as f:
        next(f)
        for line in f:
            words = line.rstrip('\n').split('\t')
            if len(words) >= 4:
                pairs.append((words[0], words[1], float(words[2]), float(words[3])))
    return pairs


def interaction_folder(outfolder):
    return f'{outfolder}/feature_interactions'


def aggregate_feature_interactions(outfolder, config, top_n_log: int = 20):
    """Ranks the pairs of all booster files: stability (share of the files with the pair among their top pairs),
    then mean strength over all files (0 where a file lacks the pair). Writes feature_interactions_ranked.tsv."""
    files = sorted(glob.glob(f'{interaction_folder(outfolder)}/*.tsv'))
    if not files:
        config.logger.info(f'No feature interaction files in {interaction_folder(outfolder)}')
        return None
    strength_sums: dict[tuple[str, str], float] = {}
    relative_sums: dict[tuple[str, str], float] = {}
    top_counts: dict[tuple[str, str], int] = {}
    present: dict[tuple[str, str], int] = {}
    for path in files:
        for rank, (feat_a, feat_b, strength, relative) in enumerate(read_interaction_file(path)):
            key = tuple(sorted((feat_a, feat_b)))
            strength_sums[key] = strength_sums.get(key, 0.) + strength
            if relative == relative:
                relative_sums[key] = relative_sums.get(key, 0.) + relative
            present[key] = present.get(key, 0) + 1
            if rank < TOP_PAIRS_FOR_STABILITY:
                top_counts[key] = top_counts.get(key, 0) + 1
    n_files = len(files)
    ranked = sorted(
        ((key, top_counts.get(key, 0) / n_files, strength_sums[key] / n_files, relative_sums.get(key, 0.) / n_files, present[key])
         for key in strength_sums),
        key=lambda row: (-row[1], -row[2]))
    out_path = f'{outfolder}/feature_interactions_ranked.tsv'
    with open(out_path, 'w') as f:
        f.write('feat_a\tfeat_b\tstability\tmean_strength\tmean_relative_strength\tn_files_present\n')
        for (feat_a, feat_b), stability, strength, relative, n_present in ranked:
            f.write(f'{feat_a}\t{feat_b}\t{stability}\t{strength}\t{relative}\t{n_present}\n')
    if config.verbosity >= 1:
        config.logger.info(f'Feature interactions of {n_files} boosters ranked in {out_path}, top {top_n_log}:')
        for (feat_a, feat_b), stability, strength, relative, n_present in ranked[:top_n_log]:
            config.logger.info(f'  {feat_a} x {feat_b}: stability {stability:.2f}, strength {strength:.4g}, relative {relative:.3f}')
    return out_path


# ---------------------------------------------------------------------------------------------------------------
# Combination features

def combination_name(op: str, feat_a: str, feat_b: str) -> str:
    return f'{COMBINATION_PREFIX}{op}||{feat_a}||{feat_b}'


def parse_combination_name(name: str):
    """(op, feature a, feature b) of a combination feature name, None for other names."""
    if not name.startswith(COMBINATION_PREFIX):
        return None
    parts = name[len(COMBINATION_PREFIX):].split('||')
    if len(parts) != 3 or parts[0] not in COMBINATION_OPS:
        return None
    return parts[0], parts[1], parts[2]


def combine(op: str, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Combination of two feature columns; NaN wherever a value is missing (and for ratio, where b is 0)."""
    with np.errstate(invalid='ignore', divide='ignore'):
        if op == 'product':
            return a * b
        if op == 'ratio':
            return np.where(b == 0, np.nan, a / np.where(b == 0, 1., b))
        if op == 'difference':
            return a - b
        if op == 'sum':
            return a + b
        if op == 'min':
            return np.where(np.isnan(a) | np.isnan(b), np.nan, np.minimum(a, b))
        if op == 'max':
            return np.where(np.isnan(a) | np.isnan(b), np.nan, np.maximum(a, b))
    raise ValueError(f'Unknown combination operation {op}')


def read_pair_file(path, n_pairs: int) -> list[tuple[str, str]]:
    """The first n_pairs feature pairs of a TSV (first two columns; a header line starting with feat_a is skipped)."""
    pairs = []
    with open(path) as f:
        for line in f:
            words = line.rstrip('\n').split('\t')
            if len(words) < 2 or words[0] == 'feat_a' or words[0].startswith('#'):
                continue
            pairs.append((words[0], words[1]))
            if len(pairs) >= n_pairs:
                break
    return pairs


def combination_names_from_config(config) -> list[str]:
    path = getattr(config, 'combination_features_file', None)
    if not path:
        return []
    ops = getattr(config, 'combination_ops', ['product', 'ratio', 'difference'])
    pairs = read_pair_file(path, getattr(config, 'combination_pairs', 25))
    return [combination_name(op, feat_a, feat_b) for feat_a, feat_b in pairs for op in ops]
