import numpy as np

def get_feat_id_vec(
        feat_names: list[str],
        feat_pos_dict: dict[str, int],
        features,
        get_cat_vec: bool =False
        ) -> tuple[np.ndarray, list[str]]:
    cat_vec: list[str] = []
    feat_id_vec: list[int] = []
    for feat_name in feat_names:
        feat_id_vec.append(feat_pos_dict[feat_name])
        if get_cat_vec:
            if features[feat_name].f_type == 'categorical':
                cat_vec.append('c')
            else:
                cat_vec.append('q')
    # dtype given explicitly: an empty list would otherwise become a float array, which
    # numpy refuses as an index
    feat_id_vec: np.ndarray = np.array(feat_id_vec, dtype=np.int64)
    return feat_id_vec, cat_vec

def get_feat_matrix_from_ids(
        feat_pos_dict: dict[str, int],
        features: dict,
        sample_pos_dict,
        raw_feature_matrix,
        sample_ids: list[str],
        feat_names: list[str],
        get_cat_vec = False
        )-> tuple[list[list[int | float | None]], list[str]]:
    if len(feat_names) == 0:
        raise ValueError(f'{len(feat_names)=}')
    
    feat_id_vec, cat_vec = get_feat_id_vec(feat_names, feat_pos_dict, features, get_cat_vec=get_cat_vec)
    
    sample_pos_vec = []
    for sample_id in sample_ids:
        try:
            sample_pos_vec.append(sample_pos_dict[sample_id])
        except KeyError:
            sample_pos_vec.append(None)

    return get_feat_matrix(raw_feature_matrix, feat_id_vec, sample_pos_vec), cat_vec

def get_raw_feat_matrix_from_sample_ids(sample_ids: list[str], sample_pos_dict: dict[str, int], raw_feature_matrix: np.ndarray):
    sample_pos_vec = []
    for sample_id in sample_ids:
        try:
            sample_pos_vec.append(sample_pos_dict[sample_id])
        except KeyError:
            sample_pos_vec.append(None)

    return get_raw_feat_matrix(raw_feature_matrix, sample_pos_vec)

def get_raw_feat_matrix_from_ids(
        feat_pos_dict: dict[str, int],
        features: dict,
        sample_pos_dict,
        raw_feature_matrix: np.ndarray,
        sample_ids: list[str],
        feat_names: list[str],
        get_cat_vec = False
        )-> list[list[int | float | None]]:
    if len(feat_names) == 0:
        raise ValueError(f'{len(feat_names)=}')
    
    feat_id_vec, cat_vec = get_feat_id_vec(feat_names, feat_pos_dict, features, get_cat_vec=get_cat_vec)

    feat_matrix = get_raw_feat_matrix_from_sample_ids(sample_ids, sample_pos_dict, raw_feature_matrix)
    
    return feat_matrix, cat_vec, feat_id_vec

def get_feat_matrix(raw_feature_matrix, feat_id_vec: list[int], sample_pos_vec) -> np.ndarray:
    """Rows sample_pos_vec and columns feat_id_vec of the feature matrix.

    A column id of -1 gives a column of 0, a column or row id of None a column or row of missing values (NaN).
    """
    raw_feature_matrix = np.asarray(raw_feature_matrix)
    feat_ids = list(feat_id_vec)
    sample_positions = list(sample_pos_vec)
    if raw_feature_matrix.ndim != 2:
        return np.full((len(sample_positions), len(feat_ids)), np.nan)

    valid_cols = [j for j, feat_id in enumerate(feat_ids) if feat_id is not None and feat_id != -1]
    valid_rows = [i for i, sample_pos in enumerate(sample_positions) if sample_pos is not None]
    if len(valid_cols) == len(feat_ids) and len(valid_rows) == len(sample_positions):
        return raw_feature_matrix[np.ix_(np.asarray(sample_positions, dtype=np.int64), np.asarray(feat_ids, dtype=np.int64))]

    feat_matrix = np.full((len(sample_positions), len(feat_ids)), np.nan, dtype=raw_feature_matrix.dtype)
    zero_cols = [j for j, feat_id in enumerate(feat_ids) if feat_id is not None and feat_id == -1]
    feat_matrix[:, zero_cols] = 0
    if len(valid_cols) > 0 and len(valid_rows) > 0:
        rows = np.asarray([sample_positions[i] for i in valid_rows], dtype=np.int64)
        cols = np.asarray([feat_ids[j] for j in valid_cols], dtype=np.int64)
        feat_matrix[np.ix_(valid_rows, valid_cols)] = raw_feature_matrix[np.ix_(rows, cols)]
    return feat_matrix

def get_raw_feat_matrix(raw_feature_matrix: np.ndarray, sample_pos_vec: list[int]) -> np.ndarray:
    feat_matrix = raw_feature_matrix[sample_pos_vec, :]
    return feat_matrix