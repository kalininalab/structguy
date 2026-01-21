def get_feat_matrix_from_ids(
        feat_pos_dict,
        features,
        sample_pos_dict,
        raw_feature_matrix,
        sample_ids: list[str],
        feat_names: list[str],
        get_cat_vec = False
        )-> list[list[int | float | None]]:
    if len(feat_names) == 0:
        raise ValueError(f'{len(feat_names)=}')
    if get_cat_vec:
        cat_vec = []
    feat_id_vec = []
    for feat_name in feat_names:
        feat_id_vec.append(feat_pos_dict[feat_name])
        if get_cat_vec:
            if features[feat_name].f_type == 'categorical':
                cat_vec.append('c')
            else:
                cat_vec.append('q')

    sample_pos_vec = []
    for sample_id in sample_ids:
        try:
            sample_pos_vec.append(sample_pos_dict[sample_id])
        except KeyError:
            sample_pos_vec.append(None)
    if get_cat_vec:
        return get_feat_matrix(raw_feature_matrix, feat_id_vec, sample_pos_vec), cat_vec
    return get_feat_matrix(raw_feature_matrix, feat_id_vec, sample_pos_vec)

def get_feat_matrix(raw_feature_matrix, feat_id_vec, sample_pos_vec) -> list[list[int | float | None]]:
    feat_matrix: list[list[int | float| None]] = []
    for sample_pos in sample_pos_vec:
        feat_vec = []
        for feat_id in feat_id_vec:
            if feat_id == -1:
                feat_vec.append(0)
                continue
            try:
                feat_vec.append(raw_feature_matrix[sample_pos][feat_id])
            except KeyError:
                feat_vec.append(None)
            except TypeError:
                feat_vec.append(None)
        feat_matrix.append(feat_vec)
    return feat_matrix