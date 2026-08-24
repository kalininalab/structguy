import os
import pickle
import time

import numpy
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, LogisticRegressionCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from structman.base_utils.base_utils import pack, unpack
from structguy import featureGenerator, sampleSpace
from structguy.featureSelection import filterCorrelatedFeats
from structguy.util import Config, calc_protein_wise_corr


def to_float_matrix(feat_matrix):
    return numpy.array(
        [[numpy.nan if v is None else float(v) for v in row] for row in feat_matrix],
        dtype=float,
    )


def build_lasso_pipeline(cat_mask, regression, cv):
    quant_idx = [pos for pos, is_cat in enumerate(cat_mask) if not is_cat]
    cat_idx = [pos for pos, is_cat in enumerate(cat_mask) if is_cat]

    transformers = []
    if len(quant_idx) > 0:
        quant_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ])
        transformers.append(("quant", quant_pipeline, quant_idx))
    if len(cat_idx) > 0:
        cat_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("cat", cat_pipeline, cat_idx))

    preprocessor = ColumnTransformer(transformers)

    if regression:
        # Default eps=1e-3 only spans 3 decades below alpha_max; the selected alpha was
        # observed sitting exactly on that lower boundary (MSE still improving at the edge
        # of the grid), so widen the search one more decade down. eps=1e-5/alphas=200 made
        # this run far slower (small alphas converge much more slowly under coordinate
        # descent), so keep the widening modest, cap iterations so a non-converging alpha
        # fails fast instead of silently burning max_iter, and log progress via verbose.
        model = LassoCV(cv=cv, eps=1e-4, n_jobs=-1, max_iter=2_000, verbose=1)
    else:
        model = LogisticRegressionCV(cv=cv, penalty="l1", solver="liblinear", max_iter=5_000)

    return Pipeline([("preprocess", preprocessor), ("model", model)])


def make_kfold(config, regression, shuffle):
    # Plain int cv would make LassoCV/LogisticRegressionCV fall back to sklearn's default
    # KFold/StratifiedKFold, which does not shuffle before splitting; build the splitter
    # explicitly so the folds are randomized whenever the sample order itself isn't already
    # randomized.
    random_state = int(time.time()) if shuffle else None
    if regression:
        return KFold(n_splits=config.crossValidation_fold, shuffle=shuffle, random_state=random_state)
    return StratifiedKFold(n_splits=config.crossValidation_fold, shuffle=shuffle, random_state=random_state)


def build_datasail_cv_splits(config, sample_id_list, regression):
    # Protein-cluster-aware fold assignment (sequence-identity based, via DataSAIL) so that
    # the internal CV LassoCV/LogisticRegressionCV use to pick alpha/C doesn't leak related
    # proteins across folds, matching the leakage precautions sampleSpace.DataSAIL_cv applies
    # for the xgboost cross validation.
    if config.random_split:
        # Skip DataSAIL entirely. build_lasso already permuted sample_id_list/X/y explicitly,
        # so no additional shuffling is needed here - KFold just slices the already-random order
        # into contiguous folds.
        config.logger.info(f'{config.random_split=}: skipping datasail_cv_splits')
        return make_kfold(config, regression, shuffle=False)

    prot_fold = sampleSpace.get_datasail_protein_split_assignment(config)

    sample_fold = numpy.array([prot_fold.get(sample_id[0], -1) for sample_id in sample_id_list])
    unassigned = int((sample_fold == -1).sum())
    if unassigned > 0:
        config.logger.warning(
            f"{unassigned} samples belong to proteins DataSAIL did not assign to a fold; "
            "excluding them from the internal CV used to pick alpha (they remain in the training set)."
        )

    cv_splits = []
    for fold in range(config.crossValidation_fold):
        test_idx = numpy.where(sample_fold == fold)[0]
        train_idx = numpy.where((sample_fold != fold) & (sample_fold != -1))[0]
        if len(test_idx) == 0 or len(train_idx) == 0:
            continue
        cv_splits.append((train_idx, test_idx))

    if len(cv_splits) < 2:
        config.logger.warning(
            "DataSAIL produced fewer than 2 usable folds for the internal LASSO alpha search; "
            "falling back to a plain, shuffled KFold."
        )
        return make_kfold(config, regression, shuffle=True)

    return cv_splits


def storeLassoModel(pipeline, feature_names, cat_mask, features, feat_stats, regression, config, fn):
    with open(fn, "wb") as output:
        pickle.dump(
            (pipeline, feature_names, cat_mask, pack(features), feat_stats, regression),
            output,
            pickle.HIGHEST_PROTOCOL,
        )
    if config.verbosity >= 1:
        config.logger.info(f"\n============\nStored LASSO model in {fn}\n============\n")


def loadLassoModel(fn):
    with open(fn, "rb") as inp:
        pipeline, feature_names, cat_mask, packed_features, feat_stats, regression = pickle.load(inp)
    features = unpack(packed_features)
    return pipeline, feature_names, cat_mask, features, feat_stats, regression


def build_lasso(config: Config):
    t0 = time.time()

    samples: sampleSpace.SampleSpace = featureGenerator.createTrainingSet(
        config, stop_matrix_transformation=(config.path_to_support_features is not None)
    )
    if config.path_to_support_features is not None:
        support_samples = featureGenerator.createTrainingSet(
            config,
            stop_matrix_transformation=True,
            other_features_path=config.path_to_support_features,
            filter_none_tv=True,
        )
        samples.fuse_samples(support_samples)

    t1 = time.time()
    config.logger.info(f"Time for loading dataset: {t1 - t0}")

    feats_to_filter = set(filterCorrelatedFeats(config, samples.feat_corr_matrix, samples.feature_names, custom_tresh=0.97))
    samples.transform_matrix_dict(exclude_feats=feats_to_filter)

    feature_names = list(samples.feat_pos_dict.keys())
    cat_mask = [samples.features[feat_name].f_type == "categorical" for feat_name in feature_names]

    X_raw, _ = samples.get_feat_matrix_from_feat_names(feature_names, config)
    X = to_float_matrix(X_raw)

    sample_id_list = sorted(samples.sample_pos_dict, key=samples.sample_pos_dict.get)
    y = [samples.samples[sample_id].targetValue for sample_id in sample_id_list]
    if config.regression:
        y = numpy.array(y, dtype=float)
    else:
        y = numpy.array(y)

    if config.random_split:
        # sample_id_list (and therefore the rows of X) is ordered protein-by-protein, since
        # that is the order samples were generated in; KFold(shuffle=True) already randomizes
        # fold assignment regardless of input order, but permute the arrays themselves here
        # too, so the sample-level randomization is explicit and does not rely on trusting
        # that internal behavior of the CV splitter.
        perm = numpy.random.default_rng(int(time.time())).permutation(len(sample_id_list))
        X = X[perm]
        y = y[perm]
        sample_id_list = [sample_id_list[pos] for pos in perm]

    config.logger.info(f"Training LASSO on {len(feature_names)} features and {len(sample_id_list)} samples")

    cv_splits = build_datasail_cv_splits(config, sample_id_list, config.regression)
    pipeline = build_lasso_pipeline(cat_mask, config.regression, cv_splits)
    pipeline.fit(X, y)

    y_fit = pipeline.predict(X)
    if config.regression:
        model = pipeline.named_steps["model"]
        alpha = model.alpha_
        r2 = r2_score(y, y_fit)
        mse = mean_squared_error(y, y_fit)
        corr, _ = stats.spearmanr(y, y_fit)
        prot_id_vec = [sample_id[0] for sample_id in sample_id_list]
        _, mean_spearman, _ = calc_protein_wise_corr(y, y_fit, prot_id_vec, stats.spearmanr)
        config.logger.info(f"Selected alpha: {alpha}")
        config.logger.info(f"Alpha grid tested: min={model.alphas_.min()}, max={model.alphas_.max()}, n={len(model.alphas_)}")
        config.logger.info(f"Train R2-Score: {r2}")
        config.logger.info(f"Train MSE: {mse}")
        config.logger.info(f"Train Spearman correlation: {corr}")
        config.logger.info(f"Train protein-wise mean Spearman: {mean_spearman}")
    else:
        acc = (y_fit == y).mean()
        config.logger.info(f"Train accuracy: {acc}")

    name_add = ''
    if config.ablation:
        name_add += '_ablated'
    if config.no_evo:
        name_add += '_no_evo'
    if config.random_split:
        name_add += '_random_split'

    modelfile = f"{config.outfolder}/StructGuy_LASSO_trained_on_{config.dataset_name}{name_add}.dump"
    storeLassoModel(pipeline, feature_names, cat_mask, samples.features, samples.feat_stats, config.regression, config, modelfile)
    config.add_entry_to_project_file("path_trained_lasso_model", modelfile)

    t2 = time.time()
    config.logger.info(f"Time for training LASSO model: {t2 - t1}")

    return modelfile


def predict_lasso(config: Config):
    t0 = time.time()
    pipeline, feature_names, cat_mask, extern_features, feat_stats, regression = loadLassoModel(config.path_to_model)
    t1 = time.time()
    config.logger.info(f"Time for loading LASSO model: {t1 - t0} {config.path_to_model=}")

    model_filename = os.path.basename(config.path_to_model).split(".")[0]
    if model_filename.count("trained_on_") > 0:
        model_name = model_filename.split("trained_on_")[1]
    else:
        model_name = model_filename

    fake_booster_list = [(None, feature_names)]
    samples, booster_specific_data = featureGenerator.load_data_for_pred(config, None, fake_booster_list, extern_features)
    test_feature_matrix, test_targets, sample_id_list, feat_id_vec, cat_vec = booster_specific_data[0]

    if len(test_feature_matrix) == 0:
        return None, None, None, None

    X = to_float_matrix(test_feature_matrix)
    y_pred = pipeline.predict(X)

    outlines = ["Protein ID\tSAV\tPredicted effect value\n"]
    for pos, sample_id in enumerate(sample_id_list):
        prot_id, aac = sample_id
        outlines.append(f"{prot_id}\t{aac}\t{y_pred[pos]}\n")

    predictions_file = f"{config.outfolder}/predictions_by_LASSO_{model_name}.tsv"
    with open(predictions_file, "w") as f:
        f.write("".join(outlines))

    r2 = mse = corr = mean_spearman = None
    known_idx = [pos for pos, tv in enumerate(test_targets) if tv is not None]
    if len(known_idx) > 0:
        known_targets = [test_targets[pos] for pos in known_idx]
        known_pred = [y_pred[pos] for pos in known_idx]
        if len(known_idx) < len(test_targets):
            config.logger.info(
                f"Excluding {len(test_targets) - len(known_idx)} samples with unknown target value from the performance measures"
            )

        if regression:
            r2 = r2_score(known_targets, known_pred)
            mse = mean_squared_error(known_targets, known_pred)
            corr, _ = stats.spearmanr(known_targets, known_pred)
            prot_id_vec = [sample_id_list[pos][0] for pos in known_idx]
            _, mean_spearman, _ = calc_protein_wise_corr(known_targets, known_pred, prot_id_vec, stats.spearmanr)

            config.logger.info(f"R2-Score: {r2}")
            config.logger.info(f"MSE: {mse}")
            config.logger.info(f"Spearman correlation: {corr}")
            config.logger.info(f"Protein-wise mean Spearman: {mean_spearman}")
        else:
            acc = numpy.mean(numpy.array(known_pred) == numpy.array(known_targets))
            config.logger.info(f"Accuracy: {acc}")

    t2 = time.time()
    config.logger.info(f"Time for LASSO prediction: {t2 - t1}")

    return mean_spearman, test_targets, y_pred, sample_id_list
