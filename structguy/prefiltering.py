import time
import ray
import sys
import traceback
from scipy import stats

from structguy.util import Config, unpack
from structguy.support_classes import CrossValidationSlice

def dummy_featureTargetCorr(train_targets, feat_name, feature_value_vector):
    val_list_1 = []
    val_list_2 = []

    for pos, tv in enumerate(train_targets):
        value = feature_value_vector[pos]
        if value is None:
            continue
        val_list_1.append(value)
        val_list_2.append(tv)
    if min(val_list_1) == max(val_list_1):
        return (
            "const",
            f"Min Val: {min(val_list_1)}, Max Val: {max(val_list_1)}, Val-type: {type(val_list_1[0])} , len vals : {len(val_list_1)}, in deac features: {feat_name in self.deactivated_features}",
        )
    try:
        corr, p_val = stats.spearmanr(val_list_1, val_list_2)
    except:
        return None, len(val_list_1)
    return corr, p_val


def dummy_addToTvmbMap(train_sample_ids, train_targets, feat_name, samples, config, name):
    try:
        feature_value_vector = samples.get_feature_value_vector(train_sample_ids, feat_name)
    except KeyError:
        return True

    try:
        target_corr, target_p_val = dummy_featureTargetCorr(train_targets, feat_name, feature_value_vector)
    except:
        if config.verbosity >= 5:
            [e, f, g] = sys.exc_info()
            g = traceback.format_exc()
            print(f"Error: cant get feature target corr for: {feat_name=} due to:\n{e}\n{f}\n{g}")
        return True
    if target_corr == "const":
        if config.verbosity >= 5:
            print(f"{name} - Removed feature: {feat_name} due to constant feature values, info: {target_p_val}")
        return True
    if target_corr is None:
        # self.removeFeature(feat_name)
        if config.verbosity >= 5:
            print(f"{name} - Removed feature: {feat_name} due to None target_corr, len of values: {target_p_val}")
        return True
    return False

@ray.remote
def remote_tvmb_calculation(packed_slice, feat_names, packed_samples, config):
    train_sample_ids, train_targets, name = ray.get(packed_slice[0])
    unpacked_samples = unpack(ray.get(packed_samples[0]))
    results = []
    for feat_name in feat_names:
        to_remove = dummy_addToTvmbMap(train_sample_ids, train_targets, feat_name, unpacked_samples, config, name)
        results.append((to_remove, feat_name))
    return results

def detectBiasedFeaturesByMeanCorrelation(
    config: Config,
    cv_slice: CrossValidationSlice,
    samples_store_id,
    samples=None,
    dummy_call=False,
    thresh=None,
    print_out=False,
    pre_filter=None,
    debug=False,
    return_list=False,
    return_score_list=False,
    remote=False
):
    if thresh is None:
        thresh = (config.tvmb_rank_threshold, config.p_val_thresh)
    if pre_filter is None:
        cv_slice.filterFeatures([])
        to_filter = []
    else:
        cv_slice.filterFeatures(pre_filter)
        to_filter = pre_filter
    if print_out:
        print("===========================================================================================")
        print(f"Biased feature detection (by mean correlation), TVMB {thresh=}, {return_score_list=}, {(pre_filter is None)=} {dummy_call=} {cv_slice.tvmb_map is None=}")
        print(f"tvmb prefiltered: {len(to_filter)=}")

    t0 = time.time()
    if config.verbosity >= 5:
        cv_slice.featureSanityCheck(verbose=True)

    if "Protein bias" not in cv_slice.slice_specific_feature_map and not dummy_call:
        cv_slice.setProteinBias(config)

        if config.verbosity >= 5:
            cv_slice.featureSanityCheck(verbose=True)

    if cv_slice.tvmb_map is None:
        if config.verbosity >= 4:
            print(f"Init tvmb_map for slice {cv_slice.name}")
        feats_to_remove = []
        cv_slice.tvmb_map = {}
        if not dummy_call or remote:
            if samples is None:
                samples = unpack(ray.get(samples_store_id))
            for feat_name in cv_slice.feature_names:
                to_remove = cv_slice.addToTvmbMap(feat_name, samples, config, dummy_call=dummy_call)
                if to_remove:
                    feats_to_remove.append(feat_name)
        else:
            remote_process_ids = []
            packed_slice = ray.put((cv_slice.train_sample_ids, cv_slice.train_targets, cv_slice.name))
            conf_store = ray.put(config)
            tvmb_procs: int = min([config.proc_n, 15])
            nr_of_feats_per_proc = len(cv_slice.feature_names) // tvmb_procs
            if len(cv_slice.feature_names) % tvmb_procs != 0:
                nr_of_feats_per_proc += 1

            feat_name_package = []
            for feat_name in cv_slice.feature_names:
                feat_name_package.append(feat_name)
                if len(feat_name_package) == nr_of_feats_per_proc:
                    remote_process_ids.append(remote_tvmb_calculation.remote([packed_slice], feat_name_package, [samples_store_id], conf_store))
                    feat_name_package = []
            if len(feat_name_package) > 0:
                remote_process_ids.append(remote_tvmb_calculation.remote([packed_slice], feat_name_package, [samples_store_id], conf_store))
                feat_name_package = []

            remote_results = ray.get(remote_process_ids)
            for results in remote_results:
                for to_remove, feat_name in results:
                    if to_remove:
                        feats_to_remove.append(feat_name)

        if config.verbosity >= 4:
            print(f"Init tvmb_map for {cv_slice.name=} {len(feats_to_remove)=}")
        for feat_name in feats_to_remove:
            cv_slice.removeFeature(feat_name)

    t1 = time.time()
    if dummy_call:
        if print_out:
            print(f"Time for tvmb filtering dummy call: {t1 - t0}")
        return samples

    cv_slice.rank_tvmb(samples_store_id, config)

    if return_score_list:
        if print_out:
            print(f"tvmb filtered: {len(cv_slice.ranked_tvmb)=}")
            print("===========================================================================================")
        return cv_slice.ranked_tvmb

    for feat_name, tvmb_score in cv_slice.ranked_tvmb[: config.tvmb_rank_threshold]:
        to_filter.append(feat_name)
        if debug or config.verbosity >= 5:
            print(cv_slice.name, "Filtered feature:", feat_name, "TVMB score:", tvmb_score)
    for feat_name in cv_slice.tvmb_map:
        tvmb_score, target_p_val = cv_slice.tvmb_map[feat_name]
        if abs(target_p_val) > config.p_val_thresh:
            to_filter.append(feat_name)
            if debug or config.verbosity >= 5:
                print(cv_slice.name, "Filtered feature:", feat_name, "p-value:", target_p_val)

    if print_out:
        print("tvmb filtered:", len(to_filter))
        print("===========================================================================================")

    if return_list:
        return to_filter

    cv_slice.filterFeatures(to_filter, print_out=print_out)

    return True