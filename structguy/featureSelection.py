import sys
import traceback
import random
import ray
import copy
import time
import psutil

from scipy import stats

from structguy import reguFeatureSelectionRegressor as reguFSreg
from structguy import reguFeatureSelectionClassificator as reguFSclf
from structguy import featureAnalysis, trainForest
from structguy.sampleSpace import splitDataSet
from structguy.support_classes import CrossValidationSlice
from structman.base_utils.base_utils import pack, unpack, add_to_times
from structguy.util import Config


def check_for_confmap_calculation(config, slice_slices, active_exp, sequence_number=0):
    if active_exp[sequence_number] == (config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp):
        return False

    for slice_slice in slice_slices:
        if slice_slice.confusion_map is None or sequence_number not in slice_slice.confusion_map:
            return True
        else:
            if (config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp) != slice_slice.active_exp[sequence_number]:
                if slice_slice.raw_confusion_map[sequence_number] is None:
                    return True

    return False


@ray.remote(max_calls=1)
def crossFold_confusion_internal_loop_wrapper(store, slice_slice, sample_store_id_list, get_loss_map):
    pre_filter, sequence_number, config, distance_map, overwrite_proc_n, force_confusion = store
    try:
        unpacked_slice_slice = unpack(slice_slice)
    except:
        unpacked_slice_slice = slice_slice
    conf_map, slice_slice, loss_map, loop_time_1, loop_time_2, loop_time_2_1, loop_time_2_2 = crossFold_confusion_internal_loop(
        unpacked_slice_slice, pre_filter, sequence_number, config, distance_map, overwrite_proc_n, samples_store_id=sample_store_id_list[0], force_confusion=force_confusion, get_loss_map=get_loss_map,
    )
    return conf_map, slice_slice, loss_map, loop_time_1, loop_time_2, loop_time_2_1, loop_time_2_2


def crossFold_confusion_internal_loop(
    slice_slice: CrossValidationSlice, pre_filter, sequence_number, config, distance_map, overwrite_proc_n, samples=None, samples_store_id=None, force_confusion=False, get_loss_map=False
):
    t_l_0 = time.time()

    # train forest on sliceslice
    if pre_filter is None:
        slice_slice.filterFeatures([])
    else:
        slice_slice.filterFeatures(pre_filter)
    t_l_1 = time.time()
    loop_time_1 = t_l_1 - t_l_0

    if force_confusion or slice_slice.raw_confusion_map is None or sequence_number not in slice_slice.raw_confusion_map:
        if slice_slice.raw_confusion_map is None:
            # slice_slice.confusion_map = {}
            slice_slice.raw_confusion_map = {}
        if sequence_number not in slice_slice.raw_confusion_map:
            # slice_slice.confusion_map[sequence_number] = None
            slice_slice.raw_confusion_map[sequence_number] = None

        calculate_confusion_map = True
    else:
        if slice_slice.raw_confusion_map[sequence_number] is not None:
            conf_map = featureAnalysis.confusion_map_from_raw_confusion_map(
                slice_slice, slice_slice.raw_confusion_map[sequence_number], config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp
            )
            calculate_confusion_map = False
        else:
            calculate_confusion_map = True

    t_l_2 = time.time()
    loop_time_2 = t_l_2 - t_l_1

    if calculate_confusion_map:
        t_l_2_0 = time.time()

        forest, _, slice_slice, _ = trainForest.trainForest(
            config, slice_slice, samples=samples, samples_store_id=samples_store_id, distance_map=distance_map, skip_feature_selection=True, skip_scoring=True, para_number=overwrite_proc_n, remote=False,
        )

        # calculate feature confusions

        t_l_2_1 = time.time()
        loop_time_2_1 = t_l_2_1 - t_l_2_0

        if config.verbosity >= 3:
            print(f"Calculating conf map: # of features {len(slice_slice.feature_names)}")
        if forest is not None:
            conf_map, raw_conf_map, loss_map = featureAnalysis.calcSliceConfusion(
                forest,
                slice_slice,
                samples=samples,
                samples_store_id=samples_store_id,
                remote=True,
                para_number=overwrite_proc_n,
                err_warping_exp=config.err_warping_exp,
                goodwill_interval=config.confusion_goodwill,
                norm_exp=config.confusion_normalization_exp,
                get_loss_map=get_loss_map,
            )
        else:
            conf_map = {}
            raw_conf_map = {}
            loss_map = {}

        if config.verbosity >= 3:
            print(f"Resulting conf map {slice_slice.name}: {len(conf_map)=}, {len(raw_conf_map)=}")

        slice_slice.raw_confusion_map[sequence_number] = raw_conf_map
        slice_slice = pack(slice_slice)

        t_l_2_2 = time.time()
        loop_time_2_2 = t_l_2_2 - t_l_2_1
    else:
        loop_time_2_1 = 0.0
        loop_time_2_2 = 0.0
        slice_slice = None
        loss_map = {}

    return conf_map, slice_slice, loss_map, loop_time_1, loop_time_2, loop_time_2_1, loop_time_2_2


def crossFoldConfusionSelect(
    config,
    cv_slice: CrossValidationSlice,
    slice_slices: None | list[CrossValidationSlice | bytes],
    samples=None,
    samples_store_id=None,
    distance_map=None,
    print_out=False,
    pre_filter=[],
    debug=False,
    return_list=False,
    overwrite_proc_n=None,
    rank_thresh=None,
    sequence_number=0,
    return_score_list=False,
    force_confusion=False,
    get_loss_map=False,
):
    times = []
    ta = time.time()

    if overwrite_proc_n is None:
        available_threads = config.proc_n
    else:
        available_threads = overwrite_proc_n

    if config.verbosity >= 2:
        print(f"Call of crossFoldconfusionSelect: {print_out=}, {overwrite_proc_n=}, {rank_thresh=}, {sequence_number=}, {force_confusion=}")

    # select by rank
    if rank_thresh is None:
        rank_thresh = int(config.confusion_rank_threshold)
    
    ta = add_to_times(times, ta)

    # make sliceslices
    if slice_slices is None:
        if not cv_slice.isSlice:
            slice_slices = cv_slice.slices
        else:
            if cv_slice.subslices is None:
                subslices = [set([x]) for x in cv_slice.train_prots]

            else:
                # print(f'Sublices: {cv_slice.subslices}')
                subslices = cv_slice.subslices

            slice_slices = []

            if samples is None:
                samples = unpack(ray.get(samples_store_id))

            for i, subslice_test_proteins in enumerate(subslices):
                try:
                    remaining_prots = set(cv_slice.train_prots) - subslice_test_proteins
                except TypeError:
                    remaining_prots = []

                if print_out and config.verbosity >= 3:
                    print(f"Subslice prots: {subslice_test_proteins}")
                if cv_slice.train_equal_test:
                    ignore_samples = None
                else:
                    ignore_samples = set(cv_slice.test_sample_ids)
                test_ids, train_ids = splitDataSet(config, samples.samples, specific_id=subslice_test_proteins, protein_wise=(not config.random_split), ignore_samples=ignore_samples)

                cv_slice_slice = CrossValidationSlice(
                    test_ids,
                    train_ids,
                    raw_feature_names=samples.feature_names,
                    sample_dict=samples.samples,
                    config=config,
                    name=f"{cv_slice.name}_subslice_{i}",
                    train_prots=remaining_prots,
                    para_number=overwrite_proc_n,
                    feature_names=cv_slice.feature_names,
                )
                
                slice_slices.append(cv_slice_slice)

        calc_at_least_once = True
    else:
        calc_at_least_once = False

    for slice_slice in slice_slices:
        try:
            slice_slice.filterFeatures(pre_filter)
        except AttributeError:
            continue

    if rank_thresh + len(pre_filter) >= len(cv_slice.feature_names):
        if return_list:
            return pre_filter, times, slice_slices

        cv_slice.filterFeatures(pre_filter, print_out=print_out)

        return True, pre_filter, times, slice_slices, None


    if force_confusion:
        calc_at_least_once = True

    ta = add_to_times(times, ta)

    if cv_slice.active_exp is None:
        cv_slice.active_exp = {}
        cv_slice.fused_confusion_map = {}
    if sequence_number not in cv_slice.active_exp:
        cv_slice.active_exp[sequence_number] = None
        cv_slice.fused_confusion_map[sequence_number] = None

    if force_confusion or cv_slice.active_exp[sequence_number] != (config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp):

        agg_conf_map = {}

        if pre_filter is None:
            cv_slice.filterFeatures([])
        else:
            if config.verbosity >= 3:
                print(f"In crossFoldConfusionSelect: {len(pre_filter)=}")
            cv_slice.filterFeatures(pre_filter)

        sample_size_threshold = config.gigs_of_ram * 3000
        n_of_samples = len(cv_slice.train_targets) + len(cv_slice.test_targets)

        if n_of_samples < sample_size_threshold:
            para_conf_calculation = True
        else:
            para_conf_calculation = False

        complete_loss_map = {}

        if not para_conf_calculation:
            updated_slice_slices = []
            for packed_slice_slice in slice_slices:
                try:
                    slice_slice = unpack(packed_slice_slice)
                except:
                    slice_slice = packed_slice_slice

                conf_map, slice_slice, loss_map, _loop_time_1, _loop_time_2, _loop_time_2_1, _loop_time_2_2 = crossFold_confusion_internal_loop(
                    slice_slice, pre_filter, sequence_number, config, distance_map, available_threads, samples=samples, samples_store_id=samples_store_id, force_confusion=force_confusion, get_loss_map=get_loss_map,
                )
                complete_loss_map.update(loss_map)
                if slice_slice is not None:
                    updated_slice_slices.append(slice_slice)



                for feat_name, confusion in conf_map:
                    if feat_name not in agg_conf_map:
                        agg_conf_map[feat_name] = []
                    agg_conf_map[feat_name].append(confusion)

            if len(updated_slice_slices) > 0:
                slice_slices = updated_slice_slices
        else:
            t_l_0 = time.time()
            n_sub_threads = available_threads // len(slice_slices)
            if calc_at_least_once:
                store = ray.put((pre_filter, sequence_number, config, distance_map, n_sub_threads, force_confusion))
                if samples_store_id is None:
                    samples_store_id = ray.put(pack(samples))
            else:
                store = ray.put((pre_filter, sequence_number, config, distance_map, n_sub_threads, force_confusion))
            loop_ray_ids = []

            for slice_slice in slice_slices:
                if calc_at_least_once and not isinstance(slice_slice, bytes):
                    packed_slice_slice = pack(slice_slice)
                else:
                    packed_slice_slice = slice_slice
                loop_ray_ids.append(crossFold_confusion_internal_loop_wrapper.remote(store, packed_slice_slice, [samples_store_id], get_loss_map))

            results = ray.get(loop_ray_ids)


            updated_slice_slices: list[bytes] = []
            for package in results:
                conf_map, slice_slice, loss_map, _loop_time_1, _loop_time_2, _loop_time_2_1, _loop_time_2_2 = package
                complete_loss_map.update(loss_map)
                if slice_slice is not None:
                    updated_slice_slices.append(slice_slice)
                for feat_name, confusion in conf_map:
                    if feat_name not in agg_conf_map:
                        agg_conf_map[feat_name] = []
                    agg_conf_map[feat_name].append(confusion)

            if len(updated_slice_slices) > 0:
                slice_slices = updated_slice_slices

        fused_confusion_map = []
        for feat_name in agg_conf_map:
            # mean_confusion = sum(agg_conf_map[feat_name])/len(agg_conf_map[feat_name])
            max_confusion = max(agg_conf_map[feat_name])
            # fused_confusion_map.append((feat_name, mean_confusion))
            fused_confusion_map.append((feat_name, max_confusion))

        fused_confusion_map = sorted(fused_confusion_map, key=lambda x: x[1], reverse=True)
        cv_slice.fused_confusion_map[sequence_number] = fused_confusion_map
        cv_slice.active_exp[sequence_number] = (config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp)
        cv_slice.loss_map = complete_loss_map
    else:
        fused_confusion_map = cv_slice.fused_confusion_map[sequence_number]
        complete_loss_map = cv_slice.loss_map

    ta = add_to_times(times, ta)

    if return_score_list:
        return fused_confusion_map, times, slice_slices

    

    if config.verbosity >= 3:
        print(f"Confusion filtering before filtering step: {len(fused_confusion_map)=}, {rank_thresh=} {config.confusion_rank_threshold=}")

    to_filter = pre_filter
    try:
        for feat_name, feature_confusion in fused_confusion_map[:-rank_thresh]:
            to_filter.append(feat_name)
            if config.verbosity >= 5:
                print("Confusion filter:", feat_name, feature_confusion)
    except:
        [e, f, g] = sys.exc_info()
        g = traceback.format_exc()
        print(f"ERROR: Illegal rank_thresh: {rank_thresh}\n{e}\n{f}\n{g}")
        sys.exit()

    # if debug:
    #    print(fused_confusion_map[-rank_thresh:])

    if print_out or config.verbosity >= 5:
        print("confusion filtered:", len(to_filter), return_list)

    ta = add_to_times(times, ta)

    if return_list:
        return to_filter, times, slice_slices

    cv_slice.filterFeatures(to_filter, print_out=print_out)

    ta = add_to_times(times, ta)

    """
    if print_out:
        confusion_rank_dict = calc_rank_dict(fused_confusion_map)

        agg_conf_map = {}
        for slice_slice in slice_slices:
            #print(f'Length of stored raw confusion map {slice_slice.name} {len(slice_slice.raw_confusion_map[sequence_number])}')
            conf_map = featureAnalysis.confusion_map_from_raw_confusion_map(slice_slice, slice_slice.raw_confusion_map[sequence_number], config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp)

            for (feat_name, confusion) in conf_map:
                if feat_name not in agg_conf_map:
                    agg_conf_map[feat_name] = []
                agg_conf_map[feat_name].append(confusion)

        fused_confusion_map = []
        feat_type_map = {}
        for feat_name in agg_conf_map:
            #mean_confusion = sum(agg_conf_map[feat_name])/len(agg_conf_map[feat_name])
            max_confusion = max(agg_conf_map[feat_name])
            #fused_confusion_map.append((feat_name, mean_confusion))
            fused_confusion_map.append((feat_name, max_confusion))
            feat_type_map[feat_name] = samples.features[feat_name].f_type

        fused_confusion_map = sorted(fused_confusion_map, key=lambda x: x[1], reverse=True)
        recalc_confusion_rank_dict = calc_rank_dict(fused_confusion_map)

        rank_dicts = {
            'confusion' : confusion_rank_dict,
            'recalc_confusion' : recalc_confusion_rank_dict
        }
        write_feature_ranks(config, rank_dicts, cv_slice.name, feat_type_map)
    """

    ta = add_to_times(times, ta)

    return True, to_filter, times, slice_slices, complete_loss_map


def confusionSelect(
    config, cv_slice, samples, distance_map=None, print_out=False, pre_filter=None, debug=False, return_list=False, overwrite_proc_n=None, rank_thresh=None, sequence_number=0, return_score_list=False
):
    if config.verbosity >= 4:
        print(f"Call of confusionSelect: print_out {print_out}, overwrite_proc_n {overwrite_proc_n}, rank_thresh {rank_thresh}, sequence_number {sequence_number}")

    # make sliceslice
    if cv_slice.slice_slice is None:
        if cv_slice.subslice is None:
            random_proteins = set([random.choice(list(cv_slice.train_prots))])

        else:
            print(f"Sublice: {cv_slice.subslice}")
            random_proteins = set(cv_slice.subslice)

        remaining_prots = set(cv_slice.train_prots) - random_proteins

        try:
            samples = ray.get(samples)
            samples = unpack(samples)
        except:
            pass
        print(f"Subslice prots: {random_proteins}")
        if cv_slice.train_equal_test:
            ignore_samples = None
        else:
            ignore_samples = set(cv_slice.test_sample_ids)
        test_ids, train_ids = splitDataSet(config, samples.samples, specific_id=random_proteins, protein_wise=(not config.random_split), ignore_samples=ignore_samples)

        cv_slice_slice = CrossValidationSlice(
            test_ids,
            train_ids,
            raw_feature_names=samples.feature_names,
            sample_dict=samples.samples,
            config=config,
            name=f"{cv_slice.name}_subslice",
            train_prots=remaining_prots,
            para_number=overwrite_proc_n,
            feature_names=cv_slice.feature_names,
        )
        cv_slice.slice_slice = cv_slice_slice
    else:
        cv_slice_slice = cv_slice.slice_slice

    # train forest on sliceslice
    if pre_filter is None:
        cv_slice.filterFeatures([])
        cv_slice_slice.filterFeatures([])
        to_filter = []
    else:
        cv_slice.filterFeatures(pre_filter)
        cv_slice_slice.filterFeatures(pre_filter)
        to_filter = pre_filter

    if print_out or config.verbosity >= 4:
        print("confusion prefiltered:", len(to_filter))

    pre_filter_count = len(to_filter)

    if cv_slice_slice.confusion_map is None or sequence_number not in cv_slice_slice.confusion_map:
        if cv_slice_slice.confusion_map is None:
            cv_slice_slice.confusion_map = {}
            cv_slice_slice.active_exp = {}
        if sequence_number not in cv_slice_slice.confusion_map:
            cv_slice_slice.confusion_map[sequence_number] = None
            cv_slice_slice.active_exp[sequence_number] = None
        calculate_confusion_map = True
    else:
        if (config.err_warping_exp, pre_filter_count) != cv_slice_slice.active_exp[sequence_number]:
            calculate_confusion_map = True
        else:
            calculate_confusion_map = False

    if calculate_confusion_map:
        forest, _, cv_slice_slice, _ = trainForest.trainForest(config, cv_slice_slice, samples, distance_map=distance_map, skip_feature_selection=True, skip_scoring=True, para_number=overwrite_proc_n)

        # calculate feature confusions

        if print_out or config.verbosity >= 4:
            print(f"Calculating conf map: # of features {len(cv_slice_slice.features)}")
        cv_slice_slice.confusion_map[sequence_number] = featureAnalysis.calcSliceConfusion(
            forest, cv_slice_slice, samples, remote=True, para_number=overwrite_proc_n, err_warping_exp=config.err_warping_exp, goodwill_interval=config.confusion_goodwill
        )
        cv_slice_slice.active_exp[sequence_number] = (config.err_warping_exp, pre_filter_count)

    if return_score_list:
        return cv_slice_slice.confusion_map[sequence_number]

    # select by rank
    if rank_thresh is None:
        rank_thresh = config.confusion_rank_threshold

    if config.verbosity >= 4:
        print(f"Confusion filtering before filtering step: size of conf map {len(cv_slice_slice.confusion_map[sequence_number])}, rank_thresh {rank_thresh}")

    try:
        for feat_name, feature_confusion in cv_slice_slice.confusion_map[sequence_number][: config.list_ranking_thresh]:
            to_filter.append(feat_name)
            if config.verbosity >= 5:
                print("Confusion filter:", feat_name, feature_confusion)
    except:
        [e, f, g] = sys.exc_info()
        g = traceback.format_exc()
        print(f"ERROR: Illegal rank_thresh: {rank_thresh}\n{e}\n{f}\n{g}")
        sys.exit()

    if print_out or config.verbosity >= 5:
        print("confusion filtered:", len(to_filter))

    if return_list:
        return to_filter

    cv_slice.filterFeatures(to_filter, print_out=print_out)

    return True


def pre_defined_feature_selection(config, cv_slice):
    f = open(config.feature_list_file, "r")
    lines = f.readlines()
    f.close()

    good_feats = set()
    for line in lines[1:]:
        words = line.split("\t")
        feat_name = words[0]
        good_feats.add(feat_name)

    to_filter = []
    for feat_name in cv_slice.feature_names:
        if feat_name not in good_feats:
            to_filter.append(feat_name)
    cv_slice.filterFeatures(to_filter)
    return True, to_filter, [], None


def regu_fs_wrapper(config, cross_val_object, samples, print_out=False, pre_filter=None, debug=False, return_list=False, overwrite_proc_n=None, return_score_list=False):
    cv_repeat = not cross_val_object.isSlice
    if not cv_repeat:
        return regu_fs(
            config, cross_val_object, samples, print_out=print_out, pre_filter=pre_filter, debug=debug, return_list=return_list, overwrite_proc_n=overwrite_proc_n, return_score_list=return_score_list
        )
    else:
        return_lists = {}
        first = True
        stored_return_list = None
        for cv_counter in cross_val_object.slices:
            if config.select_feature_for_first_slice_only:
                if first:
                    first = False
                else:
                    return_lists[cv_counter] = stored_return_list
                    continue

            if pre_filter is not None:
                to_filter = pre_filter[cv_counter]
            else:
                to_filter = None
            cv_slice = cross_val_object.slices[cv_counter]
            return_list = regu_fs(
                config, cv_slice, samples, print_out=print_out, pre_filter=to_filter, debug=debug, return_list=return_list, overwrite_proc_n=overwrite_proc_n, return_score_list=return_score_list
            )
            return_lists[cv_counter] = return_list
            if config.select_feature_for_first_slice_only:
                stored_return_list = return_list
        if return_list or return_score_list:
            return return_lists
        return True


def regu_fs(config, cv_slice, samples, print_out=False, pre_filter=None, debug=False, return_list=False, overwrite_proc_n=None, return_score_list=False):
    if cv_slice.random_subslice is None:
        min_amount_of_samples = 10000
        if len(cv_slice.train_sample_ids) < min_amount_of_samples:
            cv_sub_slice = cv_slice
        else:
            random_subsample_ids = random.sample(cv_slice.train_sample_ids, min_amount_of_samples)

            try:
                samples = ray.get(samples)
                samples = unpack(samples)
            except:
                pass

            cv_sub_slice = CrossValidationSlice(
                cv_slice.test_sample_ids,
                random_subsample_ids,
                raw_feature_names=samples.feature_names,
                sample_dict=samples.samples,
                config=config,
                name=f"{cv_slice.name}_random_subsample",
                para_number=overwrite_proc_n,
                train_equal_test=cv_slice.train_equal_test,
                feature_names=cv_slice.feature_names,
            )
            cv_slice.random_subslice = cv_sub_slice
    else:
        cv_sub_slice = cv_slice.random_subslice

    if config.regression:
        if print_out:
            print("\n--- start Regularization-FS (regression)", config.reg_alpha_exp, config.reg_thresh_exp, " ---\n")
        if pre_filter is None:
            if (config.reg_alpha_exp, config.reg_thresh_exp) == cv_sub_slice.active_reg:
                if print_out:
                    print("ReguFS skipped, FS with that alpha already active")
                if return_score_list:
                    return cv_sub_slice.c_map[(config.reg_alpha_exp, config.reg_thresh_exp)]
                return False

        if pre_filter is None:
            cv_sub_slice.active_reg = (config.reg_alpha_exp, config.reg_thresh_exp)
            to_filter = reguFSreg.wrapper(cv_sub_slice, config, print_out=print_out, pre_filter=pre_filter, debug=debug, return_score_list=return_score_list)
        else:
            if print_out:
                print("regufs prefiltered:", len(pre_filter))
            to_filter = reguFSreg.wrapper(cv_sub_slice, config, print_out=print_out, pre_filter=pre_filter, debug=debug, return_score_list=return_score_list)
        if return_score_list:
            cv_sub_slice.c_map[(config.reg_alpha_exp, config.reg_thresh_exp)] = to_filter
    else:
        if (config.reg_c_exp, config.reg_thresh_exp) == cv_sub_slice.active_reg:
            if return_score_list:
                return cv_sub_slice.c_map[(config.reg_c_exp, config.reg_thresh_exp)]
            return False
        if print_out:
            print("\n--- start Regularization-FS (classification) ---\n")
        if (config.reg_c_exp, config.reg_thresh_exp) in cv_sub_slice.c_map:
            to_filter = cv_sub_slice.c_map[(config.reg_c_exp, config.reg_thresh_exp)]
        else:
            cv_sub_slice.filterFeatures([])
            to_filter = reguFSclf.wrapper(cv_sub_slice, config, print_out=print_out)
            cv_sub_slice.c_map[(config.reg_c_exp, config.reg_thresh_exp)] = to_filter
        cv_sub_slice.active_reg = (config.reg_c_exp, config.reg_thresh_exp)

    if print_out:
        print("regufs filtered:", len(to_filter))

    if return_list or return_score_list:
        return to_filter

    cv_slice.filterFeatures(to_filter, print_out=print_out)
    return True


def filterCorrelatedFeats(
    config: Config,
    samples = None,
    samples_store_id = None,
):
    
    if samples is None:
        samples = unpack(ray.get(samples_store_id))

    feats_to_filter = set()
    for feat_nr_a, feat_name_a in enumerate(samples.feature_names):
        if feat_name_a in feats_to_filter:
            continue
        for feat_nr_b, feat_name_b in enumerate(samples.feature_names):
            if feat_nr_a == feat_nr_b:
                continue
        
            if feat_name_b in feats_to_filter:
                continue

            corr, cov_a, cov_b, cov_both, tv_corr_a, tv_corr_b = samples.feat_corr_matrix[feat_nr_a][feat_nr_b]
            feat_score_a = cov_a * abs(tv_corr_a)
            feat_score_b = cov_b * abs(tv_corr_b)

            if abs(corr) > config.corr_thresh and cov_both > 0.01:
                
                if feat_score_a > feat_score_b:
                    feats_to_filter.add(feat_name_b)
                else:
                    feats_to_filter.add(feat_name_b)

    if config.verbosity >= 3:
        print(f'Filtering correlated features {config.corr_thresh=} {len(feats_to_filter)=}')

    return list(feats_to_filter), samples


def meanCorrelationWrapper(
    config: Config,
    cross_val_object: CrossValidationSlice,
    samples_store_id,
    samples=None,
    dummy_call=False,
    print_out=False,
    pre_filter=None,
    debug=False,
    return_list=False,
    return_score_list=False,
    remote=False
):
    cv_repeat = not cross_val_object.isSlice
    if config.tvmb_rank_threshold < 0:
        return None

    if not cv_repeat:
        return detectBiasedFeaturesByMeanCorrelation(
            config, cross_val_object, samples_store_id, samples=samples, dummy_call=dummy_call, print_out=print_out, pre_filter=pre_filter, debug=debug, return_list=return_list, return_score_list=return_score_list, remote=remote
        )

    else:
        return_lists = {}
        for cv_counter in cross_val_object.slices:
            cv_slice = cross_val_object.slices[cv_counter]
            return_lists[cv_counter] = detectBiasedFeaturesByMeanCorrelation(
                config, cv_slice, samples_store_id, samples=samples, dummy_call=dummy_call, print_out=print_out, pre_filter=pre_filter, debug=debug, return_list=return_list, return_score_list=return_score_list, remote=remote
            )

        if return_list or return_score_list:
            return return_lists

        return True


def positionalMeanCorrelationWrapper(config, cross_val_object, print_out=False, pre_filter=None, debug=False, return_list=False):
    cv_repeat = not cross_val_object.isSlice
    if config.tvpmb_rank_threshold < 0:
        return None
    if not cv_repeat:
        return detectBiasedFeaturesByPositionalMeanCorrelation(config, cross_val_object, print_out=print_out, pre_filter=pre_filter, debug=debug, return_list=return_list)

    else:
        return_lists = {}
        for cv_counter in cross_val_object.slices:
            cv_slice = cross_val_object.slices[cv_counter]
            return_lists[cv_counter] = detectBiasedFeaturesByPositionalMeanCorrelation(config, cv_slice, print_out=print_out, pre_filter=pre_filter, debug=debug, return_list=return_list)

        if return_list:
            return return_lists

        return True


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
    feature_value_vector = samples.get_feature_value_vector(train_sample_ids, feat_name)

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
    config,
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
        return

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


def detectBiasedFeaturesByPositionalMeanCorrelation(config, cv_slice, thresh=None, print_out=False, pre_filter=None, debug=False, return_list=False):
    if thresh is None:
        thresh = config.tvpmb_rank_threshold
    if pre_filter is None:
        cv_slice.filterFeatures([])
        to_filter = []
    else:
        cv_slice.filterFeatures(pre_filter)
        to_filter = pre_filter

    if print_out:
        print("===========================================================================================")
        print("Biased feature detection (by mean correlation), TVPMB thresh:", thresh)
        print("tvpmb prefiltered:", len(to_filter))

    if "Position means" not in cv_slice.slice_specific_feature_map:
        cv_slice.setPositionMeans(config)

    if cv_slice.tvpmb_map is None:
        cv_slice.tvpmb_map = {}
        for feat_name in cv_slice.feature_names:
            if feat_name == "Position means":
                continue
            if feat_name not in cv_slice.tvpmb_map:
                cv_slice.addToTvpmbMap(feat_name, config)
            else:
                tvpmb_score = cv_slice.tvpmb_map[feat_name]

    cv_slice.rank_tvpmb(config)
    for feat_name, tvpmb_score in cv_slice.ranked_tvpmb[: config.tvpmb_rank_threshold]:
        to_filter.append(feat_name)
        if debug or config.verbosity >= 5:
            print(cv_slice.name, "Filtered feature:", feat_name, "TVPMB score:", tvpmb_score)

    if print_out:
        print("tvpmb filtered:", len(to_filter))
        print("===========================================================================================")

    if return_list:
        return to_filter

    cv_slice.filterFeatures(to_filter, print_out=print_out)

    return True


# Needs to be updated
def detectBiasedFeaturesByBalancedImportances(config, samples):
    something_filtered = True
    n = 0
    while something_filtered and n < 5:
        something_filtered = False
        n += 1
        # step zero: save original configs
        ori_prot_based_separation = config.prot_based_separation
        ori_balanceSubsampling = config.balanceSubsampling
        ori_filter_single_variant_prots = config.filter_single_variant_prots
        # step one: train a forest with raw unbalanced samples
        config.prot_based_separation = True
        config.balanceSubsampling = None
        config.filter_single_variant_prots = False

        forest, scores = trainForest(config, samples, repeat=config.repeat_training)

        # step two: calculate feature importance values
        raw_feat_importance_map = calcFeatureImportances(forest, samples, config)
        # step three: train a forest with balanced samples
        config.prot_based_separation = True
        config.balanceSubsampling = "balanced"
        config.filter_single_variant_prots = True

        samples.balanceSubSampleTrainSet(config)
        samples.calcVectors(config)

        forest, scores = trainForest(config, samples, repeat=config.repeat_training)

        # step four: calculate feature importance values
        balanced_feat_importance_map = calcFeatureImportances(forest, samples, config)
        # step five: compare the importance values
        diff_tuples = []
        for feat_name in raw_feat_importance_map:
            raw_score = raw_feat_importance_map[feat_name]
            bal_score = balanced_feat_importance_map[feat_name]
            diff_tuples.append([feat_name, raw_score - bal_score])

        diff_tuples.sort(key=lambda x: x[1], reverse=True)
        print("===========================================================================================")
        print("Biased feature detection")
        samples.printBalance(config)
        mean_importance = 1 / len(diff_tuples)
        print("Total amount of features:", len(diff_tuples), "Mean feature importance:", mean_importance)
        print_once = False
        for feature_name, score in diff_tuples:
            if abs(score) > mean_importance:
                if score < 0.0 and not print_once:
                    print("...")
                    print_once = True
                print(feature_name, ":", score)
            if score > config.bfd_factor * mean_importance:
                samples.removeFeature(feature_name)
                print("Removed feature:", feature_name)
                something_filtered = True
        # step six: reset original configs
        config.prot_based_separation = ori_prot_based_separation
        config.balanceSubsampling = ori_balanceSubsampling
        config.filter_single_variant_prots = ori_filter_single_variant_prots
        samples.undoBalancing(config)
        samples.printBalance(config)
        print("===========================================================================================")
    return


def select_by_sequential_confusion(config, cross_val_object, samples, pre_filter=None, print_out=False, debug=False, overwrite_proc_n=None, return_list=False, return_score_list=False, repetition=2):
    if print_out:
        print("=== Feature selection by sequential feature confusion ===")
    if config.confusion_rank_threshold < 0 or config.sequential_confusion_rank_threshold < 0:
        return None

    cv_repeat = not cross_val_object.isSlice

    if not cv_repeat:
        if pre_filter is None:
            to_filter = []
        else:
            to_filter = pre_filter
        i = -1
        for i in range(repetition - 1):
            if i == 0:
                to_filter = confusionSelect(
                    config,
                    cross_val_object,
                    samples,
                    print_out=print_out,
                    debug=debug,
                    pre_filter=to_filter,
                    return_list=True,
                    overwrite_proc_n=overwrite_proc_n,
                    rank_thresh=config.sequential_confusion_rank_threshold,
                    return_score_list=return_score_list,
                )
            else:
                to_filter = confusionSelect(
                    config,
                    cross_val_object,
                    samples,
                    print_out=print_out,
                    debug=debug,
                    pre_filter=to_filter,
                    return_list=True,
                    overwrite_proc_n=overwrite_proc_n,
                    sequence_number=i,
                    return_score_list=return_score_list,
                )
        return confusionSelect(
            config,
            cross_val_object,
            samples,
            print_out=print_out,
            debug=debug,
            pre_filter=to_filter,
            overwrite_proc_n=overwrite_proc_n,
            return_list=return_list,
            sequence_number=i + 1,
            return_score_list=return_score_list,
        )
    else:
        return_lists = {}
        first = True
        stored_return_list = None
        for cv_counter in cross_val_object.slices:
            if config.select_feature_for_first_slice_only:
                if first:
                    first = False
                else:
                    return_lists[cv_counter] = stored_return_list
                    continue
            cv_slice = cross_val_object.slices[cv_counter]
            if pre_filter is None:
                to_filter = []
            else:
                to_filter = pre_filter
            i = -1
            for i in range(repetition - 1):
                if i == 0:
                    to_filter = confusionSelect(
                        config,
                        cv_slice,
                        samples,
                        print_out=print_out,
                        debug=debug,
                        pre_filter=to_filter,
                        return_list=True,
                        overwrite_proc_n=overwrite_proc_n,
                        rank_thresh=config.sequential_confusion_rank_threshold,
                        return_score_list=return_score_list,
                    )
                else:
                    to_filter = confusionSelect(
                        config,
                        cv_slice,
                        samples,
                        print_out=print_out,
                        debug=debug,
                        pre_filter=to_filter,
                        return_list=True,
                        overwrite_proc_n=overwrite_proc_n,
                        sequence_number=i,
                        return_score_list=return_score_list,
                    )
            return_list = confusionSelect(
                config,
                cv_slice,
                samples,
                print_out=print_out,
                debug=debug,
                pre_filter=to_filter,
                overwrite_proc_n=overwrite_proc_n,
                return_list=return_list,
                sequence_number=i + 1,
                return_score_list=return_score_list,
            )
            return_lists[cv_counter] = return_list
            if config.select_feature_for_first_slice_only:
                stored_return_list = return_list
        if return_list or return_score_list:
            return return_lists
        return True


def select_features(
        config,
        cross_val_object: CrossValidationSlice,
        slice_slices,
        samples_store_id=None,
        samples=None,
        print_out=False,
        debug=False,
        overwrite_proc_n=None,
        force_confusion=False,
        get_loss_map=False,
        remote=False
        ):
    
    times = []
    ta = time.time()

    if config.feature_selection == "confusion":
        if print_out:
            print("=== Feature selection by feature confusion ===")
            print(f'{overwrite_proc_n=} {force_confusion=} {get_loss_map=}')

        filter_corr_feats, samples = filterCorrelatedFeats(config, samples = samples, samples_store_id = samples_store_id)

        ta = add_to_times(times, ta)

        meanCorrelationWrapper(
            config,
            cross_val_object,
            samples_store_id,
            samples=samples,
            dummy_call=True,
            print_out=print_out,
            debug=debug,
            return_score_list=True,
            remote=remote,
            pre_filter=filter_corr_feats
            )
        
        ta = add_to_times(times, ta)

        filtered, to_filter, conf_times, slice_slices, loss_map = crossFoldConfusionSelect(
            config,
            cross_val_object,
            slice_slices,
            samples=samples,
            samples_store_id=samples_store_id,
            print_out=print_out,
            debug=debug,
            overwrite_proc_n=overwrite_proc_n,
            force_confusion=force_confusion,
            get_loss_map=get_loss_map,
            pre_filter=filter_corr_feats,
        )

        times += conf_times

        return filtered, to_filter, times, slice_slices, loss_map

    else:
        print("=== ERROR: Feature selection with illegal key word:", config.feature_selection, "called ===")
        return None


def calc_rank_dict(score_list):
    rank_dict = {}
    min_score = None
    max_score = None
    for rank, (f_name, score) in enumerate(score_list):
        if rank > 0:
            prev_f_name, prev_score = score_list[rank - 1]
            if score == prev_score:
                prev_tied_rank = rank_dict[prev_f_name][2]
                tied_rank = prev_tied_rank
            else:
                tied_rank = rank
        else:
            tied_rank = rank
        rank_dict[f_name] = [rank, score, tied_rank]
        if min_score is None:
            min_score = score
        elif score < min_score:
            min_score = score
        if max_score is None:
            max_score = score
        elif max_score < score:
            max_score = score

    for f_name in rank_dict:
        raw_score = rank_dict[f_name][1]
        scaled_score = (raw_score - min_score) / (max_score - min_score)

        rank_dict[f_name].append(scaled_score)

    return rank_dict


def write_feature_ranks(config, rank_dicts, name, feat_type_map):
    headers = ["Feature Name", "Feature Type"]
    fs_types = list(rank_dicts.keys())
    for fs_type in fs_types:
        headers.append(f"{fs_type} Rank")
        headers.append(f"{fs_type} Tied Rank")
        headers.append(f"{fs_type} Score")
        headers.append(f"{fs_type} Scaled Score")

    header = "\t".join(headers) + "\n"

    feat_lines = {}
    for fs_type in fs_types:
        for feat_name in rank_dicts[fs_type]:
            rank, score, tied_rank, scaled_score = rank_dicts[fs_type][feat_name]
            if feat_name not in feat_lines:
                feat_lines[feat_name] = [feat_name, feat_type_map[feat_name]]
            feat_lines[feat_name].append(str(rank))
            feat_lines[feat_name].append(str(tied_rank))
            feat_lines[feat_name].append(str(score))
            feat_lines[feat_name].append(str(scaled_score))

    lines = [header]

    for feat_name in feat_lines:
        lines.append("\t".join(feat_lines[feat_name]) + "\n")

    page = "".join(lines)

    outfile = f"{config.outfolder}/{config.dataset_name}_feature_ranks_{name}.tsv"

    f = open(outfile, "w")
    f.write(page)
    f.close()
