from ossaudiodev import SNDCTL_SEQ_GETINCOUNT
import sys
import traceback
import random
import ray
import copy
import time

from structguy import reguFeatureSelectionRegressor as reguFSreg
from structguy import reguFeatureSelectionClassificator as reguFSclf
from structguy import featureAnalysis, trainForest
from structguy.sampleSpace import splitDataSet
from structguy.support_classes import CrossValidationSlice
from structman.base_utils.base_utils import pack, unpack

def check_for_confmap_calculation(config, slice_slices, active_exp, sequence_number = 0):

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

@ray.remote(max_calls = 1)
def crossFold_confusion_internal_loop_wrapper(store, slice_slice):
    pre_filter, sequence_number, config, samples, distance_map, overwrite_proc_n = store
    conf_map, slice_slice, loop_time_1, loop_time_2, loop_time_2_1, loop_time_2_2 = crossFold_confusion_internal_loop(unpack(slice_slice), pre_filter, sequence_number, config, samples, distance_map, overwrite_proc_n)
    return pack(conf_map), slice_slice, loop_time_1, loop_time_2, loop_time_2_1, loop_time_2_2

def crossFold_confusion_internal_loop(slice_slice, pre_filter, sequence_number, config, samples, distance_map, overwrite_proc_n):
    t_l_0 = time.time()

    #train forest on sliceslice
    if pre_filter is None:
        slice_slice.filterFeatures([])
    else:
        slice_slice.filterFeatures(pre_filter)
    t_l_1 = time.time()
    loop_time_1 = (t_l_1 - t_l_0)

    if slice_slice.raw_confusion_map is None or sequence_number not in slice_slice.raw_confusion_map:
        if slice_slice.raw_confusion_map is None:
            #slice_slice.confusion_map = {}
            slice_slice.raw_confusion_map = {}
        if sequence_number not in slice_slice.raw_confusion_map:
            #slice_slice.confusion_map[sequence_number] = None
            slice_slice.raw_confusion_map[sequence_number] = None

        calculate_confusion_map = True
    else:
        if slice_slice.raw_confusion_map[sequence_number] is not None:
            conf_map = featureAnalysis.confusion_map_from_raw_confusion_map(slice_slice, slice_slice.raw_confusion_map[sequence_number], config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp)
            calculate_confusion_map = False
        else:
            calculate_confusion_map = True

    t_l_2 = time.time()
    loop_time_2 = (t_l_2 - t_l_1)

    if calculate_confusion_map:
        t_l_2_0 = time.time()
    
        forest, _, slice_slice, _ = trainForest.trainForest(config, slice_slice, samples, distance_map = distance_map, skip_feature_selection = True, skip_scoring = True, para_number = overwrite_proc_n)

        #calculate feature confusions

        t_l_2_1 = time.time()
        loop_time_2_1 = (t_l_2_1 - t_l_2_0)

        if config.verbosity >= 4:
            print(f'Calculating conf map: # of features {len(slice_slice.features)}')
        
        conf_map, raw_conf_map = featureAnalysis.calcSliceConfusion(forest, slice_slice, samples, remote = True, para_number = overwrite_proc_n, err_warping_exp = config.err_warping_exp, goodwill_interval = config.confusion_goodwill, norm_exp = config.confusion_normalization_exp)
        
        if config.verbosity >= 4:
            print(f'Resulting conf map {slice_slice.name}: {len(conf_map)}, {len(raw_conf_map)}')

        slice_slice.raw_confusion_map[sequence_number] = raw_conf_map
        slice_slice = pack(slice_slice)

        t_l_2_2 = time.time()
        loop_time_2_2 = (t_l_2_2 - t_l_2_1)
    else:
        loop_time_2_1 = 0.
        loop_time_2_2 = 0.
        slice_slice = None

    return conf_map, slice_slice, loop_time_1, loop_time_2, loop_time_2_1, loop_time_2_2

def crossFoldConfusionSelect(config, cv_slice, samples, slice_slices, distance_map = None, print_out = False, pre_filter = None, debug = False, return_list = False, overwrite_proc_n = None, rank_thresh = None, sequence_number = 0, return_score_list = False):

    times = []
    t0 = time.time()

    if overwrite_proc_n is None:
        available_threads = config.proc_n
    else:
        available_threads = overwrite_proc_n

    if config.verbosity >= 4:
        print(f'Call of crossFoldconfusionSelect: print_out {print_out}, overwrite_proc_n {overwrite_proc_n}, rank_thresh {rank_thresh}, sequence_number {sequence_number}')

    try:
        samples = ray.get(samples)
    except:
        samples = samples

    t1 = time.time()
    times.append(('1',t1-t0))

    #make sliceslices
    if slice_slices is None:
        if not cv_slice.isSlice:
            slice_slices = cv_slice.slices
        else:

            if cv_slice.subslices is None:
                subslices = [set([x]) for x in cv_slice.train_prots]

            else:
                print(f'Sublices: {cv_slice.subslices}')
                subslices = cv_slice.subslices

            slice_slices = []
            for i, subslice_test_proteins in enumerate(subslices):
                remaining_prots = set(cv_slice.train_prots) - subslice_test_proteins

                
                if print_out:
                    print(f'Subslice prots: {subslice_test_proteins}')
                if cv_slice.train_equal_test:
                    ignore_samples = None
                else:
                    ignore_samples = set(cv_slice.test_sample_ids)
                test_ids, train_ids = splitDataSet(config, samples.samples, specific_id = subslice_test_proteins, protein_wise = True, ignore_samples = ignore_samples)

                cv_slice_slice = CrossValidationSlice(test_ids, train_ids, raw_feature_names = samples.feature_names, sample_dict = samples.samples, config = config, name = f'{cv_slice.name}_subslice_{i}', train_prots = remaining_prots, para_number = overwrite_proc_n, feature_names = cv_slice.feature_names)
                slice_slices.append(cv_slice_slice)

        calc_at_least_once = True
    else:
        calc_at_least_once = False

    t2 = time.time()
    times.append(('2',t2-t1))

    if cv_slice.active_exp is None:
        cv_slice.active_exp = {}
        cv_slice.fused_confusion_map = {}
    if sequence_number not in cv_slice.active_exp:
        cv_slice.active_exp[sequence_number] = None
        cv_slice.fused_confusion_map[sequence_number] = None

    if cv_slice.active_exp[sequence_number] != (config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp):
        t1_0 = time.time()
        agg_conf_map = {}

        loop_time_1 = 0.
        loop_time_2 = 0.
        loop_time_2_1 = 0.
        loop_time_2_2 = 0.
        loop_time_3 = 0.

        if pre_filter is None:
            cv_slice.filterFeatures([])
        else:
            cv_slice.filterFeatures(pre_filter)

        if available_threads < len(slice_slices):
            updated_slice_slices = []
            for packed_slice_slice in slice_slices:
                conf_map, slice_slice, _loop_time_1, _loop_time_2, _loop_time_2_1, _loop_time_2_2 = crossFold_confusion_internal_loop(unpack(packed_slice_slice), pre_filter, sequence_number, config, samples, distance_map, available_threads)

                if slice_slice is not None:
                    updated_slice_slices.append(slice_slice)

                loop_time_1 += _loop_time_1
                loop_time_2 += _loop_time_2
                loop_time_2_1 += _loop_time_2_1
                loop_time_2_2 += _loop_time_2_2

                t_l_2 = time.time()

                for (feat_name, confusion) in conf_map:
                    if feat_name not in agg_conf_map:
                        agg_conf_map[feat_name] = []
                    agg_conf_map[feat_name].append(confusion)

                t_l_3 = time.time()
                loop_time_3 += (t_l_3 - t_l_2)
            if len(updated_slice_slices) > 0:
                slice_slices = updated_slice_slices
        else:
            t_l_0 = time.time()
            n_sub_threads = available_threads // len(slice_slices)
            if calc_at_least_once:
                store = ray.put((pre_filter, sequence_number, config, samples, distance_map, n_sub_threads))
            else:
                store = ray.put((pre_filter, sequence_number, config, None, distance_map, n_sub_threads))
            loop_ray_ids = []
            for slice_slice in slice_slices:
                if calc_at_least_once:
                    packed_slice_slice = pack(slice_slice)
                else:
                    packed_slice_slice = slice_slice
                loop_ray_ids.append(crossFold_confusion_internal_loop_wrapper.remote(store, packed_slice_slice))

            t_l_1 = time.time()
            loop_time_1 = t_l_1 - t_l_0

            results = ray.get(loop_ray_ids)

            t_l_2 = time.time()
            loop_time_2 = t_l_2 - t_l_1

            updated_slice_slices = []
            for package in results:
                packed_conf_map, slice_slice, _loop_time_1, _loop_time_2, _loop_time_2_1, _loop_time_2_2 = package
                conf_map = unpack(packed_conf_map)
                if slice_slice is not None:
                    updated_slice_slices.append(slice_slice)
                for (feat_name, confusion) in conf_map:
                    if feat_name not in agg_conf_map:
                        agg_conf_map[feat_name] = []
                    agg_conf_map[feat_name].append(confusion)

            if len(updated_slice_slices) > 0:
                slice_slices = updated_slice_slices

            t_l_3 = time.time()
            loop_time_3 = (t_l_3 - t_l_2)
            

        times.append(('1.1.l1', loop_time_1))
        times.append(('1.1.l2', loop_time_2))
        times.append(('1.1.l2.1', loop_time_2_1))
        times.append(('1.1.l2.2', loop_time_2_2))
        times.append(('1.1.l3', loop_time_3))

        t1_1 = time.time()
        times.append(('1.1',t1_1-t1_0))            

        fused_confusion_map = []
        for feat_name in agg_conf_map:
            #mean_confusion = sum(agg_conf_map[feat_name])/len(agg_conf_map[feat_name])
            max_confusion = max(agg_conf_map[feat_name])
            #fused_confusion_map.append((feat_name, mean_confusion))
            fused_confusion_map.append((feat_name, max_confusion))

        fused_confusion_map = sorted(fused_confusion_map, key=lambda x: x[1], reverse=True)
        cv_slice.fused_confusion_map[sequence_number] = fused_confusion_map
        cv_slice.active_exp[sequence_number] = (config.confusion_goodwill, config.err_warping_exp, config.confusion_normalization_exp)
    else:
        fused_confusion_map = cv_slice.fused_confusion_map[sequence_number]

    t3 = time.time()
    times.append(('3',t3-t2))

    if return_score_list:
        return fused_confusion_map, times, slice_slices

    
    #select by rank
    if rank_thresh is None:
        rank_thresh = config.confusion_rank_threshold

    if config.verbosity >= 4:
        print(f'Confusion filtering before filtering step: size of conf map {len(fused_confusion_map)}, rank_thresh {rank_thresh}')

    to_filter = []
    try:
        for feat_name, feature_confusion in fused_confusion_map[:-rank_thresh]:
            to_filter.append(feat_name)
            if config.verbosity >= 5:
                print('Confusion filter:',feat_name, feature_confusion)
    except:

        [e, f, g] = sys.exc_info()
        g = traceback.format_exc()
        print(f'ERROR: Illegal rank_thresh: {rank_thresh}\n{e}\n{f}\n{g}')
        sys.exit()

    if debug:
        print(fused_confusion_map[-rank_thresh:])

    if print_out or config.verbosity >= 5:
        print('confusion filtered:',len(to_filter))

    t4 = time.time()
    times.append(('4',t4-t3))

    if return_list:
        return to_filter, times, slice_slices

    cv_slice.filterFeatures(to_filter, print_out = print_out)

    t5 = time.time()
    times.append(('5',t5-t4))

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
        
    t6 = time.time()
    times.append(('6',t6-t5))

    return True, times, slice_slices
    

def confusionSelect(config, cv_slice, samples, distance_map = None, print_out = False, pre_filter = None, debug = False, return_list = False, overwrite_proc_n = None, rank_thresh = None, sequence_number = 0, return_score_list = False):

    if config.verbosity >= 4:
        print(f'Call of confusionSelect: print_out {print_out}, overwrite_proc_n {overwrite_proc_n}, rank_thresh {rank_thresh}, sequence_number {sequence_number}')

    #make sliceslice
    if cv_slice.slice_slice is None:

        if cv_slice.subslice is None:
            random_proteins = set([random.choice(list(cv_slice.train_prots))])
            
        else:
            print(f'Sublice: {cv_slice.subslice}')
            random_proteins = set(cv_slice.subslice)

        remaining_prots = set(cv_slice.train_prots) - random_proteins

        try:
            samples = ray.get(samples)
        except:
            pass
        print(f'Subslice prots: {random_proteins}')
        if cv_slice.train_equal_test:
            ignore_samples = None
        else:
            ignore_samples = set(cv_slice.test_sample_ids)
        test_ids, train_ids = splitDataSet(config, samples.samples, specific_id = random_proteins, protein_wise = True, ignore_samples = ignore_samples)

        cv_slice_slice = CrossValidationSlice(test_ids, train_ids, raw_feature_names = samples.feature_names, sample_dict = samples.samples, config = config, name = f'{cv_slice.name}_subslice', train_prots = remaining_prots, para_number = overwrite_proc_n, feature_names = cv_slice.feature_names)
        cv_slice.slice_slice = cv_slice_slice
    else:
        cv_slice_slice = cv_slice.slice_slice

    #train forest on sliceslice
    if pre_filter is None:
        cv_slice.filterFeatures([])
        cv_slice_slice.filterFeatures([])
        to_filter = []
    else:
        cv_slice.filterFeatures(pre_filter)
        cv_slice_slice.filterFeatures(pre_filter)
        to_filter = pre_filter

    if print_out or config.verbosity >= 4:
        print('confusion prefiltered:',len(to_filter))

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
        forest, _, cv_slice_slice, _ = trainForest.trainForest(config, cv_slice_slice, samples, distance_map = distance_map, skip_feature_selection = True, skip_scoring = True, para_number = overwrite_proc_n)

        #calculate feature confusions

        if print_out or config.verbosity >= 4:
            print(f'Calculating conf map: # of features {len(cv_slice_slice.features)}')
        cv_slice_slice.confusion_map[sequence_number] = featureAnalysis.calcSliceConfusion(forest, cv_slice_slice, samples, remote = True, para_number = overwrite_proc_n, err_warping_exp = config.err_warping_exp, goodwill_interval = config.confusion_goodwill)
        cv_slice_slice.active_exp[sequence_number] = (config.err_warping_exp, pre_filter_count)

    if return_score_list:
        return cv_slice_slice.confusion_map[sequence_number]

    #select by rank
    if rank_thresh is None:
        rank_thresh = config.confusion_rank_threshold

    if config.verbosity >= 4:
        print(f'Confusion filtering before filtering step: size of conf map {len(cv_slice_slice.confusion_map[sequence_number])}, rank_thresh {rank_thresh}')

    try:
        for feat_name, feature_confusion in cv_slice_slice.confusion_map[sequence_number][:config.list_ranking_thresh]:
            to_filter.append(feat_name)
            if config.verbosity >= 5:
                print('Confusion filter:',feat_name, feature_confusion)
    except:

        [e, f, g] = sys.exc_info()
        g = traceback.format_exc()
        print(f'ERROR: Illegal rank_thresh: {rank_thresh}\n{e}\n{f}\n{g}')
        sys.exit()

    if print_out or config.verbosity >= 5:
        print('confusion filtered:',len(to_filter))

    if return_list:
        return to_filter

    cv_slice.filterFeatures(to_filter, print_out = print_out)

    return True

def regu_fs_wrapper(config, cross_val_object, samples, print_out = False, pre_filter = None, debug = False, return_list = False, overwrite_proc_n = None, return_score_list = False):
    cv_repeat = not cross_val_object.isSlice
    if not cv_repeat:
        return regu_fs(config, cross_val_object, samples, print_out = print_out, pre_filter = pre_filter, debug = debug, return_list = return_list, overwrite_proc_n = overwrite_proc_n, return_score_list = return_score_list)
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
            return_list = regu_fs(config, cv_slice, samples, print_out = print_out, pre_filter = to_filter, debug = debug, return_list = return_list, overwrite_proc_n = overwrite_proc_n, return_score_list = return_score_list)
            return_lists[cv_counter] = return_list
            if config.select_feature_for_first_slice_only:
                stored_return_list = return_list
        if return_list or return_score_list:
            return return_lists
        return True

def regu_fs(config, cv_slice, samples, print_out = False, pre_filter = None, debug = False, return_list = False, overwrite_proc_n = None, return_score_list = False):

    if cv_slice.random_subslice is None:
        min_amount_of_samples = 10000
        if len(cv_slice.train_sample_ids) < min_amount_of_samples:
            cv_sub_slice = cv_slice
        else:

            random_subsample_ids = random.sample(cv_slice.train_sample_ids, min_amount_of_samples)

            try:
                samples = ray.get(samples)
            except:
                pass

            cv_sub_slice = CrossValidationSlice(cv_slice.test_sample_ids, random_subsample_ids, raw_feature_names = samples.feature_names, sample_dict = samples.samples, config = config, name = f'{cv_slice.name}_random_subsample', para_number = overwrite_proc_n, train_equal_test = cv_slice.train_equal_test, feature_names = cv_slice.feature_names)
            cv_slice.random_subslice = cv_sub_slice
    else:
        cv_sub_slice = cv_slice.random_subslice

    if config.regression:
        if print_out:
            print('\n--- start Regularization-FS (regression)',config.reg_alpha_exp,config.reg_thresh_exp,' ---\n')
        if pre_filter is None:
            if (config.reg_alpha_exp,config.reg_thresh_exp) == cv_sub_slice.active_reg:
                if print_out:
                    print('ReguFS skipped, FS with that alpha already active')
                if return_score_list:
                    return cv_sub_slice.c_map[(config.reg_alpha_exp,config.reg_thresh_exp)]
                return False

        if pre_filter is None:
            cv_sub_slice.active_reg = (config.reg_alpha_exp,config.reg_thresh_exp)
            to_filter = reguFSreg.wrapper(cv_sub_slice, config, print_out = print_out, pre_filter = pre_filter, debug = debug, return_score_list = return_score_list)
        else:
            if print_out:
                print('regufs prefiltered:',len(pre_filter))
            to_filter = reguFSreg.wrapper(cv_sub_slice, config, print_out = print_out, pre_filter = pre_filter, debug = debug, return_score_list = return_score_list)
        if return_score_list:
            cv_sub_slice.c_map[(config.reg_alpha_exp,config.reg_thresh_exp)] = to_filter
    else:
        if (config.reg_c_exp,config.reg_thresh_exp) == cv_sub_slice.active_reg:
            if return_score_list:
                return cv_sub_slice.c_map[(config.reg_c_exp,config.reg_thresh_exp)]
            return False
        if print_out:
            print("\n--- start Regularization-FS (classification) ---\n")
        if (config.reg_c_exp,config.reg_thresh_exp) in cv_sub_slice.c_map:
            to_filter = cv_sub_slice.c_map[(config.reg_c_exp,config.reg_thresh_exp)]
        else:
            cv_sub_slice.filterFeatures([])
            to_filter = reguFSclf.wrapper(cv_sub_slice, config, print_out = print_out)
            cv_sub_slice.c_map[(config.reg_c_exp,config.reg_thresh_exp)] = to_filter
        cv_sub_slice.active_reg = (config.reg_c_exp,config.reg_thresh_exp)

    if print_out:
        print('regufs filtered:',len(to_filter))

    if return_list or return_score_list:
        return to_filter
    
    cv_slice.filterFeatures(to_filter, print_out = print_out)
    return True

def meanCorrelationWrapper(config, samples, cross_val_object, dummy_call = False, print_out = False, pre_filter = None, debug = False, return_list = False, return_score_list = False):
    cv_repeat = not cross_val_object.isSlice
    if config.tvmb_rank_threshold < 0:
        return None
    if not cv_repeat:
        return detectBiasedFeaturesByMeanCorrelation(config, samples, cross_val_object, dummy_call= dummy_call, print_out = print_out, pre_filter = pre_filter, debug = debug, return_list = return_list, return_score_list = return_score_list)

    else:
        return_lists = {}
        for cv_counter in cross_val_object.slices:
            cv_slice = cross_val_object.slices[cv_counter]
            return_lists[cv_counter] = detectBiasedFeaturesByMeanCorrelation(config, samples, cv_slice, dummy_call= dummy_call, print_out = print_out, pre_filter = pre_filter, debug = debug, return_list = return_list, return_score_list = return_score_list)

        if return_list or return_score_list:
            return return_lists

        return True

def positionalMeanCorrelationWrapper(config, cross_val_object, print_out = False, pre_filter = None, debug = False, return_list = False):
    cv_repeat = not cross_val_object.isSlice
    if config.tvpmb_rank_threshold < 0:
        return None
    if not cv_repeat:
        return detectBiasedFeaturesByPositionalMeanCorrelation(config, cross_val_object, print_out = print_out, pre_filter = pre_filter, debug = debug, return_list = return_list)

    else:
        return_lists = {}
        for cv_counter in cross_val_object.slices:
            cv_slice = cross_val_object.slices[cv_counter]
            return_lists[cv_counter] = detectBiasedFeaturesByPositionalMeanCorrelation(config, cv_slice, print_out = print_out, pre_filter = pre_filter, debug = debug, return_list = return_list)

        if return_list:
            return return_lists

        return True

def detectBiasedFeaturesByMeanCorrelation(config, samples, cv_slice, dummy_call = False, thresh=None, print_out = False, pre_filter = None, debug = False, return_list = False, return_score_list = False):
    if thresh is None:
        thresh = (config.tvmb_rank_threshold, config.p_val_thresh)
    if pre_filter is None:

        cv_slice.filterFeatures([])
        to_filter = []
    else:
        cv_slice.filterFeatures(pre_filter)
        to_filter = pre_filter
    if print_out:
        print('===========================================================================================')
        print(f'Biased feature detection (by mean correlation), TVMB thresh: {thresh}, return_score_list: {return_score_list}, pre_filter: {pre_filter}')
        print('tvmb prefiltered:',len(to_filter))

    if config.verbosity >= 5:
        cv_slice.featureSanityCheck(verbose = True)

    if not 'Protein bias' in cv_slice.slice_specific_feature_map:
        cv_slice.setProteinBias(config)

    if config.verbosity >= 5:
        cv_slice.featureSanityCheck(verbose = True)

    if cv_slice.tvmb_map is None:
        if config.verbosity >= 4:
            print(f'Init tvmb_map for slice {cv_slice.name}')
        feats_to_remove = []
        cv_slice.tvmb_map = {}
        for feat_name in cv_slice.feature_names:
            if not feat_name in cv_slice.tvmb_map:
                to_remove = cv_slice.addToTvmbMap(feat_name, samples, config, dummy_call = dummy_call)
                if to_remove:
                    feats_to_remove.append(feat_name)
            #else:
            #    tvmb_score,target_p_val = cv_slice.tvmb_map[feat_name]
        for feat_name in feats_to_remove:
            cv_slice.removeFeature(feat_name)

    if dummy_call:
        return
    
    cv_slice.rank_tvmb(samples, config)

    if return_score_list:
        return cv_slice.ranked_tvmb

    for feat_name,tvmb_score in cv_slice.ranked_tvmb[:config.tvmb_rank_threshold]:
        to_filter.append(feat_name)
        if debug or config.verbosity >= 5:
            print(cv_slice.name,'Filtered feature:',feat_name,'TVMB score:',tvmb_score)
    for feat_name in cv_slice.tvmb_map:
        tvmb_score,target_p_val = cv_slice.tvmb_map[feat_name]
        if abs(target_p_val) > config.p_val_thresh:
            to_filter.append(feat_name)
            if debug or config.verbosity >= 5:
                print(cv_slice.name,'Filtered feature:',feat_name,'p-value:',target_p_val)

    if print_out:
        print('tvmb filtered:',len(to_filter))
        print('===========================================================================================')

    if return_list:
        return to_filter

    cv_slice.filterFeatures(to_filter, print_out = print_out)

    return True

def detectBiasedFeaturesByPositionalMeanCorrelation(config, cv_slice, thresh=None, print_out = False, pre_filter = None, debug = False, return_list = False):
    if thresh is None:
        thresh = config.tvpmb_rank_threshold
    if pre_filter is None:

        cv_slice.filterFeatures([])
        to_filter = []
    else:
        cv_slice.filterFeatures(pre_filter)
        to_filter = pre_filter

    if print_out:
        print('===========================================================================================')
        print('Biased feature detection (by mean correlation), TVPMB thresh:',thresh)
        print('tvpmb prefiltered:',len(to_filter))

    if not 'Position means' in cv_slice.slice_specific_feature_map:
        cv_slice.setPositionMeans(config)

    if cv_slice.tvpmb_map is None:
        cv_slice.tvpmb_map = {}
        for feat_name in cv_slice.feature_names:
            if feat_name == 'Position means':
                continue
            if not feat_name in cv_slice.tvpmb_map:
                cv_slice.addToTvpmbMap(feat_name,config)
            else:
                tvpmb_score = cv_slice.tvpmb_map[feat_name]

    cv_slice.rank_tvpmb(config)
    for feat_name,tvpmb_score in cv_slice.ranked_tvpmb[:config.tvpmb_rank_threshold]:
        to_filter.append(feat_name)
        if debug or config.verbosity >= 5:
            print(cv_slice.name,'Filtered feature:',feat_name,'TVPMB score:',tvpmb_score)

    if print_out:
        print('tvpmb filtered:',len(to_filter))
        print('===========================================================================================')

    if return_list:
        return to_filter

    cv_slice.filterFeatures(to_filter, print_out = print_out)

    return True

#Needs to be updated
def detectBiasedFeaturesByBalancedImportances(config,samples):
    something_filtered = True
    n = 0
    while something_filtered and n < 5:
        something_filtered = False
        n += 1
        #step zero: save original configs
        ori_prot_based_separation = config.prot_based_separation
        ori_balanceSubsampling = config.balanceSubsampling
        ori_filter_single_variant_prots = config.filter_single_variant_prots
        #step one: train a forest with raw unbalanced samples
        config.prot_based_separation = True
        config.balanceSubsampling = None
        config.filter_single_variant_prots = False

        forest,scores = trainForest(config,samples,repeat = config.repeat_training)

        #step two: calculate feature importance values
        raw_feat_importance_map = calcFeatureImportances(forest,samples,config)
        #step three: train a forest with balanced samples
        config.prot_based_separation = True
        config.balanceSubsampling = 'balanced'
        config.filter_single_variant_prots = True

        samples.balanceSubSampleTrainSet(config)
        samples.calcVectors(config)

        forest,scores = trainForest(config,samples,repeat = config.repeat_training)

        #step four: calculate feature importance values
        balanced_feat_importance_map = calcFeatureImportances(forest,samples,config)
        #step five: compare the importance values
        diff_tuples = []
        for feat_name in raw_feat_importance_map:
            raw_score = raw_feat_importance_map[feat_name]
            bal_score = balanced_feat_importance_map[feat_name]
            diff_tuples.append([feat_name,raw_score-bal_score])

        diff_tuples.sort(key=lambda x:x[1],reverse=True)
        print('===========================================================================================')
        print('Biased feature detection')
        samples.printBalance(config)
        mean_importance = 1/len(diff_tuples)
        print('Total amount of features:',len(diff_tuples),'Mean feature importance:',mean_importance)
        print_once = False
        for feature_name,score in diff_tuples:
            if abs(score) > mean_importance:
                if score < 0. and not print_once:
                    print('...')
                    print_once = True
                print(feature_name,':',score)
            if score > config.bfd_factor*mean_importance:
                samples.removeFeature(feature_name)
                print('Removed feature:',feature_name)
                something_filtered = True
        #step six: reset original configs
        config.prot_based_separation = ori_prot_based_separation
        config.balanceSubsampling = ori_balanceSubsampling
        config.filter_single_variant_prots = ori_filter_single_variant_prots
        samples.undoBalancing(config)
        samples.printBalance(config)
        print('===========================================================================================')
    return

def select_by_sequential_confusion(config, cross_val_object, samples, pre_filter = None, print_out = False, debug = False, overwrite_proc_n = None, return_list = False, return_score_list = False, repetition = 2):
    if print_out:
        print('=== Feature selection by sequential feature confusion ===')
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
                to_filter = confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, return_list = True, overwrite_proc_n = overwrite_proc_n, rank_thresh = config.sequential_confusion_rank_threshold, return_score_list = return_score_list)
            else:
                to_filter = confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, return_list = True, overwrite_proc_n = overwrite_proc_n, sequence_number = i, return_score_list = return_score_list)
        return confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, overwrite_proc_n = overwrite_proc_n, return_list = return_list, sequence_number = i+1, return_score_list = return_score_list)
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
                    to_filter = confusionSelect(config, cv_slice, samples, print_out = print_out, debug = debug, pre_filter = to_filter, return_list = True, overwrite_proc_n = overwrite_proc_n, rank_thresh = config.sequential_confusion_rank_threshold, return_score_list = return_score_list)
                else:
                    to_filter = confusionSelect(config, cv_slice, samples, print_out = print_out, debug = debug, pre_filter = to_filter, return_list = True, overwrite_proc_n = overwrite_proc_n, sequence_number = i, return_score_list = return_score_list)
            return_list = confusionSelect(config, cv_slice, samples, print_out = print_out, debug = debug, pre_filter = to_filter, overwrite_proc_n = overwrite_proc_n, return_list = return_list, sequence_number = i+1, return_score_list = return_score_list)
            return_lists[cv_counter] = return_list
            if config.select_feature_for_first_slice_only:
                stored_return_list = return_list
        if return_list or return_score_list:
            return return_lists
        return True

def select_features(config, cross_val_object, samples, slice_slices, print_out = False, debug = False, overwrite_proc_n = None):

    cv_repeat = not cross_val_object.isSlice

    #if config.feature_selection == 'balancedImportances':
        #detectBiasedFeaturesByBalancedImportances(config,samples)
        #samples.calcVectors(config,print_out=print_out)
    if config.feature_selection == 'meanCorrelation':
        if print_out:
            print('=== Feature selection by mean correlation ===')
        return meanCorrelationWrapper(config, samples, cross_val_object, print_out = print_out, debug = debug)

    elif config.feature_selection == 'regularization':
        if print_out:
            print('=== Feature selection by regularization ===')
        return regu_fs_wrapper(config, cross_val_object, samples,  print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n)

    elif config.feature_selection == 'double':
        if print_out:
            print('=== Feature selection by meanCorrelation and regularization ===')

        to_filter = meanCorrelationWrapper(config, samples, cross_val_object, print_out = print_out, debug = debug, return_list = True)
        return regu_fs_wrapper(config, cross_val_object, samples,  print_out = print_out, debug = debug, pre_filter = to_filter, overwrite_proc_n = overwrite_proc_n)

    elif config.feature_selection == 'confusion':
        if print_out:
            print('=== Feature selection by feature confusion ===')
        meanCorrelationWrapper(config, samples, cross_val_object, dummy_call=True, print_out = print_out, debug = debug, return_score_list = True)
        filtered, times, slice_slices = crossFoldConfusionSelect(config, cross_val_object, samples, slice_slices, print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n)
        return filtered, times, slice_slices


    elif config.feature_selection == 'sequential_confusion':
        return select_by_sequential_confusion(config, cross_val_object, samples, print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n)

    elif config.feature_selection == 'sequential_confusion_and_regu':
        to_filter = select_by_sequential_confusion(config, cross_val_object, samples, print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n, return_list = True)
        regu_fs_wrapper(config, cross_val_object, samples, print_out = print_out, pre_filter = to_filter, debug = debug, overwrite_proc_n = overwrite_proc_n)
        return True


    elif config.feature_selection == 'confusion_and_regu':
        to_filter = select_by_sequential_confusion(config, cross_val_object, samples, print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n, return_list = True, repetition = 1)
        regu_fs_wrapper(config, cross_val_object, samples,  print_out = print_out, pre_filter = to_filter, debug = debug, overwrite_proc_n = overwrite_proc_n)
        return True

    elif config.feature_selection == 'threeStaged':
        if print_out:
            print('=== Feature selection by meanCorrelation, confusion, and regularization ===')

        to_filter = meanCorrelationWrapper(config, samples, cross_val_object, print_out = print_out, debug = debug, return_list = True)
        to_filter = select_by_sequential_confusion(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, overwrite_proc_n = overwrite_proc_n, return_list = True, repetition = 1)
        regu_fs_wrapper(config, cross_val_object, samples, print_out = print_out, pre_filter = to_filter, debug = debug, overwrite_proc_n = overwrite_proc_n)
        return True

    elif config.feature_selection == 'threeStaged_listranking':
        if print_out:
            print('=== Feature selection by list ranking of meanCorrelation, confusion, and regularization ===')

        mc_list = meanCorrelationWrapper(config, samples, cross_val_object, print_out = print_out, debug = debug, return_score_list = True)
        mc_rank_dict = calc_rank_dict(mc_list)
        if print_out:
            print(f'Length of mc_list: {len(mc_list)}, {len(mc_rank_dict)}')

        confusion_list = crossFoldConfusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n, return_score_list = True)
        confusion_rank_dict = calc_rank_dict(confusion_list)
        if print_out:
            print(f'Length of confusion_list: {len(confusion_list)}, {len(confusion_rank_dict)}')

        regu_list = regu_fs_wrapper(config, cross_val_object, samples, print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n, return_score_list = True) 
        regu_rank_dict = calc_rank_dict(regu_list)
        if print_out:
            print(f'Length of regu_list: {len(regu_list)}, {len(regu_rank_dict)}')              

        regu_conf_list = []
        feat_type_map = {}
        for feat_name in confusion_rank_dict:
            c_tied_rank = confusion_rank_dict[feat_name][2]
            r_tied_rank = regu_rank_dict[feat_name][2]
            regu_conf_list.append((feat_name, c_tied_rank + r_tied_rank))
            feat_type_map[feat_name] = samples.features[feat_name].f_type
        regu_conf_list.sort(key=lambda x:x[1],reverse=True)

        if print_out:
            regu_conf_rank_dict = calc_rank_dict(regu_conf_list)

            rank_dicts = {
                'mean correlation' : mc_rank_dict,
                'confusion' : confusion_rank_dict,
                'lasso regu' : regu_rank_dict,
                'confusion X regu' : regu_conf_rank_dict
            }
            write_feature_ranks(config, rank_dicts, cross_val_object.name, feat_type_map)

        if config.list_ranking_thresh is not None:
            to_filter = [x[0] for x in regu_conf_list[int(config.list_ranking_thresh):]]
        cross_val_object.filterFeatures(to_filter)

        return True

    else:
        print('=== ERROR: Feature selection with illegal key word:',config.feature_selection,'called ===')
        return None


def calc_rank_dict(score_list):
    rank_dict = {}
    min_score = None
    max_score = None
    for rank, (f_name, score) in enumerate(score_list):
        if rank > 0:
            prev_f_name, prev_score = score_list[rank-1]
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
        scaled_score = (raw_score - min_score)/(max_score - min_score)

        rank_dict[f_name].append(scaled_score)

    return rank_dict

def write_feature_ranks(config, rank_dicts, name, feat_type_map):
    headers = ['Feature Name', 'Feature Type']
    fs_types = list(rank_dicts.keys())
    for fs_type in fs_types:
        headers.append(f'{fs_type} Rank')
        headers.append(f'{fs_type} Tied Rank')
        headers.append(f'{fs_type} Score')
        headers.append(f'{fs_type} Scaled Score')


    header = "\t".join(headers) + '\n'

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
        lines.append('\t'.join(feat_lines[feat_name]) + '\n')

    page = ''.join(lines)

    outfile = f'{config.outfolder}/{config.dataset_name}_feature_ranks_{name}.tsv'

    f = open(outfile, 'w')
    f.write(page)
    f.close()
