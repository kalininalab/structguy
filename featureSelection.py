import reguFeatureSelectionRegressor as reguFSreg
import reguFeatureSelectionClassificator as reguFSclf
import sampleSpace
import random
import featureAnalysis
import ray
import trainForest

def confusionSelect(config, cv_slice, samples, distance_map = None, print_out = False, pre_filter = None, debug = False, return_list = False, overwrite_proc_n = False):
    #make sliceslice
    if cv_slice.slice_slice is None:
        random_protein = random.choice(list(cv_slice.train_prots))

        remaining_prots = cv_slice.train_prots.copy().remove(random_protein)

        try:
            samples = ray.get(samples)
        except:
            pass

        test_ids, train_ids = samples.splitDataSet(config, specific_id = random_protein, protein_wise = True, skip_protein = cv_slice.name)

        cv_slice_slice = sampleSpace.CrossValidationSlice(test_ids, train_ids, samples, config, name = random_protein, train_prots = remaining_prots)
        cv_slice.slice_slice = cv_slice_slice
    else:
        cv_slice_slice = cv_slice.slice_slice

    #train forest on sliceslice
    if pre_filter is None:
        cv_slice_slice.filterFeatures([])
        to_filter = []
    else:
        cv_slice_slice.filterFeatures(pre_filter)
        to_filter = pre_filter

    if print_out:
        print('confusion prefiltered:',len(to_filter))

    forest, scores = trainForest.trainForest(config, cv_slice_slice, distance_map, skip_feature_selection = True, skip_scoring = True, para_number = overwrite_proc_n)

    #calculate feature confusions

    conf_map = featureAnalysis.calcSliceConfusion(forest, cv_slice_slice, para_number = overwrite_proc_n, remote = False)

    #select by rank

    for feat_name, feature_confusion in conf_map[:config.confusion_rank_threshold]:
        to_filter.append(feat_name)
        if debug:
            print('Confusion filter:',feat_name, feature_confusion)

    if print_out:
        print('confusion filtered:',len(to_filter))

    if return_list:
        return to_filter

    cv_slice.filterFeatures(to_filter, print_out = print_out)

    return True

def regu_fs(config, cv_slice, print_out = False, pre_filter = None, debug = False, return_list = False):
    if config.regression:
        if print_out:
            print('\n--- start Regularization-FS (regression)',config.reg_alpha_exp,config.reg_thresh_exp,' ---\n')
        if pre_filter is None:
            if (config.reg_alpha_exp,config.reg_thresh_exp) == cv_slice.active_reg:
                if print_out:
                    print('ReguFS skipped, FS with that alpha already active')
                return False

        if pre_filter is None:
            cv_slice.active_reg = (config.reg_alpha_exp,config.reg_thresh_exp)
            to_filter = reguFSreg.wrapper(cv_slice, config, print_out = print_out, pre_filter = pre_filter, debug = debug)
        else:
            if print_out:
                print('regufs prefiltered:',len(pre_filter))
            to_filter = reguFSreg.wrapper(cv_slice, config, print_out = print_out, pre_filter = pre_filter, debug = debug)
    else:
        if (config.reg_c_exp,config.reg_thresh_exp) == cv_slice.active_reg:
            return False
        if print_out:
            print("\n--- start Regularization-FS (classification) ---\n")
        if (config.reg_c_exp,config.reg_thresh_exp) in cv_slice.c_map:
            to_filter = cv_slice.c_map[(config.reg_c_exp,config.reg_thresh_exp)]
        else:
            cv_slice.filterFeatures([])
            to_filter = reguFSclf.wrapper(cv_slice, config, print_out = print_out)
            cv_slice.c_map[(config.reg_c_exp,config.reg_thresh_exp)] = to_filter
        cv_slice.active_reg = (config.reg_c_exp,config.reg_thresh_exp)
    if print_out:
        print('regufs filtered:',len(to_filter))
    if return_list:
        return to_filter
    cv_slice.filterFeatures(to_filter, print_out = print_out)
    return True

def detectBiasedFeaturesByMeanCorrelation(config, cv_slice, thresh=None, print_out = False, pre_filter = None, debug = False):
    if thresh is None:
        thresh = (config.tvmb_rank_threshold, config.p_val_thresh)
    if pre_filter is None:
        if cv_slice.active_tvmb_threshold == thresh:
            return
        cv_slice.filterFeatures([])
        to_filter = []
    else:
        cv_slice.filterFeatures(pre_filter)
        to_filter = pre_filter
    if print_out:
        print('===========================================================================================')
        print('Biased feature detection (by mean correlation), TVMB thresh:',thresh)
        print('tvmb prefiltered:',len(to_filter))

    if not 'Protein bias' in cv_slice.slice_specific_feature_map:
        cv_slice.setProteinBias(config)

    if cv_slice.tvmb_map is None:
        cv_slice.tvmb_map = {}
        for feat_name in cv_slice.feature_names:
            if not feat_name in cv_slice.tvmb_map:
                cv_slice.addToTvmbMap(feat_name,config)
            else:
                tvmb_score,target_p_val = cv_slice.tvmb_map[feat_name]

    cv_slice.rank_tvmb(config)
    for feat_name,tvmb_score in cv_slice.ranked_tvmb[:config.tvmb_rank_threshold]:
        to_filter.append(feat_name)
        if debug:
            print(cv_slice.name,'Removed feature:',feat_name,'TVMB score:',tvmb_score)
    for feat_name in cv_slice.tvmb_map:
        tvmb_score,target_p_val = cv_slice.tvmb_map[feat_name]
        if abs(target_p_val) > config.p_val_thresh:
            to_filter.append(feat_name)
            if debug:
                print(cv_slice.name,'Removed feature:',feat_name,'p-value:',target_p_val)

    if print_out:
        print('tvmb filtered:',len(to_filter))
        print('===========================================================================================')
    if pre_filter is None:
        cv_slice.active_tvmb_threshold = thresh
        cv_slice.filterFeatures(to_filter, print_out = print_out)
    return to_filter

def detectBiasedFeaturesByPositionalMeanCorrelation(config, cv_slice, thresh=None, print_out = False, pre_filter = None, debug = False):
    if thresh is None:
        thresh = config.tvpmb_rank_threshold
    if pre_filter is None:
        if cv_slice.active_tvpmb_threshold == thresh:
            return
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
        if debug:
            print(cv_slice.name,'Removed feature:',feat_name,'TVPMB score:',tvpmb_score)

    if print_out:
        print('tvpmb filtered:',len(to_filter))
        print('===========================================================================================')
    if pre_filter is None:
        cv_slice.active_tvpmb_threshold = thresh
        cv_slice.filterFeatures(to_filter, print_out = print_out)
    return to_filter

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

def select_features(config, cross_val_object, samples = None, print_out = False, debug = False, overwrite_proc_n = None):

    cv_repeat = not cross_val_object.isSlice

    #if config.feature_selection == 'balancedImportances':
        #detectBiasedFeaturesByBalancedImportances(config,samples)
        #samples.calcVectors(config,print_out=print_out)
    if config.feature_selection == 'meanCorrelation':

        if config.tvmb_rank_threshold < 0 or config.tvpmb_rank_threshold < 0:
                return None
        if not cv_repeat:
            detectBiasedFeaturesByMeanCorrelation(config,cross_val_object,print_out = print_out)
            detectBiasedFeaturesByPositionalMeanCorrelation(config,cross_val_object,print_out = print_out)
        else:
            for cv_counter in cross_val_object.slices:
                cv_slice = cross_val_object.slices[cv_counter]
                detectBiasedFeaturesByMeanCorrelation(config,cv_slice,print_out = print_out)
                detectBiasedFeaturesByPositionalMeanCorrelation(config,cv_slice,print_out = print_out)
        return True

    elif config.feature_selection == 'regularization':
        if print_out:
            print('=== Feature selection by regularization ===')
        if not cv_repeat:
            return regu_fs(config,cross_val_object,print_out = print_out)
        else:
            for cv_counter in cross_val_object.slices:
                cv_slice = cross_val_object.slices[cv_counter]
                regu_fs(config,cv_slice,print_out = print_out)
            return True
    elif config.feature_selection == 'double':
        if print_out:
            print('=== Feature selection by meanCorrelation and regularization ===')

        if config.tvmb_rank_threshold < 0 or config.tvpmb_rank_threshold < 0:
                return None
        if not cv_repeat:
            to_filter = detectBiasedFeaturesByMeanCorrelation(config,cross_val_object,print_out = print_out, pre_filter = [], debug = debug)
            to_filter = detectBiasedFeaturesByPositionalMeanCorrelation(config,cross_val_object,print_out = print_out, pre_filter = to_filter, debug = debug)
            return regu_fs(config,cross_val_object,print_out = print_out, pre_filter = to_filter, debug = debug)
        else:
            for cv_counter in cross_val_object.slices:
                cv_slice = cross_val_object.slices[cv_counter]
                to_filter = detectBiasedFeaturesByMeanCorrelation(config,cv_slice,print_out = print_out, pre_filter = [], debug = debug)
                to_filter = detectBiasedFeaturesByPositionalMeanCorrelation(config,cv_slice,print_out = print_out, pre_filter = to_filter, debug = debug)
                regu_fs(config,cv_slice,print_out = print_out, pre_filter = to_filter, debug = debug)
            return True

    elif config.feature_selection == 'confusion':
        if print_out:
            print('=== Feature selection by feature confusion ===')
        if config.confusion_rank_threshold < 0:
            return None
        if not cv_repeat:
            return confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n)
        else:
            for cv_counter in cross_val_object.slices:
                cv_slice = cross_val_object.slices[cv_counter]
                confusionSelect(config, cv_slice, samples, print_out = print_out, debug = debug, overwrite_proc_n = overwrite_proc_n)
            return True

    elif config.feature_selection == 'sequential_confusion':
        if print_out:
            print('=== Feature selection by sequential feature confusion ===')
        if config.confusion_rank_threshold < 0:
            return None
        if not cv_repeat:
            to_filter = []
            for i in range(1):
                to_filter = confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, return_list = True)
            return confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, overwrite_proc_n = overwrite_proc_n)
        else:
            for cv_counter in cross_val_object.slices:
                cv_slice = cross_val_object.slices[cv_counter]
                to_filter = []
                for i in range(1):
                    to_filter = confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, return_list = True)
                confusionSelect(config, cv_slice, samples, print_out = print_out, debug = debug, pre_filter = to_filter, overwrite_proc_n = overwrite_proc_n)
            return True

    elif config.feature_selection == 'threeStaged':
        if print_out:
            print('=== Feature selection by meanCorrelation and regularization ===')

        if config.tvmb_rank_threshold < 0 or config.tvpmb_rank_threshold < 0 or config.confusion_rank_threshold < 0:
            return None
        if not cv_repeat:
            to_filter = detectBiasedFeaturesByMeanCorrelation(config,cross_val_object,print_out = print_out, pre_filter = [], debug = debug)
            to_filter = detectBiasedFeaturesByPositionalMeanCorrelation(config,cross_val_object,print_out = print_out, pre_filter = to_filter, debug = debug)
            #to_filter = confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, return_list = True, overwrite_proc_n = overwrite_proc_n)
            #return regu_fs(config,cross_val_object,print_out = print_out, pre_filter = to_filter, debug = debug)
            to_filter = regu_fs(config,cross_val_object,print_out = print_out, pre_filter = to_filter, debug = debug, return_list = True)
            return confusionSelect(config, cross_val_object, samples, print_out = print_out, debug = debug, pre_filter = to_filter, overwrite_proc_n = overwrite_proc_n)
        else:
            for cv_counter in cross_val_object.slices:
                cv_slice = cross_val_object.slices[cv_counter]
                to_filter = detectBiasedFeaturesByMeanCorrelation(config,cv_slice,print_out = print_out, pre_filter = [], debug = debug)
                to_filter = detectBiasedFeaturesByPositionalMeanCorrelation(config,cv_slice,print_out = print_out, pre_filter = to_filter, debug = debug)
                #to_filter = confusionSelect(config, cv_slice, samples, print_out = print_out, debug = debug, pre_filter = to_filter, return_list = True, overwrite_proc_n = overwrite_proc_n)
                #regu_fs(config,cv_slice,print_out = print_out, pre_filter = to_filter, debug = debug)
                to_filter = regu_fs(config,cv_slice,print_out = print_out, pre_filter = to_filter, debug = debug, return_list = True)
                confusionSelect(config, cv_slice, samples, print_out = print_out, debug = debug, pre_filter = to_filter, overwrite_proc_n = overwrite_proc_n)
            return True

    else:
        print('=== ERROR: Feature selection with illegal key word:',config.feature_selection,'called ===')
        return None




