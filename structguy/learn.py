import ray
import pickle
import sys

from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

from scipy import stats

from structguy import util, featureGenerator, sampleSpace, trainForest, featureAnalysis
from structguy import hyperParameterOptimization as hpo
from structguy.results_analysis import Results, write_protein_wise_performances
from structman.base_utils.base_utils import pack

def calcFeatureImportances(forest, samples, cv_slice, config, print_them=False):
    feature_scores = forest.feature_importances_

    feat_score_tuples = []
    feature_importance_map = {}
    #print feature_names

    feature_type_importance = {}

    for pos,feature_name in enumerate(cv_slice.feature_names):
        #print feature_name,': ',feature_scores[pos]
        try:
            feature_importance_map[feature_name] = feature_scores[pos]
        except:
            print('Some Error:',len(feature_scores),len(cv_slice.feature_names),len(cv_slice.features))
            return feature_importance_map
        if print_them:
            feat_score_tuples.append((feature_name,feature_scores[pos]))

        score = feature_importance_map[feature_name]
        feature_type = samples.features[feature_name].group

        if not feature_type in feature_type_importance:
            feature_type_importance[feature_type] = 0.
        feature_type_importance[feature_type] += score


    if print_them:
        feat_score_tuples.sort(key=lambda x:x[1],reverse=True)

        for feature_name,score in feat_score_tuples:
            if score == 0.0:
                continue
            if score < config.print_scores_greater_than:
                continue
            print(f'{feature_name}: {score} | Spearmans corr to true values: {cv_slice.featureTargetCorr(feature_name, samples, config)}')

        print('Total feature importance by feature type:', feature_type_importance)

    return feature_importance_map


def save_feature_importances(outfile, feature_importance_map):
    lines = ['Feature\tImportance\n']
    for feat_name in feature_importance_map:
        lines.append(f'{feat_name}\t{feature_importance_map[feat_name]}\n')

    f = open(outfile, 'w')
    f.write(''.join(lines))
    f.close()

def learn(config, effectRegressor=None, test_config = None):
    crossValidation = config.crossValidation

    if config.verbosity >= 1:
        print('================================ Start Learn ==============================================')
        print('Protein-based randomization: ',config.prot_based_separation)
        print('add Protein mean values: ',config.addBias)
        if isinstance(crossValidation,int):
            print('Cross Validation: %s-fold' % str(crossValidation))
        else:
            print('Cross Validation:',crossValidation)
        print('filter structural features: ',config.filterStructuralFeatures)
        print('filter samples with no mapped structures: ',config.structure_threshold)
        print('Clean type-2 circularity from training set: ',config.remove_t2)
        print('Trainset subsampling: ', config.balanceSubsampling,' (filter single variant proteins: ',config.filter_single_variant_prots,')')
        print('===========================================================================================')
        if config.path_structural_feature_table is not None:
            print(f'Using feature file: {config.path_structural_feature_table}')
            print(f'Writing processed features to processed feature file: {config.path_structural_feature_table.rsplit(".",1)[0]}_processed.tsv')
        elif config.path_to_processed_features_file is not None:
            print(f'Using feature file: {config.path_to_processed_features_file}')
        print(f'Writing output to: {config.outfolder}')

    samples: sampleSpace.SampleSpace = featureGenerator.createTrainingSet(config, stop_matrix_transformation = (config.path_to_support_features is not None))
    if config.path_to_support_features is not None:
        support_samples = featureGenerator.createTrainingSet(config, stop_matrix_transformation = True, other_features_path = config.path_to_support_features, filter_none_tv = True)
        samples.fuse_samples(support_samples)


    if config.weighting == 'geometric' and config.regression:
        samples.setGeometricDistanceMap(config)
        distance_map = ray.put(samples.geometric_distance_map)
    else:
        distance_map = None

    out_value = None

    if not config.skip_cv:

        if crossValidation == 'LOPO':
            cross_val_obj = sampleSpace.LOPO(samples, config)
        elif crossValidation == 'DataSAIL':
            cross_val_obj = sampleSpace.DataSAIL_cv(sampleSpace = samples, config = config)
        elif crossValidation == 'specific':
            cross_val_obj = sampleSpace.Given_split(samples, config)
        else:
            cross_val_obj = sampleSpace.X_fold_cv(samples, config, crossValidation)

        cv_slice = cross_val_obj.getCurrentSlice()

        if config.verbosity >= 1:
            print(f'{len(cv_slice.train_targets)=}')
            print('Testset length: ',len(cv_slice.test_targets))

        debug = config.debug_mode
        if config.feature_selection == 'confusion' or config.feature_selection == 'confusion_and_regu' or config.feature_selection == 'sequential_confusion' or config.feature_selection == 'threeStaged' or config.feature_selection == 'sequential_confusion_and_regu' or config.feature_selection == 'threeStaged_listranking':
            samples_store_id = ray.put(pack(samples))
        else:
            samples_store_id = None

        if config.cv_hpo:
            initial_training_input = cross_val_obj
        else:
            initial_training_input = cv_slice

        if config.verbosity >= 3:
            print(f'Before initial model training: {config.cv_hpo=} {initial_training_input.slice_slices=}')

        if config.hyperOptimization == 'bayesianComplete':
            forest,scores,initial_training_input, slice_slices = trainForest.trainForest(config, initial_training_input, samples_store_id = samples_store_id, slice_slices = initial_training_input.slice_slices, samples = samples, distance_map = distance_map, repeat = config.repeat_training, cv_repeat = config.cv_hpo, print_out = True, debug = debug)
            hpo.bayesianComplete(config, initial_training_input, scores, samples = samples_store_id, distance_map = distance_map)
        elif config.hyperOptimization == 'threeDim':
            forest, scores, initial_training_input, slice_slices = trainForest.trainForest(config, initial_training_input, samples_store_id = samples_store_id, slice_slices = initial_training_input.slice_slices, samples = samples, distance_map = distance_map, repeat = config.repeat_training, cv_repeat = config.cv_hpo, print_out = True, debug = debug)
            hpo.threeDimHyperOptimization(config, initial_training_input, scores, slice_slices, samples = samples, samples_store_id = samples_store_id, distance_map = distance_map, debug = debug)
        else:
            if debug:
                forest,scores,initial_training_input, slice_slices = trainForest.trainForest(config, initial_training_input, samples_store_id = samples_store_id, slice_slices = initial_training_input.slice_slices, samples = samples, distance_map = distance_map, repeat = config.repeat_training, cv_repeat = config.cv_hpo, print_out = True, debug = debug)
                scores.printOut()
            if config.verbosity >= 1:
                print('Hyperparamter optimization skipped')

        lopo_scores = {}

        accum_y_pred = []
        accum_true_vals = []


        if config.regression:
            mses = []
            pearsons = []
            spears = []
        else:
            rocs = []
            accs = []
            fs = []
        
        forests = {}

        append = False

        if samples_store_id is None:
            samples_store_id = ray.put(samples)

        for cv_counter in cross_val_obj.slices:
            cv_slice = cross_val_obj.slices[cv_counter]

            if config.verbosity >= 1:
                print(cv_counter, len(cv_slice.train_targets))
                print('Testset length: ',len(cv_slice.test_targets))

                cv_slice.printBalance(config)

            print_out = config.verbosity >= 2
            forest, scores, cv_slice, slice_slices = trainForest.trainForest(config, cv_slice, samples = samples, samples_store_id = samples_store_id, slice_slices = cv_slice.slice_slices, distance_map = distance_map, print_out = print_out, debug = debug)

            if forest is None:
                continue

            forests[cv_counter] = forest

            if config.crossValidation == 'LOPO':
                lopo_scores[tuple(cv_counter)] = scores

            test_feature_matrix = cv_slice.get_test_feature_matrix(samples)

            y_pred = forest.predict(test_feature_matrix)

            if config.regression:
                mses.append((scores.mse,len(cv_slice.test_targets)))
                pearsons.append((scores.pearson_r,1))
                spears.append(scores.corr)
                prot_wise_spearmans, mean_spearman, raw_corrs = util.calc_protein_wise_corr(cv_slice.test_targets, y_pred, cv_slice.test_sample_ids, stats.spearmanr)

                if config.verbosity >= 1:
                    print(f'Prot-wise RHO: {mean_spearman}')
                    print(prot_wise_spearmans)
                    for prot_id, spear_ in raw_corrs:
                        if spear_ < 0.3:
                            print(f'Low prot-wise rho: {prot_id} - {spear_}')


            else:
                rocs.append((scores.roc,len(cv_slice.test_targets)))
                accs.append((scores.acc,len(cv_slice.test_targets)))
                fs.append((scores.f1,len(cv_slice.test_targets)))

            for pred in y_pred:
                accum_y_pred.append(pred)
            for true_value in cv_slice.test_targets:
                accum_true_vals.append(true_value)

            #if config.regression:
            #    confusion_map, raw_conf_map = featureAnalysis.calcSliceConfusion(forest, cv_slice, remote = True, err_warping_exp = config.err_warping_exp, goodwill_interval = config.confusion_goodwill)
            #    print('Top 10 confusing features:')
            #    for i in range(10):
            #        print(confusion_map[i])

            if config.outfolder != None:

                if config.produce_scatterplot and config.regression and config.crossValidation == 'LOPO':
                    base_name = f'{config.outfolder}/{config.dataset_name}'
                    scatterfile = '%s_%s.png' % (base_name,str(cv_counter))
                    hexbinfile = '%s_%s_hexbin.png' % (base_name,str(cv_counter))

                    y_pred_median = util.median(y_pred)
                    tv_median = util.median(cv_slice.test_targets)
                    util.scatterplot(y_pred, test_feature_matrix, cv_slice.feature_names,
                                     cv_slice.test_targets, config.target_values, y_pred_median, tv_median, scatterfile)
                    util.hexbinplot(y_pred, cv_slice.test_targets, config.target_values, hexbinfile)

                writeOutput(config, y_pred, cv_slice, samples, append=append)
                if not append: #Append is only False in the first loop iteration
                    append = True

        if config.regression:
            if config.verbosity >= 1:
                util.printMean(mses,'MSE')
                util.printMean(pearsons,"Pearson's correlation")
            if len(accum_y_pred) > 0:
                cum_spear, _ = stats.spearmanr(accum_y_pred, accum_true_vals)
            else:
                cum_spear = None
            if len(spears) > 0:
                mean_spear = sum(spears)/len(spears)
            else:
                mean_spear = None

            out_value = (cum_spear, mean_spear)
        else:
            util.printMean(rocs,'auROC')
            util.printMean(accs,'ACC')
            util.printMean(fs,'F-Score')

        if config.outfolder != None and config.produce_scatterplot and config.regression:
            base_name = f'{config.outfolder}/{config.dataset_name}'
            scatterfile = '%s.png' % (base_name)
            hexbinfile = '%s_hexbin.png' % (base_name)

            #scatterplot(y_pred,test_feature_matrix,feature_names,test_targets,target_values,0.5,observed_value_threshold,scatterfile,feature_highlight='Class')
            #middle_value = (max(accum_y_pred) + min(accum_y_pred))/2.
            if len(accum_y_pred) > 0:
                y_pred_median = util.median(accum_y_pred)
                tv_median = util.median(accum_true_vals)
                util.scatterplot(accum_y_pred,cv_slice.get_test_feature_matrix(samples),cv_slice.feature_names,accum_true_vals,config.target_values,y_pred_median,tv_median,scatterfile)
                util.hexbinplot(accum_y_pred,accum_true_vals,config.target_values,hexbinfile)
                if config.crossValidation == 'LOPO':
                    labels = []
                    values = []
                    for lopo_id in lopo_scores:
                        pearson_r = lopo_scores[lopo_id].pearson_r
                        labels.append(str(lopo_id))
                        values.append(pearson_r)
                    title = 'Pearson\'s correlation'
                    radarfile = '%s_radar.png' % (base_name)
                    util.radar(labels,values,title,radarfile)
    elif config.feature_selection == 'confusion' or config.feature_selection == 'confusion_and_regu' or config.feature_selection == 'sequential_confusion' or config.feature_selection == 'threeStaged' or config.feature_selection == 'sequential_confusion_and_regu' or config.feature_selection == 'threeStaged_listranking':
        if crossValidation == 'LOPO':
            cross_val_obj = sampleSpace.LOPO(samples, config)
        elif crossValidation == 'DataSAIL':
            cross_val_obj = sampleSpace.DataSAIL_cv(sampleSpace = samples, config = config)
        elif crossValidation == 'specific':
            cross_val_obj = sampleSpace.Given_split(samples, config)
        else:
            cross_val_obj = sampleSpace.X_fold_cv(samples, config, crossValidation)
    else:
        cross_val_obj = None

    if config.outfolder != None and not config.skip_final_model:
        base_name = f'{config.outfolder}/{config.dataset_name}'
        modelfile = f'{config.outfolder}/StructGuy_trained_on_{config.dataset_name}.dump'

        filtered_features_file = '%s_filtered_features.tsv' % (base_name)

        forest = buildFinalModel(samples, config, internal_cv = cross_val_obj, outfile = modelfile, filtered_features_file = filtered_features_file)

        config.add_entry_to_project_file('path_trained_model', modelfile)

        if not config.skip_cv:
            cv_file = '%s_full_cv_forests.dump' % (base_name)
            storeCV(forests, config, cross_val_obj, cv_file)

    return out_value

def evaluate_dataset(config):
    forest, extern_feature_names_list, impute_map, model_config = loadModel(config.path_to_model)

    if config.verbosity >= 2:
        print(f'{extern_feature_names_list[:5]}\n...\n{extern_feature_names_list[-5:]}')

    samples = featureGenerator.createTrainingSet(config, external_impute = impute_map, for_prediction=True)

    for sample_id in samples.samples:
        samples.samples[sample_id].testtrain = 'test'

    test_feature_matrix, test_targets, sample_id_list = samples.get_test_data_for_feature_list(extern_feature_names_list)

    if len(test_feature_matrix) == 0:
        return None, None, None

    if config.verbosity >= 1:
        print('Shape of the feature matrix:',len(test_feature_matrix),len(test_feature_matrix[0]))
    y_pred = forest.predict(test_feature_matrix)

    if config.path_to_multi_savs_table is not None:
        multi_savs = util.parse_multi_savs_table(config)

        effect_dict = {}
        mm_y_pred = []
        combined_y_pred = []
        mm_test_targets = []
        combined_test_targets = []
        mm_sample_id_list = []
        combined_sample_id_list = []
        new_sample_id_list = []
        new_y_pred = []
        new_test_targets = []
        for pos, sample_id in enumerate(sample_id_list):
            pred_value = y_pred[pos]
            effect_dict[sample_id] = pred_value
            true_value = test_targets[pos]

            if true_value is not None:
                combined_test_targets.append(true_value)
                combined_sample_id_list.append(sample_id)
                combined_y_pred.append(pred_value)
                new_test_targets.append(true_value)
                new_sample_id_list.append(sample_id)
                new_y_pred.append(pred_value)
            elif config.target_values is None:
                new_sample_id_list.append(sample_id)
                new_y_pred.append(pred_value)
                combined_sample_id_list.append(sample_id)
                combined_y_pred.append(pred_value)

        for (prot_id, aacs, effect) in multi_savs:
            individual_effect_preds = []
            for aac in aacs:
                individual_effect_preds.append(effect_dict[(prot_id, aac)])
            combined_effect = util.combine_individual_effects(individual_effect_preds)

            combined_test_targets.append(effect)
            combined_sample_id_list.append((prot_id, ":".join(aacs)))
            combined_y_pred.append(combined_effect)

            mm_test_targets.append(effect)
            mm_sample_id_list.append((prot_id, ":".join(aacs)))
            mm_y_pred.append(combined_effect)

        test_targets = new_test_targets
        y_pred = new_y_pred
        sample_id_list = new_sample_id_list

    else:
        mm_y_pred = None

    protein_info = {}
    protein_wise_results = {}
    for pos, sample_id in enumerate(sample_id_list):
        if config.target_values is not None:
            true_value = test_targets[pos]

            if true_value is None:
                print(f'Ground truth is None for: {sample_id}')
                sys.exit()
        else:
            true_value = None

        pred_value = y_pred[pos]

        if pred_value is None:
            print(f'Predicted value is None for: {sample_id}')

        prot_id, aac = sample_id
        
        if true_value is not None:
            if prot_id not in protein_info:
                protein_size = samples.get_feature_value(sample_id, 'Protein Size')
                protein_info[prot_id] = [protein_size, 0]
            protein_info[prot_id][1] += 1

        if not prot_id in protein_wise_results:
            protein_wise_results[prot_id] = Results()

        protein_wise_results[prot_id].add_result(aac, true_value, pred_value)

    if config.regression:
        if config.target_values is not None:
            r2 = r2_score(test_targets,y_pred)
            mse = mean_squared_error(test_targets,y_pred)
            corr,p_value = stats.spearmanr(test_targets,y_pred)
        else:
            r2 = None
            mse = None
            corr = None
            p_value = None

        if config.verbosity >= 1:
            print('R2-Score: ',r2)
            print('MSE: ',mse)
            print('Spearman correlation and p-value: ',corr,p_value)

        if config.target_values is not None:
            prot_wise_spearmans, mean_spearman, _ = util.calc_protein_wise_corr(test_targets, y_pred, sample_id_list, stats.spearmanr)
            prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(test_targets, y_pred, sample_id_list, stats.pearsonr)
        else:
            prot_wise_spearmans = None
            mean_spearman = None
            prot_wise_pearsons = None
            mean_pearson = None

        if config.verbosity >= 1:
            print(f'Number of samples: {len(sample_id_list)} {len(test_targets)} {len(y_pred)}')

            print(f'Prot-wise mean pearson: {mean_pearson}')
            print(f'Prot-wise mean spearman: {mean_spearman}')

        if mm_y_pred is not None:
            prot_wise_spearmans, mean_mm_spearman, _ = util.calc_protein_wise_corr(mm_test_targets, mm_y_pred, mm_sample_id_list, stats.spearmanr)
            prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(mm_test_targets, mm_y_pred, mm_sample_id_list, stats.pearsonr)

            if config.verbosity >= 1:
                print(f'Number of samples: {len(mm_sample_id_list)} {len(mm_test_targets)} {len(mm_y_pred)}')

                print(f'Prot-wise mean pearson for multi savs: {mean_pearson}')
                print(f'Prot-wise mean spearman for multi savs: {mean_mm_spearman}')

            prot_wise_spearmans, mean_comb_spearman, _ = util.calc_protein_wise_corr(combined_test_targets, combined_y_pred, combined_sample_id_list, stats.spearmanr)
            prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(combined_test_targets, combined_y_pred, combined_sample_id_list, stats.pearsonr)

            if config.verbosity >= 1:
                print(f'Number of samples: {len(combined_sample_id_list)} {len(combined_test_targets)} {len(combined_y_pred)}')

                print(f'Prot-wise mean pearson for all variants: {mean_pearson}')
                print(f'Prot-wise mean spearman for all variants: {mean_comb_spearman}')

        write_protein_wise_performances(f'{config.outfolder}/protein_wise_results.tsv', prot_wise_spearmans, protein_info)

        if config.trace_decisions:
            decisions, pred_std_vector = featureAnalysis.explain_decisions(config, forest, y_pred, test_feature_matrix, extern_feature_names_list)

        header = 'Protein ID\tSAV\tPredicted effect value\tTree-wise standard deviation\tFeature 1\tFeature 2\t Feature 3\t Feature 4\t Feature 5\n'
        lines = [header]
        if mm_y_pred is None:
            combined_sample_id_list = sample_id_list
            combined_y_pred = y_pred
            combined_test_targets = test_targets

        for pos, sample_id in enumerate(combined_sample_id_list):
            pred_value = combined_y_pred[pos]
            prot_id, aac = sample_id
            if config.trace_decisions:
                pred_std = pred_std_vector[pos]
                feature_decisions = decisions[pos]
            else:
                pred_std = ''
                feature_decisions = ''
            words = [prot_id, aac, str(pred_value), str(pred_std)]
            for feat_name, weight, left_thresh, right_thresh in feature_decisions[:50]:
                if left_thresh is None and right_thresh is None:
                    decision_string = f'{feat_name} is None (Weight: {weight})'
                elif left_thresh is None:
                    decision_string = f'{feat_name} < {right_thresh} (Weight: {weight})'
                elif right_thresh is None:
                    decision_string = f'{feat_name} >= {left_thresh} (Weight: {weight})'
                else:
                    decision_string = f'{feat_name} in [{left_thresh}, {right_thresh}] (Weight: {weight})'
                words.append(decision_string)
            line = '\t'.join(words) + '\n'
            lines.append(line)

        predictions_file = f'{config.outfolder}/predictions.tsv'
        f = open(predictions_file, 'w')
        f.write(''.join(lines))
        f.close()

        if config.produce_scatterplot and config.target_values is not None:
            scatterfile = f'{config.outfolder}/predicted_value_scatterplot.png'
            hexbinfile = f'{config.outfolder}/predicted_value_hexbinplot.png'

            y_pred_median = util.median(y_pred)
            tv_median = util.median(test_targets)
            util.scatterplot(y_pred, test_feature_matrix, extern_feature_names_list, test_targets, config.target_values, y_pred_median, tv_median, scatterfile)
            util.hexbinplot(y_pred, test_targets, config.target_values, hexbinfile)

        return mean_spearman, combined_test_targets, combined_y_pred
    elif model_config.regression:
        int_targets = []
        for x in test_targets:
            if x == 'Benign':
                int_targets.append(1)
            else: 
                int_targets.append(0)
        total_roc_auc = roc_auc_score(int_targets, y_pred)
        prot_wise_roc_aucs, mean_roc_auc, _ = util.calc_protein_wise_corr(int_targets, y_pred, sample_id_list, roc_auc_score, mono_return_score_function = True)
        if config.verbosity >= 1:
            print(f'Total roc_auc: {total_roc_auc}')
            print(f'Prot-wise mean roc_auc: {mean_roc_auc}')

        header = 'Protein ID\tSAV\tPredicted effect value\n'
        lines = [header]
        if mm_y_pred is None:
            combined_sample_id_list = sample_id_list
            combined_y_pred = y_pred
            combined_test_targets = test_targets

        for pos, sample_id in enumerate(combined_sample_id_list):
            pred_value = combined_y_pred[pos]
            prot_id, aac = sample_id
            words = [prot_id, aac, str(pred_value)]
            
            line = '\t'.join(words) + '\n'
            lines.append(line)

        predictions_file = f'{config.outfolder}/predictions.tsv'
        f = open(predictions_file, 'w')
        f.write(''.join(lines))
        f.close()

        #write_protein_wise_pearsons(f'{config.outfolder}/protein_wise_results.tsv', prot_wise_roc_aucs, protein_info)
        return mean_roc_auc, int_targets, y_pred

    else:
        acc = accuracy_score(test_targets, y_pred)
        int_targets = classToInt(test_targets, samples)
        int_preds = classToInt(y_pred,samples)
        roc = roc_auc_score(int_targets,int_preds)

        f1 = f1_score(int_targets,int_preds)

        precision = precision_score(int_targets,int_preds)
        recall = recall_score(int_targets,int_preds)

        mcc = matthews_corrcoef(int_targets,int_preds)

        if config.verbosity >= 1:
            print('F-Score: ',f1)
            print('Accuracy: ',acc)
            print('Precision:',precision)
            print('Recall:',recall)
            print('MCC:',mcc)
    return


def writeOutput(config, y_pred, cv_slice, sampleSpace, append=False):

    feature_names = cv_slice.feature_names

    long_feat_name_string = "\t".join(feature_names)

    header = f'Protein Identifier\tAmino Acid Change\tTarget value\tPredicted value\tError\t{long_feat_name_string}\n'
    if not append:
        outlines = [header]
    else:
        outlines = []

    color_map = {}
    for pos,sample_id in enumerate(cv_slice.test_sample_ids):
        u_ac,aac = sample_id
        target_value = cv_slice.test_targets[pos]
        aac_base = aac[:-1]
        if not u_ac in color_map:
            color_map[u_ac] = {}
        if not aac_base in color_map[u_ac]:
            color_map[u_ac][aac_base] = []
        
        predicted_value = y_pred[pos]
        if config.regression:
            error = abs(target_value-predicted_value)
        else:
            error = target_value == predicted_value

        color_map[u_ac][aac_base].append(error)

        feature_vector = []
        for feature_name in feature_names:
            feat = sampleSpace.features[feature_name]
            feature_value = sampleSpace.get_feature_value(sample_id, feature_name)
            feature_vector.append(feat.string_convert(feature_value))

        outlines.append('%s\t%s\t%s\t%s\t%s\t%s\n' % (u_ac,aac,str(target_value),str(y_pred[pos]),str(error),'\t'.join(feature_vector)))
    outfile = f'{config.outfolder}/{config.dataset_name}_stratified_predictions.tsv'
    if not append :
        f = open(outfile,'w')
    else:
        f = open(outfile,'a')
    f.write(''.join(outlines))
    f.close()


def storeCV(forests, config, cross_val_obj, cv_file):
    with open(cv_file, 'wb') as output:
        pickle.dump((forests, cross_val_obj, config), output, pickle.HIGHEST_PROTOCOL)
    if config.verbosity >= 1:
        print('\n============\nStored full CV results in %s\n============\n' % cv_file)

def loadCV(fn):
    with open(fn, 'rb') as inp:
        forests, cross_val_object, config = pickle.load(inp)
    if config.verbosity >= 1:    
        print('\n============\nLoaded full CV from %s\n============\n' % fn)
    return forests, cross_val_object, config


def storeModel(model, feature_names, config, fn):
    with open(fn, 'wb') as output:
        pickle.dump((model, feature_names, config), output, pickle.HIGHEST_PROTOCOL)
    if config.verbosity >= 1:
        print('\n============\nStored model in %s\n============\n' % fn)

def loadModel(fn):
    with open(fn, 'rb') as inp:
        model, feature_names, config = pickle.load(inp)

    try:
        path_to_impute_map = config.path_to_impute_map
        with open(path_to_impute_map, 'rb') as inp:
            impute_map = pickle.load(inp)
    except:
        impute_map = None

    if config.verbosity >= 1:
        print('\n============\nLoaded model from %s\n============\n' % fn)
    return model, feature_names, impute_map, config

def buildFinalModel(samples, config, internal_cv = None, outfile = None, filtered_features_file = None):
    #Some feature selection strategies require an internal cross validation-like slicing
    #An example is the confusion-based feature section

    cross_val_obj = sampleSpace.FullSlice(samples, config, internal_cv = internal_cv)

    full_slice = cross_val_obj.slices[0]

    if config.verbosity >= 1:
        full_slice.printBalance(config)

    if config.feature_selection == 'confusion':
        full_slice.subslices = []
        for slice_slice in full_slice.slice_slices:
            full_slice.subslices.append(slice_slice.test_prots)

    print_out = config.verbosity >= 1
    forest, scores, full_slice, slice_slices = trainForest.trainForest(config, full_slice, samples = samples, distance_map = samples.geometric_distance_map, print_out = print_out ,skip_scoring = True)

    if config.verbosity >= 1:
        print('Full Slice Info after training:')
        full_slice.printBalance(config)

    feat_importance_map = calcFeatureImportances(forest, samples, full_slice, config, print_them = print_out)

    if outfile != None:
        storeModel(forest, full_slice.feature_names, config, outfile)


    if filtered_features_file != None:
        save_feature_importances(filtered_features_file, feat_importance_map)

    return forest
