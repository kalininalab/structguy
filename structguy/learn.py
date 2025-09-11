import pickle
import sys
import os
import time
import ray
import random
import shap
import numpy
from scipy import stats
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.ensemble import RandomForestRegressor
from supertree import SuperTree
from structman.base_utils.base_utils import pack

from structguy import featureAnalysis, featureGenerator, sampleSpace, trainForest, util
from structguy import hyperParameterOptimization as hpo
from structguy.results_analysis import Results, write_protein_wise_performances
from structguy.support_classes import CrossValidationSlice
from structguy.util import Config, loadModel, storeModel
from structguy.consts import feature_categories

from xgboost import plot_tree, DMatrix
import matplotlib.pyplot as plt

def calcFeatureImportances(forest, samples, cv_slice, config, print_them=False):
    feature_scores = forest.feature_importances_

    feat_score_tuples = []
    feature_importance_map = {}
    # print feature_names

    feature_type_importance = {}

    for pos, feature_name in enumerate(cv_slice.feature_names):
        # print feature_name,': ',feature_scores[pos]
        try:
            feature_importance_map[feature_name] = feature_scores[pos]
        except KeyError:
            print(
                "Some Error:",
                len(feature_scores),
                len(cv_slice.feature_names),
                len(cv_slice.features),
            )
            return feature_importance_map
        except IndexError:
            feature_importance_map[feature_name] = 0.

        if print_them:
            feat_score_tuples.append((feature_name, feature_scores[pos]))

        score = feature_importance_map[feature_name]
        feature_type = samples.features[feature_name].group

        if feature_type not in feature_type_importance:
            feature_type_importance[feature_type] = 0.0
        feature_type_importance[feature_type] += score

    if print_them:
        feat_score_tuples.sort(key=lambda x: x[1], reverse=True)

        for feature_name, score in feat_score_tuples:
            if score == 0.0:
                continue
            if score < config.print_scores_greater_than:
                continue
            print(f"{feature_name}: {score}")

        print("Total feature importance by feature type:", feature_type_importance)

    return feature_importance_map


def save_feature_importances(outfile, feature_importance_map):
    lines = ["Feature\tImportance\n"]
    for feat_name in feature_importance_map:
        lines.append(f"{feat_name}\t{feature_importance_map[feat_name]}\n")

    f = open(outfile, "w")
    f.write("".join(lines))
    f.close()


def learn(config, effectRegressor=None, test_config=None):
    crossValidation = config.crossValidation

    if config.verbosity >= 1:
        print("================================ Start Learn ==============================================")
        print("Protein-based randomization: ", config.prot_based_separation)
        print("add Protein mean values: ", config.addBias)
        if isinstance(crossValidation, int):
            print("Cross Validation: %s-fold" % str(crossValidation))
        else:
            print("Cross Validation:", crossValidation)
        print("filter structural features: ", config.filterStructuralFeatures)
        print("filter samples with no mapped structures: ", config.structure_threshold)
        print("Clean type-2 circularity from training set: ", config.remove_t2)
        print(
            "Trainset subsampling: ",
            config.balanceSubsampling,
            " (filter single variant proteins: ",
            config.filter_single_variant_prots,
            ")",
        )
        print(f'{config.random_split=}')
        print("===========================================================================================")
        if config.path_structural_feature_table is not None:
            print(f"Using feature file: {config.path_structural_feature_table}")
            print(f"Writing processed features to processed feature file: {config.path_structural_feature_table.rsplit('.', 1)[0]}_processed.tsv")
        elif config.path_to_processed_features_file is not None:
            print(f"Using feature file: {config.path_to_processed_features_file}")
        print(f"Writing output to: {config.outfolder}")

    t0 = time.time()
    samples: sampleSpace.SampleSpace = featureGenerator.createTrainingSet(config, stop_matrix_transformation=(config.path_to_support_features is not None))
    if config.path_to_support_features is not None:
        support_samples = featureGenerator.createTrainingSet(
            config,
            stop_matrix_transformation=True,
            other_features_path=config.path_to_support_features,
            filter_none_tv=True,
        )
        samples.fuse_samples(support_samples)

    t_0 = time.time()
    print(f'Time for loading dataset: {t_0 - t0} {config.n_of_features=}')
    
    if config.weighting == "geometric" and config.regression:
        samples.setGeometricDistanceMap(config)
        distance_map = ray.put(samples.geometric_distance_map)
    else:
        distance_map = None

    out_value = None
    samples_store_id = None

    if not config.skip_cv:
        if crossValidation == "LOPO":
            cross_val_obj = sampleSpace.LOPO(samples, config)
        elif crossValidation == "DataSAIL":
            cross_val_obj = sampleSpace.DataSAIL_cv(sampleSpace=samples, config=config)
        elif crossValidation == "specific":
            cross_val_obj = sampleSpace.Given_split(samples, config)
        else:
            cross_val_obj = sampleSpace.X_fold_cv(samples, config, crossValidation)

        cv_slice = cross_val_obj.getCurrentSlice()

        if config.verbosity >= 1:
            print(f"{len(cv_slice.train_targets)=}")
            print("Testset length: ", len(cv_slice.test_targets))

        debug = config.debug_mode
        if (
            config.feature_selection == "confusion"
            or config.feature_selection == "confusion_and_regu"
            or config.feature_selection == "sequential_confusion"
            or config.feature_selection == "threeStaged"
            or config.feature_selection == "sequential_confusion_and_regu"
            or config.feature_selection == "threeStaged_listranking"
        ):
            samples_store_id = ray.put(pack(samples))
        else:
            samples_store_id = None

        if config.cv_hpo:
            initial_training_input = cross_val_obj
        else:
            initial_training_input = cv_slice

        if config.verbosity >= 3:
            print(f"Before initial model training: {config.cv_hpo=} {initial_training_input.slice_slices=}")

        if config.hyperOptimization == "bayesianComplete":
            forest, scores, initial_training_input, slice_slices = trainForest.trainForest(
                config,
                initial_training_input,
                samples_store_id=samples_store_id,
                slice_slices=initial_training_input.slice_slices,
                samples=samples,
                distance_map=distance_map,
                repeat=config.repeat_training,
                cv_repeat=config.cv_hpo,
                print_out=True,
                debug=debug,
                remote=False,
            )
            hpo.bayesianComplete(
                config,
                initial_training_input,
                scores,
                slice_slices,
                samples=samples,
                samples_store_id=samples_store_id,
                distance_map=distance_map,
                debug=debug,
            )
        elif config.hyperOptimization == "threeDim":
            forest, (first_scores, scores), initial_training_input, slice_slices = trainForest.trainForest(
                config,
                initial_training_input,
                samples_store_id=samples_store_id,
                slice_slices=initial_training_input.slice_slices,
                samples=samples,
                distance_map=distance_map,
                repeat=config.repeat_training,
                cv_repeat=config.cv_hpo,
                print_out=True,
                debug=debug,
                remote=False,
                get_first_scores=True
            )
            hpo.threeDimHyperOptimization(
                config,
                initial_training_input,
                scores,
                first_scores,
                slice_slices,
                samples=samples,
                samples_store_id=samples_store_id,
                distance_map=distance_map,
                debug=debug,
            )
        else:
            if debug:
                forest, scores, initial_training_input, slice_slices = trainForest.trainForest(
                    config,
                    initial_training_input,
                    samples_store_id=samples_store_id,
                    slice_slices=initial_training_input.slice_slices,
                    samples=samples,
                    distance_map=distance_map,
                    repeat=config.repeat_training,
                    cv_repeat=config.cv_hpo,
                    print_out=True,
                    debug=debug,
                )
                scores.printOut()
            if config.verbosity >= 1:
                print("Hyperparamter optimization skipped")

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
            cv_slice: CrossValidationSlice = cross_val_obj.slices[cv_counter]

            if config.verbosity >= 1:
                print(cv_counter, len(cv_slice.train_targets))
                print("Testset length: ", len(cv_slice.test_targets))

                cv_slice.printBalance(config)

            print_out = config.verbosity >= 2
            forest, scores, cv_slice, slice_slices = trainForest.trainForest(
                config,
                cv_slice,
                samples=samples,
                samples_store_id=samples_store_id,
                slice_slices=cv_slice.slice_slices,
                distance_map=distance_map,
                print_out=print_out,
                debug=debug,
                remote=False
            )

            if forest is None:
                continue

            forests[cv_counter] = forest

            if config.crossValidation == "LOPO":
                lopo_scores[tuple(cv_counter)] = scores

            test_feature_matrix = cv_slice.get_test_feature_matrix(samples)

            y_pred = forest.predict(test_feature_matrix)

            if config.regression:
                mses.append((scores.mse, len(cv_slice.test_targets)))
                pearsons.append((scores.pearson_r, 1))
                spears.append(scores.corr)
                prot_wise_spearmans, mean_spearman, raw_corrs = util.calc_protein_wise_corr(
                    cv_slice.test_targets,
                    y_pred,
                    cv_slice.test_sample_ids,
                    stats.spearmanr,
                )

                if config.verbosity >= 1:
                    print(f"Prot-wise RHO: {mean_spearman}")
                    print(prot_wise_spearmans)
                    for prot_id, spear_ in raw_corrs:
                        if spear_ < 0.3:
                            print(f"Low prot-wise rho: {prot_id} - {spear_}")

            else:
                rocs.append((scores.roc, len(cv_slice.test_targets)))
                accs.append((scores.acc, len(cv_slice.test_targets)))
                fs.append((scores.f1, len(cv_slice.test_targets)))

            for pred in y_pred:
                accum_y_pred.append(pred)
            for true_value in cv_slice.test_targets:
                accum_true_vals.append(true_value)

            # if config.regression:
            #    confusion_map, raw_conf_map = featureAnalysis.calcSliceConfusion(forest, cv_slice, remote = True, err_warping_exp = config.err_warping_exp, goodwill_interval = config.confusion_goodwill)
            #    print('Top 10 confusing features:')
            #    for i in range(10):
            #        print(confusion_map[i])

            if config.outfolder is not None:
                if config.produce_scatterplot and config.regression and config.crossValidation == "LOPO":
                    base_name = f"{config.outfolder}/{config.dataset_name}"
                    scatterfile = "%s_%s.png" % (base_name, str(cv_counter))
                    hexbinfile = "%s_%s_hexbin.png" % (base_name, str(cv_counter))

                    y_pred_median = util.median(y_pred)
                    tv_median = util.median(cv_slice.test_targets)
                    util.scatterplot(
                        y_pred,
                        test_feature_matrix,
                        cv_slice.feature_names,
                        cv_slice.test_targets,
                        config.target_values,
                        y_pred_median,
                        tv_median,
                        scatterfile,
                    )
                    util.hexbinplot(y_pred, cv_slice.test_targets, config.target_values, hexbinfile)

                writeOutput(config, y_pred, cv_slice, samples, append=append)
                if not append:  # Append is only False in the first loop iteration
                    append = True

        if config.regression:
            if config.verbosity >= 1:
                util.printMean(mses, "MSE")
                util.printMean(pearsons, "Pearson's correlation")
            if len(accum_y_pred) > 0:
                cum_spear, _ = stats.spearmanr(accum_y_pred, accum_true_vals)
            else:
                cum_spear = None
            if len(spears) > 0:
                mean_spear = sum(spears) / len(spears)
            else:
                mean_spear = None

            out_value = (cum_spear, mean_spear)
        else:
            util.printMean(rocs, "auROC")
            util.printMean(accs, "ACC")
            util.printMean(fs, "F-Score")

        if config.outfolder is not None and config.produce_scatterplot and config.regression:
            base_name = f"{config.outfolder}/{config.dataset_name}"
            scatterfile = "%s.png" % (base_name)
            hexbinfile = "%s_hexbin.png" % (base_name)

            # scatterplot(y_pred,test_feature_matrix,feature_names,test_targets,target_values,0.5,observed_value_threshold,scatterfile,feature_highlight='Class')
            # middle_value = (max(accum_y_pred) + min(accum_y_pred))/2.
            if len(accum_y_pred) > 0:
                y_pred_median = util.median(accum_y_pred)
                tv_median = util.median(accum_true_vals)
                util.scatterplot(
                    accum_y_pred,
                    cv_slice.get_test_feature_matrix(samples),
                    cv_slice.feature_names,
                    accum_true_vals,
                    config.target_values,
                    y_pred_median,
                    tv_median,
                    scatterfile,
                )
                util.hexbinplot(accum_y_pred, accum_true_vals, config.target_values, hexbinfile)
                if config.crossValidation == "LOPO":
                    labels = []
                    values = []
                    for lopo_id in lopo_scores:
                        pearson_r = lopo_scores[lopo_id].pearson_r
                        labels.append(str(lopo_id))
                        values.append(pearson_r)
                    title = "Pearson's correlation"
                    radarfile = "%s_radar.png" % (base_name)
                    util.radar(labels, values, title, radarfile)
    elif (
        config.feature_selection == "confusion"
        or config.feature_selection == "confusion_and_regu"
        or config.feature_selection == "sequential_confusion"
        or config.feature_selection == "threeStaged"
        or config.feature_selection == "sequential_confusion_and_regu"
        or config.feature_selection == "threeStaged_listranking"
    ):
        if crossValidation == "LOPO":
            cross_val_obj = sampleSpace.LOPO(samples, config)
        elif crossValidation == "DataSAIL":
            cross_val_obj = sampleSpace.DataSAIL_cv(sampleSpace=samples, config=config)
        elif crossValidation == "specific":
            cross_val_obj = sampleSpace.Given_split(samples, config)
        else:
            cross_val_obj = sampleSpace.X_fold_cv(samples, config)
    else:
        cross_val_obj = None

    if config.outfolder is not None and not config.skip_final_model:
        base_name = f"{config.outfolder}/{config.dataset_name}"
        name_add = ''
        if config.random_split:
            name_add = '_random_split'

        modelfile = f"{config.outfolder}/StructGuy_trained_on_{config.dataset_name}{name_add}.dump"

        filtered_features_file = "%s_filtered_features.tsv" % (base_name)

        if samples_store_id is None:
            samples_store_id = ray.put(pack(samples))

        forest = buildFinalModel(
            samples,
            samples_store_id,
            config,
            internal_cv=cross_val_obj,
            outfile=modelfile,
            filtered_features_file=filtered_features_file,
        )

        config.add_entry_to_project_file("path_trained_model", modelfile)

        if not config.skip_cv:
            cv_file = "%s_full_cv_forests.dump" % (base_name)
            storeCV(forests, config, cross_val_obj, cv_file)

    return out_value


def load_data_for_pred(
    config: Config,
    impute_map,
    feature_names: list[str],
    ):
    t0 = time.time()
    samples = featureGenerator.createTrainingSet(config, external_impute=impute_map, for_prediction=True, filter_synon=config.trace_decisions)
    t1 = time.time()

    print(f'Time for loading dataset: {t1-t0}')

    for sample_id in samples.samples:
        samples.samples[sample_id].testtrain = "test"

    test_feature_matrix, test_targets, sample_id_list = samples.get_test_data_for_feature_list(feature_names)

    return samples, test_feature_matrix, test_targets, sample_id_list

def evaluate_dataset(config: Config):
    t0 = time.time()
    forest, extern_feature_names_list, impute_map, model_config, feat_stats = loadModel(config.path_to_model)
    t1 = time.time()

    print(f'Time for loading model: {t1-t0}')

    if config.verbosity >= 4:
        print(f'{feat_stats=}')

    model_filename = os.path.basename(config.path_to_model).split('.')[0]
    if model_filename.count('trained_on_') > 0:
        model_name = model_filename.split('trained_on_')[1]
    else:
        model_name = model_filename

    if config.verbosity >= 2:
        print(f"{extern_feature_names_list[:5]}\n...\n{extern_feature_names_list[-5:]}")

    samples, test_feature_matrix, test_targets, sample_id_list = load_data_for_pred(config, impute_map, extern_feature_names_list)
    
    if len(test_feature_matrix) == 0:
        return None, None, None

    if config.verbosity >= 1:
        print(
            "Shape of the feature matrix:",
            len(test_feature_matrix),
            len(test_feature_matrix[0]),
        )
    """
    max_test_feat_len = 100_000

    if len(test_feature_matrix) > max_test_feat_len:
        nr_splits = len(test_feature_matrix) // max_test_feat_len
        if max_test_feat_len % max_test_feat_len != 0:
            nr_splits += 1
        split_size = (len(test_feature_matrix) // nr_splits) + 1
        total_y_pred = []
        for i in range(nr_splits):
            if config.verbosity >= 1:
                print(f"Shape of the feature matrix split: {split_size=}")
            left = i * split_size
            right = (i+1) * split_size
            y_pred = forest.predict(test_feature_matrix[left:right])
            total_y_pred += y_pred
        y_pred = total_y_pred
    else:
    """
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

        for prot_id, aacs, effect in multi_savs:
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
                print(f"Ground truth is None for: {sample_id}")
                sys.exit()
        else:
            true_value = None

        pred_value = y_pred[pos]

        if pred_value is None:
            print(f"Predicted value is None for: {sample_id}")

        prot_id, aac = sample_id

        if true_value is not None:
            if prot_id not in protein_info:
                protein_size = samples.get_feature_value(sample_id, "Protein Size")
                protein_info[prot_id] = [protein_size, 0]
            protein_info[prot_id][1] += 1

        if prot_id not in protein_wise_results:
            protein_wise_results[prot_id] = Results()

        protein_wise_results[prot_id].add_result(aac, true_value, pred_value)

    if config.regression:
        if config.target_values is not None:
            r2 = r2_score(test_targets, y_pred)
            mse = mean_squared_error(test_targets, y_pred)
            corr, p_value = stats.spearmanr(test_targets, y_pred)
        else:
            r2 = None
            mse = None
            corr = None
            p_value = None

        if config.verbosity >= 1:
            print("R2-Score: ", r2)
            print("MSE: ", mse)
            print("Spearman correlation and p-value: ", corr, p_value)

        if config.target_values is not None:
            prot_wise_spearmans, mean_spearman, _ = util.calc_protein_wise_corr(test_targets, y_pred, sample_id_list, stats.spearmanr)
            prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(test_targets, y_pred, sample_id_list, stats.pearsonr)
        else:
            prot_wise_spearmans = None
            mean_spearman = None
            prot_wise_pearsons = None
            mean_pearson = None

        if config.verbosity >= 1:
            print(f"Number of samples: {len(sample_id_list)} {len(test_targets)} {len(y_pred)}")

            print(f"Prot-wise mean pearson: {mean_pearson}")
            print(f"Prot-wise mean spearman: {mean_spearman}")

        if mm_y_pred is not None:
            prot_wise_spearmans, mean_mm_spearman, _ = util.calc_protein_wise_corr(mm_test_targets, mm_y_pred, mm_sample_id_list, stats.spearmanr)
            prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(mm_test_targets, mm_y_pred, mm_sample_id_list, stats.pearsonr)

            if config.verbosity >= 1:
                print(f"Number of samples: {len(mm_sample_id_list)} {len(mm_test_targets)} {len(mm_y_pred)}")

                print(f"Prot-wise mean pearson for multi savs: {mean_pearson}")
                print(f"Prot-wise mean spearman for multi savs: {mean_mm_spearman}")

            prot_wise_spearmans, mean_comb_spearman, _ = util.calc_protein_wise_corr(
                combined_test_targets,
                combined_y_pred,
                combined_sample_id_list,
                stats.spearmanr,
            )
            prot_wise_pearsons, mean_pearson, _ = util.calc_protein_wise_corr(
                combined_test_targets,
                combined_y_pred,
                combined_sample_id_list,
                stats.pearsonr,
            )

            if config.verbosity >= 1:
                print(f"Number of samples: {len(combined_sample_id_list)} {len(combined_test_targets)} {len(combined_y_pred)}")

                print(f"Prot-wise mean pearson for all variants: {mean_pearson}")
                print(f"Prot-wise mean spearman for all variants: {mean_comb_spearman}")

        write_protein_wise_performances(
            f"{config.outfolder}/protein_wise_results.tsv",
            prot_wise_spearmans,
            protein_info,
        )

        if config.trace_decisions:
            if isinstance(forest, RandomForestRegressor):
                decisions, pred_std_vector = featureAnalysis.explain_decisions(config, forest, y_pred, test_feature_matrix, extern_feature_names_list, feat_stats)
            else:
                #sample_wise_feature_influence, _ = trainForest.perturb_xgb(config, config.path_to_model, test_feature_matrix, extern_feature_names_list, y_pred, test_targets, get_sample_wise_data=True, get_feat_impacts=False)
                forest.get_booster().feature_names = extern_feature_names_list

                #explainer = shap.TreeExplainer(forest)
                #explanation = explainer(test_feature_matrix)
                dtest_feature_matrix = DMatrix(test_feature_matrix, feature_names = extern_feature_names_list)
                explanation = forest.get_booster().predict(dtest_feature_matrix, pred_contribs=True)
                #print(explanation[0])

                #for pos, shap_val in enumerate(explanation[0][:-1]):
                #    print(f'{shap_val=} {extern_feature_names_list[pos]}')

                #cat_exp, cat_shaps = util.categorize_shap_from_xgb(explanation[0][:-1], extern_feature_names_list)

                #shap.plots.force(explanation[0][-1], cat_shaps, matplotlib=True, show=False, feature_names=feature_categories)
                #shap.plots.force(explanation[0], matplotlib=True, show=False, feature_names=extern_feature_names_list)
                #plt.savefig(f"{config.outfolder}/force_plot.png")


                joined_sample_ids = [x[0] + x[1] for x in sample_id_list]
                print(f'{len(joined_sample_ids)=} {len(y_pred)=} {len(test_feature_matrix[0])=} {len(extern_feature_names_list)=}')

                

                #plt.figure(figsize=(120, 80))
                #plot_tree(forest, num_trees = 2)
                #plt.savefig('xgb_viz.png')

                #"""
                st = SuperTree(
                    forest,
                    test_feature_matrix,
                    y_pred,
                    extern_feature_names_list,
                    joined_sample_ids
                )
                #st.show_tree(2)

                super_tree_folder = f"{config.outfolder}/supertrees"
                if not os.path.isdir(super_tree_folder):
                    os.makedirs(super_tree_folder)
                tree_id = 0
                booster_obj = forest.get_booster()
                for tree in booster_obj:
                    outfile = f'{super_tree_folder}/super_tree_{tree_id}.html'
                    st.save_html(which_tree = tree_id, filename=outfile)
                    tree_id += 1

                tree_df = booster_obj.trees_to_dataframe()
                #print(tree_df)

                feat_tree_map = {}

                tree_id_vec = tree_df['Tree']
                feat_name_vec = tree_df['Feature']

                for pos, tree_id in enumerate(tree_id_vec):
                    feat_name = feat_name_vec[pos]
                    if feat_name not in feat_tree_map:
                        feat_tree_map[feat_name] = set()
                    feat_tree_map[feat_name].add(tree_id)


                for feat_name in feat_tree_map:
                    print(f'{feat_name} {feat_tree_map[feat_name]}')
                #st.save_html()
                #"""
        else:
            sample_wise_feature_influence = []
            explanation = None
            
        if mm_y_pred is None or config.trace_decisions:
            combined_sample_id_list = sample_id_list
            combined_y_pred = y_pred
            combined_test_targets = test_targets

        if isinstance(forest, RandomForestRegressor):
            header = "Protein ID\tSAV\tPredicted effect value\tTree-wise standard deviation\tFeature 1\tFeature 2\t Feature 3\t Feature 4\t Feature 5\n"
            lines = [header]
            
            for pos, sample_id in enumerate(combined_sample_id_list):
                pred_value = combined_y_pred[pos]
                prot_id, aac = sample_id
                if config.trace_decisions:
                    pred_std = pred_std_vector[pos]
                    feature_decisions = decisions[pos]
                else:
                    pred_std = ""
                    feature_decisions = ""
                words = [prot_id, aac, str(pred_value), str(pred_std)]
                for feat_name, weight, left_thresh, right_thresh in feature_decisions[:50]:
                    if left_thresh is None and right_thresh is None:
                        decision_string = f"{feat_name} is None (Weight: {weight})"
                    elif left_thresh is None:
                        decision_string = f"{feat_name} < {right_thresh} (Weight: {weight})"
                    elif right_thresh is None:
                        decision_string = f"{feat_name} >= {left_thresh} (Weight: {weight})"
                    else:
                        decision_string = f"{feat_name} in [{left_thresh}, {right_thresh}] (Weight: {weight})"
                    words.append(decision_string)
                line = "\t".join(words) + "\n"
                lines.append(line)
        else:
            #number_of_displayed_features = 20
            header = "Protein ID\tSAV\tPredicted effect value"

            if explanation is not None:
                #for i in range(number_of_displayed_features):
                #    header += f"\tFeature {i+1}"
                for i in range(len(feature_categories)):
                    #header += f"\tFeature category {i+1}\tImpact sum\tMean impact\tMax impact feature"
                    header += f"\tFeature category {i+1}\tShap value\tTop feature of category {i+1}"
                header += '\n'
                lines = [header]

                if config.plot_sample_forces:
                    force_plot_folder = f"{config.outfolder}/force_plots"
                    if not os.path.isdir(force_plot_folder):
                        os.makedirs(force_plot_folder)

                for pos, sample_id in enumerate(combined_sample_id_list):
                    pred_value = combined_y_pred[pos]
                    prot_id, aac = sample_id
                        
                    words = [prot_id, aac, str(pred_value)]

                    cat_exp, cat_shaps = util.categorize_shap_from_xgb(explanation[pos][:-1], extern_feature_names_list)

                    if config.plot_sample_forces:
                        modified_feat_labels = []
                        cat_shaps = []
                        for pos, shap_val, (max_feat_shap, max_feat, feat_pos) in cat_exp:
                            feat_cat = feature_categories[pos]
                            if feat_pos is not None:
                                val = test_feature_matrix[pos][feat_pos]
                            else:
                                val = None
                            modified_feat_labels.append(f'{feat_cat}\n{max_feat} (shap={max_feat_shap}, val={val})')
                            cat_shaps.append(shap_val)

                        shap.plots.force(explanation[pos][-1], cat_shaps, matplotlib=True, show=False, feature_names=modified_feat_labels)
                        plt.savefig(f"{force_plot_folder}/{prot_id}_{aac}_cat_force_plot.png")
                        shap.plots.force(explanation[pos][-1], explanation[pos][:-1], matplotlib=True, show=False, feature_names=extern_feature_names_list)
                        plt.savefig(f"{force_plot_folder}/{prot_id}_{aac}_force_plot.png")
                        plt.clf()


                    for pos, shap_val, (max_feat_shap, max_feat, feat_pos) in cat_exp:
                        feat_cat = feature_categories[pos]
                        if shap_val < 0:
                            words.append(f'{feat_cat} features skews prediction towards functional consequence')
                        else:
                            words.append(f'{feat_cat} features skews prediction towards wiltype-like effect')
                        words.append(str(shap_val))

                        
                        if feat_pos is not None:
                            val = test_feature_matrix[pos][feat_pos]
                        else:
                            val = None
                        words.append(f'{max_feat} (shap={max_feat_shap}, {val=})')
                    """
                    #count = 0
                    if pos < len(sample_wise_feature_influence):

                        categorized_impacts = {}
                        for feat_name, feat_impact, feat_val in sample_wise_feature_influence[pos]:
                            if feat_name[:3] == 'oh_' and feat_val == 0:
                                continue
                            try:
                                feat_cat = feature_categories[util.catogrize_feat_by_name(feat_name)]
                            except TypeError:
                                print(f'{feat_name} could not be categorized')
                                continue
                            if feat_cat not in categorized_impacts:
                                categorized_impacts[feat_cat] = [[], 0., None, None]
                            categorized_impacts[feat_cat][0].append(feat_impact)
                            if abs(feat_impact) >= abs(categorized_impacts[feat_cat][1]):
                                categorized_impacts[feat_cat][1] = feat_impact
                                categorized_impacts[feat_cat][2] = feat_name
                                categorized_impacts[feat_cat][3] = feat_val

                        categorized_impacts_list = []
                        for feat_cat in categorized_impacts:
                            feat_impacts, max_impact, max_feat, max_feat_val = categorized_impacts[feat_cat]
                            impact_sum = sum(feat_impacts)
                            mean_impact = impact_sum/len(feat_impacts)

                            categorized_impacts_list.append((feat_cat, impact_sum, mean_impact, max_impact, max_feat, max_feat_val))

                        categorized_impacts_list = sorted(categorized_impacts_list, key=lambda x:abs(x[1]), reverse=True)

                        for (feat_cat, impact_sum, mean_impact, max_impact, max_feat, max_feat_val) in categorized_impacts_list:
                            if impact_sum < 0:
                                words.append(f'{feat_cat} features skews prediction towards functional consequence')
                            else:
                                words.append(f'{feat_cat} features skews prediction towards wiltype-like effect')
                            words.append(str(impact_sum))
                            words.append(str(mean_impact))
                            words.append(f'{max_feat} (val={max_feat_val}) has impact {max_impact}')
                    """
                            
                                
                    line = "\t".join(words) + "\n"
                    lines.append(line)

        predictions_file = f"{config.outfolder}/predictions_by_{model_name}.tsv"
        f = open(predictions_file, "w")
        f.write("".join(lines))
        f.close()

        if config.produce_scatterplot and config.target_values is not None:
            scatterfile = f"{config.outfolder}/predicted_value_scatterplot.png"
            hexbinfile = f"{config.outfolder}/predicted_value_hexbinplot.png"

            y_pred_median = util.median(y_pred)
            tv_median = util.median(test_targets)
            util.scatterplot(
                y_pred,
                test_feature_matrix,
                extern_feature_names_list,
                test_targets,
                config.target_values,
                y_pred_median,
                tv_median,
                scatterfile,
            )
            util.hexbinplot(y_pred, test_targets, config.target_values, hexbinfile)

        return mean_spearman, combined_test_targets, combined_y_pred
    elif model_config.regression:
        int_targets = []
        for x in test_targets:
            if x == "Benign":
                int_targets.append(1)
            else:
                int_targets.append(0)
        total_roc_auc = roc_auc_score(int_targets, y_pred)
        prot_wise_roc_aucs, mean_roc_auc, _ = util.calc_protein_wise_corr(
            int_targets,
            y_pred,
            sample_id_list,
            roc_auc_score,
            mono_return_score_function=True,
        )
        if config.verbosity >= 1:
            print(f"Total roc_auc: {total_roc_auc}")
            print(f"Prot-wise mean roc_auc: {mean_roc_auc}")

        header = "Protein ID\tSAV\tPredicted effect value\n"
        lines = [header]
        if mm_y_pred is None:
            combined_sample_id_list = sample_id_list
            combined_y_pred = y_pred
            combined_test_targets = test_targets

        for pos, sample_id in enumerate(combined_sample_id_list):
            pred_value = combined_y_pred[pos]
            prot_id, aac = sample_id
            words = [prot_id, aac, str(pred_value)]

            line = "\t".join(words) + "\n"
            lines.append(line)

        predictions_file = f"{config.outfolder}/predictions_by_{model_name}.tsv"
        f = open(predictions_file, "w")
        f.write("".join(lines))
        f.close()

        # write_protein_wise_pearsons(f'{config.outfolder}/protein_wise_results.tsv', prot_wise_roc_aucs, protein_info)
        return mean_roc_auc, int_targets, y_pred

    else:
        acc = accuracy_score(test_targets, y_pred)
        int_targets = classToInt(test_targets, samples)
        int_preds = classToInt(y_pred, samples)
        roc = roc_auc_score(int_targets, int_preds)

        f1 = f1_score(int_targets, int_preds)

        precision = precision_score(int_targets, int_preds)
        recall = recall_score(int_targets, int_preds)

        mcc = matthews_corrcoef(int_targets, int_preds)

        if config.verbosity >= 1:
            print("F-Score: ", f1)
            print("Accuracy: ", acc)
            print("Precision:", precision)
            print("Recall:", recall)
            print("MCC:", mcc)
    return


def writeOutput(config, y_pred, cv_slice, sampleSpace, append=False):
    feature_names = cv_slice.feature_names

    long_feat_name_string = "\t".join(feature_names)

    header = f"Protein Identifier\tAmino Acid Change\tTarget value\tPredicted value\tError\t{long_feat_name_string}\n"
    if not append:
        outlines = [header]
    else:
        outlines = []

    color_map = {}
    for pos, sample_id in enumerate(cv_slice.test_sample_ids):
        u_ac, aac = sample_id
        target_value = cv_slice.test_targets[pos]
        aac_base = aac[:-1]
        if u_ac not in color_map:
            color_map[u_ac] = {}
        if aac_base not in color_map[u_ac]:
            color_map[u_ac][aac_base] = []

        predicted_value = y_pred[pos]
        if config.regression:
            error = abs(target_value - predicted_value)
        else:
            error = target_value == predicted_value

        color_map[u_ac][aac_base].append(error)

        feature_vector = []
        for feature_name in feature_names:
            feat = sampleSpace.features[feature_name]
            feature_value = sampleSpace.get_feature_value(sample_id, feature_name)
            feature_vector.append(feat.string_convert(feature_value))

        outlines.append(
            "%s\t%s\t%s\t%s\t%s\t%s\n"
            % (
                u_ac,
                aac,
                str(target_value),
                str(y_pred[pos]),
                str(error),
                "\t".join(feature_vector),
            )
        )
    outfile = f"{config.outfolder}/{config.dataset_name}_stratified_predictions.tsv"
    if not append:
        f = open(outfile, "w")
    else:
        f = open(outfile, "a")
    f.write("".join(outlines))
    f.close()


def storeCV(forests, config, cross_val_obj, cv_file):
    with open(cv_file, "wb") as output:
        pickle.dump((forests, cross_val_obj, config), output, pickle.HIGHEST_PROTOCOL)
    if config.verbosity >= 1:
        print("\n============\nStored full CV results in %s\n============\n" % cv_file)


def loadCV(fn):
    with open(fn, "rb") as inp:
        forests, cross_val_object, config = pickle.load(inp)
    if config.verbosity >= 1:
        print("\n============\nLoaded full CV from %s\n============\n" % fn)
    return forests, cross_val_object, config



def buildFinalModel(samples, samples_store_id, config, internal_cv=None, outfile=None, filtered_features_file=None):
    # Some feature selection strategies require an internal cross validation-like slicing
    # An example is the confusion-based feature section

    cross_val_obj: sampleSpace.FullSlice = sampleSpace.FullSlice(samples, config, internal_cv=internal_cv)

    full_slice: CrossValidationSlice = cross_val_obj.slices[0]

    if config.verbosity >= 1:
        full_slice.printBalance(config)

    if config.feature_selection == "confusion":
        full_slice.subslices = []
        for slice_slice in full_slice.slice_slices:
            full_slice.subslices.append(slice_slice.test_prots)

    print_out = config.verbosity >= 1
    forest, scores, full_slice, slice_slices = trainForest.trainForest(
        config,
        full_slice,
        samples=samples,
        samples_store_id=samples_store_id,
        distance_map=samples.geometric_distance_map,
        print_out=print_out,
        skip_scoring=True,
    )

    if config.verbosity >= 1:
        print("Full Slice Info after training:")
        full_slice.printBalance(config)

    feat_importance_map = calcFeatureImportances(forest, samples, full_slice, config, print_them=print_out)

    if outfile is not None:
        storeModel(forest, full_slice.feature_names, config, outfile, samples.feat_stats)

    if filtered_features_file is not None:
        save_feature_importances(filtered_features_file, feat_importance_map)

    return forest
