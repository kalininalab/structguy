import pickle
import sys
import os
import time
import ray
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

from structguy import featureAnalysis, featureGenerator, sampleSpace, trainForest
from structguy.featureGenerator import load_data_for_pred
from structguy import hyperParameterOptimization as hpo
from structguy.results_analysis import Results, write_protein_wise_performances
from structguy.support_classes import CrossValidationSlice
from structguy.util import (
    Config, loadModel, storeModel, calc_protein_wise_corr, scatterplot, hexbinplot,
    printMean, radar, parse_multi_savs_table, combine_individual_effects, cat_shap_full_matrix,
    categorize_shap_from_xgb, protein_wise_scatter_plot
)
from structguy.consts import feature_categories

from xgboost import plot_tree, DMatrix
import matplotlib.pyplot as plt


def xgbFeatureImportances(bst, config):
    feat_importance_map = bst.get_score(importance_type='gain')
    feat_score_tuples = []
    for feature_name in feat_importance_map:
        score = feat_importance_map[feature_name]
        feat_score_tuples.append((feature_name, score))
    feat_score_tuples.sort(key=lambda x: x[1], reverse=True)

    for feature_name, score in feat_score_tuples:
        if score == 0.0:
            continue
        if score < config.print_scores_greater_than:
            continue
        config.logger.info(f"{feature_name}: {score}")
    return feat_importance_map

def calcFeatureImportances(booster_list, samples, cv_slice, config, print_them=False):
    forest = booster_list[0][0]
    try:
        feature_scores = forest.feature_importances_
    except AttributeError:
        config.logger.error(f'ERROR: forest was {forest} in calcFeatureImportances')
        return xgbFeatureImportances(forest,config)
    
    feat_score_tuples = []
    feature_importance_map = {}
    # print feature_names

    feature_type_importance = {}

    for pos, feature_name in enumerate(cv_slice.feature_names):
        # print feature_name,': ',feature_scores[pos]
        try:
            feature_importance_map[feature_name] = feature_scores[pos]
        except KeyError:
            config.logger.info(
                "Some Error:",
                len(feature_scores),
                len(cv_slice.feature_names),
                len(cv_slice.features),
            )
            return feature_importance_map
        except IndexError:
            feature_importance_map[feature_name] = 0.0

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
            config.logger.info(f"{feature_name}: {score}")

        config.logger.info("Total feature importance by feature type:", feature_type_importance)

    return feature_importance_map


def save_feature_importances(outfile, feature_importance_map):
    lines = ["Feature\tImportance\n"]
    for feat_name in feature_importance_map:
        lines.append(f"{feat_name}\t{feature_importance_map[feat_name]}\n")

    f = open(outfile, "w")
    f.write("".join(lines))
    f.close()


def learn(config: Config):
    crossValidation = config.crossValidation

    if config.verbosity >= 1:
        config.logger.info("================================ Start Learn ==============================================")
        config.logger.info(f'{sys.argv}')
        config.logger.info(f"Protein-based randomization: {config.prot_based_separation}")
        config.logger.info(f"add Protein mean values: {config.addBias}")
        config.logger.info(f"Cross Validation: {crossValidation}")
        config.logger.info(f"filter structural features: {config.filterStructuralFeatures}")
        config.logger.info(f"filter samples with no mapped structures: {config.structure_threshold}")
        config.logger.info(f"Clean type-2 circularity from training set: {config.remove_t2}")

        config.logger.info(f"{config.random_split=}")
        config.logger.info("===========================================================================================")
        if config.path_structural_feature_table is not None:
            config.logger.info(f"Using feature file: {config.path_structural_feature_table}")
            config.logger.info(f"Writing processed features to processed feature file: {config.path_structural_feature_table.rsplit('.', 1)[0]}_processed.tsv")
        elif config.path_to_processed_features_file is not None:
            config.logger.info(f"Using feature file: {config.path_to_processed_features_file}")
        config.logger.info(f"Writing output to: {config.outfolder}")

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
    config.logger.info(f"Time for loading dataset: {t_0 - t0} {config.n_of_features=} {config.select_samples=}")

    if config.select_samples:
        feat_coverage_file = f"{config.outfolder}/prot_wise_feat_coverage.tsv"
        prot_wise_feature_coverage = samples.write_feature_coverage_matrix(feat_coverage_file)
        samples.select_bad_prots(prot_wise_feature_coverage, config)
        config.logger.info(f'Wrote prot-wise feature coverages to {feat_coverage_file}')

    if config.weighting == "geometric" and config.regression:
        samples.setGeometricDistanceMap(config)
        distance_map = ray.put(samples.geometric_distance_map)
    else:
        distance_map = None

    out_value = None
    samples_store_id = None

    if not config.skip_cv:
        samples_store_id = ray.put(pack(samples))

        if crossValidation == "LOPO":
            cross_val_obj = sampleSpace.LOPO(samples, config)
        elif crossValidation == "DataSAIL":
            cross_val_obj = sampleSpace.DataSAIL_cv(samples_store_id=samples_store_id, sampleSpace=samples, config=config)
        elif crossValidation == "specific":
            cross_val_obj = sampleSpace.Given_split(samples, config)
        else:
            cross_val_obj = sampleSpace.X_fold_cv(samples, config, crossValidation)

        cv_slice = cross_val_obj.getCurrentSlice()

        if config.verbosity >= 1:
            config.logger.info(f"{len(cv_slice.train_targets)=}")
            config.logger.info(f"{len(cv_slice.test_targets)=}")

        debug = config.debug_mode

        if config.cv_hpo:
            initial_training_input = cross_val_obj
        else:
            initial_training_input = cv_slice

        if config.verbosity >= 3:
            config.logger.info(f"Before initial model training: {config.cv_hpo=} {initial_training_input.slice_slices=}")

        if config.hyperOptimization == "bayesianComplete":
            forest, scores, initial_training_input = trainForest.trainForest(
                config,
                initial_training_input,
                samples.feat_corr_matrix, samples.feature_names,
                samples_store_id=samples_store_id,
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
                samples=samples,
                samples_store_id=samples_store_id,
                distance_map=distance_map,
                debug=debug,
            )
        elif config.hyperOptimization == "threeDim":
            if config.multi_gpu > 0:
                gpu_share = config.multi_gpu
            else:
                gpu_share = None
            forest, (first_scores, scores), initial_training_input = trainForest.trainForest(
                config,
                initial_training_input,
                samples.feat_corr_matrix, samples.feature_names,
                samples_store_id=samples_store_id,
                samples=samples,
                repeat=config.repeat_training,
                cv_repeat=config.cv_hpo,
                print_out=True,
                debug=debug,
                #remote=(config.multi_gpu >= len(initial_training_input.slices)),
                remote=(config.multi_gpu > 0),
                get_first_scores=True,
                gpu_share=gpu_share
            )
            hpo.threeDimHyperOptimization(
                config,
                initial_training_input,
                scores,
                first_scores,
                samples=samples,
                samples_store_id=samples_store_id,
                distance_map=distance_map,
                debug=debug,
            )
        else:
            if debug:
                forest, scores, initial_training_input = trainForest.trainForest(
                    config,
                    initial_training_input,
                    samples.feat_corr_matrix, samples.feature_names,
                    samples_store_id=samples_store_id,
                    samples=samples,
                    distance_map=distance_map,
                    repeat=config.repeat_training,
                    cv_repeat=config.cv_hpo,
                    print_out=True,
                    debug=debug,
                )
                scores.printOut()
            if config.verbosity >= 1:
                config.logger.info("Hyperparamter optimization skipped")

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
                config.logger.info(f"{cv_counter=}, {len(cv_slice.train_targets)=}")
                config.logger.info(f"Testset length: {len(cv_slice.test_targets)}")

                cv_slice.printBalance(config)

            print_out = config.verbosity >= 2
            if config.multi_gpu > 0:
                gpu_share = config.multi_gpu
            else:
                gpu_share = None
            booster_list, scores, cv_slice = trainForest.trainForest(
                config,
                cv_slice,
                samples.feat_corr_matrix,
                samples.feature_names,
                samples=samples,
                samples_store_id=samples_store_id,
                distance_map=distance_map,
                repeat=config.repeat_training,
                print_out=print_out,
                debug=debug,
                remote=False,
                gpu_share=gpu_share
            )

            if booster_list is None:
                continue

            forests[cv_counter] = booster_list

            if config.crossValidation == "LOPO":
                lopo_scores[tuple(cv_counter)] = scores

            y_pred = trainForest.booster_list_process_and_predict(booster_list, cv_slice, samples)

            name_add = ""
            if config.random_split:
                name_add = "_random_split"
            elif config.penalize_train_test_gap:
                name_add = '_ptt'
            else:
                name_add = '_ot'

            modelfile = f"{config.outfolder}/StructGuy_trained_on_{config.dataset_name}{name_add}_slice_{cv_counter}.dump"

            storeModel(booster_list, cv_slice.feature_names, config, modelfile, samples.feat_stats, samples.features)

            if config.regression:
                mses.append((scores.mse, len(cv_slice.test_targets)))
                pearsons.append((scores.pearson_r, 1))
                spears.append(scores.corr)
                prot_wise_spearmans, mean_spearman, raw_corrs = calc_protein_wise_corr(
                    cv_slice.test_targets,
                    y_pred,
                    cv_slice.test_sample_ids,
                    stats.spearmanr,
                )

                if config.verbosity >= 1:
                    config.logger.info(f"Prot-wise RHO: {mean_spearman}")
                    config.logger.info(prot_wise_spearmans)
                    if scores.mean_spear_repeat_std is not None:
                        config.logger.info(f'{scores.mean_spear_repeat_std=}')
                    for prot_id, spear_ in raw_corrs:
                        if spear_ < 0.3:
                            config.logger.info(f"Low prot-wise rho: {prot_id} - {spear_}")

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
            #    config.logger.info('Top 10 confusing features:')
            #    for i in range(10):
            #        config.logger.info(confusion_map[i])

            if config.outfolder is not None:
                if config.produce_scatterplot and config.regression and config.crossValidation == "LOPO":
                    base_name = f"{config.outfolder}/{config.dataset_name}"
                    scatterfile = "%s_%s.png" % (base_name, str(cv_counter))
                    hexbinfile = "%s_%s_hexbin.png" % (base_name, str(cv_counter))

                    scatterplot(
                        y_pred,
                        cv_slice.test_targets,
                        config.target_values,
                        scatterfile
                    )
                    hexbinplot(y_pred, cv_slice.test_targets, config.target_values, hexbinfile)

                writeOutput(config, y_pred, cv_slice, samples, append=append)
                if not append:  # Append is only False in the first loop iteration
                    append = True

        if config.regression:
            if config.verbosity >= 1:
                printMean(mses, "MSE")
                printMean(pearsons, "Pearson's correlation")
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
            printMean(rocs, "auROC")
            printMean(accs, "ACC")
            printMean(fs, "F-Score")

        if config.outfolder is not None and config.produce_scatterplot and config.regression:
            base_name = f"{config.outfolder}/{config.dataset_name}"
            scatterfile = "%s.png" % (base_name)
            hexbinfile = "%s_hexbin.png" % (base_name)

            # scatterplot(y_pred,test_feature_matrix,feature_names,test_targets,target_values,0.5,observed_value_threshold,scatterfile,feature_highlight='Class')
            # middle_value = (max(accum_y_pred) + min(accum_y_pred))/2.
            if len(accum_y_pred) > 0:

                scatterplot(
                    accum_y_pred,
                    accum_true_vals,
                    config.target_values,
                    scatterfile,
                )
                hexbinplot(accum_y_pred, accum_true_vals, config.target_values, hexbinfile)
                if config.crossValidation == "LOPO":
                    labels = []
                    values = []
                    for lopo_id in lopo_scores:
                        pearson_r = lopo_scores[lopo_id].pearson_r
                        labels.append(str(lopo_id))
                        values.append(pearson_r)
                    title = "Pearson's correlation"
                    radarfile = "%s_radar.png" % (base_name)
                    radar(labels, values, title, radarfile)
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
            cross_val_obj = sampleSpace.DataSAIL_cv(samples_store_id=samples_store_id, sampleSpace=samples, config=config)
        elif crossValidation == "specific":
            cross_val_obj = sampleSpace.Given_split(samples, config)
        else:
            cross_val_obj = sampleSpace.X_fold_cv(samples, config)
    else:
        cross_val_obj = None

    if config.outfolder is not None and not config.skip_final_model:
        base_name = f"{config.outfolder}/{config.dataset_name}"
        name_add = ""
        if config.random_split:
            name_add = "_random_split"
        elif config.penalize_train_test_gap:
            name_add = '_ptt'
        else:
            name_add = '_ot'


        modelfile = f"{config.outfolder}/StructGuy_trained_on_{config.dataset_name}{name_add}.dump"

        filtered_features_file = "%s_filtered_features.tsv" % (base_name)

        if samples_store_id is None:
            samples_store_id = ray.put(pack(samples))

        booster_list = buildFinalModel(
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

def val_to_str(val):
    try:
        int_val = int(str(val))
        return str(int_val)
    except ValueError:
        pass
    if val is None:
        val_str = "None"
    else:
        if val < -10:
            prec = 1
        elif val < -1:
            prec = 2
        elif val < -0.1:
            prec = 3
        elif val < -0.01:
            prec = 4
        elif val < -0.001:
            prec = 5
        elif val < 0.0001:
            prec = 6
        elif val < 0.001:
            prec = 5
        elif val < 0.01:
            prec = 4
        elif val < 0.1:
            prec = 3
        elif val < 1:
            prec = 2
        else:
            prec = 1

        val_str = f"{val:.{prec}f}"
    return val_str

def evaluate_dataset(config: Config):
    t0 = time.time()
    extern_feature_names_list: list[str]
    booster_list, extern_feature_names_list, impute_map, model_config, feat_stats, extern_features = loadModel(config.path_to_model)
    t1 = time.time()

    config.logger.info(f"Time for loading model: {t1 - t0}")

    if config.verbosity >= 4:
        config.logger.info(f"{feat_stats=}")

    model_filename = os.path.basename(config.path_to_model).split(".")[0]
    if model_filename.count("trained_on_") > 0:
        model_name = model_filename.split("trained_on_")[1]
    else:
        model_name = model_filename

    if config.verbosity >= 2:
        config.logger.info(f"{type(extern_feature_names_list)=} {extern_feature_names_list[:5]}\n...\n{extern_feature_names_list[-5:]}")

    if config.verbosity >= 5:
        config.logger.info(f'{extern_feature_names_list=}')

    samples, booster_specific_data = load_data_for_pred(config, impute_map, booster_list, extern_features)

    test_feature_matrix, test_targets, sample_id_list, feat_id_vec, cat_vec = booster_specific_data[0]

    if config.verbosity >= 3:
        feat_coverage_file = f"{config.outfolder}/prot_wise_feat_coverage.tsv"
        samples.write_feature_coverage_matrix(feat_coverage_file)
        config.logger.info(f'Wrote prot-wise feature coverages to {feat_coverage_file}')

    external_feat_pos_dict = dict(zip(extern_feature_names_list, range(len(extern_feature_names_list))))

    if config.verbosity >= 5:
        print(f'{feat_id_vec=}')
        print(f'{samples.feat_pos_dict['oh_Wildtype AA_I']=}')

        for pos, sample_id in enumerate(sample_id_list):
            feat_pos = external_feat_pos_dict['oh_Wildtype AA_I']
            print(f'{sample_id=} {feat_pos=} {test_feature_matrix[pos][feat_pos]=}')
            print(f'{samples.raw_feature_matrix[pos][samples.feat_pos_dict['oh_Wildtype AA_I']]=}')

    if len(test_feature_matrix) == 0:
        return None, None, None

    if config.verbosity >= 1:
        config.logger.info(f"Shape of the feature matrix: {len(test_feature_matrix)} {len(test_feature_matrix[0])}")

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
                config.logger.info(f"Shape of the feature matrix split: {split_size=}")
            left = i * split_size
            right = (i+1) * split_size
            y_pred = forest.predict(test_feature_matrix[left:right])99
            total_y_pred += y_pred
        y_pred = total_y_pred
    else:
    """
    test_feat_mats = []
    for booster_index, (_, extern_feature_names_list) in enumerate(booster_list):
        test_feature_matrix, test_targets, sample_id_list, feat_id_vec, cat_vec = booster_specific_data[booster_index]
        dtest_feature_matrix = DMatrix(
            numpy.array(test_feature_matrix),
            feature_types=cat_vec,
            enable_categorical=True,
            feature_names = extern_feature_names_list)
        test_feat_mats.append(dtest_feature_matrix)
    y_pred = trainForest.booster_list_predict(booster_list, test_feat_mats)

    if config.path_to_multi_savs_table is not None:
        multi_savs = parse_multi_savs_table(config)

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
            combined_effect = combine_individual_effects(individual_effect_preds)

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
                config.logger.info(f"Ground truth is None for: {sample_id}")
                sys.exit()
        else:
            true_value = None

        pred_value = y_pred[pos]

        if pred_value is None:
            config.logger.info(f"Predicted value is None for: {sample_id}")

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
            config.logger.info(f"R2-Score: {r2}")
            config.logger.info(f"MSE: {mse}")
            config.logger.info(f"Spearman correlation and p-value: {corr} {p_value}")

        if config.target_values is not None:
            prot_wise_spearmans, mean_spearman, _ = calc_protein_wise_corr(test_targets, y_pred, sample_id_list, stats.spearmanr)
            prot_wise_pearsons, mean_pearson, _ = calc_protein_wise_corr(test_targets, y_pred, sample_id_list, stats.pearsonr)
        else:
            prot_wise_spearmans = None
            mean_spearman = None
            prot_wise_pearsons = None
            mean_pearson = None

        if config.verbosity >= 1:
            config.logger.info(f"Number of samples: {len(sample_id_list)} {len(test_targets)} {len(y_pred)}")

            config.logger.info(f"Prot-wise mean pearson: {mean_pearson}")
            config.logger.info(f"Prot-wise mean spearman: {mean_spearman}")

        if mm_y_pred is not None:
            prot_wise_spearmans, mean_mm_spearman, _ = calc_protein_wise_corr(mm_test_targets, mm_y_pred, mm_sample_id_list, stats.spearmanr)
            prot_wise_pearsons, mean_pearson, _ = calc_protein_wise_corr(mm_test_targets, mm_y_pred, mm_sample_id_list, stats.pearsonr)

            if config.verbosity >= 1:
                config.logger.info(f"Number of samples: {len(mm_sample_id_list)} {len(mm_test_targets)} {len(mm_y_pred)}")

                config.logger.info(f"Prot-wise mean pearson for multi savs: {mean_pearson}")
                config.logger.info(f"Prot-wise mean spearman for multi savs: {mean_mm_spearman}")

            prot_wise_spearmans, mean_comb_spearman, _ = calc_protein_wise_corr(
                combined_test_targets,
                combined_y_pred,
                combined_sample_id_list,
                stats.spearmanr,
            )
            prot_wise_pearsons, mean_pearson, _ = calc_protein_wise_corr(
                combined_test_targets,
                combined_y_pred,
                combined_sample_id_list,
                stats.pearsonr,
            )

            if config.verbosity >= 1:
                config.logger.info(f"Number of samples: {len(combined_sample_id_list)} {len(combined_test_targets)} {len(combined_y_pred)}")

                config.logger.info(f"Prot-wise mean pearson for all variants: {mean_pearson}")
                config.logger.info(f"Prot-wise mean spearman for all variants: {mean_comb_spearman}")

        write_protein_wise_performances(
            f"{config.outfolder}/protein_wise_results.tsv",
            prot_wise_spearmans,
            protein_info,
        )

        if config.trace_decisions:
            if isinstance(forest, RandomForestRegressor):
                decisions, pred_std_vector = featureAnalysis.explain_decisions(config, forest, y_pred, test_feature_matrix, extern_feature_names_list, feat_stats)
            else:
                # sample_wise_feature_influence, _ = trainForest.perturb_xgb(config, config.path_to_model, test_feature_matrix, extern_feature_names_list, y_pred, test_targets, get_sample_wise_data=True, get_feat_impacts=False)
                forest.feature_names = extern_feature_names_list

                # explainer = shap.TreeExplainer(forest)
                # explanation = explainer(test_feature_matrix)
                #dtest_feature_matrix = DMatrix(test_feature_matrix, feature_names=extern_feature_names_list)
                
                explanation = forest.predict(dtest_feature_matrix, pred_contribs=True)
                shap_expl = shap.Explanation(explanation[:,:-1], data = test_feature_matrix, feature_names=extern_feature_names_list)
                
                ax = shap.plots.beeswarm(shap_expl, show=False, max_display= 30)
                plt.subplots_adjust(left=0.5, right=0.9)
                plt.savefig(f"{config.outfolder}/beeswarm.png")
                plt.clf()

                cat_expl = cat_shap_full_matrix(explanation, extern_feature_names_list)

                cat_shap_expl = shap.Explanation(cat_expl[:,:-1], feature_names=feature_categories)
                ax = shap.plots.beeswarm(cat_shap_expl, show=False, max_display = len(feature_categories))
                plt.subplots_adjust(left=0.5, right=0.9)
                plt.savefig(f"{config.outfolder}/cat_beeswarm.png")
                plt.clf()

                # config.logger.info(explanation[0])

                # for pos, shap_val in enumerate(explanation[0][:-1]):
                #    config.logger.info(f'{shap_val=} {extern_feature_names_list[pos]}')

                # cat_exp, cat_shaps = categorize_shap_from_xgb(explanation[0][:-1], extern_feature_names_list)

                # shap.plots.force(explanation[0][-1], cat_shaps, matplotlib=True, show=False, feature_names=feature_categories)
                # shap.plots.force(explanation[0], matplotlib=True, show=False, feature_names=extern_feature_names_list)
                # plt.savefig(f"{config.outfolder}/force_plot.png")

                joined_sample_ids = [x[0] + x[1] for x in sample_id_list]
                config.logger.info(f"{len(joined_sample_ids)=} {len(y_pred)=} {len(test_feature_matrix[0])=} {len(extern_feature_names_list)=}")

                # plt.figure(figsize=(120, 80))
                # plot_tree(forest, num_trees = 2)
                # plt.savefig('xgb_viz.png')

                # """
                


                if config.plot_trees:
                    st = SuperTree(forest, test_feature_matrix, y_pred, extern_feature_names_list, joined_sample_ids)
                    # st.show_tree(2)

                    super_tree_folder = f"{config.outfolder}/supertrees"
                    if not os.path.isdir(super_tree_folder):
                        os.makedirs(super_tree_folder)
                    tree_id = 0
                    
                    for tree in forest:
                        outfile = f"{super_tree_folder}/super_tree_{tree_id}.html"
                        st.save_html(which_tree=tree_id, filename=outfile)
                        tree_id += 1

                    
                    # config.logger.info(tree_df)

                    feat_tree_map = {}
                    
                    tree_df = forest.trees_to_dataframe()

                    tree_id_vec = tree_df["Tree"]
                    feat_name_vec = tree_df["Feature"]

                    for pos, tree_id in enumerate(tree_id_vec):
                        feat_name = feat_name_vec[pos]
                        if feat_name not in feat_tree_map:
                            feat_tree_map[feat_name] = set()
                        feat_tree_map[feat_name].add(tree_id)

                    for feat_name in feat_tree_map:
                        config.logger.info(f"{feat_name} {feat_tree_map[feat_name]}")


                # st.save_html()
                # """
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
            # number_of_displayed_features = 20
            if config.calc_sd:
                header = "Protein ID\tSAV\tPredicted effect value\tTree STD"
            else:
                header = "Protein ID\tSAV\tPredicted effect value"

            
            # for i in range(number_of_displayed_features):
            #    header += f"\tFeature {i+1}"
            for i in range(len(feature_categories)):
                # header += f"\tFeature category {i+1}\tImpact sum\tMean impact\tMax impact feature"
                header += f"\tFeature category {i + 1}\tShap value\tTop feature of category {i + 1}"
            header += "\n"
            lines = [header]

            if config.target_values is not None:
                eval_header = "Protein ID\tSAV\tPredicted effect value\tTrue value"
                for i in range(len(feature_categories)):
                    eval_header += f"\tFeature category {i + 1}\tShap value\tTop feature of category {i + 1}\tFeature Shap value\tFeat value"
                eval_header += "\n"
                eval_lines = [eval_header]

                full_eval_header = "Protein ID\tSAV\tPredicted effect value\tTrue value"
                for feat_name in extern_feature_names_list:
                    full_eval_header += f'\t{feat_name}'
                full_eval_header += "\n"
                full_eval_lines = [full_eval_header]

            if config.plot_sample_forces:
                force_plot_folder = f"{config.outfolder}/force_plots"
                if not os.path.isdir(force_plot_folder):
                    os.makedirs(force_plot_folder)

            if config.calc_sd:
                ind_preds = []
                for tree_id, tree in enumerate(forest):
                    ind_pred = tree.predict(dtest_feature_matrix)
                    ind_preds.append(ind_pred)

                ind_preds = numpy.array(ind_preds).transpose()

            for pos, sample_id in enumerate(combined_sample_id_list):
                pred_value = combined_y_pred[pos]
                prot_id, aac = sample_id

                words = [prot_id, aac, str(pred_value)]

                if config.target_values is not None:
                    eval_words = words[:]
                    eval_words.append(str(combined_test_targets[pos]))
                    full_eval_words = eval_words[:]

                if config.calc_sd:
                    pred_std = numpy.std(ind_preds[pos])
                    words.append(str(pred_std))

                if explanation is not None:
                    #print(f'{sample_id} {pos=} {len(explanation[pos][:-1])=} {len(extern_feature_names_list)=}')
                    cat_exp, cat_shaps = categorize_shap_from_xgb(explanation[pos][:-1], extern_feature_names_list)
                    if config.plot_sample_forces:
                        modified_feat_labels = []
                        cat_shaps = []
                        for cat_pos, shap_val, (max_feat_shap, max_feat, feat_pos) in cat_exp:
                            if max_feat is not None:
                                feat_cat = feature_categories[cat_pos]
                                if feat_pos is not None:
                                    val = test_feature_matrix[pos][feat_pos]
                                    val_str = val_to_str(val)
                                    
                                else:
                                    val_str = "None"

                                perc_shap = (100*max_feat_shap)/shap_val

                                feat_st = feat_stats[max_feat]
                                mean_val = feat_st[2]
                                mean_val_str = val_to_str(mean_val)

                                modified_feat_labels.append(f"{feat_cat}\n{max_feat}\nshap={max_feat_shap:.3f} ({perc_shap:.1f}%)\nval={val_str} (mean={mean_val_str})")
                            else:
                                modified_feat_labels.append('None')
                            cat_shaps.append(shap_val)

                        shap.plots.force(explanation[pos][-1], numpy.array(cat_shaps), matplotlib=True, show=False, feature_names=modified_feat_labels, figsize=(28,5))
                        plt.savefig(f"{force_plot_folder}/{prot_id}_{aac}_cat_force_plot.png")
                        shap.plots.force(explanation[pos][-1], explanation[pos][:-1], matplotlib=True, show=False, feature_names=extern_feature_names_list)
                        plt.savefig(f"{force_plot_folder}/{prot_id}_{aac}_force_plot.png")
                        plt.clf()

                    for cat_pos, shap_val, (max_feat_shap, max_feat, feat_pos) in cat_exp:
                        feat_cat = feature_categories[cat_pos]
                        if shap_val < 0:
                            words.append(f"{feat_cat} features skews prediction towards functional consequence")
                        else:
                            words.append(f"{feat_cat} features skews prediction towards wiltype-like effect")
                        words.append(str(shap_val))

                        if feat_pos is not None:
                            val = test_feature_matrix[pos][feat_pos]
                        else:
                            val = None
                        words.append(f"{max_feat} (shap={max_feat_shap}, {val=})")

                        if config.target_values is not None:
                            eval_words.append(str(feat_cat))
                            eval_words.append(str(shap_val))
                            eval_words.append(str(max_feat))
                            eval_words.append(str(max_feat_shap))
                            eval_words.append(str(val))

                    for feat_pos, feat_name in enumerate(extern_feature_names_list):
                        shap_val = explanation[pos][feat_pos]
                        feat_val = test_feature_matrix[pos][feat_pos]
                        if config.target_values is not None:
                            full_eval_words.append(f'{shap_val},{feat_val}')

                line = "\t".join(words) + "\n"
                lines.append(line)
                if config.target_values is not None:
                    eval_line = "\t".join(eval_words) + "\n"
                    eval_lines.append(eval_line)

                    full_eval_line = "\t".join(full_eval_words) + "\n"
                    full_eval_lines.append(full_eval_line)


        predictions_file = f"{config.outfolder}/predictions_by_{model_name}.tsv"
        f = open(predictions_file, "w")
        f.write("".join(lines))
        f.close()
        if config.target_values is not None:
            eval_predictions_file = f"{config.outfolder}/predictions_by_{model_name}_with_eval.tsv"
            f = open(eval_predictions_file, "w")
            f.write("".join(eval_lines))
            f.close()

            full_eval_predictions_file = f"{config.outfolder}/predictions_by_{model_name}_full_eval.tsv"
            f = open(full_eval_predictions_file, "w")
            f.write("".join(full_eval_lines))
            f.close()

        if config.produce_scatterplot and config.target_values is not None:
            scatterfile = f"{config.outfolder}/predicted_value_scatterplot.png"
            hexbinfile = f"{config.outfolder}/predicted_value_hexbinplot.png"

            scatterplot(
                y_pred,
                test_targets,
                config.target_values,
                scatterfile,
            )
            hexbinplot(y_pred, test_targets, config.target_values, hexbinfile)

            scatter_folder = f"{config.outfolder}/scatter_plots"
            if not os.path.isdir(scatter_folder):
                os.makedirs(scatter_folder)
            protein_wise_scatter_plot(test_targets, y_pred, combined_sample_id_list, scatter_folder, config.target_values)

        return mean_spearman, combined_test_targets, combined_y_pred
    elif model_config.regression:
        int_targets = []
        for x in test_targets:
            if x == "Benign":
                int_targets.append(1)
            else:
                int_targets.append(0)
        total_roc_auc = roc_auc_score(int_targets, y_pred)
        prot_wise_roc_aucs, mean_roc_auc, _ = calc_protein_wise_corr(
            int_targets,
            y_pred,
            sample_id_list,
            roc_auc_score,
            mono_return_score_function=True,
        )
        if config.verbosity >= 1:
            config.logger.info(f"Total roc_auc: {total_roc_auc}")
            config.logger.info(f"Prot-wise mean roc_auc: {mean_roc_auc}")

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
            config.logger.info("F-Score: ", f1)
            config.logger.info("Accuracy: ", acc)
            config.logger.info("Precision:", precision)
            config.logger.info("Recall:", recall)
            config.logger.info("MCC:", mcc)
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
        config.logger.info("\n============\nStored full CV results in %s\n============\n" % cv_file)


def loadCV(fn):
    with open(fn, "rb") as inp:
        forests, cross_val_object, config = pickle.load(inp)
    if config.verbosity >= 1:
        config.logger.info("\n============\nLoaded full CV from %s\n============\n" % fn)
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
    if config.multi_gpu > 0:
        gpu_share = config.multi_gpu
    else:
        gpu_share = None
    booster_list, scores, full_slice = trainForest.trainForest(
        config,
        full_slice,
        samples.feat_corr_matrix,
        samples.feature_names,
        samples=samples,
        samples_store_id=samples_store_id,
        distance_map=samples.geometric_distance_map,
        print_out=print_out,
        skip_scoring=True,
        gpu_share=gpu_share
    )

    if config.verbosity >= 1:
        config.logger.info("Full Slice Info after training:")
        full_slice.printBalance(config)

    feat_importance_map = calcFeatureImportances(booster_list, samples, full_slice, config, print_them=print_out)

    if outfile is not None:
        storeModel(booster_list, full_slice.feature_names, config, outfile, samples.feat_stats, samples.features)

    if filtered_features_file is not None:
        save_feature_importances(filtered_features_file, feat_importance_map)

    return booster_list
