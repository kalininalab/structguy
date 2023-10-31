#!/usr/bin/python3
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import matthews_corrcoef

from scipy import stats
import resource
import sys
import os
import random

import util
import getopt

import featureGenerator
import hyperParameterOptimization as hpo
import sampleSpace
import trainForest
from results_analysis import Results, write_protein_wise_pearsons
import featureAnalysis

import cProfile
import pstats

import ray
import pickle


def calcFeatureImportances(forest, samples, cv_slice, config, print_them=False):
    feature_scores = forest.feature_importances_

    feat_score_tuples = []
    feature_value_map = {}
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
            feature_value_map[feature_name] = []
            for values in cv_slice.test_feature_matrix:
                feature_value_map[feature_name].append(values[pos])

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
            print(feature_name, ':', score, ' ;Spearman: ', stats.spearmanr(cv_slice.test_targets, feature_value_map[feature_name]))

        print('Total feature importance by feature type:', feature_type_importance)

    return feature_importance_map


def learn(config, effectRegressor=None):
    crossValidation = config.crossValidation

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
    elif config.path_to_processed_feature_file is not None:
        print(f'Using feature file: {config.path_to_processed_feature_file}')
    print(f'Writing output to: {config.outfolder}')

    samples = featureGenerator.createTrainingSet(config)

    if config.geometric_weighting and config.regression:
        samples.setGeometricDistanceMap(config)
        distance_map = ray.put(samples.geometric_distance_map)
    else:
        distance_map = None

    if not config.skip_cv:

        if crossValidation == 'LOPO':
            cross_val_obj = sampleSpace.LOPO(samples, config)
        elif crossValidation == 'DataSAIL':
            cross_val_obj = sampleSpace.DataSAIL_cv(samples, config)
        else:
            cross_val_obj = sampleSpace.X_fold_cv(samples, config, crossValidation)

        cv_slice = cross_val_obj.getCurrentSlice()

        print(len(cv_slice.train_targets))
        print('Testset length: ',len(cv_slice.test_targets))

        debug = False
        if config.feature_selection == 'confusion' or config.feature_selection == 'confusion_and_regu' or config.feature_selection == 'sequential_confusion' or config.feature_selection == 'threeStaged' or config.feature_selection == 'sequential_confusion_and_regu' or config.feature_selection == 'threeStaged_listranking':
            samples_store_id = ray.put(samples)
        else:
            samples_store_id = None

        if config.cv_hpo:
            initial_training_input = cross_val_obj
        else:
            initial_training_input = cv_slice

        if config.hyperOptimization == 'bayesianComplete':
            forest,scores = trainForest.trainForest(config, initial_training_input, samples = samples_store_id, distance_map = distance_map, repeat = config.repeat_training, cv_repeat = config.cv_hpo, print_out = True, debug = debug)
            hpo.bayesianComplete(config, initial_training_input, scores, samples = samples_store_id, distance_map = distance_map)
        elif config.hyperOptimization == 'threeDim':
            forest, scores = trainForest.trainForest(config, initial_training_input, samples = samples_store_id, distance_map = distance_map, repeat = config.repeat_training, cv_repeat = config.cv_hpo, print_out = True, debug = debug)
            hpo.threeDimHyperOptimization(config, initial_training_input, scores, samples = samples_store_id, distance_map = distance_map, debug = debug)
        else:
            if debug:
                forest,scores = trainForest.trainForest(config, initial_training_input, samples = samples_store_id, distance_map = distance_map, repeat = config.repeat_training, cv_repeat = config.cv_hpo, print_out = True, debug = debug)
                scores.printOut()

            print('Hyperparamter optimization skipped')

        lopo_scores = {}

        accum_y_pred = []
        accum_true_vals = []


        if config.regression:
            mses = []
            pearsons = []
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

            print(cv_counter, len(cv_slice.train_targets))
            print('Testset length: ',len(cv_slice.test_targets))

            cv_slice.printBalance(config)

            forest,scores = trainForest.trainForest(config, cv_slice, samples = samples_store_id, distance_map = distance_map, print_out = True, debug = debug)

            forests[cv_counter] = forest

            if config.crossValidation == 'LOPO':
                lopo_scores[tuple(cv_counter)] = scores

            y_pred = forest.predict(cv_slice.test_feature_matrix)

            if config.regression:
                mses.append((scores.mse,len(cv_slice.test_targets)))
                pearsons.append((scores.pearson_r,1))
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
                    util.scatterplot(y_pred, cv_slice.test_feature_matrix, cv_slice.feature_names,
                                     cv_slice.test_targets, config.target_values, y_pred_median, tv_median, scatterfile)
                    util.hexbinplot(y_pred, cv_slice.test_targets, config.target_values, hexbinfile)

                writeOutput(config, y_pred, cv_slice, samples, append=append)
                if not append: #Append is only False in the first loop iteration
                    append = True

        if config.regression:
            util.printMean(mses,'MSE')
            util.printMean(pearsons,"Pearson's correlation")
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
            y_pred_median = util.median(accum_y_pred)
            tv_median = util.median(accum_true_vals)
            util.scatterplot(accum_y_pred,cv_slice.test_feature_matrix,cv_slice.feature_names,accum_true_vals,config.target_values,y_pred_median,tv_median,scatterfile)
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
            cross_val_obj = sampleSpace.DataSAIL_cv(samples, config)
        else:
            cross_val_obj = sampleSpace.X_fold_cv(samples, config, crossValidation)
    else:
        cross_val_obj = None

    if config.outfolder != None:
        base_name = f'{config.outfolder}/{config.dataset_name}'
        modelfile = f'{config.outfolder}/StructGuy_trained_on_{config.dataset_name}.dump'

        filtered_features_file = '%s_filtered_features.tsv' % (base_name)

        forest = buildFinalModel(samples, config, internal_cv = cross_val_obj, outfile = modelfile, filtered_features_file = filtered_features_file)

        config.add_entry_to_project_file('path_trained_model', modelfile)

        if not config.skip_cv:
            cv_file = '%s_full_cv_forests.dump' % (base_name)
            storeCV(forests, config, cross_val_obj, cv_file)


def evaluate_dataset(config):
    samples = featureGenerator.createTrainingSet(config)

    for sample_id in samples.samples:
        samples.samples[sample_id].testtrain = 'test'

    forest, extern_feature_names_list, model_config = loadModel(config.path_to_model)

    test_feature_matrix, test_targets, sample_id_list = samples.get_test_data_for_feature_list(extern_feature_names_list)


    print('Shape of the feature matrix:',len(test_feature_matrix),len(test_feature_matrix[0]))
    y_pred = forest.predict(test_feature_matrix)

    protein_info = {}
    protein_wise_results = {}
    for pos, sample_id in enumerate(sample_id_list):
        true_value = test_targets[pos]
        pred_value = y_pred[pos]
        prot_id, aac = sample_id
        protein_size = samples.features['Protein Size'].value_map[sample_id]

        if true_value is not None:
            if prot_id not in protein_info:
                protein_info[prot_id] = [protein_size, 0]
            protein_info[prot_id][1] += 1

        if not prot_id in protein_wise_results:
            protein_wise_results[prot_id] = Results()

        protein_wise_results[prot_id].add_result(aac, true_value, pred_value)

    if config.regression:
        r2 = r2_score(test_targets,y_pred)
        mse = mean_squared_error(test_targets,y_pred)
        corr,p_value = stats.spearmanr(test_targets,y_pred)

        print('R2-Score: ',r2)
        print('MSE: ',mse)
        print('Spearman correlation and p-value: ',corr,p_value)

        prot_wise_spearmans, mean_spearman = util.calc_protein_wise_corr(test_targets, y_pred, sample_id_list, stats.spearmanr)
        prot_wise_pearsons, mean_pearson = util.calc_protein_wise_corr(test_targets, y_pred, sample_id_list, stats.pearsonr)

        print(f'Prot-wise mean pearson: {mean_pearson}')
        print(f'Prot-wise mean spearman: {mean_spearman}')

        write_protein_wise_pearsons(f'{config.outfolder}/protein_wise_results.tsv', protein_wise_results, protein_info)

        decisions, pred_std_vector = featureAnalysis.explain_decisions(config, forest, y_pred, test_feature_matrix, extern_feature_names_list)

        header = 'Protein ID\tSAV\tPredicted effect value\tTree-wise standard deviation\tFeature 1\tFeature 2\t Feature 3\t Feature 4\t Feature 5\n'
        lines = [header]
        for pos, sample_id in enumerate(sample_id_list):
            pred_value = y_pred[pos]
            prot_id, aac = sample_id
            pred_std = pred_std_vector[pos]
            feature_decisions = decisions[pos]
            words = [prot_id, aac, str(pred_value), str(pred_std)]
            for feat_name, weight, left_thresh, right_thresh in feature_decisions[:5]:
                if left_thresh is None:
                    decision_string = f'{feat_name} < {right_thresh}'
                elif right_thresh is None:
                    decision_string = f'{feat_name} >= {left_thresh}'
                else:
                    decision_string = f'{feat_name} in [{left_thresh}, {right_thresh}]'
                words.append(decision_string)
            line = '\t'.join(words) + '\n'
            lines.append(line)

        predictions_file = f'{config.outfolder}/predictions.tsv'
        f = open(predictions_file, 'w')
        f.write(''.join(lines))
        f.close()

        if config.produce_scatterplot:
            scatterfile = f'{config.outfolder}/predicted_value_scatterplot.png'
            hexbinfile = f'{config.outfolder}/predicted_value_hexbinplot.png'

            y_pred_median = util.median(y_pred)
            tv_median = util.median(test_targets)
            util.scatterplot(y_pred, test_feature_matrix, extern_feature_names_list, test_targets, config.target_values, y_pred_median, tv_median, scatterfile)
            util.hexbinplot(y_pred, test_targets, config.target_values, hexbinfile)

    else:
        acc = accuracy_score(test_targets, y_pred)
        int_targets = classToInt(test_targets, samples)
        int_preds = classToInt(y_pred,samples)
        roc = roc_auc_score(int_targets,int_preds)

        f1 = f1_score(int_targets,int_preds)

        precision = precision_score(int_targets,int_preds)
        recall = recall_score(int_targets,int_preds)

        mcc = matthews_corrcoef(int_targets,int_preds)

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
            feature_value = feat.value_map[sample_id]
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
    print('\n============\nStored full CV results in %s\n============\n' % cv_file)

def loadCV(fn):
    with open(fn, 'rb') as inp:
        forests, cross_val_object, config = pickle.load(inp)
    print('\n============\nLoaded full CV from %s\n============\n' % fn)
    return forests, cross_val_object, config


def storeModel(model, feature_names, config, fn):
    with open(fn, 'wb') as output:
        pickle.dump((model, feature_names, config), output, pickle.HIGHEST_PROTOCOL)
    print('\n============\nStored model in %s\n============\n' % fn)

def loadModel(fn):
    with open(fn, 'rb') as inp:
        model, feature_names, config = pickle.load(inp)
    print('\n============\nLoaded model from %s\n============\n' % fn)
    return model, feature_names, config

def buildFinalModel(samples, config, internal_cv = None, outfile = None, filtered_features_file = None):
    #Some feature selection strategies require an internal cross validation-like slicing
    #An example is the confusion-based feature section
    cross_val_obj = sampleSpace.FullSlice(samples, config, internal_cv = internal_cv)
    full_slice = cross_val_obj.slices[0]

    full_slice.printBalance(config)

    forest,scores = trainForest.trainForest(config, full_slice, samples = samples, distance_map = samples.geometric_distance_map, print_out = True,skip_scoring = True)

    feat_importance_map = calcFeatureImportances(forest, samples, full_slice, config, print_them=True)

    if outfile != None:
        storeModel(forest, full_slice.feature_names, config, outfile)

    if filtered_features_file != None:
        full_slice.write(filtered_features_file)

    return forest
