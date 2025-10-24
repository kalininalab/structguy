#!/usr/bin/python
import sys
import getopt
import os
import time
import logging
from datetime import datetime

from structguy import featureGenerator, util, learn, featureAnalysis
from structguy.sequence_feature_generation import prepare_gemme
from ray.train import ScalingConfig


import structman.base_utils.ray_utils as ray_utils
## Import the Forest-Guided Clustering package
from fgclustering import (
    forest_guided_clustering, 
    forest_guided_feature_importance, 
    plot_forest_guided_feature_importance, 
    plot_forest_guided_decision_paths,
    DistanceRandomForestProximity,
    ClusteringKMedoids,
    ClusteringClara
)

disclaimer = """
structguy_main.py generate_features [-i -o --verbosity]\n
structguy_main.py build_model [-i -o --verbosity]\n
MORE TODO\n
"""

def parse_arguments(argument_start = 2, manual_args = None):
    if manual_args is None:
        argv = sys.argv[argument_start:]
        try:
            long_paras = [
                'verbosity=', #from 1 to 5
                'hp=',
                'nocv', # Skip Cross Validation
                'nohpo', # Skip Hyperparameter Optimization
                'lopo', # Activate LOPO training setup,
                'endless_hpo',
                'overwrite',
                'feats=',
                'type=',
                'test_config=',
                'support_features=',
                'splits=',
                'random_split',
                'skip_final_model',
                'trace_decisions',
                'filter_syn',
                'penalize_ttg',
                'nofshpo',
                'forces',
                'gpu=',
                'sd',
                'repeat='
            ]
            opts, args = getopt.getopt(argv, "i:n:m:d", long_paras)

        except getopt.GetoptError as e:
            print(f"Illegal Input\n{e}\n\n{disclaimer}")
            return
    else:
        opts = manual_args

    path_to_project_file = None

    verbosity_overwrite = None
    path_to_hyperparameters_file = None
    path_to_model = None
    feature_list_file = None

    skip_cv = None
    skip_hpo = None
    force_lopo = False
    overwrite = False

    debug = False

    overwrite_proc_n = None
    forest_type = None
    test_config_path = None

    path_to_splits_file = None
    path_to_support_features = None
    skip_final_model = None

    trace_decisions = False
    plot_sample_forces = False
    calc_sd = False

    filter_syn = False
    random_split = False
    bayesianComplete = False
    penalize_train_test_gap = False
    skip_fshpo = False

    multi_gpu = 0
    repeat = None

    for opt, arg in opts:
        if opt == '-i':
            path_to_project_file = arg
            if not os.path.isfile(path_to_project_file):
                print(f'Error: {path_to_project_file} is not a valid file path')
                return

        if opt == '-m':
            path_to_model = arg
            if not os.path.isfile(path_to_model):
                print('ERROR: path to model file is invalid')
                return

        if opt == '-d':
            debug = True

        if opt == '--verbosity':
            verbosity_overwrite = int(arg)

        if opt == '--hp':
            path_to_hyperparameters_file = arg

        if opt == '-n':
            overwrite_proc_n = int(arg)
            
        if opt == '--nocv':
            skip_cv = True

        if opt == '--nohpo':
            skip_hpo = True

        if opt == '--lopo':
            force_lopo = True
        
        if opt == '--overwrite':
            overwrite = True

        if opt == '--feats':
            feature_list_file = arg

        if opt == '--type':
            forest_type = arg

        if opt == '--test_config':
            test_config_path = arg

        if opt == '--splits':
            path_to_splits_file = arg

        if opt == '--support_features':
            path_to_support_features = arg

        if opt == '--skip_final_model':
            skip_final_model = True

        if opt == '--trace_decisions':
            trace_decisions = True

        if opt == '--filter_syn':
            filter_syn = True

        if opt == '--random_split':
            random_split = True

        if opt == '--endless_hpo':
            bayesianComplete = True

        if opt == '--penalize_ttg':
            penalize_train_test_gap = True

        if opt == '--nofshpo':
            skip_fshpo = True

        if opt == '--forces':
            plot_sample_forces = True

        if opt == '--sd':
            calc_sd = True

        if opt == '--gpu':
            multi_gpu = int(arg)

        if opt == '--repeat':
            repeat = int(arg)

    if path_to_model is not None:
        if path_to_model.count('/') > 0:
            model_name = path_to_model.rsplit("/",1)[1].rsplit('.',1)[0]
        else:
            model_name = path_to_model.rsplit('.',1)[0]
    else:
        model_name = None


    print(f'Parsing config: {path_to_project_file=} {random_split=}')

    config: util.Config = util.Config(path_to_project_file, hyperparameters_path = path_to_hyperparameters_file)

    if skip_final_model is not None:
        config.skip_final_model = True

    config.path_to_model = path_to_model
    config.model_name = model_name
    config.overwrite = overwrite
    config.random_split = random_split
    config.debug_mode = debug
    config.filter_synonymous = filter_syn
    config.penalize_train_test_gap = penalize_train_test_gap

    config.multi_gpu = multi_gpu

    if repeat is not None:
        config.repeat_training = repeat

    if skip_fshpo:
        config.hpo_do_feat_selection = False

    config.path_to_splits_file = path_to_splits_file
    if path_to_splits_file is not None:
        config.crossValidation = 'specific'
    config.path_to_support_features = path_to_support_features

    config.trace_decisions = trace_decisions
    config.plot_sample_forces = plot_sample_forces
    config.calc_sd = calc_sd

    if force_lopo:
        config.crossValidation = 'LOPO'

    if random_split:
        config.crossValidation = 'Random'
        config.prot_based_separation = False

    if skip_cv is not None:
        config.skip_cv = True

    if skip_hpo is not None:
        config.hyperOptimization = None

    if bayesianComplete:
        config.hyperOptimization = 'bayesianComplete'

    if overwrite_proc_n is not None:
        config.proc_n = overwrite_proc_n

    if verbosity_overwrite is not None:
        config.verbosity = verbosity_overwrite

    if feature_list_file is not None:
        config.feature_selection = 'pre_defined'
        config.feature_list_file = feature_list_file

    if forest_type is not None:
        config.forest_type = forest_type
        if forest_type == 'gradient_boost' or forest_type == 'xgboost':
            #config.impute_missing_values = True
            try:
                import torch
                config.gpu_mode = torch.cuda.is_available()
                config.multi_gpu = max([1, config.multi_gpu])
            except ModuleNotFoundError:
                config.gpu_mode = False

    if config.multi_gpu > 1:
        #import dask
        #from dask_cuda import LocalCUDACluster
        #with LocalCUDACluster(n_workers=config.multi_gpu, threads_per_worker=config.proc_n) as cluster:
        #    pass
        config.gpu_mode = True
        cpu_per_worker = max([1, (config.proc_n//config.multi_gpu) - 2])
        config.scaling_config = ScalingConfig(num_workers=config.multi_gpu, use_gpu=True, resources_per_worker={"CPU": cpu_per_worker})

    if test_config_path is not None:
        test_config = util.Config(test_config_path)
    else:
        test_config = None

    main_logger = logging.getLogger(__name__)

    time_stamp = str(datetime.now()).replace(' ','_')

    log_file = f'{config.outfolder}/main_log_{time_stamp}.log'
    logging.basicConfig(filename=log_file, encoding='utf-8', level=logging.DEBUG)

    config.logger = main_logger

    return config, test_config

def generate_violins():
    config, _ = parse_arguments()
    featureAnalysis.plot_violins(config)

def feature_generator_main():
    config, test_config = parse_arguments()

    if config.path_structural_feature_table is not None or config.overwrite:
        featureGenerator.expand_structural_feature_table(config)

def build_model_main(manual_args = None):
    config, test_config = parse_arguments(manual_args = manual_args)

    config.saveHyperParameter()

    logging_level = 0
    if config.verbosity >= 4:
        logging_level = 20

    ray_utils.ray_init(config, overwrite_logging_level = logging_level, total_memory_quantile = 0.74, num_gpus=config.multi_gpu)

    out_value = learn.learn(config, test_config=test_config)

    config.saveHyperParameter()
    return out_value

def predict_main(manual_args = None):
    config, test_config = parse_arguments(manual_args = manual_args)
    config.predict_mode = True
    ray_utils.ray_init(config, overwrite_logging_level = 0)
    score, y_true, y_pred = learn.evaluate_dataset(config)
    return score, y_true, y_pred

def prep_gemme():
    config: util.Config
    config, _ = parse_arguments()
    prepare_gemme(config)


def generate_info():
    config: util.Config
    config, _ = parse_arguments()

    t0 = time.time()

    forest, extern_feature_names_list, impute_map, model_config, feat_stats = learn.loadModel(config.path_to_model)

    t1 = time.time()

    print(f'Loaded model: time={t1-t0}')

    n_of_trees, n_of_nodes = featureAnalysis.get_base_stats(forest)

    t2 = time.time()
    print(f'Random Forest model consits of {n_of_trees} trees and a total of {n_of_nodes} Nodes, time: {t2-t1}')

    samples, test_feature_matrix, test_targets, sample_id_list = learn.load_data_for_pred(config, impute_map, extern_feature_names_list)
    
    t3 = time.time()
    print(f'Loaded data: time={t3-t2}')

    # compute the forest-guided clusters
    fgc = forest_guided_clustering(
        estimator=forest, 
        X=test_feature_matrix, 
        y=test_targets,
        n_jobs=config.proc_n,
        clustering_distance_metric=DistanceRandomForestProximity(memory_efficient=True, dir_distance_matrix="./"), 
        clustering_strategy=ClusteringClara(sub_sample_size=0.6, sampling_iter=5, method="fasterpam"),
    )

    t4 = time.time()
    print(f'Time for clustering: {t4-t3}')

    # evaluate feature importance
    feature_importance = forest_guided_feature_importance(
        X=test_feature_matrix, 
        y=test_targets,
        cluster_labels=fgc.cluster_labels,
        model_type=fgc.model_type,
    )

    # visualize the results
    plot_forest_guided_feature_importance(
        feature_importance_local=feature_importance.feature_importance_local,
        feature_importance_global=feature_importance.feature_importance_global
    )

    plot_forest_guided_decision_paths(
        data_clustering=feature_importance.data_clustering,
        model_type=fgc.model_type,
    )

    model_config.saveHyperParameter(f'{config.model_name}_hyperparameter.conf')

def main():

    start_time = time.time()
    possible_key_words = set(['generate_features', 'build_model', 'predict', 'info', 'violins', 'prep_gemme'])

    if len(sys.argv) < 2:
        print(disclaimer)
        return

    key_word = sys.argv[1]

    if key_word not in possible_key_words:
        print(disclaimer)
        return

    if key_word == 'generate_features':
        feature_generator_main()

    if key_word == 'build_model':
        build_model_main()

    if key_word == 'predict':
        predict_main()

    if key_word == 'info':
        generate_info()

    if key_word == 'violins':
        generate_violins()

    if key_word == 'prep_gemme':
        prep_gemme()

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
