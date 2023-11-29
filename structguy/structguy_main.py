#!/usr/bin/python
import sys
import getopt
import os
import time

from structguy import featureGenerator, util, learn, featureAnalysis

import structman.base_utils.ray_utils as ray_utils

disclaimer = """
structguy_main.py generate_features [-i -o --verbosity]\n
structguy_main.py build_model [-i -o --verbosity]\n
MORE TODO\n
"""

def parse_arguments(argument_start = 2):
    argv = sys.argv[argument_start:]
    try:
        long_paras = [
            'verbosity=', #from 1 to 5
            'hp=',
            'nocv' # Skip Cross Validation
        ]
        opts, args = getopt.getopt(argv, "i:n:m:", long_paras)

    except getopt.GetoptError:
        print("Illegal Input\n\n", disclaimer)
        return

    path_to_project_file = None

    verbosity_overwrite = None
    path_to_hyperparameters_file = None
    path_to_model = None

    skip_cv = None

    overwrite_proc_n = None

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

        if opt == '--verbosity':
            verbosity_overwrite = int(arg)

        if opt == '--hp':
            path_to_hyperparameters_file = arg

        if opt == '-n':
            overwrite_proc_n = int(arg)
            
        if opt == '--nocv':
            skip_cv = True


    if path_to_model is not None:
        if path_to_model.count('/') > 0:
            model_name = path_to_model.rsplit("/",1)[1].rsplit('.',1)[0]
        else:
            model_name = path_to_model.rsplit('.',1)[0]
    else:
        model_name = None


    config = util.Config(path_to_project_file, hyperparameters_path = path_to_hyperparameters_file)

    config.path_to_model = path_to_model
    config.model_name = model_name
    
    if skip_cv is not None:
        config.skip_cv = True

    if overwrite_proc_n is not None:
        config.proc_n = overwrite_proc_n

    if verbosity_overwrite is not None:
        config.verbosity = verbosity_overwrite

    return config

def feature_generator_main():
    config = parse_arguments()

    ray_utils.ray_init(config, overwrite_logging_level = 0)

    if config.path_structural_feature_table is not None:
        featureGenerator.expand_structural_feature_table(config)

def build_model_main():
    config = parse_arguments()
    # if config.verbosity > 0:
    #     print(config.printHyperParameter())

    config.saveHyperParameter()

    ray_utils.ray_init(config, overwrite_logging_level = 0)

    learn.learn(config)

    config.saveHyperParameter()

def predict_main():
    config = parse_arguments()

    ray_utils.ray_init(config, overwrite_logging_level = 0)
    learn.evaluate_dataset(config)

def generate_info():
    config = parse_arguments()

    forest, extern_feature_names_list, model_config = learn.loadModel(config.path_to_model)
    n_of_trees, n_of_nodes = featureAnalysis.get_base_stats(forest)

    print(f'Random Forest model consits of {n_of_trees} trees and a total of {n_of_nodes} Nodes')

    model_config.saveHyperParameter(f'{config.model_name}_hyperparameter.conf')

def main():

    start_time = time.time()
    possible_key_words = set(['generate_features', 'build_model', 'predict', 'info'])

    if len(sys.argv) < 2:
        print(disclaimer)
        return

    key_word = sys.argv[1]

    if not key_word in possible_key_words:
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

    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()
