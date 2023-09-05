#!/usr/bin/python
import sys
import getopt
import os

import featureGenerator
import util
import learn

import structman.structman_main as str_main

disclaimer = """
structguy_main.py generate_features [-f -c -o -s --verbosity --sm_conf]\n
structguy_main.py build_model [(-f or -p) -c -o -s --verbosity --sm_conf]\n
MORE TODO\n
"""

def parse_arguments(argument_start = 2):
    argv = sys.argv[argument_start:]
    try:
        long_paras = [
            'verbosity=',
            'sm_conf='
        ]
        opts, args = getopt.getopt(argv, "p:f:c:o:s:n:m:", long_paras)

    except getopt.GetoptError:
        print("Illegal Input\n\n", disclaimer)
        return

    path_structural_feature_table = None
    path_to_config = None
    path_to_structman_config = None

    path_to_sequence_fasta = None
    verbosity_overwrite = None
    path_to_processed_feature_file = None
    path_to_model = None

    overwrite_proc_n = None

    for opt, arg in opts:
        if opt == '-f':
            path_structural_feature_table = arg
            if not os.path.isfile(path_structural_feature_table):
                print('ERROR: path to structural feature table is invalid')
                return

        if opt == '-c':
            path_to_config = arg
            if not os.path.isfile(path_to_config):
                print('ERROR: path to config file is invalid')
                return

        if opt == '-s':
            path_to_sequence_fasta = arg
            if not os.path.isfile(path_to_sequence_fasta):
                print('ERROR: path to sequence fasta file is invalid')
                return

        if opt == '-m':
            path_to_model = arg
            if not os.path.isfile(path_to_model):
                print('ERROR: path to model file is invalid')
                return

        if opt == '-o':
            outfolder = arg

        if opt == '--verbosity':
            verbosity_overwrite = int(arg)

        if opt == '--sm_conf':
            path_to_structman_config = arg

        if opt == '-p':
            path_to_processed_feature_file = arg

        if opt == '-n':
            overwrite_proc_n = int(arg)

    if path_to_processed_feature_file is not None:
        primary_dataset_file_path = path_to_processed_feature_file
    else:
        primary_dataset_file_path = path_structural_feature_table

    if primary_dataset_file_path.count('/') > 0:
        _outfolder, primary_dataset_filename = primary_dataset_file_path.rsplit("/",1)
    else:
        _outfolder = os.getcwd()
        primary_dataset_filename = primary_dataset_file_path

    dataset_name = primary_dataset_filename.rsplit('.',1)[0]

    if outfolder is None:
        outfolder = _outfolder

    config = util.Config(path_to_config)

    config.outfolder = outfolder
    config.dataset_name = dataset_name
    config.path_structural_feature_table = path_structural_feature_table
    config.path_to_config = path_to_config
    config.path_to_structman_config = path_to_structman_config
    config.path_to_sequence_fasta = path_to_sequence_fasta
    config.path_to_processed_feature_file = path_to_processed_feature_file
    config.path_to_model = path_to_model

    if overwrite_proc_n is not None:
        config.proc_n = overwrite_proc_n

    if verbosity_overwrite is not None:
        config.verbosity = verbosity_overwrite

    return config

def feature_generator_main():
    config = parse_arguments()

    if config.path_structural_feature_table is not None:
        config.structman_config = str_main.Config(config.path_to_structman_config, external_call = True, verbosity = config.verbosity, num_of_cores = config.proc_n)
        featureGenerator.expand_structural_feature_table(config)

def build_model_main():
    config = parse_arguments()
    config.structman_config = str_main.Config(config.path_to_structman_config, external_call = True, verbosity = config.verbosity, num_of_cores = config.proc_n)

    import structman.base_utils.ray_utils as ray_utils

    ray_utils.ray_init(config.structman_config, overwrite_logging_level = 0)

    model,feature_names = learn.learn(config)

def predict_main():
    config = parse_arguments()
    config.structman_config = str_main.Config(config.path_to_structman_config, external_call = True, verbosity = config.verbosity, num_of_cores = config.proc_n)

    import structman.base_utils.ray_utils as ray_utils

    ray_utils.ray_init(config.structman_config, overwrite_logging_level = 0)
    learn.evaluate_dataset(config)

def main():

    possible_key_words = set(['generate_features', 'build_model', 'predict'])

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

if __name__ == "__main__":
    main()
