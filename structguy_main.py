#!/usr/bin/python3
import sys
import getopt
import os

import featureGenerator
import util

import structman.structman_main as str_main

disclaimer = 'TODO'

def feature_generator_main():
    argv = sys.argv[2:]
    try:
        long_paras = [
            'verbosity=',
            'sm_conf='
        ]
        opts, args = getopt.getopt(argv, "f:c:o:s:", long_paras)

    except getopt.GetoptError:
        print("Illegal Input\n\n", disclaimer)
        return

    path_structural_feature_table = None
    path_to_config = None
    path_to_structman_config = None
    path_to_outfile = None
    path_to_sequence_fasta = None
    verbosity_overwrite = None

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

        if opt == '-o':
            path_to_outfile = arg

        if opt == '--verbosity':
            verbosity_overwrite = int(arg)

        if opt == '--sm_conf':
            path_to_structman_config = arg

    config = util.Config(path_to_config)

    if verbosity_overwrite is not None:
        config.verbosity = verbosity_overwrite

    if path_structural_feature_table is not None:
        config.structman_config = str_main.Config(path_to_structman_config, external_call = True, verbosity = config.verbosity)
        featureGenerator.expand_structural_feature_table(path_structural_feature_table, config, path_to_outfile, path_to_sequence_fasta = path_to_sequence_fasta)

def main():

    possible_key_words = set(['generate_features'])

    key_word = sys.argv[1]

    if not key_word in possible_key_words:
        print(disclaimer)
        return

    if key_word == 'generate_features':
        feature_generator_main()

if __name__ == "__main__":
    main()
