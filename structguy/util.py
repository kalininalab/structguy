#!/usr/bin/python3
import sys, os
import getopt
import statistics
import numpy as np
from psutil import virtual_memory

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from structguy.scripts import radarplot
from structman.base_utils.base_utils import Errorlog, resolve_path

class OutputCapture:
    def __init__(self):
        self.captured_output = ""

    def write(self, text):
        self.captured_output += text

    def replace(self, ReplaceFrom, replaceTo):
        self.captured_output.replace(ReplaceFrom, replaceTo)

    def __str__(self):
        return self.captured_output

def parse_conf(filepath):
    f = open(filepath, 'r')
    lines = f.read().split('\n')
    f.close()

    opt_args = []

    for line in lines:
        if len(line) == 0:
            continue
        if line[0] == '#':
            continue
        
        words = line.split('=')
        if len(words) < 2:
            continue

        # Set config values and remove leading/trailling whitespaces
        opt = words[0].strip()
        arg = words[1].replace("\n","").strip()
        opt_args.append((opt, arg))
    return opt_args

class Config:
    def __init__(self, path_to_project_file, hyperparameters_path = None):
        util_scriptpath = os.path.abspath(resolve_path(__file__))
        settings_path = f'{util_scriptpath.rsplit('/',1)[0]}/resources/search_db_settings.conf'
        search_db_opt_args = parse_conf(settings_path)

        self.profiling = False
        self.predict_mode = False
        self.iupred_path = ""
        self.ray_local_mode = False
        mem = virtual_memory()
        self.gigs_of_ram = mem.total / 1024 / 1024 / 1024
        self.errorlog = Errorlog()

        self.dataset_name = ''
        self.path_to_sequence_fasta = None
        self.path_to_features_file = None
        self.path_to_processed_features_file = None
        self.path_to_imputed_features_file = None
        self.path_structural_feature_table = None
        self.path_to_multi_savs_table = None
        self.outfolder = None

        self.debug = 1
        self.proc_n = 48
        self.seq_feat_processes = 48

        self.mmseqs_path = ""

        self.search_dbs = []
        self.msa_dbs = [] #['ref50']
        self.gpw_dbs = ['ref50', 'ref90']

        self.mmseqs_search_db_ref50 = ""
        self.mmseqs_search_db_ref90 = ""
        self.mmseqs_search_db_ref100 = ''

        for (opt, arg) in search_db_opt_args:
            if opt == 'search_db_folder':
                self.mmseqs_search_db_ref50 = f'{arg}/uniref50_search_db'
                self.mmseqs_search_db_ref90 = f'{arg}/uniref90_search_db'

        self.mmseqs_tmp_folder = ''

        self.msa_db = ''

        self.fasta_mode = False

        self.verbosity = 1

        self.crossValidation=True
        self.crossValidation_fold = 5
        self.multiple_lopo = 1

        self.split_rate = 0.2

        self.mafft_path = ''
        self.blast_path = ''
        self.psic_source = f'{util_scriptpath.rsplit('/',1)[0]}/resources/psic'
        self.blosum_path = f'{self.psic_source}/Blosum62.txt'

        self.pdb_path = ''


        self.skip_cv = False
        self.suppress_remote_forests = False

        self.feature_selection = 'threeStaged' #Possible key words: threeStaged, 'meanCorrelation', 'regularization', 'double', 'confusion', confusion_and_regu sequential_confusion, sequential_confusion_and_regu, 'confusion_and_regu'
        self.select_feature_for_first_slice_only = True

        self.impute_missing_values = False

        self.hpo_do_feat_selection = True
        self.hpo_do_forest_param = True
        self.hpo_do_sample_weighting = True

        self.hp_outputFile = "hyperparameters.conf"

        #reguFS stuff
        self.maxIter = 1000

        self.bfd_factor = 2.0

        self.tvmb_rank_threshold = 0 #2
        self.tvpmb_rank_threshold = 0

        #Confusion feature selection
        self.sequential_confusion_rank_threshold = 91
        self.confusion_rank_threshold = 311
        self.confusion_goodwill = 0.3720368#0.62#0.25        
        self.err_warping_exp = 1.584#1.9855 #4.0
        self.confusion_normalization_exp = 1.2

        self.structure_threshold = None
        self.blacklist = []
        self.color_structures = False

        self.tag_filter = set([])
        self.tag_based_separation = None
        self.target_translator = {}
        self.tag_based_crossValidation = None
        self.targetFilter = set(['nan'])
        self.target_values = None
        self.class_labels = None
        self.binary_thresh = 0.5
        self.protein_filter = set([])

        self.print_scores_greater_than = 0.0005

        self.add_more_sample_files = []
        self.filterStructuralFeatures = False
        self.addBias = False
        self.remove_t2 = False
        self.prot_based_separation = False
        self.filter_single_variant_prots = False
        self.balanceSubsampling = None
        self.fusePositions = False
        #self.standard_feature_filter = {'dPSIC GPW ref50':0.0}

        self.transform = False
        self.produce_scatterplot = False

        self.geometric_weighting = False
        self.geometric_exponent = 2

        #HPO setup
        self.hyperOptimization='twoDim'
        self.cv_hpo = False
        self.cv_hpo_limiter = None
        self.cv_counters = None
        self.optimize_mean = False
        self.repeat_training = 1

        self.feature_penalty = 0.00002

        #Forest hyperparameters
        self.tree_depth = 194
        self.min_sample_split = 4
        self.tree_min_leaf_samples = 1
        self.num_of_trees = 382
        self.rel_max_leaf_node_pruning = None#2.
        self.max_leaf_nodes = None #int((2**(depth))/rel_max_leaf_node_pruning)
        self.class_weight = 'balanced_subsample'
        self.max_feature_parameter = 'log2'
        self.max_feature_cont_parameter = 0.362233199721295
        self.bootstrap_parameter = True
        self.min_impurity_decrease_exp = 11.7787
        self.oob_score = False
        self.ccp_alpha_exp = 30#10.365171642057746
        self.max_sample_parameter = 0.9276380091150952
        self.number_of_bins = 2
        self.p_val_thresh = 0.81533
        self.sample_weight_parameter = 1.0
        self.reg_alpha_exp = 3.5 #1.5215717821360353
        self.reg_c_exp = 3.
        self.reg_thresh_exp = 20.
        self.list_ranking_thresh = 150

        self.maximal_exp = 19.0

        #intervals for hyperparameter Optimization
        self.class_weights = ['balanced','balanced_subsample',None]
        self.bootstrap_parameters = [True,False]
        self.max_feature_parameters = ['auto','log2','sqrt']
        self.max_feature_cont_parameter_bounds = [0.,1.]
        self.min_sample_split_half_step = [2,200]
        self.min_sample_leaf_half_step = [1,50]
        self.tree_depth_half_step = [1,400]
        self.forest_size_half_step = [10,500]
        self.min_impurity_decrease_exp_half_step = [2.0,20.0]
        self.oob_scores = [True,False]
        self.ccp_alpha_exp_half_step = [1.,30.]
        self.max_sample_half_step = [0.,1.]
        self.geometric_exponent_bounds = [0.,4.]
        self.reg_alpha_exp_half_step = [0.,20.]
        self.reg_c_exp_half_step = [-1.,20.]
        self.reg_thresh_exp_half_step = [0.,20.]

        self.list_ranking_thresh_bounds = [0, 'max']
        self.confusion_goodwill_bounds = [0., 1.0]

        self.tvmb_rank_half_step = [0, 'max']
        self.tvpmb_rank_half_step = [0, 'max']
        self.confusion_rank_threshold_bounds = [0, 'max']
        self.sequential_confusion_rank_threshold_bounds = [0, 'max']
        self.err_warping_exp_bounds = [0.,4.]
        self.confusion_normalization_exp_bounds = [0.,4.]

        self.number_of_bins_half_step = [1,20]
        self.p_val_thresh_half_step = [0.,1.]
        self.sample_weight_parameter_half_step = [0.,1.]

        overwrite_objective_function = None

        self.path_to_project_file = path_to_project_file

        opt_args = parse_conf(path_to_project_file)
        for opt, arg in opt_args:
            if opt == 'proc_n':
                self.proc_n = int(arg)

            elif opt == 'profiling':
                if arg == 'True':
                    self.profiling = True

            elif opt == 'seq_feat_processes':
                self.seq_feat_processes = int(arg)

            elif opt == 'mmseqs_path':
                self.mmseqs_path = arg

            elif opt == 'mmseqs_tmp_folder':
                self.mmseqs_tmp_folder = arg

            elif opt == 'mmseqs_search_db_ref50':
                self.mmseqs_search_db_ref50 = arg
            elif opt == 'mmseqs_search_db_ref90':
                self.mmseqs_search_db_ref90 = arg
            elif opt == 'mmseqs_search_db_ref100':
                self.mmseqs_search_db_ref100 = arg

            elif opt == 'msa_db':
                self.msa_db = arg
                if not os.path.isdir(self.msa_db):
                    os.mkdir(self.msa_db)

            elif opt == 'crossValidation':
                try:
                    self.crossValidation = int(arg)
                except:
                    if arg.count(',') > 0:
                        self.tag_based_crossValidation = arg.split(',')
                    else:
                        self.crossValidation = arg

            elif opt == 'cv_hpo':
                if arg == 'True':
                    self.cv_hpo = True
                elif arg == 'False':
                    self.cv_hpo = False

            elif opt == 'multiple_lopo':
                self.multiple_lopo = int(arg)

            elif opt == 'binary_thresh':
                self.binary_thresh = float(arg)

            elif opt == 'split_rate':
                self.split_rate = float(arg)

            elif opt == 'mafft_path':
                self.mafft_path = arg

            elif opt == 'blast_path':
                self.blast_path = arg

            elif opt == 'psic_source':
                self.psic_source = arg

            elif opt == 'blosum_path':
                self.blosum_path = arg

            elif opt == 'pdb_path':
                self.pdb_path = arg

            elif opt == 'produce_scatterplot':
                if arg == 'True':
                    self.produce_scatterplot = True

            elif opt == 'target_values':
                self.target_values = arg

            elif opt == 'class_labels':
                self.class_labels = arg.split(',')

            elif opt == 'blacklist':
                self.blacklist = arg.split(',')

            elif opt == 'feature_selection':
                self.feature_selection = arg

            elif opt == 'bfd_factor':
                self.bfd_factor = float(arg)

            elif opt == 'prot_based_separation':
                if arg == 'True':
                    self.prot_based_separation = True
                elif arg == 'False':
                    self.prot_based_separation = False

            elif opt == 'filterStructuralFeatures':
                if arg == 'True':
                    self.filterStructuralFeatures = True
                elif arg == 'False':
                    self.filterStructuralFeatures = False

            elif opt == 'addBias':
                if arg == 'True':
                    self.addBias = True
                elif arg == 'False':
                    self.addBias = False

            elif opt == 'balanceSubsampling':
                if arg == 'None':
                    self.balanceSubsampling = None
                else:
                    self.balanceSubsampling = arg

            elif opt == 'structure_threshold':
                if arg == 'None':
                    self.structure_threshold = None
                else:
                    self.structure_threshold = int(arg)

            elif opt == 'fusePositions':
                if arg == 'True':
                    self.fusePositions = True
                elif arg == 'False':
                    self.fusePositions = False

            elif opt == 'color_structures':
                if arg == 'True':
                    self.color_structures = True
                elif arg == 'False':
                    self.color_structures = False

            elif opt == 'hyperOptimization':
                self.hyperOptimization = arg

            elif opt == 'optimize_mean':
                if arg == 'True':
                    self.optimize_mean = True
                elif arg == 'False':
                    self.optimize_mean = False

            elif opt == 'transform':
                if arg == 'True':
                    self.transform = True
                elif arg == 'False':
                    self.transform = False

            elif opt == 'filter_single_variant_prots':
                if arg == 'True':
                    self.filter_single_variant_prots = True
                elif arg == 'False':
                    self.filter_single_variant_prots = False

            elif opt == 'remove_t2':
                if arg == 'True':
                    self.remove_t2 = True
                elif arg == 'False':
                    self.remove_t2 = False

            elif opt == 'objective_function':
                overwrite_objective_function = arg

            elif opt == 'target_map':
                dict_tuples = arg.split(',')
                for dt in dict_tuples:
                    key,value = dt.split(':')
                    self.target_translator[key] = value

            elif opt == 'tag_filter':
                for tag in arg.split(','):
                    if tag == '':
                        continue
                    self.tag_filter.add(tag)

            elif opt == 'protein_filter':
                for u_ac in arg.split(','):
                    if u_ac == '':
                        continue
                    self.protein_filter.add(u_ac)

            elif opt == 'additional_samples':
                datafile = arg
                self.add_more_sample_files.append(datafile)

            elif opt == 'tree_depth':
                self.tree_depth = int(arg)

            elif opt == 'min_sample_split':
                self.min_sample_split = int(arg)

            elif opt == 'tree_min_leaf_samples':
                self.tree_min_leaf_samples = int(arg)

            elif opt == 'num_of_trees':
                self.num_of_trees = int(arg)

            elif opt == 'class_weight':
                self.class_weight = arg

            elif opt == 'max_feature_parameter':
                self.max_feature_parameter = arg

            elif opt == 'max_feature_cont_parameter':
                self.max_feature_cont_parameter = float(arg)

            elif opt == 'min_impurity_decrease_exp':
                self.min_impurity_decrease_exp = float(arg)

            elif opt == 'ccp_alpha_exp':
                self.ccp_alpha_exp = float(arg)

            elif opt == 'max_sample_parameter':
                self.max_sample_parameter = float(arg)

            elif opt == 'tvmb_rank_threshold':
                self.tvmb_rank_threshold = int(arg)

            elif opt == 'tvpmb_rank_threshold':
                self.tvpmb_rank_threshold = int(arg)

            elif opt == 'confusion_rank_threshold':
                self.confusion_rank_threshold = int(arg)

            elif opt == 'sequential_confusion_rank_threshold':
                self.sequential_confusion_rank_threshold = int(arg)

            elif opt == 'err_warping_exp':
                self.err_warping_exp = float(arg)

            elif opt == 'confusion_normalization_exp':
                self.confusion_normalization_exp = float(arg)

            elif opt == 'reg_alpha_exp':
                self.reg_alpha_exp = float(arg)

            elif opt == 'reg_c_exp':
                self.reg_c_exp = float(arg)

            elif opt == 'reg_thresh_exp':
                self.reg_thresh_exp = float(arg)

            elif opt == 'geometric_exp':
                self.geometric_exponent = float(arg)

            elif opt == 'skip_cv':
                if arg == 'True':
                    self.skip_cv = True
                elif arg == 'False':
                    self.skip_cv = False

            elif opt == 'hpo_do_feat_selection':
                if arg == 'True':
                    self.hpo_do_feat_selection = True
                elif arg == 'False':
                    self.hpo_do_feat_selection = False

            elif opt == 'hpo_do_forest_param':
                if arg == 'True':
                    self.hpo_do_forest_param = True
                elif arg == 'False':
                    self.hpo_do_forest_param = False

            elif opt == 'hpo_do_sample_weighting':
                if arg == 'True':
                    self.hpo_do_sample_weighting = True
                elif arg == 'False':
                    self.hpo_do_sample_weighting = False

            elif opt == 'Path_to_dataset_file':
                self.path_to_dataset = arg

            elif opt == 'Path_to_structman_features_file':
                self.path_structural_feature_table = arg

            elif opt == 'Path_to_sequences_file':
                self.path_to_sequence_fasta = arg

            elif opt == 'dataset_name':
                self.dataset_name = arg

            elif opt == 'path_to_features_file':
                self.path_to_features_file = arg

            elif opt == 'path_to_processed_features_file':
                self.path_to_processed_features_file = arg

            elif opt == 'outfolder':
                self.outfolder = arg

            elif opt == 'path_to_imputed_features_file':
                self.path_to_imputed_features_file = arg

            elif opt == 'path_to_impute_map':
                self.path_to_impute_map = arg

            elif opt == 'Path_to_multi_savs_table':
                self.path_to_multi_savs_table = arg

        if self.mmseqs_search_db_ref50 != '':
            self.search_dbs.append('ref50')
        if self.mmseqs_search_db_ref90 != '':
            self.search_dbs.append('ref90')
        if self.mmseqs_search_db_ref100 != '':
            self.search_dbs.append('ref100')

        self.regression = (self.class_labels is None)
        if not self.regression:
            self.objective_function = 'MCC'
            self.criterion = 'gini'
            self.criteria = ['gini','entropy']
        else:
            self.objective_function = 'Mean Spearman'#'Pearson'#'Spearman'#'MSE'
            self.criterion = 'friedman_mse'
            self.criteria = ['friedman_mse']#,'mae']

        if overwrite_objective_function != None:
            self.objective_function = overwrite_objective_function

        #Hyperparameters setup if an HP file is provided
        self.hyperparameters_path = hyperparameters_path
        if hyperparameters_path is not None:
            try:
                f_hp = open(hyperparameters_path, 'r')
                lines_hp = f_hp.read().split('\n')
                f_hp.close()
            except:
                print(f'Error trying to read HP file: {hyperparameters_path}')
                lines_hp = []

            for line in lines_hp:
                if len(line) == 0:
                    continue
                if line[0] == '#':
                    continue
                
                if line.count('=') == 1:
                    words = line.split('=')
                else:
                    words = line.split()
                # CHeck the 'param = value' format
                if len(words) != 2:
                    continue
                
                # Set hyperparameter value
                opt = words[0].strip()
                arg = words[1].replace("\n","").strip()

                if opt == 'tree_depth':
                    self.tree_depth = int(arg)
                    continue
                if opt == 'min_sample_split':
                    self.min_sample_split = int(arg)
                    continue
                if opt == 'tree_min_leaf_samples':
                    self.tree_min_leaf_samples = int(arg)
                    continue
                if opt == 'num_of_trees':
                    self.num_of_trees = int(arg)
                    continue
                # if opt == 'rel_max_leaf_node_pruning':
                #     self.rel_max_leaf_node_pruning = None
                #     continue
                if opt == 'max_leaf_nodes':
                    try:
                        self.max_leaf_nodes = int(arg)
                    except:
                        pass
                    continue
                if opt == 'class_weight':
                    if arg == "None":
                        self.class_weight = None
                    elif arg in ('balanced','balanced_subsample'):
                        self.class_weight = arg
                    continue
                if opt == 'max_feature_parameter':
                    if arg in ('auto','log2','sqrt'):
                        self.max_feature_parameter = arg
                    continue
                if opt == 'max_feature_cont_parameter':
                    flarg = float(arg)
                    if flarg >= 0 and flarg <= 1:
                        self.max_feature_cont_parameter = flarg
                    continue
                if opt == 'bootstrap_parameter':
                    if arg == "True":
                        self.bootstrap_parameter = True
                    if arg == "False":
                        self.bootstrap_parameter = False
                    continue
                if opt == 'min_impurity_decrease_exp':
                    self.min_impurity_decrease_exp = float(arg)
                    continue
                if opt == 'oob_score':
                    if arg == "True":
                        self.oob_score = True
                    if arg == "False":
                        self.oob_score = False
                    continue
                if opt == 'ccp_alpha_exp':
                    self.ccp_alpha_exp = float(arg)
                    continue
                if opt == 'max_sample_parameter':
                    flarg = float(arg)
                    if flarg >= 0 and flarg <= 1:
                        self.max_sample_parameter = flarg
                    continue
                if opt == 'number_of_bins':
                    self.number_of_bins = int(arg)
                    continue
                if opt == 'p_val_thresh':
                    self.p_val_thresh = float(arg)
                    continue
                if opt == 'sample_weight_parameter':
                    self.sample_weight_parameter = float(arg)
                    continue
                if opt == 'reg_alpha_exp':
                    self.reg_alpha_exp = float(arg)
                    continue
                if opt == 'reg_c_exp':
                    self.reg_c_exp = float(arg)
                    continue
                if opt == 'reg_thresh_exp':
                    self.reg_thresh_exp = float(arg)
                    continue
                if opt == 'confusion_goodwill':
                    flarg = float(arg)
                    if flarg >= 0 and flarg <= 1:
                        self.confusion_goodwill = flarg
                    continue
                if opt == 'list_ranking_thresh':
                    self.list_ranking_thresh = int(arg)
                    continue
                if opt == 'sequential_confusion_rank_threshold':
                    self.sequential_confusion_rank_threshold = int(arg)
                    continue
                if opt == 'confusion_rank_threshold':
                    self.confusion_rank_threshold = int(arg)
                    continue
                if opt == 'err_warping_exp':
                    self.err_warping_exp = float(arg)
                    continue
                if opt == 'confusion_normalization_exp':
                    self.confusion_normalization_exp = float(arg)
                    continue

        #self.blacklist = ['P28482']#set(['P28482','P42212','P38398','P06654','Q9UK59','P04386','P00552'])
    
        #self.prot_based_separation = True
        #self.filterStructuralFeatures = True
        #self.addBias = False

        #self.balanceSubsampling = 'balanced'
        
        #self.structure_threshold = 1
        #self.fusePositions = False
        #self.color_structures = True
        #self.hyperOptimization=True
        #self.transform = False

        #self.filter_single_variant_prots = True
        #self.remove_t2 = True

    def setByString(self,parameter_name,value):
        attributs = vars(self)
        attribute_types = {attribute_name: type(attributs[attribute_name]) for attribute_name in attributs}

        if parameter_name in attribute_types:
            target_type = attribute_types[parameter_name]
            if isinstance(value, target_type) or isinstance(value, type(None)) or (value == None):
                setattr(self, parameter_name, value)
            elif isinstance(value, float) and target_type is int:
                setattr(self, parameter_name, round(value))
            elif str(target_type).count('float') > 0 and isinstance(value, float):
                setattr(self, parameter_name, value)
            else:
                raise TypeError(f"Value for {parameter_name} should be typed as a {target_type}, but given was: {value} ({type(value)})")
        else:
            raise ValueError(f"Parameter {parameter_name} not known.")

    def getByString(self, parameter_name):
        return getattr(self, parameter_name, None)

    def getScoreTuple(self):
        sct = (self.tree_depth,self.min_sample_split,self.num_of_trees,self.class_weight,
                self.max_feature_parameter, self.max_feature_cont_parameter, self.bootstrap_parameter,self.min_impurity_decrease_exp,
                self.oob_score,self.ccp_alpha_exp,self.max_sample_parameter, self.criterion,
                self.tree_min_leaf_samples,self.tvmb_rank_threshold,self.tvpmb_rank_threshold, self.confusion_rank_threshold,
                self.sequential_confusion_rank_threshold, self.err_warping_exp, self.confusion_normalization_exp,
                self.number_of_bins, self.list_ranking_thresh, self.confusion_goodwill,
                self.p_val_thresh,self.sample_weight_parameter,self.reg_alpha_exp, self.reg_c_exp, self.reg_thresh_exp, self.geometric_exponent)
        return sct

    def printParameter(self):
        print('Depth:',self.tree_depth)
        print('Min sample split:',self.min_sample_split)
        print('Min sample leaf:',self.tree_min_leaf_samples)
        print('Forest size:',self.num_of_trees)
        print('Class weight:',self.class_weight)
        print('Max features:',self.max_feature_parameter)
        print('Max features continuous:', self.max_feature_cont_parameter)
        print('Bootstrap:',self.bootstrap_parameter)
        print('Min impurity decrease exponent:',self.min_impurity_decrease_exp)
        print('Out of bag:',self.oob_score)
        print('CCP alpha exponent:',self.ccp_alpha_exp)

        print('Max sample:',self.max_sample_parameter)

        print('TVMB rank threshold:',self.tvmb_rank_threshold)
        print('TVPMB rank threshold:',self.tvpmb_rank_threshold)
        print('Confusion rank threshold:', self.confusion_rank_threshold)
        print('Sequential confusion rank threshold:', self.sequential_confusion_rank_threshold)
        print('Confusion error warping exponent:', self.err_warping_exp)
        print('Confusion normalization exponent:', self.confusion_normalization_exp)
        print('Criterion:',self.criterion)
        print('Number of bins:',self.number_of_bins)
        print('p-value threshold:',self.p_val_thresh)
        print('Sample weight parameter:',self.sample_weight_parameter)
        print('Regularization alpha exponent:', self.reg_alpha_exp)
        print('Regularization C exponent:', self.reg_c_exp)
        print('Regularization threshold exponent:', self.reg_thresh_exp)
        print('Geometric exponent:', self.geometric_exponent)
        print(f'Confusion goodwill: {self.confusion_goodwill}')
        print(f'List ranking thresh: {self.list_ranking_thresh}')
        return

    def printHyperParameter(self):
        print("tree_depth", self.tree_depth)
        print("min_sample_split", self.min_sample_split)
        print("tree_min_leaf_samples", self.tree_min_leaf_samples)
        print("num_of_trees", self.num_of_trees)
        print("max_leaf_nodes", self.max_leaf_nodes)
        print("class_weight", self.class_weight)
        print("max_feature_parameter", self.max_feature_parameter)
        print("max_feature_cont_parameter", self.max_feature_cont_parameter)
        print("bootstrap_parameter", self.bootstrap_parameter)
        print("min_impurity_decrease_exp", self.min_impurity_decrease_exp)
        print("oob_score", self.oob_score)
        print("ccp_alpha_exp", self.ccp_alpha_exp)
        print("max_sample_parameter", self.max_sample_parameter)
        print("number_of_bins", self.number_of_bins)
        print("p_val_thresh", self.p_val_thresh)
        print("sample_weight_parameter", self.sample_weight_parameter)
        print("reg_alpha_exp", self.reg_alpha_exp)
        print("reg_c_exp", self.reg_c_exp)
        print("reg_thresh_exp", self.reg_thresh_exp)
        print("confusion_goodwill", self.confusion_goodwill)
        print("list_ranking_thresh", self.list_ranking_thresh)
        print("sequential_confusion_rank_threshold", self.sequential_confusion_rank_threshold)
        print("confusion_rank_threshold", self.confusion_rank_threshold)
        print("err_warping_exp", self.err_warping_exp)
        print("confusion_normalization_exp", self.confusion_normalization_exp)
        return

    def saveHyperParameter(self, outputFileName = None):
        if outputFileName is None:
            outputFileName = self.hp_outputFile

        capture = OutputCapture()
        sys.stdout = capture  # Redirige la sortie standard vers la variable capture
        self.printHyperParameter()
        sys.stdout = sys.__stdout__  # Restaure la sortie standard

        capture.replace(" ", "=")


        hp_file_outpath = f"{self.outfolder}/./{outputFileName}"
        print(f'Saving HP file to {hp_file_outpath}')

        with open(hp_file_outpath, "w") as hpfo:
            print(capture, file = hpfo)
            return

    def add_entry_to_project_file(self, field, value):
        new_lines = []
        
        opt_args = parse_conf(self.path_to_project_file)
        overwritten = False

        for opt, arg in opt_args:
            if opt == field:
                new_lines.append(f'{opt} = {value}\n')
                overwritten = True
            else:
                new_lines.append(f'{opt} = {arg}\n')

        if not overwritten:
            new_lines.append(f'{field} = {value}\n')

        f = open(self.path_to_project_file, 'w')
        f.write(''.join(new_lines))
        f.close()


def tags_to_effect(config, tags):
    if config.target_values is not None:
        target_values = []
        for tag in tags.split(','):
            if tag == '':
                continue
            if tag[0] != '#':
                continue
            try:
                tag_id, tag_value = tag.split(':')
            except:
                try:
                    tag_id, tag_value = tag.split('=')
                except:
                    continue
            if tag_id != config.target_values:
                continue
            try:
                tag_value = float(tag_value)
            except:
                print(f'Given effect tag had non-float tag value: {tag_value}')
                continue
            target_values.append(tag_value)
        if len(target_values) == 0:
            target_value = None
        else:
            target_value = sum(target_values)/len(target_values)
    else:
        target_value = None
        for tag in tags.split(','):
            if tag[:6] == 'label:':
                target_value = tag[6:]

    return target_value

def parse_multi_savs_table(config):
    f = open(config.path_to_multi_savs_table, 'r')
    lines = f.readlines()
    f.close()

    multi_savs = []

    for line in lines[1:]:
        words = line[:-1].split('\t')
        prot_id = words[0]
        aacs = words[1].split(':')
        tags = words[2]
        effect = tags_to_effect(config, tags)
        multi_savs.append((prot_id, aacs, effect))

    return multi_savs

def combine_individual_effects(individual_effects, multiply = True):
    if len(individual_effects) == 1:
        return individual_effects[0]
    if multiply:
        individual_effects = sorted(individual_effects)[:2]
        c = max([individual_effects[0], 0.])
        for e in individual_effects[1:]:
            em = max([e, 0.])
            c = c*em
    else:
        c = sum(individual_effects) + 1.0 - float(len(individual_effects))

    return c

class Scores:
    __slots__ = [
                    'mse', 'wmse', 'r2', 'wr2', 'corr', 'acc', 'roc', 'precision', 'recall', 'f1',
                    'mcc', 'pearson_r', 'n_of_features', 'mean_spearman', 'mean_pearson', 'feature_penalty'
                ]
    def __init__(
                    self, mse = None, r2 = None, corr = None, acc = None, roc = None, precision = None,
                    recall = None, f1 = None, mcc = None, pearson_r = None, zero = False, wmse = None,
                    wr2 = None, optimal = False, n_of_features = None, mean_spearman = None, 
                    mean_pearson = None, feature_penalty = None
                ):
        self.n_of_features = n_of_features
        self.feature_penalty = feature_penalty
        if zero:
            self.mse = float('inf')
            self.wmse = float('inf')
            self.r2 = -1.0
            self.wr2 = -1.0
            self.corr = 0.0
            self.acc = 0.0
            self.roc = 0.0
            self.precision = 0.0
            self.recall = 0.0
            self.f1 = 0.0
            self.mcc = -1.0
            self.pearson_r = 0.0
            self.mean_spearman = 0.0
            self.mean_pearson = 0.0
            return
        elif optimal:
            self.mse = 0.
            self.wmse = 0.
            self.r2 = 1.0
            self.wr2 = 1.0
            self.corr = 1.0
            self.acc = 1.0
            self.roc = 1.0
            self.precision = 1.0
            self.recall = 1.0
            self.f1 = 1.0
            self.mcc = 1.0
            self.pearson_r = 1.0
            self.mean_spearman = 1.0
            self.mean_pearson = 1.0
            return
        self.mse = mse
        self.wmse = wmse
        self.r2 = r2
        self.wr2 = wr2
        self.corr = corr
        self.acc = acc
        self.roc = roc
        self.precision = precision
        self.recall = recall
        self.f1 = f1
        self.mcc = mcc
        self.pearson_r = pearson_r
        self.mean_spearman = mean_spearman
        self.mean_pearson = mean_pearson
        return

    def printOut(self):
        print('------------Scores------------')
        print(f'Generated for {self.n_of_features} number of features')
        if self.mse != None:
            print('-MSE:',self.mse)
        if self.wmse != None:
            print('-weighted MSE:',self.wmse)
        if self.r2 != None:
            print('-R2:',self.r2)
        if self.wr2 != None:
            print('-weighted R2:',self.wr2)
        if self.corr != None:
            print('-Spearmans Correlation:',self.corr)
        if self.mean_spearman is not None:
            print(f'-Mean Protein-Wise Spearmans Corr: {self.mean_spearman}')
        if self.pearson_r != None:
            print('-Pearsons Correlation:',self.pearson_r)
        if self.mean_pearson is not None:
            print(f"-Mean Protein-Wise Pearsons Corr: {self.mean_pearson}")
        if self.acc != None:
            print('-Accuracy:',self.acc)
        if self.roc != None:
            print('-auROC:',self.roc)
        if self.precision != None:
            print('-Precision:',self.precision)
        if self.recall != None:
            print('-Recall:',self.recall)
        if self.f1 != None:
            print('-F1:',self.f1)
        if self.mcc != None:
            print('-MCC:',self.mcc)
        if self.feature_penalty != None:
            print(f'Feature penalty term: {self.feature_penalty}')

        print('------------------------------')
        return

    def objective_value(self,config):
        if config.objective_function == 'MCC':
            return self.mcc
        if config.objective_function == 'MSE':
            return self.mse
        if config.objective_function == 'F-Score':
            return self.f1
        if config.objective_function == 'Accuracy':
            return self.acc
        if config.objective_function == 'Spearman':
            return self.corr
        if config.objective_function == 'Pearson':
            return self.pearson_r
        if config.objective_function == 'Mean Spearman':
            return self.mean_spearman
        if config.objective_function == 'Mean Pearson':
            return self.mean_pearson
        if config.objective_function == 'R2':
            return self.r2
        if config.objective_function == 'auROC':
            return self.roc


def calc_protein_wise_corr(y_test, y_pred, sample_ids, corr_function, mono_return_score_function = False):
    test_pred_pairs = {}
    for sample_nr, yt_value in enumerate(y_test):
        prot_id, _ = sample_ids[sample_nr]
        if not prot_id in test_pred_pairs:
            test_pred_pairs[prot_id] = [], []
        test_pred_pairs[prot_id][0].append(yt_value)
        test_pred_pairs[prot_id][1].append(y_pred[sample_nr])

    prot_wise_corrs = []
    corrs = []
    for prot_id in test_pred_pairs:
        if not mono_return_score_function:
            corr, _ = corr_function(test_pred_pairs[prot_id][0], test_pred_pairs[prot_id][1])
        else:
            corr = corr_function(test_pred_pairs[prot_id][0], test_pred_pairs[prot_id][1])
        corr = abs(corr)
        prot_wise_corrs.append((prot_id, corr))
        corrs.append(corr)
    if len(corrs) > 0:
        mean_corr = sum(corrs)/len(corrs)
    else:
        mean_corr = 0
    return prot_wise_corrs, mean_corr

def objective_function_criterium(config, scores, best_scores, feature_penalty = None):
    if best_scores is None:
        if scores is None:
            return False
        return True
    if scores is None:
        return False

    return get_objective_score(config, scores, feature_penalty = feature_penalty) > get_objective_score(config, best_scores, feature_penalty = feature_penalty)

def get_objective_score(config, scores, feature_penalty = None):
    if scores is None:
        return False

    greater_is_better = True

    if config.objective_function == 'MCC':
        score = scores.mcc
    elif config.objective_function == 'MSE':
        score = scores.mse
        greater_is_better = False
    elif config.objective_function == 'F-Score':
        score = scores.f1
    elif config.objective_function == 'Accuracy':
        score = scores.acc
    elif config.objective_function == 'Spearman':
        score = scores.corr
    elif config.objective_function == 'Mean Spearman':
        score = scores.mean_spearman
    elif config.objective_function == 'Mean Pearson':
        score = scores.mean_pearson
    elif config.objective_function == 'Pearson':
        score = scores.pearson_r
    elif config.objective_function == 'R2':
        score = scores.r2
    elif config.objective_function == 'auROC':
        score = scores.roc
    else:
        raise 'Unknown objective function'

    if feature_penalty is not None and scores.n_of_features is not None and score is not None:
        scores.feature_penalty = scores.n_of_features*feature_penalty
        if greater_is_better:
            score -= scores.feature_penalty
        else:
            score += scores.feature_penalty

    return score


def objective_function_delta(config,scores,scores_2):
    if scores is None:
        if scores_2 is None:
            return None
        return scores_2.objective_value(config)
    if scores_2 is None:
        return scores.objective_value(config)

    if config.objective_function == 'MCC':
        return abs(scores.mcc-scores_2.mcc)
    if config.objective_function == 'MSE':
        return abs(scores.mse-scores_2.mse)
    if config.objective_function == 'F-Score':
        return abs(scores.f1-scores_2.f1)
    if config.objective_function == 'Accuracy':
        return abs(scores.acc-scores_2.acc)
    if config.objective_function == 'Spearman':
        return abs(scores.corr-scores_2.corr)
    if config.objective_function == 'Mean Spearman':
        return abs(scores.mean_spearman-scores_2.mean_spearman)
    if config.objective_function == 'Pearson':
        return abs(scores.pearson_r-scores_2.pearson_r)
    if config.objective_function == 'Mean Pearson':
        return abs(scores.mean_pearson-scores_2.mean_pearson)
    if config.objective_function == 'R2':
        return abs(scores.r2-scores_2.r2)
    if config.objective_function == 'auROC':
        return abs(scores.roc-scores_2.roc)

def mean_scores(scores_list):
    def mean(l):
        if len(l) == 0:
            return None
        return sum(l)/len(l)
    mses = []
    r2s = []
    corrs = []
    mean_spearmans = []
    accs = []
    rocs = []
    precisions = []
    recalls = []
    f1s = []
    mccs = []
    pearson_rs = []
    mean_pearsons = []
    n_of_features_s = []

    for scores_obj in scores_list:
        if scores_obj.mse != None:
            mses.append(scores_obj.mse)
        if scores_obj.r2 != None:
            r2s.append(scores_obj.r2)
        if scores_obj.corr != None:
            corrs.append(scores_obj.corr)
        if scores_obj.mean_spearman is not None:
            mean_spearmans.append(scores_obj.mean_spearman)
        if scores_obj.pearson_r != None:
            pearson_rs.append(scores_obj.pearson_r)
        if scores_obj.mean_pearson is not None:
            mean_pearsons.append(scores_obj.mean_pearson)
        if scores_obj.acc != None:
            accs.append(scores_obj.acc)
        if scores_obj.roc != None:
            rocs.append(scores_obj.roc)
        if scores_obj.precision != None:
            precisions.append(scores_obj.precision)
        if scores_obj.recall != None:
            recalls.append(scores_obj.recall)
        if scores_obj.f1 != None:
            f1s.append(scores_obj.f1)
        if scores_obj.mcc != None:
            mccs.append(scores_obj.mcc)
        if scores_obj.n_of_features is not None:
            n_of_features_s.append(scores_obj.n_of_features)

    if len(n_of_features_s) == 0:
        n_of_features = None
    else:
        n_of_features = mean(n_of_features_s)

    scores_obj = Scores(mse = mean(mses), r2 = mean(r2s), corr = mean(corrs), acc = mean(accs), roc = mean(rocs),
                        precision = mean(precisions), recall = mean(recalls), f1 = mean(f1s), mcc = mean(mccs),
                        pearson_r = mean(pearson_rs), mean_pearson = mean(mean_pearsons), mean_spearman = mean(mean_spearmans),
                        n_of_features = n_of_features)
    return scores_obj

def writeOutput(outfile,feature_names,feature_matrix,id_vector,reg_vector,seq_id_vector,prediction,test_ids,sub_file_id=None):
    outlines = ['Uniprot Ac\tAAC\tobserved value\tpredicted value\tmax seq id\t%s' % '\t'.join(feature_names)]
    test_ids = sorted(test_ids)
    for pred_pos,pred_value in enumerate(prediction):
        pos = test_ids[pred_pos]

        feature_vector = feature_matrix[pos]
        u_ac,aac,tags = id_vector[pos]
        observed_value = str(reg_vector[pos])
        max_seq = str(seq_id_vector[pos])
        outlines.append('%s\t%s\t%s\t%s\t%s\t%s' % (u_ac,aac,observed_value,str(pred_value),max_seq,'\t'.join([str(x) for x in feature_vector])))

    base_name,f_type = outfile.rsplit('.',1)
    if not os.path.isdir(base_name):
        os.mkdir(base_name)
    outfile = '%s/%s' % (base_name,outfile)

    if sub_file_id != None:
        if sub_file_id in class_order:
            sub_file_id = class_order[sub_file_id]
        outfile = '%s/%s_%s.%s' % (base_name,base_name,sub_file_id,f_type)

    f = open(outfile,'w')
    f.write('\n'.join(outlines))
    f.close()


def hexbinplot(prediction_values,true_values,target_value_name,outfile):
    fig = plt.figure()
    ax = plt.subplot()

    #plt.legend(loc="upper left",fontsize = 12,framealpha=1,fancybox=True)
    
    min_pred = min(prediction_values)
    max_pred = max(prediction_values)
    min_value = min(true_values)
    max_value = max(true_values)

    hb = ax.hexbin(true_values,prediction_values, gridsize=50, cmap='inferno', bins = 'log')
    ax.axis([min_value, max_value, min_pred, max_pred])
    plt.xlabel(target_value_name, fontsize=15)

    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('log10(counts)')
    plt.ylabel('Predicted value', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(outfile,dpi=300)
    plt.clf()

def radar(labels,values,title,outfile):
    N = len(values)

    tupl_list = list(zip(labels,values))
    tupl_list.sort(key= lambda x:x[1])
    labels = [x[0] for x in tupl_list]
    values = [x[1] for x in tupl_list]

    theta = radarplot.radar_factory(N, frame = 'polygon')
    fig, ax = plt.subplots(figsize=(9, 9), nrows=1, ncols=1,
                             subplot_kw=dict(projection='radar'))
    fig.subplots_adjust(wspace=0.25, hspace=0.20, top=0.85, bottom=0.05)

    #ax = axes[0]
    ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
    ax.set_title(title, weight='bold', size='medium', position=(0.5, 1.1),
                 horizontalalignment='center', verticalalignment='center')

    ax.plot(theta, values, color='g')
    ax.fill(theta, values, facecolor='g', alpha=0.25)
    ax.set_varlabels(labels)

    # add legend relative to top-left plot
    #ax = axes[0, 0]
    #legend = ax.legend(labels, loc=(0.9, .95),
    #                   labelspacing=0.1, fontsize='small')

    fig.text(0.5, 0.965, 'Radarplot',
             horizontalalignment='center', color='black', weight='bold',
             size='large')
    plt.tight_layout()
    plt.savefig(outfile,dpi=300)
    plt.clf()

def scatterplot(prediction_values,feature_matrix,feature_names,true_values,target_value_name,bal_point,threshold,outfile,feature_highlight=None,subs_map={}):

    highlight_pos = None
    if feature_highlight != None:
        for pos,feature_name in enumerate(feature_names):
            if feature_name == feature_highlight:
                highlight_pos = pos

                print('Highlight feauture: ',feature_name)

    if subs_map != {}:
        subs_backmap = {int(y):int(x) for x,y in subs_map.items()}
    else:
        subs_backmap = None

    if highlight_pos != None:

        highlight_map = {}

        for pos,predicted_value in enumerate(prediction_values):
            highlight = feature_matrix[pos][highlight_pos]

            if subs_backmap != None:
                highlight = subs_backmap[highlight]
            if feature_highlight.count('Acid') > 0:
                highlight = aa_order[highlight]

            true_value = true_values[pos]

            if not highlight in highlight_map:
                highlight_map[highlight] = {}
                highlight_map[highlight]['TN'] = []
                highlight_map[highlight]['FN'] = []
                highlight_map[highlight]['FP'] = []
                highlight_map[highlight]['TP'] = []

            if true_value < threshold and predicted_value < bal_point:
                highlight_map[highlight]['TP'].append([true_value,predicted_value])
            elif true_value < threshold and predicted_value >= bal_point:
                highlight_map[highlight]['FP'].append([true_value,predicted_value])
            elif true_value >= threshold and predicted_value >= bal_point:
                highlight_map[highlight]['TN'].append([true_value,predicted_value])
            elif true_value >= threshold and predicted_value < bal_point:
                highlight_map[highlight]['FN'].append([true_value,predicted_value])



        ax = plt.subplot()

        for highlight in highlight_map:
            TN = highlight_map[highlight]['TN']
            FN = highlight_map[highlight]['FN']
            FP = highlight_map[highlight]['FP']
            TP = highlight_map[highlight]['TP']

            for x,y in TP:
                plt.text(x,y,str(highlight),alpha=0.5,color='darkgreen')
            for x,y in FP:
                plt.text(x,y,str(highlight),alpha=0.5,color='navy')
            for x,y in TN:
                plt.text(x,y,str(highlight),alpha=0.5,color='green')
            for x,y in FN:
                plt.text(x,y,str(highlight),alpha=0.5,color='blue')

        #plt.scatter([],[],alpha=1,color='darkgreen',label='TP: %s' % str(len(TP[0])))
        #plt.scatter([],[],alpha=1,color='navy',label='FP: %s' % str(len(FP[0])))
        #plt.scatter([],[],alpha=1,color='green',label='TN: %s' % str(len(TN[0])))
        #plt.scatter([],[],alpha=1,color='blue',label='FN: %s' % str(len(FN[0])))

    else:

        TN = [[],[]]
        FN = [[],[]]
        FP = [[],[]]
        TP = [[],[]]

        for pos,predicted_value in enumerate(prediction_values):
            true_value = true_values[pos]

            if true_value < threshold and predicted_value < bal_point:
                TP[0].append(true_value)
                TP[1].append(predicted_value)
            elif true_value < threshold and predicted_value >= bal_point:
                FP[0].append(true_value)
                FP[1].append(predicted_value)
            elif true_value >= threshold and predicted_value >= bal_point:
                TN[0].append(true_value)
                TN[1].append(predicted_value)
            elif true_value >= threshold and predicted_value < bal_point:
                FN[0].append(true_value)
                FN[1].append(predicted_value)



        ax = plt.subplot()

        plt.scatter(TP[0],TP[1],alpha=0.5,color='darkgreen')
        plt.scatter(FP[0],FP[1],alpha=0.5,color='navy')
        plt.scatter(TN[0],TN[1],alpha=0.5,color='green')
        plt.scatter(FN[0],FN[1],alpha=0.5,color='blue')

        plt.scatter([],[],alpha=1,color='darkgreen',label='TP: %s' % str(len(TP[0])))
        plt.scatter([],[],alpha=1,color='navy',label='FP: %s' % str(len(FP[0])))
        plt.scatter([],[],alpha=1,color='green',label='TN: %s' % str(len(TN[0])))
        plt.scatter([],[],alpha=1,color='blue',label='FN: %s' % str(len(FN[0])))
    plt.legend(loc="upper left",fontsize = 12,framealpha=1,fancybox=True)
    
    min_pred = min(prediction_values)
    max_pred = max(prediction_values)
    min_value = min(true_values)
    max_value = max(true_values)

    #print min_pred,max_pred,min_value,max_value

    plt.plot([threshold, threshold],[min_pred, max_pred] , color='red',lw=2)
    plt.plot([min_value, max_value], [ bal_point, bal_point], color='red',lw=2)
    


    plt.xlabel(target_value_name, fontsize=15)

    ax.yaxis.tick_left()
    ax.yaxis.set_label_position('left')

    plt.ylabel('Predicted value', fontsize=15)
    
    plt.tight_layout()
    plt.savefig(outfile,dpi=300)
    plt.clf()

def plotMPP(config,indatafile,outfile):
    if config.regression:
        return

    f = open(indatafile,'r')
    lines = f.readlines()
    f.close()

    prot_dist = {}
    target_value_pos_map = {}
    for pos,tv in enumerate(config.target_values):
        target_value_pos_map[tv] = pos

    for line in lines:
        words = line.replace('\t',' ').split()
        u_ac = words[0]
        aac = words[1]
        if len(words) > 2:
            tags = words[2]
        else:
            tags = None
        tv = parseTVfromTags(config,tags)
        if not u_ac in prot_dist:
            prot_dist[u_ac] = [0]*len(config.target_values)
        prot_dist[u_ac][target_value_pos_map[tv]] += 1

    histo = {}
    y_max = 0
    max_N = 41
    print('Total number of proteins:',len(prot_dist))
    n_of_pure_pathogenic = 0
    n_of_pure_benign = 0
    n_of_majorly_pathogenic = 0
    n_of_majorly_benign = 0
    n_of_mixed = 0
    for u_ac in prot_dist:
        N = sum(prot_dist[u_ac])
        if N >= max_N:
            N = max_N
        if not N in histo:
            histo[N] = [0] + ([0]*len(prot_dist[u_ac]))
        histo[N][0] += 1
        if histo[N][0] > y_max:
            y_max = histo[N][0]
        for pos,tv_N in enumerate(prot_dist[u_ac]):
            histo[N][pos+1] += tv_N
        if prot_dist[u_ac][0] == 0:
            n_of_pure_pathogenic += 1
        elif prot_dist[u_ac][1] == 0:
            n_of_pure_benign += 1
        else:
            n_of_mixed += 1
        if prot_dist[u_ac][0] >= prot_dist[u_ac][1] * 4:
            n_of_majorly_benign += 1
        if prot_dist[u_ac][1] >= prot_dist[u_ac][0] *4:
            n_of_majorly_pathogenic += 1

    print('Number of pure pathogenic proteins:',n_of_pure_pathogenic)
    print('Number of pure benign proteins:',n_of_pure_benign)
    print('Number of mixed proteins:',n_of_mixed)

    print('Number of majorly pathogenic proteins:',n_of_majorly_pathogenic)
    print('Number of majorly benign proteins:',n_of_majorly_benign)

    color_map = ['orange','green','cyan','purple']
    title = 'Mutations per protein distribution'
    x_labels = []

    fontsize = 18
    small_fontsize = 12

    ind = np.arange(len(histo))    # the x locations for the groups
    width = 0.8 #1.0/float(N)       # the width of the bars: can also be len(x) sequence

    fig = plt.figure(figsize=(0.5*len(histo),9.))
    ax = plt.subplot(111)

    bars = []
    new_bar = True
    plots = []


    for n in range(1,max_N + 1):
        if not n in histo:
            continue
        x_label = n
        x_labels.append(str(x_label))

        total_height = histo[n][0]
        sub_bar_heights = []
        tv_n_sum = sum(histo[n][1:])
        for tv_n in histo[n][1:]:
            sub_bar_heights.append(total_height*tv_n/tv_n_sum)

        last_bar_small = False
        last_bar_ha = 'center'
        for pos,sub_bar_height in enumerate(sub_bar_heights):
            
            value = sub_bar_height
            col = color_map[pos]
            if new_bar:
                p = ax.bar([n-1+width*0.5], value, width,color = col)
                cumulated_values = value
                new_bar = False
            else:
                p = ax.bar([n-1+width*0.5], value, width,bottom=cumulated_values,color = col)
                cumulated_values += value
            '''
            if value[0] > 0.045*y_max:
                plt.text(n+width*0.5,cumulated_values[0]-value[0]*0.5,'%.2f%%' % (value[0]*100.0),ha="center", va="center", color="white", fontsize=fontsize)
                last_bar_small = False
                last_bar_ha = 'center'
            elif hide_small_numbers:
                pass
            elif center_small_sumbers:
                plt.text(n+width*0.5,cumulated_values[0]-value[0]*0.5,'%.2f%%' % (value[0]*100.0),ha="center", va="center", color="white", fontsize=small_fontsize)
            elif not last_bar_small or last_bar_ha == 'right':
                plt.text(n+width*0.5,cumulated_values[0]-value[0]*0.5,'%.2f%%' % (value[0]*100.0),ha="center", va="center", color="white", fontsize=small_fontsize)
                last_bar_small = True
                last_bar_ha = 'center'
            elif last_bar_ha == 'center':
                plt.text(n,cumulated_values[0]-value[0]*0.5,'%.2f%%' % (value[0]*100.0),ha="left", va="center", color="white", fontsize=small_fontsize)
                last_bar_small = True
                last_bar_ha = 'left'
            elif last_bar_ha == 'left':
                plt.text(n+width,cumulated_values[0]-value[0]*0.5,'%.2f%%' % (value[0]*100.0),ha="right", va="center", color="white", fontsize=small_fontsize)
                last_bar_small = True
                last_bar_ha = 'right'
            '''
            plots.append(p)
        new_bar = True

    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.6, box.height])

    plt.ylabel('number of proteins',fontsize=fontsize)
    plt.xlabel('number of mutations per protein',fontsize=fontsize)
    plt.title(title,fontsize=fontsize)
    x_labels[-1] = '>=%s' % str(max_N)
    plt.xticks([x+(width/2.0) for x in ind],['\n'.join(x.split(' ')) for x in x_labels],fontsize=small_fontsize)
    plt.yticks(np.arange(0, y_max+1, y_max//10),["%s" % str(x) for x in np.arange(0, y_max+1, y_max//10)],fontsize=fontsize)

    plt.xlim((0,len(histo)-1+width))
    plt.ylim((0,y_max))
    '''
    if not hide_legend:
        ax.legend([p[0] for p in reversed(plots)], classifications,loc = 7,fontsize=fontsize,bbox_to_anchor = (1.55,0.5))#(2.-N*0.15,0.5)) #good x-values: N=1: 2.7 N=2: 1.85, N=3: 1.55, N=6: 1.275
    '''
    #plt.show()

    plt.savefig(outfile,bbox_inches='tight')
    return


def get_msa_path(out_directory, prot_id, ref_db_id, gpw = False, psic = False, unpacked = False):
    prot_id = prot_id.replace('/','_')
    prot_id = prot_id.replace(',','_')

    if gpw:
        gpw_extension = '_gpw'
    else:
        gpw_extension = ''

    if psic:
        file_type = 'psic'
    else:
        file_type = 'fasta'

    if unpacked:
        gzip_extension = ''
    else:
        gzip_extension = '.gz'

    filename = f'{out_directory}/{prot_id}_{ref_db_id}{gpw_extension}.{file_type}{gzip_extension}'

    filename = filename.replace('(','_')
    filename = filename.replace(')','_')

    return filename

def parseTVfromTags(config,tags):
    for tag in tags.split(','):
        if config.regression:
            parts = tag.split(':')
            if len(parts) < 2:
                continue
            if parts[0] != config.target_values:
                continue
            target_value = float(parts[1])

        else:
            if not tag in config.target_values:
                continue
            target_value = tag
    return target_value

def addReverseMutations(config,indatafile,outfile):
    if config.regression:
        f = open(indatafile,'r')
        lines = f.readlines()
        f.close()
        new_lines = []
        for line in lines:
            if line == '':
                continue
            words = line.split()
            u_ac = words[0]
            aac = words[1]
            tags = words[2]
            target_value = parseTVfromTags(config,tags)
            reverse_aac = '%s%s%s' % (aac[-1],aac[1:-1],aac[0])
            reverse_target_value = config.binary_thresh - target_value

            new_lines.append(line)
            new_lines.append('%s\t%s\t%s:%s\n' % (u_ac,reverse_aac,config.target_values,str(reverse_target_value)))

        f = open(outfile,'w')
        f.write(''.join(new_lines))
        f.close()
    return

def parseFeatureList(featureFile):
    f = open(featureFile,'r')
    lines = f.readlines()
    f.close()

    feature_list = []
    for line in lines:
        if line == '':
            continue
        feature_list.append(line.strip())
    return feature_list


#Needs to be updated
"""
def calculateCorrelationMatrix(config,indatafile,outfile,list_of_features,addTargetValue=False):
    samples = learn.createTrainingSet(config,config.session,infile=indatafile,debug=config.debug)
    samples.oneHotifyAll()

    if list_of_features == None:
        list_of_features = samples.feature_names

    if config.addBias:
        (test_feature_matrix,test_targets,train_feature_matrix,train_targets) = samples.setProteinBias(config)
        list_of_features.append('Protein bias')

    corr_matrix = {}
    if addTargetValue:
        for feat_name in list_of_features:
            corr,p_val = samples.featureTargetCorr(feat_name,config)
            corr_matrix[(feat_name,'Target Value')] = corr
            corr_matrix[('Target Value',feat_name)] = corr

    for feat_name_1 in list_of_features:
        for feat_name_2 in list_of_features:
            if (feat_name_2,feat_name_1) in corr_matrix:
                continue
            if feat_name_1 == feat_name_2:
                corr = 1.0
            else:
                corr,p_val = samples.featureCorr(feat_name_1,feat_name_2)
            corr_matrix[(feat_name_1,feat_name_2)] = corr
            corr_matrix[(feat_name_2,feat_name_1)] = corr

    if addTargetValue:
        list_of_features.append('Target Value')
        corr_matrix[('Target Value','Target Value')] = 1.0

    header = 'Spearman correlation\t%s\n' % '\t'.join(list_of_features)
    lines = [header]
    for feat_name_1 in list_of_features:
        corr_list = []
        for feat_name_2 in list_of_features:
            corr_list.append('%.2f' % (corr_matrix[(feat_name_1,feat_name_2)]))
        lines.append('%s\t%s\n' % (feat_name_1,'\t'.join(corr_list)))

    f = open(outfile,'w')
    f.write(''.join(lines))
    f.close()
    return
"""

def printMean(score_list,name):
    N = len(score_list)
    total_quant = 0.

    for score,quant in score_list:
        total_quant += quant

    weighted_scores = []
    for score,quant in score_list:
        weighted_score = score*quant*N/total_quant
        weighted_scores.append(weighted_score)

    print('Mean %s:' % name, statistics.mean(weighted_scores),'STD:',statistics.stdev(weighted_scores))
    return

def median(l):
    n = len(l)
    l = sorted(l)
    if n % 2 == 0:
        med = (l[(n//2)-1]+l[n//2])/2.0
    else:
        med = l[(n-1)//2]
    return med

if __name__ == "__main__":
    disclaimer = 'Here comes the disclaimer'
    control = sys.argv[1]
    argv = sys.argv[2:]
    try:
        opts,args = getopt.getopt(argv,"c:i:o:f:n:ht",['help'])
    except getopt.GetoptError:
        print("Illegal Input\n\n",disclaimer)
        sys.exit(2)

    indatafile = ''
    outfile = ''
    config_path = ''
    featureFile = None
    addTargetValue = False
    overwrite_proc_n = None

    for opt,arg in opts:
        if opt == '-c':
            config_path = arg
        if opt == '-i':
            indatafile = arg
        if opt == '-o':
            outfile = arg
        if opt == '-f':
            featureFile = arg
        if opt == '-t':
            addTargetValue = True
        if opt == '-n':
            overwrite_proc_n = int(arg)
        if opt == '-h' or opt == '--help':
            print(disclaimer)
            sys.exit(0)

    config = Config(config_path)

    if overwrite_proc_n is not None:
        config.proc_n = overwrite_proc_n

    if control == 'addReverse':
        addReverseMutations(config,indatafile,outfile)
    if control == 'corrMatrix':
        if featureFile != None:
            list_of_features = parseFeatureList(featureFile)
        else:
            list_of_features = None
        calculateCorrelationMatrix(config,indatafile,outfile,list_of_features,addTargetValue = addTargetValue)
    if control == 'MPP_distribution':
        plotMPP(config,indatafile,outfile)




