import ray
import sys
import os
import traceback
import time
import random
import pickle
import numpy
import cupy as cp
import xgboost as xgb
from scipy import stats

from typing import Callable

from structguy import dicts
from structguy.util import get_gpu_memory, Config
from structguy.class_utils import get_feat_matrix_from_ids, get_raw_feat_matrix_from_ids, get_feat_id_vec

from structman.lib.sdsc.sdsc_utils import Slotted_obj
from structman.lib.serializedPipeline import sizeof_fmt

MAX_QUANTILE_BATCHES = 256

def calculate_chunksizes(n_of_chunks, n_of_items):
    small_chunksize = n_of_items // n_of_chunks
    big_chunksize = small_chunksize + 1
    n_of_small_chunks = n_of_chunks * big_chunksize - n_of_items
    n_of_big_chunks = n_of_chunks - n_of_small_chunks
    if n_of_big_chunks == 0:
        big_chunksize = 0
    return small_chunksize, big_chunksize, n_of_small_chunks, n_of_big_chunks

@ray.remote(max_calls = 1)
def distance_weighting_subroutine(store, package):
    target_value_map, rounded_exp, set_size, subsamplesize = store
    random_samples = random.sample(range(set_size),subsamplesize)
    outputs = []
    for target_value_1, pos_1 in package:
        d_sum = 0.
        for pos_2 in random_samples:
            target_value_2 = target_value_map[pos_2]
            d = abs(target_value_1-target_value_2)
            d_sum += d**rounded_exp
        outputs.append((pos_1, d_sum/subsamplesize))
    return outputs

possible_na_values = set(['-', 'None', 'inf'])
class Feature(Slotted_obj):
    __slots__ = ['name', 'f_type', 'group', 'default', 'mutation_specific', 'category_map', 'category_counter', 'category_backmap']
    def __init__(self, name = None, f_type = None,group=None,default_value=None,mutation_specific=False):
        self.name = name
        self.f_type = f_type
        self.group = group
        self.default = default_value
        self.mutation_specific=mutation_specific

        if f_type == 'categorical':
            self.category_map = {}
            self.category_counter = 0
            self.category_backmap = {}


    def value_from_string(self, string):
        value = None
        if string in possible_na_values:
            return None

        if self.name == 'Blosum62':
            try:
                value = float(string)
            except ValueError:
                try:
                    value = dicts.BLOSUM62[(string[0],string[-1])]
                except KeyError:
                    value = dicts.BLOSUM62[(string[-1],string[0])]
            return value

        try:
            if self.f_type == 'categorical':
                value = string
            elif self.f_type == 'real':
                value = float(string)
            elif self.f_type == 'integer' or self.f_type == 'binary':
                try:
                    value = int(string)
                except ValueError:
                    value = float(string)
                    self.f_type = 'real'
            elif self.f_type == 'unknown':
                try:
                    value = int(string)
                    if value != self.default:
                        self.f_type = 'integer'
                except:
                    try:
                        value = float(string)
                        if value != self.default:
                            self.f_type = 'real'
                    except:
                        self.f_type = 'categorical'
                        self.category_map = {}
                        self.category_counter = 0
                        self.category_backmap = {}

                        value = string
            else:
                print(f'Error in value_from_string: {self.f_type=} {self.name=} {string=}')
                value = None
        except:
            print(f'Error in value_from_string: {self.f_type=} {self.name=} {string=}')
            value = None
        return value

    def string_convert(self,value):
        if not self.f_type == 'categorical':
            return str(value)
        elif value is None:
            return 'None'
        else:
            return str(self.category_backmap[value])
            #return str(value)

class Iterator(xgb.DataIter):
    """A custom iterator for loading files in batches."""

    def __init__(
        self, file_paths: list[tuple[str, str, str]], device: str = 'cuda'
    ) -> None:
        self.device = device

        self._file_paths = file_paths
        self._it = 0
        self._ext_dat = None
        # XGBoost will generate some cache files under the current directory with the
        # prefix "cache"
        super().__init__(cache_prefix=os.path.join(".", "cache"))

    def load_file(self) -> tuple[numpy.ndarray, numpy.ndarray, list[str], list[str]]:
        """Load a single batch of data."""
        X_path, y_path, ext_path = self._file_paths[self._it]

        if self._ext_dat is None:
            with open(ext_path, 'rb') as inp:
                feat_names, cat_vec, feat_id_vec = pickle.load(inp)
            self._ext_dat = feat_names, cat_vec, feat_id_vec
        else:
            feat_names, cat_vec, feat_id_vec = self._ext_dat

        # When the `ExtMemQuantileDMatrix` is used, the device must match. GPU cannot
        # consume CPU input data and vice-versa.
        if self.device == "cpu":
            X = numpy.load(X_path)
            y = numpy.load(y_path)
        else:
            import cupy as cp
            X = numpy.load(X_path, allow_pickle=True)
            y = cp.load(y_path, allow_pickle=True)

            #trim down to selected features
            X = cp.array(X[:, feat_id_vec])

        assert X.shape[0] == y.shape[0]

        return X, y, feat_names, cat_vec

    def next(self, input_data: Callable) -> bool:
        """Advance the iterator by 1 step and pass the data to XGBoost.  This function
        is called by XGBoost during the construction of ``DMatrix``

        """
        if self._it == len(self._file_paths):
            # return False to let XGBoost know this is the end of iteration
            return False

        # input_data is a keyword-only function passed in by XGBoost and has the similar
        # signature to the ``DMatrix`` constructor.
        X, y, feat_names, cat_vec = self.load_file()
        input_data(data=X,
            label=y,
            feature_names = feat_names,
            feature_types=cat_vec
            )
        self._it += 1
        return True

    def reset(self) -> None:
        """Reset the iterator to its beginning"""
        self._it = 0

class CrossValidationSlice(Slotted_obj):
    __slots__ = [
        'isSlice',                      'test_targets',                 'test_sample_ids',
        'train_targets',                'train_sample_ids',             'feature_names',
        'name',                         'train_prots',                  'test_prots',
        'train_equal_test',             'slice_slice',                  'slice_slices',
        'subslice',                     'subslices',                    'random_subslice',
        'features',                     'deactivated_features',         'slice_specific_features',
        'slice_specific_feature_names', 'slice_specific_feature_map',   'current_geometric_exponent',
        'weight_vector_store',          'geometric_distance_map',       'int_map',
        'int_counter',                  'tvmb_map',                     'ranked_tvmb',
        'active_tvmb_threshold',        'tvpmb_map',                    'ranked_tvpmb',
        'active_tvpmb_threshold',       'current_number_of_bins',       'active_exp',
        'active_reg',                   'alpha_map',                    'c_map',
        'confusion_map',                'train_class_weight_vector',    'test_class_weight_vector',
        'fused_confusion_map',          'raw_confusion_map',            'loss_map',
        'sub_sampled_train_ids',        'sub_sampled_train_targets',    'sub_sampled_train_class_weight_vector',
        'test_prot_vec',                'test_slice_id',                'train_slice_ids',
        'code_map'
        ]
    
    slot_mask = [
        True, False, False,
        False, False, True,
        True, True, True,
        True, True, False,
        True, False, True,
        True, True, False,
        False, False, True,
        True, True, True,
        True, True, True,
        True, True, True,
        True, True, True,
        True, True, True,
        True, True, True,
        True, True, True,
        True, True, True,
        False,True, True,
        False
    ]

    def __init__(self, test_ids = None, train_ids = None, raw_feature_names = None, sample_dict = None, geometric_distance_map = None, config = None, name = '', train_prots = None, test_prots = None, train_equal_test = False, para_number = None, feature_names = None, raw_init = False):

        for slot in self.__slots__:
            self.__setattr__(slot, None)
        if raw_init or config is None:
            self.feature_names = []
            return
        self.isSlice = True

        self.test_targets = []
        if test_ids is None:
            test_ids = []
        self.test_sample_ids = [x for x in test_ids]

        self.train_targets = []
        if train_ids is None:
            train_ids = []
        self.train_sample_ids = [x for x in train_ids]

        self.sub_sampled_train_ids = None
        self.sub_sampled_train_class_weight_vector = None

        features_to_remove = []
        if feature_names is None:
            keep_features = None
        else:
            keep_features = set(feature_names)

        self.feature_names = []
        if raw_feature_names is None:
            raw_feature_names = []
        for feat_name in raw_feature_names:
            self.feature_names.append(feat_name)
            if keep_features is not None:
                if feat_name not in keep_features:
                    features_to_remove.append(feat_name)

        self.feature_names.sort()

        if sample_dict is None:
            sample_dict = {}

        self.name = name
        if train_prots is None:
            self.train_prots =  set([])
            test_samples = set(self.test_sample_ids)
            for (prot_id, aac) in sample_dict:
                if (prot_id, aac) in test_samples:
                    continue
                self.train_prots.add(prot_id)
        else:
            self.train_prots = train_prots

        self.test_prots = test_prots
        self.test_prot_vec = None
        self.code_map = {}

        self.train_equal_test = train_equal_test

        self.slice_slice = None
        self.slice_slices = None
        self.subslice = None
        self.subslices = None
        self.random_subslice = None

        self.deactivated_features = set()
        self.slice_specific_features = {}
        self.slice_specific_feature_names = []
        self.slice_specific_feature_map = {}

        self.current_geometric_exponent = None
        self.weight_vector_store = {}
        self.geometric_distance_map = {}
        self.int_map = {}
        self.int_counter = 0

        self.train_class_weight_vector = None
        self.test_class_weight_vector = None

        self.tvmb_map = None
        self.ranked_tvmb = None
        self.active_tvmb_threshold = None
        self.tvpmb_map = None
        self.ranked_tvpmb = None
        self.active_tvpmb_threshold = None
        self.current_number_of_bins = None
        self.active_exp = None
        self.active_reg = None
        self.alpha_map = {}
        self.c_map = {}

        self.loss_map = None
        self.confusion_map = None
        self.fused_confusion_map = None
        self.raw_confusion_map = None

        t0 = time.time()

        if not self.train_equal_test:
            print_out = config.verbosity >= 1
            self.checkCircularity(remove_t2=config.remove_t2, print_out=print_out)

        t1 = time.time()
        if config.verbosity >= 3:
            print(f'Init CV slice part 1: {t1-t0}')

        if config.verbosity >=3:
            print(f'In CVSLice init of {self.name=}: {train_equal_test=}, Test set: {len(self.test_sample_ids)=}, Train set: {len(self.train_sample_ids)=}')

        if not train_equal_test:
            for sample_id in self.test_sample_ids:
                try:
                    sample = sample_dict[sample_id]
                    self.test_targets.append(sample.targetValue)
                except:
                    print(f'In init CrossValidationSlice - Sample id: {sample_id} was not in the sample_dict')
                    self.test_targets.append(1.0)
                    continue
                
                self.slice_specific_features[sample_id] = {}

        t2 = time.time()
        if config.verbosity >= 3:   
            print(f'Init CV slice part 2 {train_equal_test}: {t2-t1}')

        for sample_id in self.train_sample_ids:
            sample = sample_dict[sample_id]
            self.train_targets.append(sample.targetValue)
            self.slice_specific_features[sample_id] = {}
            if train_equal_test:
                self.test_targets.append(sample.targetValue)

        self.train_targets = numpy.array(self.train_targets)
        self.test_targets = numpy.array(self.test_targets)

        self.sub_sampled_train_targets = []

        t3 = time.time()
        if config.verbosity >= 3:
            print(f'Init CV slice part 3: {t3-t2}')

        if train_equal_test:
            self.test_sample_ids = [x for x in self.train_sample_ids]


        t4 = time.time()
        if config.verbosity >= 3:
            print(f'Init CV slice part 4: {t4-t3}')

        if config.addBias:
            self.setProteinBias(config)

        t5 = time.time()
        if config.verbosity >= 3:
            print(f'Init CV slice part 5: {t5-t4}')

        if config.balanceSubsampling is not None:
            self.balanceSubSampleTrainSet(config)

        t6 = time.time()
        if config.verbosity >= 3:

            print(f'Init CV slice part 6: {t6-t5}')

        if config.verbosity >= 1:
            self.printBalance(config)

        t7 = time.time()
        if config.verbosity >= 3:
            print(f'Init CV slice part 7: {t7-t6}')

        if config.regression:
            if config.weighting == 'geometric':
                self.calcSampleWeights(config, geometric_distance_map)
            elif config.weighting == 'subsample_distance':
                self.calcSubsampleDistanceWeights(config, para_number = para_number)

        for feat_name in features_to_remove:
            self.removeFeature(feat_name)

        t8 = time.time()
        if config.verbosity >= 3:
            print(f'Init CV slice part 8: {t8-t7}')


    def featureSanityCheck(self, verbose = False):  
            
        return

    def checkCircularity(self,remove_t1=True,remove_t2=False,print_out=False):
        if not remove_t1 and not remove_t2:
            return False
        train_set = set(self.train_sample_ids)

        t1 = set()
        t2 = set()
        t2_prots = set()

        train_prots = set()
        for (u_ac,aac) in train_set:
            train_prots.add(u_ac)
        for (u_ac,aac) in self.test_sample_ids:
            if (u_ac,aac) in train_set:
                t1.add((u_ac,aac))
            if u_ac in train_prots:
                t2.add((u_ac,aac))
                t2_prots.add(u_ac)
        if print_out:
            print('Circularity check - Type 1: ',len(t1),', Type 2: ',len(t2))
        removed_entries = 0
        removed_pos = []
        if remove_t1 and len(t1) > 0:
            removed_entries += len(t1)
            for pos,sample_id in enumerate(self.train_sample_ids):
                removed_pos.append(pos)
            if print_out:
                print('Removed Type 1 Circularity samples from the train set')
        if remove_t2 and len(t2) > 0:
            removed_entries += len(t2)
            for pos,sample_id in enumerate(self.train_sample_ids):
                u_ac,aac = sample_id
                if u_ac in t2_prots:
                    removed_pos.append(pos)
            if print_out:
                print('Removed Type 2 Circularity samples from the train set')

        last_pos = None
        for pos in sorted(removed_pos,reverse=True):
            if pos != last_pos:
                del self.train_sample_ids[pos]
            last_pos = pos

        return (removed_entries > 0)

    def balanceSubSampleTrainSet(self,config):
        train_sub_sample_ids = set()
        if config.regression:
            prot_bins = {}
            for pos,(u_ac,aac) in enumerate(self.train_sample_ids):
                target = self.train_targets[pos]
                if u_ac not in prot_bins:
                    prot_bins[u_ac] = [{},{}]
                if target < config.binary_thresh:#split samples of a protein into left and right of threshold
                    prot_bins[u_ac][0][aac] = target
                else:
                    prot_bins[u_ac][1][aac] = target

            for u_ac in prot_bins:
                #Filter every protein that contains only samples left (or only right) of the threshold
                len_l = len(prot_bins[u_ac][0])
                len_r = len(prot_bins[u_ac][1])
                if len_l == 0:
                    continue
                if len_r == 0:
                    continue
                #choose bin with less entries as pivot bin
                if len_l < len_r:
                    pivot_bin = 0
                    level_bin = 1
                else:
                    pivot_bin = 1
                    level_bin = 0

                #choose and level iteration
                current_mean = 0.
                N = 0
                #print(prot_bins[u_ac][pivot_bin])
                #print(prot_bins[u_ac][level_bin])
                while True:
                    #choose pivot and add to sampling set
                    pivot_aac,pivot = prot_bins[u_ac][pivot_bin].popitem()
                    N += 1
                    current_mean = (current_mean*(N-1)+pivot)/N
                    #print('Pivot',pivot,current_mean)
                    train_sub_sample_ids.add((u_ac,pivot_aac))
                    #level the mean as good as possible by choosing a sample from the level bin
                    best_level_aac = None
                    best_level = None
                    #print('best level',best_level)
                    for level_aac in prot_bins[u_ac][level_bin]:
                        level_value = prot_bins[u_ac][level_bin][level_aac]
                        level = (current_mean*N+level_value)/(N+1)
                        #print('Level',level_value,level)
                        if best_level == None or abs(level-config.binary_thresh) < best_level:
                            best_level = abs(level-config.binary_thresh)
                            best_level_aac = level_aac
                    train_sub_sample_ids.add((u_ac,best_level_aac))
                    N += 1
                    current_mean = (current_mean*(N-1)+prot_bins[u_ac][level_bin][best_level_aac])/N
                    del prot_bins[u_ac][level_bin][best_level_aac]
                    #When the pivot bin is empty break the iteration
                    if len(prot_bins[u_ac][pivot_bin]) == 0:
                        break
            
        else:
            balance_map = {}
            for pos,(u_ac,aac) in enumerate(self.train_sample_ids):
                target = self.train_targets[pos]
                if u_ac not in balance_map:
                    balance_map[u_ac] = {}
                if target not in balance_map[u_ac]:
                    balance_map[u_ac][target] = set()
                balance_map[u_ac][target].add(aac)

            for u_ac in balance_map:
                min_label = None
                max_label = None
                min_n = None
                max_n = None
                for label in balance_map[u_ac]:
                    label_n = len(balance_map[u_ac][label])
                    if min_label is None or min_n > label_n:
                        min_label = label
                        min_n = label_n
                    if max_label is None or max_n < label_n:
                        max_label = label
                        max_n = label_n
                if min_label == max_label: #pure proteins
                    if min_n == 1 and not config.filter_single_variant_prots:
                        for aac in balance_map[u_ac][min_label]:
                            train_sub_sample_ids.add((u_ac,aac))
                    continue
                for label in balance_map[u_ac]:
                    for pos,aac in enumerate(balance_map[u_ac][label]):
                        
                        if config.balanceSubsampling == 'pure':
                            train_sub_sample_ids.add((u_ac,aac))
                        elif config.balanceSubsampling == 'balanced':
                            if pos < min_n:
                                train_sub_sample_ids.add((u_ac,aac))


        print('Performed training set balance subsampling, remaining samples: ',len(train_sub_sample_ids))

        self.filtered_samples = []
        kept_pos = []
        for pos,sample_id in enumerate(self.train_sample_ids):
            if sample_id not in train_sub_sample_ids:
                self.filtered_samples.append(sample_id)
            else:
                kept_pos.append(pos)

        self.train_targets = self.train_targets[kept_pos]
        self.train_sample_ids = self.train_sample_ids[kept_pos]

        return

    def printBalance(self, config):
        if config.regression:
            if len(self.test_targets) > 0:
                config.logger.info(f'{self.name} Mean train target value: {sum(self.train_targets)/len(self.train_targets)} Mean test target value: {sum(self.test_targets)/len(self.test_targets)}')
            config.logger.info(f'{self.name} Test set size: {len(self.test_targets)}, Train set size: {len(self.train_targets)}, Feats: {len(self.feature_names)}')
            try:
                config.logger.info(f'{self.feature_names[:5]}\n...\n{self.feature_names[-5:]}')
            except:
                config.logger.info(f'{self.feature_names}')
            return
        balance_map = {}
        for ttv in self.train_targets:
            if ttv not in balance_map:
                balance_map[ttv] = 0
            balance_map[ttv] += 1
        config.logger.info('Train set balance: ',balance_map)
        if len(balance_map) == 2:
            tv_1,tv_2 = balance_map.keys()
            
            if balance_map[tv_1] < balance_map[tv_2]:
                self.int_map[tv_1] = 1
                self.int_map[tv_2] = 0
            else:
                self.int_map[tv_2] = 1
                self.int_map[tv_1] = 0 
        balance_map = {}
        for ttv in self.test_targets:
            if ttv not in balance_map:
                balance_map[ttv] = 0
            balance_map[ttv] += 1
        config.logger.info('Test set balance: ',balance_map)

    def getGeometricDistanceMap(self,config):
        if self.geometric_distance_map is not None:
            return self.geometric_distance_map
        self.geometric_distance_map = {}
        targets = self.train_targets + self.test_targets
        sample_ids = self.train_sample_ids + self.test_sample_ids
        for pos_1,tv_1 in enumerate(targets):
            for pos_2,tv_2 in enumerate(targets):
                if pos_1 == pos_2:
                    self.geometric_distance_map[(sample_ids[pos_1],sample_ids[pos_2])] = 0.
                    continue
                if pos_2 < pos_1:
                    continue
                geometric_distance = abs(tv_1-tv_2)
                self.geometric_distance_map[(sample_ids[pos_1],sample_ids[pos_2])] = geometric_distance
                self.geometric_distance_map[(sample_ids[pos_2],sample_ids[pos_1])] = geometric_distance
        return self.geometric_distance_map

    def calcSampleWeights(self, config, geometric_distance_map, complete=False):
        if self.current_geometric_exponent == config.geometric_exponent:
            return
        print('Calc sample weights:',self.name,'with exponent:',config.geometric_exponent)
        try:
            geometric_distance_map = ray.get(geometric_distance_map)
        except:
            pass
        N = len(self.train_targets)
        self.train_class_weight_vector = []
        for pos,tv in enumerate(self.train_targets):
            d_sum = 0.
            for pos_2 in range(N):
                d_sum += geometric_distance_map[(self.train_sample_ids[pos],self.train_sample_ids[pos_2])]**config.geometric_exponent
            self.train_class_weight_vector.append(d_sum/N)

        N = len(self.test_targets)
        self.test_class_weight_vector = []
        for pos,tv in enumerate(self.test_targets):
            pos
            d_sum = 0.
            for pos_2 in range(N):
                d_sum += geometric_distance_map[(self.train_sample_ids[pos],self.train_sample_ids[pos_2])]**config.geometric_exponent
            self.test_class_weight_vector.append(d_sum/N)
        self.current_geometric_exponent = config.geometric_exponent
        if complete:
            self.class_weight_vector = self.train_class_weight_vector + self.test_class_weight_vector

    def check_auto_weights(self):
        if self.train_class_weight_vector is not None:
            return
        weight_vector = []

        count_dict = {}
        for sample_id in self.train_sample_ids:
            prot_id, aac = sample_id
            if prot_id not in count_dict:
                count_dict[prot_id] = 0
            count_dict[prot_id] += 1

        for sample_id in self.test_sample_ids:
            prot_id, aac = sample_id
            if prot_id not in count_dict:
                count_dict[prot_id] = 0
            count_dict[prot_id] += 1

        for sample_id in self.train_sample_ids:
            prot_id, _ = sample_id
            weight = len(self.train_sample_ids) / count_dict[prot_id]
            weight_vector.append(weight)
        self.train_class_weight_vector = weight_vector

        weight_vector = []
        for sample_id in self.test_sample_ids:
            prot_id, _ = sample_id
            weight = len(self.test_sample_ids) / count_dict[prot_id]
            weight_vector.append(weight)
        self.test_class_weight_vector = weight_vector


    def update_train_weight_vector(self, weight_map):
        weight_vector = []
        for sample_id in self.train_sample_ids:
            weight = weight_map[sample_id]
            weight_vector.append(weight)
        self.train_class_weight_vector = weight_vector

    def calcSubsampleDistanceWeights(self, config, para_number = None):
        rounded_exp = round(10*config.geometric_exponent)/10
        if rounded_exp in self.weight_vector_store:
            self.train_class_weight_vector, self.test_class_weight_vector = self.weight_vector_store[rounded_exp]
            return False

        set_size = len(self.train_targets)
        raw_subsamplesize = set_size * 0.005

        subsamplesize = min([set_size,max([100,round(raw_subsamplesize)])])

        store = ray.put((self.train_targets, rounded_exp, set_size, subsamplesize))

        if para_number is None:
            proc_n = config.proc_n
        else:
            proc_n = para_number

        if config.verbosity >= 2:
            print(f'Calc subsample distance weights: {self.name}, with exponent: {rounded_exp}, train_set_size: {set_size}, test_set_size: {len(self.test_targets)}, proc_n: {proc_n}')

        small_chunksize, big_chunksize, n_of_small_chunks, n_of_big_chunks = calculate_chunksizes(proc_n, len(self.train_targets))

        subroutine_results = []

        package = []
        started_big_procs = 0
        if n_of_big_chunks > 0:
            for pos, tv_1 in enumerate(self.train_targets):
                package.append((tv_1, pos))
                if len(package) == big_chunksize:
                    subroutine_results.append(distance_weighting_subroutine.remote(store, package))
                    package = []
                    started_big_procs += 1
                    if started_big_procs == n_of_big_chunks:
                        break

        border = n_of_big_chunks*big_chunksize

        for pos, tv_1 in enumerate(self.train_targets[border:]):
            package.append((tv_1, border + pos))
            if len(package) == small_chunksize:
                subroutine_results.append(distance_weighting_subroutine.remote(store, package))
                package = []

        if config.verbosity >= 2:
            print(f'Calc distance weighting remote info: started remotes: {len(subroutine_results)}, {small_chunksize}, {big_chunksize}, {n_of_small_chunks}, {n_of_big_chunks}')

        self.train_class_weight_vector = []

        outputs = ray.get(subroutine_results)
        pos_dict = {}
        for vector_entries in outputs:
            for pos, vector_entry in vector_entries:
                pos_dict[pos] = vector_entry
        for i in range(len(pos_dict)):
            self.train_class_weight_vector.append(pos_dict[i])

        if config.verbosity >= 2:
            print(f'Finished calculating weights for trainset, size: {len(self.train_class_weight_vector)}')

        set_size = len(self.test_targets)
        subsamplesize = min([set_size,max([100,round(set_size * raw_subsamplesize)])])

        store = ray.put((self.test_targets, rounded_exp, set_size, subsamplesize))

        small_chunksize, big_chunksize, n_of_small_chunks, n_of_big_chunks = calculate_chunksizes(proc_n, len(self.test_targets))

        subroutine_results = []

        package = []
        started_big_procs = 0
        if n_of_big_chunks > 0:
            for pos, tv_1 in enumerate(self.test_targets):
                package.append((tv_1, pos))
                if len(package) == big_chunksize:
                    subroutine_results.append(distance_weighting_subroutine.remote(store, package))
                    package = []
                    started_big_procs += 1
                    if started_big_procs == n_of_big_chunks:
                        break

        border = n_of_big_chunks*big_chunksize

        for pos, tv_1 in enumerate(self.test_targets[border:]):
            package.append((tv_1, pos + border))
            if len(package) == small_chunksize:
                subroutine_results.append(distance_weighting_subroutine.remote(store, package))
                package = []

        if config.verbosity >= 2:
            print(f'Calc distance weighting remote info for test set: started remotes: {len(subroutine_results)}, {small_chunksize}, {big_chunksize}, {n_of_small_chunks}, {n_of_big_chunks}')

        self.test_class_weight_vector = []

        outputs = ray.get(subroutine_results)
        pos_dict = {}
        for vector_entries in outputs:
            for pos, vector_entry in vector_entries:
                pos_dict[pos] = vector_entry
        for i in range(len(pos_dict)):
            self.test_class_weight_vector.append(pos_dict[i])

        if config.verbosity >= 2:
            print(f'Finished calculating weights for testset, size: {len(self.test_class_weight_vector)}')

        self.weight_vector_store[rounded_exp] = self.train_class_weight_vector, self.test_class_weight_vector
        del subroutine_results
        del store

        return True

    def setProteinBias(self, config):
        if self.slice_specific_feature_map is None:
            self.slice_specific_feature_map = {}
            self.slice_specific_feature_names = []
            self.slice_specific_features = {}
        if 'Protein bias' not in self.slice_specific_feature_map:
            self.slice_specific_feature_map['Protein bias'] = len(self.slice_specific_feature_names)
            try:
                self.slice_specific_feature_names.append('Protein bias')
            except:
                self.slice_specific_feature_names = ['Protein bias']
        else:
            return
        bias_map = {}
        if not config.regression:
            int_map = {}
            for pos,tv in enumerate(config.target_values):
                int_map[tv] = pos
        else:
            min_tv = float('inf')
            max_tv = -float('inf')
        for pos,target_value in enumerate(self.train_targets):
            u_ac,aac = self.train_sample_ids[pos]
            target_value = self.train_targets[pos]
            if u_ac not in bias_map:
                bias_map[u_ac] = []

            if not config.regression:
                if target_value in int_map:
                    bias_map[u_ac].append(int_map[target_value])
            else:
                if target_value > max_tv:
                    max_tv = target_value
                if target_value < min_tv:
                    min_tv = target_value
                bias_map[u_ac].append(target_value)

        for u_ac in bias_map:
            bias_map[u_ac] = sum(bias_map[u_ac])/len(bias_map[u_ac])

        for pos,sample_id in enumerate(self.train_sample_ids):
            u_ac,aac = sample_id
            if u_ac in bias_map:
                value = bias_map[u_ac]
            else:
                value = 0.5

            self.slice_specific_features[sample_id]['Protein bias'] = value

        for pos,sample_id in enumerate(self.test_sample_ids):
            u_ac,aac = sample_id
            if u_ac in bias_map:
                value = bias_map[u_ac]
            else:
                value = 0.5

            self.slice_specific_features[sample_id]['Protein bias'] = value

    def setPositionMeans(self,config):
        if 'Position means' not in self.slice_specific_feature_map:
            self.slice_specific_feature_map['Position means'] = len(self.slice_specific_feature_names)
            self.slice_specific_feature_names.append('Position means')
        else:
            return
        bias_map = {}
        if not config.regression:
            int_map = {}
            for pos,tv in enumerate(config.target_values):
                int_map[tv] = pos
        else:
            min_tv = float('inf')
            max_tv = -float('inf')
        for sample_pos,target_value in enumerate(self.train_targets):
            u_ac,aac = self.train_sample_ids[sample_pos]
            pos = int(aac[1:-1])
            if (u_ac,pos) not in bias_map:
                bias_map[(u_ac,pos)] = []

            if not config.regression:
                if target_value in int_map:
                    bias_map[(u_ac,pos)].append(int_map[target_value])
            else:
                if target_value > max_tv:
                    max_tv = target_value
                if target_value < min_tv:
                    min_tv = target_value
                bias_map[(u_ac,pos)].append(target_value)

        for (u_ac,pos) in bias_map:
            bias_map[(u_ac,pos)] = sum(bias_map[(u_ac,pos)])/len(bias_map[(u_ac,pos)])


        for sample_pos,sample_id in enumerate(self.train_sample_ids):
            u_ac,aac = sample_id
            pos = int(aac[1:-1])
            if (u_ac,pos) in bias_map:
                value = bias_map[(u_ac,pos)]
            else:
                value = 0.5

            self.slice_specific_features[sample_id]['Position means'] = value

        for sample_pos,sample_id in enumerate(self.test_sample_ids):
            u_ac,aac = sample_id
            pos = int(aac[1:-1])
            if (u_ac,pos) in bias_map:
                value = bias_map[(u_ac,pos)]
            else:
                value = 0.5

            self.slice_specific_features[sample_id]['Position means'] = value

        
    def featureTargetCorr(self, feat_name, feature_value_vector, config):
        val_list_1 = []
        val_list_2 = []

        for pos,tv in enumerate(self.train_targets):
            value = feature_value_vector[pos]
            if value is None:
                continue
            val_list_1.append(value)
            if not config.regression:
                if tv not in self.int_map:
                    self.int_map[tv] = self.int_counter
                    self.int_counter += 1
                tv = self.int_map[tv]
            val_list_2.append(tv)
        if min(val_list_1) == max(val_list_1):
            return 'const', f'Min Val: {min(val_list_1)}, Max Val: {max(val_list_1)}, Val-type: {type(val_list_1[0])} , len vals : {len(val_list_1)}, in deac features: {feat_name in self.deactivated_features}'
        try:
            corr,p_val = stats.spearmanr(val_list_1,val_list_2)
        except:
            return None, len(val_list_1)
        return corr,p_val

    def featureCorr(self, samples, feat_name_1, feat_name_2):

        val_list_1 = samples.get_feature_value_vector(self.train_sample_ids, feat_name_1)
        val_list_2 = samples.get_feature_value_vector(self.train_sample_ids, feat_name_2)

        if min(val_list_1) == max(val_list_1):
            return None, None
        try:
            corr,p_val = stats.spearmanr(val_list_1,val_list_2)
        except:
            return None, None
        return corr,p_val

    def set_to_features(self, dedicated_features):
        dedicated_features = set(dedicated_features)
        reacs = []
        for deac_feat in self.deactivated_features:
            if deac_feat in dedicated_features:
                reacs.append(deac_feat)

        for reac in reacs:
            self.reactivateFeature(reac)

        for ff in self.feature_names[:]:
            if ff not in dedicated_features:
                self.deactivateFeature(ff)

        if len(self.feature_names) > 0:
            if isinstance(self.feature_names, tuple):
                self.feature_names = list(self.feature_names)
            self.feature_names.sort()


    def filterFeatures(self,filtered_features, print_out = False):
        dont_reactivate = set(filtered_features)
        reacs = []
        for deac_feat in self.deactivated_features:
            if deac_feat not in dont_reactivate:
                reacs.append(deac_feat)

        for reac in reacs:
            self.reactivateFeature(reac)

        for ff in filtered_features:
            self.deactivateFeature(ff)

        if len(self.feature_names) > 0:
            if isinstance(self.feature_names, tuple):
                self.feature_names = list(self.feature_names)
            self.feature_names.sort()

        if print_out:
            print('======\n',self.name,'filter features:',len(filtered_features),'remaining features:',len(self.feature_names),'\n=====')

    def removeFeature(self,feat_name):
        del_pos = None
        for feat_pos, feature_name in enumerate(self.feature_names):
            if feature_name == feat_name:
                del_pos = feat_pos
                break
        if del_pos is None:
            return

        try:
            del self.feature_names[del_pos]
        except TypeError:
            self.feature_names = list(self.feature_names)
            del self.feature_names[del_pos]
        except:    
            if feat_name not in self.feature_names:
                return
            [e, f, g] = sys.exc_info()
            g = traceback.format_exc()
            print('Remove feature failed: ', feat_name, e, f, g, '\n', feat_pos, feat_name in self.feature_names, '\n', self.feature_names)
            sys.exit()

    def deactivateFeature(self,feat_name, print_out = False):
        if print_out:
            print('Slice:',self.name,'Deactivate feature:',feat_name)

        self.deactivated_features.add(feat_name)
        self.removeFeature(feat_name)

    def reactivateFeature(self,feat_name, print_out = False):
        if print_out:
            print('Slice:',self.name,'Reactivate feature:',feat_name)
        if feat_name not in self.deactivated_features:
            return
        try:
            self.feature_names.append(feat_name)
        except AttributeError:
            self.feature_names = list(self.feature_names)
            self.feature_names.append(feat_name)
        self.deactivated_features.remove(feat_name)


    def reset_confusion_maps(self):
        if self.slice_slice is not None:
            self.slice_slice.confusion_map = None
        if self.slice_slices is not None:
            for slice_slice in self.slice_slices:
                slice_slice.confusion_map = None
        if not self.isSlice:
            for cv_counter in self.slices:
                cv_slice = self.slices[cv_counter]
                cv_slice.reset_confusion_maps()

    def addToTvmbMap(self, feat_name, samples, config, dummy_call = False):
        if config.verbosity >= 5:
            print(f'Calling of addToTvmbMap of feature: {feat_name} in slice {self.name}')

        feature_value_vector = samples.get_feature_value_vector(self.train_sample_ids, feat_name)

        try:
            target_corr,target_p_val = self.featureTargetCorr(feat_name, feature_value_vector, config)
        except:
            #self.removeFeature(feat_name)
            if config.verbosity >= 5:
                [e, f, g] = sys.exc_info()
                g = traceback.format_exc()
                print(f'Error: cant get feature target corr for: {feat_name=} due to:\n{e}\n{f}\n{g}')
            return True 
        if target_corr == 'const':
            #self.removeFeature(feat_name)
            if config.verbosity >= 5:
                print(f'{self.name} - Removed feature: {feat_name} due to constant feature values, info: {target_p_val}')
            return True
        if target_corr is None:
            #self.removeFeature(feat_name)
            if config.verbosity >= 5:
                print(f'{self.name} - Removed feature: {feat_name} due to None target_corr, len of values: {target_p_val}')
            return True
        if dummy_call:
            return False
        
        mean_value_corr,p_val = self.featureCorr(feat_name,'Protein bias',slice_specific_feature = True)
        if mean_value_corr != mean_value_corr:
            #self.removeFeature(feat_name)
            if config.verbosity >= 5:
                print(self.name,'Removed feature:',feat_name,'due to nan mean_value_corr')
            return True
        tvmb_score = abs(mean_value_corr)-abs(target_corr) #target value mean bias score
        if tvmb_score != tvmb_score: #test for 'nan'
            #self.removeFeature(feat_name)
            if config.verbosity >= 5:
                print(self.name,'Removed feature:',feat_name,'due to nan tvmb score')
            return True
        else:
            self.tvmb_map[feat_name] = tvmb_score,target_p_val
            return False

    def rank_tvmb(self, samples, config):
        if config.verbosity >= 5:
            print(f'Call of rank_tvmb in slice {self.name}')
        self.ranked_tvmb = []
        for feat_name in list(self.feature_names):
            if feat_name not in self.tvmb_map:
                self.addToTvmbMap(feat_name, samples, config)
                if feat_name not in self.tvmb_map:
                    continue
            if feat_name not in self.tvmb_map:
                print('Error debug out:',self.name,len(self.feature_names))
            tvmb_score,_ = self.tvmb_map[feat_name]
            self.ranked_tvmb.append((feat_name,tvmb_score))
        self.ranked_tvmb.sort(key=lambda x:x[1],reverse=True)

    def addToTvpmbMap(self,feat_name,config):
        target_corr,p_val = self.featureTargetCorr(feat_name,config)
        mean_value_corr,p_val = self.featureCorr(feat_name,'Position means', slice_specific_feature = True)
        if mean_value_corr != mean_value_corr:
            mean_value_corr = 0.0
            p_val = 1.0
        tvpmb_score = abs(mean_value_corr)-abs(target_corr) #target value mean bias score
        if tvpmb_score != tvpmb_score: #test for 'nan'
            self.removeFeature(feat_name)
            print(self.name,'Removed feature:',feat_name,'due to nan tvpmb score')

        self.tvpmb_map[feat_name] = tvpmb_score

    def rank_tvpmb(self,config):
        self.ranked_tvpmb = []
        for feat_name in self.feature_names:
            if feat_name not in self.tvpmb_map:
                self.addToTvpmbMap(feat_name,config)
            tvpmb_score = self.tvpmb_map[feat_name]
            self.ranked_tvpmb.append((feat_name,tvpmb_score))
        self.ranked_tvpmb.sort(key=lambda x:x[1],reverse=True)

    def classToInt(self,data):
        int_data = []
        for class_name in data:
            if class_name not in self.int_map:
                self.int_map[class_name] = self.int_counter
                self.int_counter += 1
            int_data.append(self.int_map[class_name])
        return int_data
    
    def get_test_feature_matrix(self, samples):
        feat_matrix, _ = get_feat_matrix_from_ids(
            samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix,
            self.test_sample_ids, self.feature_names)
        return feat_matrix
    
    def get_encoded_test_prot_vec(self):
        if self.code_map is None:
            self.code_map = {}
        if self.test_prot_vec is None:
            code_vec = []
            for prot_id, _ in self.test_sample_ids:
                if prot_id not in self.code_map:
                    self.code_map[prot_id] = len(self.code_map)
                code = self.code_map[prot_id]
                code_vec.append(code)
            code_vec = numpy.array(code_vec)
            self.test_prot_vec = code_vec

        return self.test_prot_vec

    def get_dtest(self,
            feat_pos_dict: dict[str, int],
            features,
            sample_pos_dict,
            raw_feature_matrix,
            dtrain: xgb.DMatrix
            ):
        feat_matrix, cat_vec = get_feat_matrix_from_ids(
            feat_pos_dict, features, sample_pos_dict, raw_feature_matrix,
            self.test_sample_ids, self.feature_names, get_cat_vec=True)
        encoded_prot_vec = self.get_encoded_test_prot_vec()
        print(f'{len(self.test_sample_ids)=} {len(self.test_targets)=} {len(encoded_prot_vec)=} {numpy.array(feat_matrix).shape=} {len(self.feature_names)=}')
        dtest_feature_matrix = xgb.QuantileDMatrix(
            numpy.array(feat_matrix),
            label=numpy.array(self.test_targets),
            feature_types=cat_vec,
            enable_categorical=True,
            feature_names = self.feature_names,
            ref=dtrain)
        dtest_feature_matrix.encoded_prot_vec = encoded_prot_vec
        return dtest_feature_matrix

    def get_extmem_dtest(self,
            dump_precursor: str,
            config: Config,
            feat_pos_dict: dict[str, int],
            features,
            sub_share: float,
            dtrain: xgb.DMatrix,
            get_sliced_test_matrices: bool=False
            ) -> xgb.ExtMemQuantileDMatrix:
        
        if config.verbosity >= 4:
            config.logger.info(f'Call of get_extmem_dtest: {dump_precursor}')

        file_paths, encoded_prot_vec = self.prepare_test_ext_mem_qdmatrix(
            dump_precursor,
            config,
            feat_pos_dict,
            features
            )
        
        if config.verbosity >= 4:
            config.logger.info(f'After prep in get_extmem_dtest: {dump_precursor} {sub_share=} {len(file_paths)=}')

        # Make sure XGBoost is using RMM for all allocations.
        with xgb.config_context(use_rmm=True):

        # Make sure XGBoost is using the CUDA async pool for all allocations.
        #with xgb.config_context(use_cuda_async_pool=True):
            it = Iterator(device="cuda", file_paths=file_paths)

            if config.verbosity >= 4:
                config.logger.info(f'Iterator is setup in get_extmem_dtrain: {dump_precursor}')

            ext_dtest = xgb.ExtMemQuantileDMatrix(it, ref=dtrain, enable_categorical=True, max_bin=256, max_quantile_batches = MAX_QUANTILE_BATCHES)
        
            ext_dtest.encoded_prot_vec = encoded_prot_vec

            if get_sliced_test_matrices:
                ext_test_matrices = []
                for fp_tuple in file_paths:
                    it = Iterator(device="cuda", file_paths=[fp_tuple])

                    ext_dtest_slice = xgb.ExtMemQuantileDMatrix(it, ref=dtrain, enable_categorical=True, max_bin=256, max_quantile_batches = 1)
                    ext_test_matrices.append(ext_dtest_slice)
            else:
                ext_test_matrices = None

        return ext_dtest, file_paths, ext_test_matrices

    def set_sub_sampled_train_ids(self, sub_sampling_factor: float):
        k = int(len(self.train_sample_ids)*sub_sampling_factor)
        sub_sampled_ids = random.sample(range(len(self.train_sample_ids)), k)
        self.sub_sampled_train_ids = [self.train_sample_ids[pos] for pos in sub_sampled_ids]
        self.sub_sampled_train_targets = [self.train_targets[pos] for pos in sub_sampled_ids]
        self.sub_sampled_train_class_weight_vector = [self.train_class_weight_vector[pos] for pos in sub_sampled_ids]

    def get_train_feature_matrix(self,
            feat_pos_dict: dict[str, int],
            features,
            sample_pos_dict,
            raw_feature_matrix,
            sub_sampling = 1.0) -> list[list[int | float | None]]:
        if len(self.train_sample_ids) == 0:
            raise ValueError(f'No training samples in get_train_feature_matrix: {self.train_sample_ids=}')
        if len(self.feature_names) == 0:
            raise ValueError(f'No features in get_train_feature_matrix: {self.feature_names=}')
        if not sub_sampling == 1.0:
            if self.sub_sampled_train_ids is None:
                self.set_sub_sampled_train_ids(sub_sampling)
            sample_ids = self.sub_sampled_train_ids
        else:
            sample_ids = self.train_sample_ids

        feat_matrix, cat_vec = get_feat_matrix_from_ids(
            feat_pos_dict, features, sample_pos_dict, raw_feature_matrix,
            sample_ids, self.feature_names, get_cat_vec=True)

        return feat_matrix, cat_vec
    
    def get_raw_train_feature_matrix(self,
            feat_pos_dict: dict[str, int],
            features,
            sample_pos_dict,
            raw_feature_matrix: numpy.ndarray,
            sub_sampling = 1.0) -> list[list[int | float | None]]:
        if len(self.train_sample_ids) == 0:
            raise ValueError(f'No training samples in get_train_feature_matrix: {self.train_sample_ids=}')
        if len(self.feature_names) == 0:
            raise ValueError(f'No features in get_train_feature_matrix: {self.feature_names=}')
        if not sub_sampling == 1.0:
            if self.sub_sampled_train_ids is None:
                self.set_sub_sampled_train_ids(sub_sampling)
            sample_ids = self.sub_sampled_train_ids
        else:
            sample_ids = self.train_sample_ids

        raw_feat_matrix, cat_vec, feat_id_vec = get_raw_feat_matrix_from_ids(
            feat_pos_dict, features, sample_pos_dict, raw_feature_matrix,
            sample_ids, self.feature_names, get_cat_vec=True)

        return raw_feat_matrix, cat_vec, feat_id_vec

    def get_dtrain(self,
            feat_pos_dict: dict[str, int],
            features,
            sample_pos_dict,
            raw_feature_matrix,
            sub_sampling: float = 1.0
            ) -> list[list[int | float | None]]:
        train_feature_matrix: list[list[int | float | None]]
        train_feature_matrix, cat_vec = self.get_train_feature_matrix(feat_pos_dict, features, sample_pos_dict, raw_feature_matrix, sub_sampling=sub_sampling)
        if sub_sampling == 1.0:
            train_targets = self.train_targets
        else:
            train_targets = self.sub_sampled_train_targets

        #for pos, feat_name in enumerate(self.feature_names):
        #    print(f'{feat_name=}, {cat_vec[pos]=}')

        dtrain: xgb.DMatrix = xgb.QuantileDMatrix(
            numpy.array(train_feature_matrix),
            label= numpy.array(train_targets),
            feature_names = self.feature_names,
            feature_types=cat_vec,
            enable_categorical=True)
        return dtrain
    
    def get_extmem_dtrain(self,
            dump_precursor: str,
            config: Config,
            feat_pos_dict: dict[str, int],
            features,
            sub_share: float
            ) -> xgb.ExtMemQuantileDMatrix:
        
        if config.verbosity >= 4:
            config.logger.info(f'Call of get_extmem_dtrain: {dump_precursor}')

        file_paths, prot_id_vec = self.prepare_ext_mem_qdmatrix(
            dump_precursor,
            config,
            feat_pos_dict,
            features
            )
        
        # Make sure XGBoost is using RMM for all allocations.
        with xgb.config_context(use_rmm=True):

        ## Make sure XGBoost is using the CUDA async pool for all allocations.
        #with xgb.config_context(use_cuda_async_pool=True):
            it = Iterator(device="cuda", file_paths=file_paths)

            if config.verbosity >= 4:
                config.logger.info(f'Iterator is setup in get_extmem_dtrain: {dump_precursor}')

            ext_dtrain = xgb.ExtMemQuantileDMatrix(it, enable_categorical=True, max_bin=256, max_quantile_batches = MAX_QUANTILE_BATCHES)
            ext_dtrain.encoded_prot_vec = prot_id_vec

        return ext_dtrain, file_paths

    def prepare_ext_mem_qdmatrix(self,
            dump_precursor: str,
            config: Config,
            feat_pos_dict: dict[str, int],
            features
            ) -> list[tuple[str, str, str]]:
        
        feat_id_vec, cat_vec = get_feat_id_vec(self.feature_names, feat_pos_dict, features, get_cat_vec = True)
        
        file_paths: list[tuple[str, str, str]] = []

        extra_data_path = f'{dump_precursor}_ext.dump'
        with open(extra_data_path, 'wb') as outf:
            pickle.dump((self.feature_names, cat_vec, feat_id_vec), outf)

        prot_id_vec = None
        for train_cv_id in self.train_slice_ids:
            data_paths, prot_vec_path = config.file_path_dict[train_cv_id]
            for tr_fp, te_fp in data_paths:
                file_paths.append((tr_fp, te_fp, extra_data_path))

            if prot_id_vec is None:
                prot_id_vec = numpy.load(prot_vec_path, allow_pickle=True)
            else:
                prot_id_vec = numpy.concatenate((prot_id_vec, numpy.load(prot_vec_path, allow_pickle=True)))
        return file_paths, prot_id_vec
    
    def prepare_test_ext_mem_qdmatrix(self,
            dump_precursor: str,
            config: Config,
            feat_pos_dict: dict[str, int],
            features
            ) -> list[tuple[str, str, str]]:
        feat_id_vec, cat_vec = get_feat_id_vec(self.feature_names, feat_pos_dict, features, get_cat_vec = True)
        #encoded_prot_vec = self.get_encoded_test_prot_vec()

        file_paths: list[tuple[str, str, str]] = []

        extra_data_path = f'{dump_precursor}_ext_test.dump'
        with open(extra_data_path, 'wb') as outf:
            pickle.dump((self.feature_names, cat_vec, feat_id_vec), outf)

        data_paths, prot_vec_path = config.file_path_dict[self.test_slice_id]
        for tr_fp, te_fp in data_paths:
            file_paths.append((tr_fp, te_fp, extra_data_path))
            
        return file_paths, numpy.load(prot_vec_path, allow_pickle=True)


    def get_skewed_feat_matrices(self, samples, thresh):
        feat_matrices = samples.get_skewed_feat_matrices_from_ids(self.train_sample_ids, self.feature_names, thresh)
        return feat_matrices
    
    def get_prot_wise_test_data_tuples(self, samples):
        test_pred_pairs = {}
        for sample_nr, yt_value in enumerate(self.test_targets):
            prot_id, _ = self.test_sample_ids[sample_nr]
            if prot_id not in test_pred_pairs:
                test_pred_pairs[prot_id] = [[], []]
            test_pred_pairs[prot_id][1].append(yt_value)
            test_pred_pairs[prot_id][0].append(self.test_sample_ids[sample_nr])

        for prot_id in test_pred_pairs:
            feat_matrix, _ = get_feat_matrix_from_ids(
                samples.feat_pos_dict, samples.features, samples.sample_pos_dict, samples.raw_feature_matrix,
                test_pred_pairs[prot_id][0], self.feature_names)
            test_pred_pairs[prot_id][0] = feat_matrix
        return test_pred_pairs