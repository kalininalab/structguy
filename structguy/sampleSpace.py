import os
import sys
import numpy as np
import math
import random
import ray
import time
import pickle
import copy
import contextlib
import zstd


from scipy import stats

from datasail.sail import datasail

from structguy.sequence_util import parseFromFasta
from structguy.base_classes import Feature, possible_na_values
from structguy.support_classes import CrossValidationSlice
from structguy.util import get_gpu_memory, Config
from structguy.prefiltering import detectBiasedFeaturesByMeanCorrelation
from structguy.class_utils import get_feat_matrix_from_ids, get_feat_matrix, get_raw_feat_matrix_from_sample_ids
from structguy.feature_correlation import compute_feature_correlations
from structguy.feature_interactions import parse_combination_name, combine

from structman.base_utils.base_utils import pack, unpack, Slotted_obj

MAX_PACKED_MATRIX_BYTES = 2**31


def load_samples_dump(path):
    """Loads a SampleSpace.dump file, which is either packed or (for large feature matrices) a pickle."""
    with open(path, 'rb') as f:
        if f.peek(1)[:1] == b'\x80':
            return pickle.load(f)
        packed = f.read()
    try:
        return unpack(packed)
    except zstd.Error:
        return pickle.loads(packed)


class Sample(Slotted_obj):
    __slots__ = ['sample_id', 'targetValue', 'nr', 'amount_of_structures', 'tags']
    def __init__(self,sample_id = None, nr = None):
        self.sample_id = sample_id
        self.targetValue = None
        self.nr = nr
        self.amount_of_structures = 0
        self.tags = ''

    def addTargetValue(self,value):
        self.targetValue = value

@ray.remote(max_calls = 1)
def calc_gdm_submatrix(store,i):
    targets, sample_ids, bin_size, config = store
    a = i*bin_size
    b = min([(i+1)*bin_size,len(targets)])
    geometric_distance_map = {}
    for pos_1 in range(a,b):
        tv_1 = targets[pos_1]
        for pos_2,tv_2 in enumerate(targets):
            if pos_1 == pos_2:
                geometric_distance_map[(sample_ids[pos_1],sample_ids[pos_2])] = 0.
                continue
            geometric_distance = abs(tv_1-tv_2)**config.geometric_exponent
            geometric_distance_map[(sample_ids[pos_1],sample_ids[pos_2])] = geometric_distance
    return geometric_distance_map


def splitDataSet(config, sample_id_list: list[tuple[str, str]], specific_id=None, protein_wise=False, debug=0, skip_protein = None, ignore_samples = None):
    split_rate = config.split_rate
    total_size = len(sample_id_list)
    test_size = max([1,int(total_size*split_rate)])
    train_size = total_size-test_size

    test_ids = []
    train_ids = []

    if not protein_wise:
        #simple random split
        if specific_id is None:
            test_nrs = set(np.random.choice(total_size,test_size,replace=False))
            for nr, sample_id in enumerate(sample_id_list):
                
                if nr in test_nrs:
                    test_ids.append(sample_id)
                else:
                    train_ids.append(sample_id)
        else:
            for sample_id in sample_id_list:
                if sample_id in specific_id:
                    test_ids.append(sample_id)
                else:
                    train_ids.append(sample_id)

        if debug >= 1:
            config.logger.info(f'Train size: {train_size}, Test size: {test_size}')
    else:
        # Protein-nested split
        test_proteins = set()
        test_set_sum = 0
        if specific_id is None:
            protein_sizes = {}
            for u_ac,aac in sample_id_list:
                if u_ac not in protein_sizes:
                    protein_sizes[u_ac] = 1
                else:
                    protein_sizes[u_ac] += 1
            while test_set_sum < (test_size - test_size*split_rate):
                random_protein = random.choice(list(protein_sizes.keys()))
                if random_protein not in test_proteins:
                    test_proteins.add(random_protein)
                    test_set_sum += protein_sizes[random_protein]

        else:
            test_proteins = specific_id

        for (u_ac,aac) in sample_id_list:
            if skip_protein is not None and u_ac == skip_protein:
                continue
            if ignore_samples is not None and (u_ac, aac) in ignore_samples:
                continue

            if u_ac in test_proteins:
                test_ids.append((u_ac,aac))
            else:
                train_ids.append((u_ac,aac))

        #print 'Protein Id based sample splitting:\nTest set Ids: ',test_proteins,'\nTestset size: ',test_set_sum
        if debug >= 1:
            config.logger.info(f'Train size: {train_size}, Test size: {test_size}')
    if len(test_ids) == 0:
        if ignore_samples is not None:
            ignored_prots = set()
            for (prot_id, aac) in ignore_samples:
                ignored_prots.add(prot_id)
        else:
            ignored_prots = None
        config.logger.info(f'Splitted into empty test set: {specific_id}, {ignored_prots}')
    return test_ids, train_ids

class SampleSpace(Slotted_obj):
    __slots__ = [
        'samples', 'features', 'feature_matrix_dict',
        'raw_feature_matrix', 'sample_nr', 'feature_names',
        'feat_pos_dict', 'sample_pos_dict', 'vector_store',
        'current_vector',
        'geometric_distance_map', 'current_geometric_exponent', 'sequence_map',
        'impute_map', 'feat_corr_matrix', 'feat_stats'
    ]

    slot_mask = [
        True, True, False,
        True, True, True,
        True, True, False,
        False,
        False, False, False,
        False, True, True
    ]
    def __init__(self, config=None):
        for att in self.__slots__:
            self.__setattr__(att, None)
        self.samples = {}
        self.features: dict[str, Feature] = {}
        self.feature_matrix_dict = {}
        self.raw_feature_matrix: np.ndarray = np.array([])
        self.sample_nr = 0
        self.feature_names = []
        self.feat_pos_dict = {}
        self.sample_pos_dict = {}

        self.vector_store = {}
        self.current_vector = None

        self.geometric_distance_map = {}
        self.current_geometric_exponent = None

        if config is not None:
            if config.addBias:
                self.addFeature('Protein bias','real',group='amino acid property',default_value='0.5')
            seq_map , _ = parseFromFasta(config.path_to_sequence_fasta)
            self.sequence_map = seq_map
        else:
            self.sequence_map = None
        self.impute_map = {}

    # Pickling (ray.put, dump of large matrices) sends the same slots as structman's pack, the ones on in slot_mask
    def __getstate__(self):
        return {att: getattr(self, att, None) for att, active in zip(self.__slots__, self.slot_mask) if active}

    def __setstate__(self, state):
        for att in self.__slots__:
            setattr(self, att, state.get(att))

    def fuse_samples(self, other_samples):
        if self.sequence_map is not None and other_samples.sequence_map is not None:
            self.sequence_map.update(other_samples.sequence_map)
        if self.feature_matrix_dict is not None or other_samples.feature_matrix_dict is not None:
            self.matrix_to_dict()
            other_samples.matrix_to_dict()
            self.samples.update(other_samples.samples)
            self.features.update(other_samples.features)
            self.feature_matrix_dict.update(other_samples.feature_matrix_dict)
            self.fillDefaultValues()
            self.feature_names = list(self.features.keys())
            self.transform_matrix_dict()
            return

        for feat_name, feat in other_samples.features.items():
            if feat_name not in self.features:
                self.features[feat_name] = copy.deepcopy(feat)
                if feat.f_type == 'categorical':
                    self.features[feat_name].category_map = {}
                    self.features[feat_name].category_counter = 0
                    self.features[feat_name].category_backmap = {}
                self.feature_names.append(feat_name)
        self.ensure_feature_columns()

        # The rows of other_samples in the column layout of self, categorical codes translated into the maps of self
        other_ids = list(other_samples.samples.keys())
        other_rows = [other_samples.sample_pos_dict[sample_id] for sample_id in other_ids]
        other_matrix = other_samples.raw_feature_matrix[other_rows]
        block = np.empty((len(other_ids), len(self.feat_pos_dict)), dtype=np.float64)
        block[:] = self.default_value_vector(list(self.feat_pos_dict.keys()))
        file_cols = []
        for feat_name, other_pos in other_samples.feat_pos_dict.items():
            col = self.feat_pos_dict[feat_name]
            values = other_matrix[:, other_pos]
            other_feat = other_samples.features[feat_name]
            if other_feat.f_type == 'categorical':
                decoded = np.array([other_feat.category_backmap.get(int(v)) if v == v else None for v in values.tolist()], dtype=object)
                self.features[feat_name].set_categorical()
                values = self.encode_categories(self.features[feat_name], decoded)
            block[:, col] = values
            file_cols.append(col)

        sample_meta = [
            (sample_id, sample.targetValue, sample.amount_of_structures, sample.tags)
            for sample_id, sample in ((sample_id, other_samples.samples[sample_id]) for sample_id in other_ids)
        ]
        self.add_parsed_rows(sample_meta, block, np.array(sorted(file_cols), dtype=np.int64))

    def addFeature(self,name,f_type,group=None,default_value=None,mutation_specific=False):
        feat = Feature(name = name, f_type = f_type,group=group,default_value=default_value,mutation_specific=mutation_specific)
        self.features[name] = feat
        self.feature_names.append(name)

    def default_value_vector(self, feat_names: list[str]) -> np.ndarray:
        """The default value of every given feature, as a row of the feature matrix."""
        defaults = np.full(len(feat_names), np.nan, dtype=np.float64)
        for j, feat_name in enumerate(feat_names):
            feat = self.features.get(feat_name)
            if feat is None or feat.default is None:
                continue
            if feat.f_type == 'categorical':
                defaults[j] = feat.encode(feat.default)
            else:
                try:
                    defaults[j] = float(feat.default)
                except (TypeError, ValueError):
                    pass
        return defaults

    def encode_categories(self, feat: Feature, values: np.ndarray) -> np.ndarray:
        """Encodes raw categorical values (strings, missing values as NA string or None) to the codes of the feature.

        New categories get codes in the order of their first appearance and a missing value is a category of its own,
        as in addValue.
        """
        strings = np.array(['\x00None' if value is None or value in possible_na_values else str(value) for value in values.tolist()], dtype=str)
        uniques, first_index, inverse = np.unique(strings, return_index=True, return_inverse=True)
        unique_codes = np.empty(len(uniques), dtype=np.float64)
        for u in np.argsort(first_index):
            value = None if uniques[u] == '\x00None' else str(uniques[u])
            unique_codes[u] = feat.encode(value)
        return unique_codes[inverse.reshape(-1)]

    def ensure_feature_columns(self):
        """Gives every feature a column in raw_feature_matrix, new columns are filled with the feature defaults."""
        if self.feat_pos_dict is None:
            self.feat_pos_dict = {}
        if self.sample_pos_dict is None:
            self.sample_pos_dict = {}
        n_rows = len(self.sample_pos_dict)
        if self.raw_feature_matrix is None or self.raw_feature_matrix.ndim != 2:
            self.raw_feature_matrix = np.empty((n_rows, 0), dtype=np.float64)
        missing = [feat_name for feat_name in self.feature_names if feat_name not in self.feat_pos_dict]
        if len(missing) == 0:
            return
        new_columns = np.empty((n_rows, len(missing)), dtype=np.float64)
        new_columns[:] = self.default_value_vector(missing)
        if self.raw_feature_matrix.shape[1] == 0:
            self.raw_feature_matrix = new_columns
        else:
            self.raw_feature_matrix = np.hstack([self.raw_feature_matrix, new_columns])
        for feat_name in missing:
            self.feat_pos_dict[feat_name] = len(self.feat_pos_dict)

    def add_parsed_rows(self, sample_meta: list[tuple], block: np.ndarray, file_cols: np.ndarray):
        """Adds parsed samples, block holds their rows in the column layout of feat_pos_dict.

        A sample that is already present gets the values of the columns file_cols overwritten, like addValue does.
        """
        self.feature_matrix_dict = None
        new_rows = []
        for i, (sample_id, target_value, amount_of_structures, tags) in enumerate(sample_meta):
            if sample_id in self.sample_pos_dict:
                self.raw_feature_matrix[self.sample_pos_dict[sample_id], file_cols] = block[i, file_cols]
                sample = self.samples[sample_id]
            else:
                self.sample_pos_dict[sample_id] = len(self.sample_pos_dict)
                new_rows.append(i)
                sample = Sample(sample_id, self.sample_nr)
                self.sample_nr += 1
                self.samples[sample_id] = sample
            sample.addTargetValue(target_value)
            sample.amount_of_structures = amount_of_structures
            sample.tags = tags

        if len(new_rows) == len(sample_meta) and self.raw_feature_matrix.shape[0] == 0:
            self.raw_feature_matrix = np.ascontiguousarray(block)
        elif len(new_rows) > 0:
            self.raw_feature_matrix = np.vstack([self.raw_feature_matrix, block[new_rows]])

    def matrix_to_dict(self):
        """Converts the feature matrix back to feature_matrix_dict, for the code paths that need it (fusePositions, addValue)."""
        if self.feature_matrix_dict is not None:
            return
        feat_items = list(self.feat_pos_dict.items())
        self.feature_matrix_dict = {}
        for sample_id in self.samples:
            row = self.raw_feature_matrix[self.sample_pos_dict[sample_id]].tolist()
            sample_dict = {}
            for feat_name, pos in feat_items:
                value = row[pos]
                if value != value:
                    value = None
                elif self.features[feat_name].f_type in ('categorical', 'integer', 'binary') and value.is_integer():
                    value = int(value)
                sample_dict[feat_name] = value
            self.feature_matrix_dict[sample_id] = sample_dict
        self.raw_feature_matrix = np.array([])
        self.feat_pos_dict = {}
        self.sample_pos_dict = {}

    def compact_matrix_rows(self):
        """Removes the rows of samples that were deleted from self.samples."""
        if len(self.sample_pos_dict) == len(self.samples):
            return
        keep_rows = [self.sample_pos_dict[sample_id] for sample_id in self.samples]
        self.raw_feature_matrix = self.raw_feature_matrix[keep_rows]
        self.sample_pos_dict = {sample_id: pos for pos, sample_id in enumerate(self.samples)}

    def quantitative_columns(self):
        """(feat_name, column) of all non-categorical features in the feature matrix."""
        return [
            (feat_name, self.feat_pos_dict[feat_name]) for feat_name, feat in self.features.items()
            if feat.f_type != 'categorical' and feat_name in self.feat_pos_dict
        ]

    def removeFeature(self, feat_name):
        del self.features[feat_name]
        self.feature_names.remove(feat_name)


    def print_feat_types(self, config):
        for featname in self.features:
            config.logger.info(f'{featname} - {self.features[featname].f_type}')

    def print_stats(self, config):
        prot_ids = set()
        for sample_id in self.samples:
            prot_id, aac = sample_id
            prot_ids.add(prot_id)

        config.logger.info(f'State of samples object: Proteins: {len(prot_ids)}, Samples: {len(self.samples)}, Features: {len(self.feature_names)}')

    def oneHotify(self,feat_name, config):
        if config.verbosity >= 4:
            config.logger.info(f'oneHotify of {feat_name}')
        feat = self.features[feat_name]
        f_type = feat.f_type
        if not f_type == 'categorical':
            config.logger.info('Warning: Cannot oneHotify non-categorical features')
            return
        if config.verbosity >= 4:
            config.logger.info(f'{feat.category_map=}')
        for category in feat.category_map:
            oh_name = 'oh_%s_%s' % (feat_name,category)
            self.addFeature(oh_name,'binary',group=feat.group,default_value=0)

        for sample_id in self.feature_matrix_dict:
            cat_value = self.feature_matrix_dict[sample_id][feat_name]
            category = feat.category_backmap[cat_value]
            oh_name = 'oh_%s_%s' % (feat_name,category)
            self.addValue(sample_id,1,oh_name)
            if config.verbosity >= 5:
                config.logger.info(f'Converting {feat_name=} of {sample_id=}: {cat_value=} {category=}')

    def oneHotifyAll(self, config):
        config.logger.info('Convert all categorical features to 1-hot encodings')
        del_list = []
        for feat_name in self.features:
            if self.features[feat_name].f_type == 'categorical':
                del_list.append(feat_name)
        for feat_name in del_list:
            self.oneHotify(feat_name, config)
            del self.features[feat_name]
            for sample_id in self.feature_matrix_dict:
                del self.feature_matrix_dict[sample_id][feat_name]

        self.feature_names = list(self.features.keys())
        self.fillDefaultValues()

    def encode_cat_feats(self):
        for feat_name in self.features:
            if self.features[feat_name].f_type == 'categorical':
                feat = self.features[feat_name]
                for sample_id in self.feature_matrix_dict:
                    cat_value = self.feature_matrix_dict[sample_id][feat_name]
                    category = feat.category_backmap[cat_value]

    def draw_subsamples(self, n_of_subsamples: int = 50_000):
        sample_list: list[tuple[str, str]] = list(self.samples.keys())
        if n_of_subsamples >= len(self.samples):
            return sample_list
        subsamples = random.sample(sample_list, n_of_subsamples)
        return subsamples

    def add_combination_features(self, names: list[str], config=None) -> int:
        """Adds the combination features (comb||<op>||<feature a>||<feature b>) that are missing as matrix columns.

        Pairs with an unknown or categorical feature are skipped. Returns the number of added features.
        """
        if self.feature_matrix_dict is not None:
            self.transform_matrix_dict(config=config)
        new_columns = []
        for name in dict.fromkeys(names):
            if name in self.features:
                continue
            parsed = parse_combination_name(name)
            if parsed is None:
                continue
            op, feat_a, feat_b = parsed
            if feat_a not in self.feat_pos_dict or feat_b not in self.feat_pos_dict:
                if config is not None and config.verbosity >= 2:
                    config.logger.info(f'Combination feature {name} skipped: a feature of the pair is not in the feature matrix')
                continue
            if self.features[feat_a].f_type == 'categorical' or self.features[feat_b].f_type == 'categorical':
                if config is not None and config.verbosity >= 2:
                    config.logger.info(f'Combination feature {name} skipped: categorical features are not combined')
                continue
            column = combine(op, self.raw_feature_matrix[:, self.feat_pos_dict[feat_a]], self.raw_feature_matrix[:, self.feat_pos_dict[feat_b]])
            new_columns.append((name, column, self.features[feat_a].group))
        if not new_columns:
            return 0
        self.raw_feature_matrix = np.hstack([self.raw_feature_matrix, np.column_stack([column for _, column, _ in new_columns])])
        for name, _, group in new_columns:
            self.addFeature(name, 'real', group=group)
            self.feat_pos_dict[name] = len(self.feat_pos_dict)
        if config is not None and config.verbosity >= 1:
            config.logger.info(f'Added {len(new_columns)} combination features, feature matrix: {self.raw_feature_matrix.shape}')
        return len(new_columns)

    def ensure_feat_corr_matrix(self, config):
        """Computes the feature correlations if missing: prediction runs and fused support samples skip them."""
        if self.feat_corr_matrix is None:
            self.transform_matrix_dict(config=config)
            self.calc_subsamples_feat_corr_matrix(config)

    def calc_subsamples_feat_corr_matrix(self, config, n_of_subsamples = 50_000) -> np.ndarray:
        """Returns a float32 array of shape (n_feats, n_feats, 6) holding (corr, cov_a, cov_b, cov_both, tv_corr_a, tv_corr_b) per feature pair."""
        subsamples = self.draw_subsamples(n_of_subsamples=n_of_subsamples)
        feat_matrix, cat_vec = get_feat_matrix_from_ids(
            self.feat_pos_dict,
            self.features,
            self.sample_pos_dict,
            self.raw_feature_matrix,
            subsamples,
            self.feature_names)
        # float dtype turns None into NaN; an object array would be unpickled as a full copy in every worker
        feat_matrix = np.ascontiguousarray(np.array(feat_matrix, dtype=np.float64).transpose())
        tv_vector = np.array([self.samples[sample_id].targetValue for sample_id in subsamples], dtype=np.float64)

        feat_corr_matrix, feat_stats = compute_feature_correlations(feat_matrix, tv_vector, config)
        del feat_matrix

        self.feat_stats = {feat_name: feat_stats[feat_nr] for feat_nr, feat_name in enumerate(self.feature_names)}
        self.feat_corr_matrix = feat_corr_matrix

        config.logger.info(f'{len(self.feat_stats)=} {len(self.feat_corr_matrix)=} {len(self.feature_names)}')
        return feat_corr_matrix

    def external_impute(self, external_impute):
        if self.feature_matrix_dict is None:
            for feat_name, col in self.quantitative_columns():
                column = self.raw_feature_matrix[:, col]
                column[np.isnan(column)] = external_impute.get(feat_name, 0.)
            return
        for feat_name in self.features:
            feat = self.features[feat_name]
            if feat.f_type == 'categorical':
                continue
            if feat_name in external_impute:
                impute_value = external_impute[feat_name]
            else:
                impute_value = 0.
            for sample_id in self.feature_matrix_dict:
                if feat_name not in self.feature_matrix_dict[sample_id]:
                    self.feature_matrix_dict[sample_id][feat_name] = impute_value
                elif self.feature_matrix_dict[sample_id][feat_name] is None:
                    self.feature_matrix_dict[sample_id][feat_name] = impute_value

    def get_bounds(self, feat_name):
        if self.feature_matrix_dict is None:
            column = self.raw_feature_matrix[:, self.feat_pos_dict[feat_name]]
            if np.isnan(column).all():
                return None, None
            return float(np.nanmin(column)), float(np.nanmax(column))
        min_value = None
        max_value = None
        for sample_id in self.feature_matrix_dict:
            value = self.feature_matrix_dict[sample_id][feat_name]
            if value is None:
                continue
            if min_value is None:
                min_value = value
            elif value < min_value:
                min_value = value
            if max_value is None:
                max_value = value
            elif value > max_value:
                max_value = value
        return min_value, max_value

    def impute_all(self):
        if self.feature_matrix_dict is None:
            for feat_name, col in self.quantitative_columns():
                min_val, max_val = self.get_bounds(feat_name)
                if min_val is not None:
                    impute_value = min_val - abs(max_val - min_val)
                else:
                    impute_value = 0
                self.impute_map[feat_name] = impute_value
                column = self.raw_feature_matrix[:, col]
                column[np.isnan(column)] = impute_value
            return
        for feat_name in self.features:
            feat = self.features[feat_name]
            if feat.f_type == 'categorical':
                continue
            #config.logger.info(feat_name)
            min_val, max_val = self.get_bounds(feat_name)
            if min_val is not None:
                impute_value = min_val - abs(max_val - min_val)
            else:
                impute_value = 0
            self.impute_map[feat_name] = impute_value
            for sample_id in self.feature_matrix_dict:
                if feat_name not in self.feature_matrix_dict[sample_id]:
                    self.feature_matrix_dict[sample_id][feat_name] = impute_value
                elif self.feature_matrix_dict[sample_id][feat_name] is None:
                    self.feature_matrix_dict[sample_id][feat_name] = impute_value

    def dump_impute_map(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self.impute_map, output, pickle.HIGHEST_PROTOCOL)

    def fusePositions(self):
        self.matrix_to_dict()
        pos_map = {}
        for sample_id in self.samples:
            (u_ac,aac) = sample_id
            target_value = self.samples[sample_id].targetValue
            aac_base = aac[:-1]
            if (u_ac,aac_base) not in pos_map:
                pos_map[(u_ac,aac_base)] = [1,aac[-1]]#['all neutral',aac[-1]]
            if target_value < 0.25:
                pos_map[(u_ac,aac_base)][0] = 0#'possibly damaging'

        self.samples = {}
        self.sample_nr = 0
        new_feat_names = []
        del_feat_names = []
        done = False
        for sample_id in pos_map:
            u_ac,aac_base = sample_id
            new_target,example_mut_aa = pos_map[sample_id]
            for feat_name in self.features:
                feat = self.features[feat_name]
                if not feat.mutation_specific:

                    value = self.feature_matrix_dict[(u_ac,'%s%s' % (aac_base,example_mut_aa))][feat_name]
                    self.addValue(sample_id,value,feat_name)
                    if not done:
                        new_feat_names.append(feat_name)
                elif not done:
                    del_feat_names.append(feat_name)
            done = True
            self.addTargetValue(sample_id,new_target)
        self.feature_names = new_feat_names
        for feat_name in del_feat_names:
            del self.features[feat_name]
        self.cleanFeatureValues()    

    def removeSamples(self,del_list):
        for sample_id in del_list:
            del self.samples[sample_id]
        self.cleanFeatureValues()

    def cleanFeatureValues(self):
        if self.feature_matrix_dict is None:
            self.compact_matrix_rows()
            return
        del_values = []
        for sample_id in self.feature_matrix_dict:
            if sample_id not in self.samples:
                del_values.append(sample_id)
        for sample_id in del_values:
            del self.feature_matrix_dict[sample_id]
        return

    def removeFeaturesByType(self, feat_type, config):
        config.logger.info('Remove features of type: ',feat_type)
        del_list = []
        for feat_name in self.features:
            feat = self.features[feat_name]
            if feat.group == feat_type:
                del_list.append(feat_name)
        for feat_name in del_list:
            del self.features[feat_name]
        self.feature_names = list(self.features.keys())
        return

    def addValue(self,sample_id,value,feat_name, config = None):
        if sample_id not in self.samples:
            self.samples[sample_id] = Sample(sample_id,self.sample_nr)
            self.sample_nr += 1
            self.feature_matrix_dict[sample_id] = {}
            if config is not None:
                config.logger.info(f'Added new sample: {sample_id=} {self.sample_nr=}')
        if not self.features[feat_name].f_type == 'categorical':
            self.feature_matrix_dict[sample_id][feat_name] = value
        else:
            if value not in self.features[feat_name].category_map:
                self.features[feat_name].category_map[value] = self.features[feat_name].category_counter
                self.features[feat_name].category_backmap[self.features[feat_name].category_counter] = value
                self.features[feat_name].category_counter += 1
            self.feature_matrix_dict[sample_id][feat_name] = self.features[feat_name].category_map[value]


    def addTargetValue(self,sample_id,value):
        self.samples[sample_id].addTargetValue(value)

    def targetTransformation(self,config):
        config.binary_thresh = math.sqrt(config.binary_thresh)
        for sample_id in self.samples:
            val = self.samples[sample_id].targetValue
            if val < 0.:
                val = 0.
            self.samples[sample_id].targetValue = math.sqrt(val)

    def fillDefaultValues(self):
        if self.feature_matrix_dict is None:
            self.ensure_feature_columns()
            return
        for sample_id in self.samples:
            for feat_name in self.features:
                if feat_name not in self.feature_matrix_dict[sample_id]:
                    self.feature_matrix_dict[sample_id][feat_name] = self.features[feat_name].default

    def dump(self, outfile):
        bu_slotmask = self.deactivate_slot_mask()
        try:
            # msgpack cannot hold bytes objects over 4 GiB, which a large feature matrix becomes in pack
            packed = None
            if getattr(self.raw_feature_matrix, 'nbytes', 0) < MAX_PACKED_MATRIX_BYTES:
                try:
                    packed = pack(self)
                except ValueError:
                    packed = None
            with open(outfile, 'wb') as f:
                if packed is not None:
                    f.write(packed)
                else:
                    pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self.reactivate_slot_mask(bu_slotmask)

    def write(self, outfile, block_size: int = 1000):
        if self.feature_matrix_dict is not None:
            self.write_from_dict(outfile)
            return

        self.ensure_feature_columns()
        feature_names = list(self.features.keys())
        cols = [self.feat_pos_dict[feature_name] for feature_name in feature_names]
        header = 'Uniprot Ac\tAAC\tTarget value\tTags\tAmount of mapped structures\t%s' % '\t'.join(feature_names)

        int_cols = [j for j, feature_name in enumerate(feature_names) if self.features[feature_name].f_type in ('integer', 'binary')]
        cat_cols = [j for j, feature_name in enumerate(feature_names) if self.features[feature_name].f_type == 'categorical']
        class_col = feature_names.index('RIN-based simple classification') if 'RIN-based simple classification' in feature_names else None

        sample_ids = list(self.samples.keys())
        class_files = {}
        if class_col is not None:
            base_name, file_type = outfile.rsplit('.', 1)
            if not os.path.isdir(base_name):
                os.mkdir(base_name)

        with open(outfile, 'w') as f:
            f.write(header)
            for start in range(0, len(sample_ids), block_size):
                block_ids = sample_ids[start:start + block_size]
                block = self.raw_feature_matrix[[self.sample_pos_dict[sample_id] for sample_id in block_ids]][:, cols]
                # Python floats and ints, so str() formats them like the values of the old dict based writer
                values = block.astype(object)
                if int_cols:
                    sub = block[:, int_cols]
                    with np.errstate(invalid='ignore'):
                        integral = ~np.isnan(sub) & (sub == np.floor(sub))
                    int_values = values[:, int_cols]
                    int_values[integral] = np.where(integral, sub, 0).astype(np.int64).astype(object)[integral]
                    values[:, int_cols] = int_values
                for j in cat_cols:
                    backmap = self.features[feature_names[j]].category_backmap
                    values[:, j] = ['None' if v != v else str(backmap[int(v)]) for v in block[:, j].tolist()]

                lines = []
                for sample_id, row in zip(block_ids, values.tolist()):
                    u_ac, aac = sample_id
                    sample = self.samples[sample_id]
                    feature_str = '\t'.join(map(str, row))
                    if 'nan' in feature_str:
                        # Missing values are written as None; twice, as adjacent cells share their tab
                        feature_str = f'\t{feature_str}\t'.replace('\tnan\t', '\tNone\t').replace('\tnan\t', '\tNone\t')[1:-1]
                    target_value_str = str(sample.targetValue)
                    lines.append('\n%s\t%s\t%s\t%s\t%s\t%s' % (u_ac, aac, target_value_str, sample.tags, str(sample.amount_of_structures), feature_str))

                    if class_col is None:
                        continue
                    class_name = row[class_col]
                    if class_name == 'None':
                        continue
                    if class_name not in class_files:
                        class_files[class_name] = open('%s_%s.%s' % (base_name, class_name, file_type), 'w')
                        class_files[class_name].write(header)
                    class_files[class_name].write('\n%s\t%s\t%s\t%s' % (u_ac, aac, target_value_str, feature_str))
                f.write(''.join(lines))

        for class_file in class_files.values():
            class_file.close()

    def write_from_dict(self, outfile):

        self.fillDefaultValues()
        class_out_lines = {}

        feature_names = list(self.features.keys())

        header = 'Uniprot Ac\tAAC\tTarget value\tTags\tAmount of mapped structures\t%s' % '\t'.join(feature_names)
        outlines = [header]

        for pos,feature_name in enumerate(feature_names):
            if feature_name == 'RIN-based simple classification':
                class_pos = pos

        for sample_id in self.samples:
            class_name = None
            u_ac,aac = sample_id
            target_value_str = str(self.samples[sample_id].targetValue)

            feature_vector = []
            for feature_name in feature_names:
                feat = self.features[feature_name]
                feature_value = self.feature_matrix_dict[sample_id][feature_name]
                feature_vector.append(feat.string_convert(feature_value))
                if feature_name == 'RIN-based simple classification':
                    class_name = feat.category_backmap[feature_value]
            outlines.append('%s\t%s\t%s\t%s\t%s\t%s' % (u_ac,aac,target_value_str,self.samples[sample_id].tags,str(self.samples[sample_id].amount_of_structures),'\t'.join(feature_vector)))

            if class_name is None:
                continue
            if class_name not in class_out_lines:
                class_out_lines[class_name] = [header]
            class_out_lines[class_name].append('%s\t%s\t%s\t%s' % (u_ac,aac,target_value_str,'\t'.join(feature_vector)))


        f = open(outfile,'w')
        f.write('\n'.join(outlines))
        f.close()

        if len(class_out_lines) > 0:
            base_name,file_type = outfile.rsplit('.',1)
            if not os.path.isdir(base_name):
                os.mkdir(base_name)

            for class_name in class_out_lines:
                class_outfile = '%s_%s.%s' % (base_name,class_name,file_type)
                f = open(class_outfile,'w')
                f.write('\n'.join(class_out_lines[class_name]))
                f.close()

    def write_feature_coverage_matrix(self, outfile, config: Config):
        prot_wise_feature_coverage = {'total':{}}
        feat_name_list = list(self.features.keys())

        sample_id_list = list(self.samples.keys())

        if self.feature_matrix_dict is None:
            # [non-missing, zeros, total] per protein and feature, counted on the row blocks of each protein
            prot_rows = {}
            for sample_id in sample_id_list:
                prot_rows.setdefault(sample_id[0], []).append(self.sample_pos_dict[sample_id])
            present = [(j, self.feat_pos_dict[feat_name]) for j, feat_name in enumerate(feat_name_list) if feat_name in self.feat_pos_dict]
            present_j = np.array([j for j, _ in present], dtype=np.int64)
            present_cols = np.array([col for _, col in present], dtype=np.int64)
            total_counts = np.zeros((len(feat_name_list), 3), dtype=np.int64)
            for prot_id, rows in prot_rows.items():
                counts = np.zeros((len(feat_name_list), 3), dtype=np.int64)
                if len(present_cols) > 0:
                    block = self.raw_feature_matrix[np.ix_(rows, present_cols)]
                    counts[present_j, 0] = (~np.isnan(block)).sum(axis=0)
                    counts[present_j, 1] = (block == 0).sum(axis=0)
                counts[:, 2] = len(rows)
                total_counts += counts
                prot_wise_feature_coverage[prot_id] = {feat_name: counts[j].tolist() for j, feat_name in enumerate(feat_name_list)}
            prot_wise_feature_coverage['total'] = {feat_name: total_counts[j].tolist() for j, feat_name in enumerate(feat_name_list)}
            feat_iter = []
        else:
            feat_iter = self.features

        for feat_name in feat_iter:
            val_list = self.get_feature_value_vector(sample_id_list, feat_name)
            for index, sample_id in enumerate(sample_id_list):
                prot_id ,aac = sample_id

                if prot_id not in prot_wise_feature_coverage:
                    prot_wise_feature_coverage[prot_id] = {}
                
                if feat_name not in prot_wise_feature_coverage[prot_id]:
                    prot_wise_feature_coverage[prot_id][feat_name] = [0, 0, 0]

                if feat_name not in prot_wise_feature_coverage['total']:
                    prot_wise_feature_coverage['total'][feat_name] = [0, 0, 0]

                if val_list is not None:
                    if val_list[index] is not None and not np.isnan(val_list[index]):
                        prot_wise_feature_coverage[prot_id][feat_name][0] += 1
                        prot_wise_feature_coverage['total'][feat_name][0] += 1
                    if val_list[index] == 0:
                        prot_wise_feature_coverage[prot_id][feat_name][1] += 1
                        prot_wise_feature_coverage['total'][feat_name][1] += 1
                prot_wise_feature_coverage[prot_id][feat_name][2] += 1
                prot_wise_feature_coverage['total'][feat_name][2] += 1
            
        for feat_name in prot_wise_feature_coverage['total']:
            n_non_nulls, n_zeros, total_n = prot_wise_feature_coverage['total'][feat_name]
            cov = n_non_nulls/total_n
            zero_cov = n_zeros/total_n
            if cov < 0.2:
                config.logger.info(f'Low coverage feature: {feat_name} {cov=} {zero_cov=} {n_non_nulls=}')
            if zero_cov > 0.8:
                config.logger.info(f'High zero features: {feat_name} {cov=} {zero_cov=}')

        prot_id_list = list(prot_wise_feature_coverage.keys())
        
        header_words = ['']
        for feat_name in feat_name_list:
            header_words.append(f'{feat_name} cov')
            header_words.append(f'{feat_name} zeros')
            header_words.append(f'{feat_name} total')
        header = '\t'.join(header_words) + '\n'
        outlines = [header]
        for prot_id in prot_id_list:
            words = [prot_id]
            for feat_name in feat_name_list:
                cov = prot_wise_feature_coverage[prot_id][feat_name][0] / prot_wise_feature_coverage[prot_id][feat_name][2]
                prot_wise_feature_coverage[prot_id][feat_name][0] = cov
                zeros = prot_wise_feature_coverage[prot_id][feat_name][1] / prot_wise_feature_coverage[prot_id][feat_name][2]
                prot_wise_feature_coverage[prot_id][feat_name][1] = zeros
                words.append(str(cov))
                words.append(str(zeros))
                words.append(str(prot_wise_feature_coverage[prot_id][feat_name][2]))

            outlines.append('\t'.join(words) + '\n')

        f = open(outfile, 'w')
        f.write(''.join(outlines))
        f.close()

        return prot_wise_feature_coverage
    
    def select_bad_prots(self, prot_wise_feature_coverage, config):
        filter_criteria = {
            'gemme_Ind' : (0.5, 0.5),
        }

        prots_to_remove = []
        for prot_id in prot_wise_feature_coverage:
            for feat_name in prot_wise_feature_coverage[prot_id]:
                if feat_name not in filter_criteria:
                    continue
                cov, zeros, _ = prot_wise_feature_coverage[prot_id][feat_name]

                cov_lower_bound, zeros_upper_bound = filter_criteria[feat_name]

                if cov < cov_lower_bound or zeros > zeros_upper_bound:
                    prots_to_remove.append(prot_id)
        config.logger.info(f'Removing {prots_to_remove} from samples object')
        self.remove_samples_by_prot_ids_after_matrix_transform(prots_to_remove)

    def remove_samples_by_prot_ids_after_matrix_transform(self, prots_to_remove):
        prots_to_remove = set(prots_to_remove)
        samples_to_remove = [sample_id for sample_id in self.samples if sample_id[0] in prots_to_remove]
        for sample_id in samples_to_remove:
            del self.samples[sample_id]
        self.compact_matrix_rows()

    def transform_matrix_dict(self, exclude_feats = None, config= None):
        if self.feature_matrix_dict is None:
            # Already a matrix: only drop the excluded feature columns
            if self.raw_feature_matrix.dtype == object:
                self.raw_feature_matrix = np.array(self.raw_feature_matrix, dtype=np.float64)
            kept_feats = [feat_name for feat_name in self.feat_pos_dict if exclude_feats is None or feat_name not in exclude_feats]
            if len(kept_feats) < len(self.feat_pos_dict):
                kept_cols = [self.feat_pos_dict[feat_name] for feat_name in kept_feats]
                self.raw_feature_matrix = np.ascontiguousarray(self.raw_feature_matrix[:, kept_cols])
                self.feat_pos_dict = {feat_name: pos for pos, feat_name in enumerate(kept_feats)}
            if config is not None and config.verbosity >= 1:
                config.logger.info(f'Feature matrix: {self.raw_feature_matrix.shape}, {len(self.feat_pos_dict)} features')
            return

        raw_feature_matrix = []
        fixed_feat_names = []
        feature_matrix_dict = self.feature_matrix_dict

        self.feat_pos_dict = {}
        self.sample_pos_dict = {}

        for sample_pos, sample_id in enumerate(feature_matrix_dict):
            self.sample_pos_dict[sample_id] = sample_pos
            raw_feature_matrix.append([])
            if sample_pos == 0:
                #print(f'{sample_id=} {feature_matrix_dict[sample_id]=}')
                for feat_name in feature_matrix_dict[sample_id]:
                    if exclude_feats is not None:
                        if feat_name in exclude_feats:
                            continue
                    self.feat_pos_dict[feat_name] = len(self.feat_pos_dict)

                    if config is not None and config.verbosity >= 4:
                        config.logger.info(f'Setting {feat_name=} to feat_pos_dict')

                    fixed_feat_names.append(feat_name)
            for feat_name in fixed_feat_names:
                try:
                    raw_feature_matrix[sample_pos].append(feature_matrix_dict[sample_id][feat_name])
                except KeyError:
                    raw_feature_matrix[sample_pos].append(0.)
        # None becomes NaN
        self.raw_feature_matrix = np.array(raw_feature_matrix, dtype=np.float64)
        self.feature_matrix_dict = None


    def get_feature_value(self, sample_id, feat_name):
        try:
            feat_value = self.raw_feature_matrix[self.sample_pos_dict[sample_id]][self.feat_pos_dict[feat_name]]
        except KeyError:
            feat_value = None
        # NaN is the missing value of the matrix
        if feat_value is not None and feat_value != feat_value:
            feat_value = None
        return feat_value

    def get_feature_value_vector(self, sample_ids, feat_name):
        try:
            feature_id = self.feat_pos_dict[feat_name]
        except KeyError as err:
            print(f'Error in get_feature_value_vector: {err=}')
            return None
        sample_positions = [self.sample_pos_dict[sample_id] for sample_id in sample_ids]
        values = self.raw_feature_matrix[sample_positions, feature_id].tolist()
        return [None if value is None or value != value else value for value in values]

    def get_feat_matrix_from_feat_names(self, feat_names : list[str], config, get_feat_id_dict = False):
        feat_id_vec = []
        if get_feat_id_dict:
            feat_id_dict = {}
        for feat_id, feat_name in enumerate(feat_names):
            if get_feat_id_dict:
                feat_id_dict[feat_name] = feat_id
            try:
                feat_id_vec.append(self.feat_pos_dict[feat_name])
            except KeyError:
                if config.verbosity >= 3:
                    config.logger.warning(f'In get_feat_matrix_from_feat_names: {feat_name} not in feat_pos_dict')
                if feat_name[0:3] == 'oh_':
                    feat_id_vec.append(-1)
                else:
                    feat_id_vec.append(None)
            except TypeError:
                config.logger.warning(f'{self.feat_pos_dict[:100]=}')

        sample_pos_vec = list(range(len(self.raw_feature_matrix)))
        if get_feat_id_dict:
            return get_feat_matrix(self.raw_feature_matrix, feat_id_vec, sample_pos_vec), feat_id_vec, feat_id_dict
        return get_feat_matrix(self.raw_feature_matrix, feat_id_vec, sample_pos_vec), feat_id_vec

    def get_skewed_feat_matrices_from_ids(self, sample_ids, feat_names, thresh):
        feat_id_vec = []
        for feat_name in feat_names:
            feat_id_vec.append(self.feat_pos_dict[feat_name])

        sample_pos_vec_l = []
        sample_pos_vec_r = []
        for sample_id in sample_ids:
            target_value = self.samples[sample_id].targetValue
            if target_value < thresh:
                try:
                    sample_pos_vec_l.append(self.sample_pos_dict[sample_id])
                except KeyError:
                    sample_pos_vec_l.append(None)
            else:
                try:
                    sample_pos_vec_r.append(self.sample_pos_dict[sample_id])
                except KeyError:
                    sample_pos_vec_r.append(None)

        return get_feat_matrix(self.raw_feature_matrix, feat_id_vec, sample_pos_vec_l), get_feat_matrix(self.raw_feature_matrix, feat_id_vec, sample_pos_vec_r)

    def setGeometricDistanceMap(self,config):
        if self.geometric_distance_map is not None and self.current_geometric_exponent == config.geometric_exponent:
            return
        t0 = time.time()
        config.logger.info('Start of GDM calculation with gexp:', config.geometric_exponent)
        self.geometric_distance_map = {}
        bin_size = len(self.samples) // config.proc_n
        if len(self.samples) % config.proc_n != 0:
            bin_size += 1
        targets = [self.samples[sample_id].targetValue for sample_id in self.samples]
        sample_ids = list(self.samples.keys())
        store = ray.put((targets,sample_ids,bin_size,config))

        t1 = time.time()
        config.logger.info('GDM store created, time:',t1-t0)

        block_ids = []

        for i in range(config.proc_n):
            block_ids.append(calc_gdm_submatrix.remote(store,i))

        block_results = ray.get(block_ids)

        t2 = time.time()
        config.logger.info('GDM submatrices calculated, time:',t2-t1)

        for submatrix in block_results:
            self.geometric_distance_map.update(submatrix)

        self.current_geometric_exponent = config.geometric_exponent

        t3 = time.time()
        config.logger.info('GDM calculation complete, time:',t3-t2)

    def detectOutliers(self,config):
        N = len(self.samples)
        if N==0:
            raise Exception('Called detectOutliers with empty samples object')
        num_of_bins = int(math.log10(N)*4)

        min_bin_size = int(math.log10(N))

        lower_bin_thresh = float('inf')
        upper_bin_thresh = -float('inf')
        for sample_id in self.samples:
            tv = self.samples[sample_id].targetValue
            if tv > upper_bin_thresh:
                upper_bin_thresh = tv
            if tv < lower_bin_thresh:
                lower_bin_thresh = tv

        bin_size = (upper_bin_thresh - lower_bin_thresh)/num_of_bins
        bins = {}
        for sample_id in self.samples:
            tv = self.samples[sample_id].targetValue
            bin_lower = int((tv-lower_bin_thresh)//bin_size)
            if bin_lower not in bins:
                bins[bin_lower] = []
            bins[bin_lower].append(sample_id)

        deletion_list = []
        for bin_lower in bins:
            if len(bins[bin_lower]) <= min_bin_size:
                for sample_id in bins[bin_lower]:
                    if config.verbosity >= 3 or len(deletion_list) < 10:
                        config.logger.info(f'Outlier detection: {sample_id=} {self.samples[sample_id].targetValue=}')
                    deletion_list.append(sample_id)
        config.logger.info(f'Outlier detection removed {len(deletion_list)} of {N} samples')
        self.removeSamples(deletion_list)

    def printPureMixedProportion(self,config):
        pure_map = {}
        mixed_map = set()
        prot_set = set()
        target_value_map = {}
        for (u_ac,aac) in self.samples:
            prot_set.add(u_ac)
            targetValue = self.samples[(u_ac,aac)].targetValue
            if targetValue not in target_value_map:
                target_value_map[targetValue] = 0
            target_value_map[targetValue] += 1
            if u_ac in mixed_map:
                continue
            if u_ac not in pure_map:
                pure_map[u_ac] = targetValue
            else:
                if targetValue == pure_map[u_ac]:
                    continue
                else:
                    mixed_map.add(u_ac)
                    del pure_map[u_ac]

        n_prot = len(prot_set)

        if n_prot == 0:
            return

        ppr = len(pure_map)/n_prot
        mpr = len(mixed_map)/n_prot
        if not config.regression:
            config.logger.info(f'# of Proteins: {n_prot}, # of Variants: {len(self.samples)}, target value balance: {target_value_map}')
        else:
            config.logger.info(f'# of Proteins: {n_prot}, # of Variants: {len(self.samples)}')
        config.logger.info(f'Pure protein proportion: {ppr}, Mixed protein proportion: {mpr}')
        return

    def undoBalancing(self,config):
        for sample_id in self.filtered_samples:
            self.samples[sample_id].testtrain = 'train'
        self.calcVectors(config)
        return

    def getVectors(self,config,forceCalc=False):
        if self.test_feature_matrix is None or forceCalc:
            self.calcVectors(config)
        return self.test_feature_matrix,self.test_targets,self.train_feature_matrix,self.train_targets

    def get_test_data_for_feature_list(self, extern_feature_list: list[str], config, extern_features: dict[str, Feature]):
        test_feature_matrix, feat_id_vec, feat_id_dict = self.get_feat_matrix_from_feat_names(extern_feature_list, config, get_feat_id_dict=True)
        test_targets = []
        sample_id_list = []
        for sample_id in self.samples:
            sample_id_list.append(sample_id)
            test_targets.append(self.samples[sample_id].targetValue)

        cat_vec = []
        for feat_name in extern_feature_list:
            if extern_features[feat_name].category_map is None:
                extern_features[feat_name].category_map = {}
                extern_features[feat_name].category_counter = 0
                extern_features[feat_name].category_backmap = {}
            if self.features[feat_name].f_type == 'categorical':
                cat_vec.append('c')
                for sample_id, _ in enumerate(test_feature_matrix):
                    feat_id = feat_id_dict[feat_name]
                    old_encoded_value = test_feature_matrix[sample_id][feat_id]
                    if old_encoded_value is None or old_encoded_value != old_encoded_value:
                        decoded_value = None
                    else:
                        decoded_value = self.features[feat_name].category_backmap[int(old_encoded_value)]
                    if decoded_value not in extern_features[feat_name].category_map:
                        extern_features[feat_name].category_map[decoded_value] = extern_features[feat_name].category_counter
                        extern_features[feat_name].category_backmap[extern_features[feat_name].category_counter] = decoded_value
                        extern_features[feat_name].category_counter += 1
                    new_encoded_value = extern_features[feat_name].category_map[decoded_value]
                    test_feature_matrix[sample_id][feat_id] = new_encoded_value
            else:
                cat_vec.append('q')

        return test_feature_matrix, test_targets, sample_id_list, feat_id_vec, cat_vec

    def filterSamplesByMappedStructures(self,config):
        del_list = []
        for sample_id in self.samples:
            if self.samples[sample_id].amount_of_structures is None:
                self.samples[sample_id].amount_of_structures = 0
            if self.samples[sample_id].amount_of_structures < config.structure_threshold:
                del_list.append(sample_id)
        config.logger.info('Filtered ',len(del_list),' samples in the structure filtering')
        self.removeSamples(del_list)

    def standardFilter(self, config, filter_synon = False):
        del_list = []
        num_tv_viol = 0
        num_ff = 0
        num_tf = 0
        num_pf = 0
        num_sf = 0
        N = len(self.samples)
        for sample_id in self.samples:
            sample = self.samples[sample_id]
            u_ac, aac = sample_id

            if filter_synon:
                if aac[0] == aac[-1]:
                    del_list.append(sample_id)
                    num_sf += 1
                    continue

            for tag in sample.tags.split(','):
                if tag in config.tag_filter:
                    del_list.append(sample_id)
                    num_tf += 1
                    break

            if u_ac in config.protein_filter:
                del_list.append(sample_id)
                num_pf += 1
                continue

            if ((sample.targetValue is None or sample.targetValue == 'None') and not config.predict_mode) or sample.targetValue != sample.targetValue or sample.targetValue in config.targetFilter:
                if sample.targetValue in config.target_translator:
                    sample.targetValue = config.target_translator[sample.targetValue]
                else:
                    del_list.append(sample_id)
                    num_tv_viol += 1
                    continue
                
            
        config.logger.info(f'Filtered {len(del_list)} of {N} samples in the standard filtering')
        config.logger.info(f'{num_tv_viol} due to target value violation')
        config.logger.info(f'{num_ff} due to feature filter')
        config.logger.info(f'{num_tf} due to tag filter')
        config.logger.info(f'{num_pf} due to protein filter')
        config.logger.info(f'{num_sf} due to synonoumus filter')
        self.removeSamples(del_list)

    def adjustParameterRanges(self,config):
        if config.tvmb_rank_half_step[-1] == 'max':
            config.tvmb_rank_half_step[-1] = len(self.feature_names) -1
        if config.tvpmb_rank_half_step[-1] == 'max':
            config.tvpmb_rank_half_step[-1] = len(self.feature_names) -1
        if config.confusion_rank_threshold_bounds[-1] == 'max':
            config.confusion_rank_threshold_bounds[-1] = len(self.feature_names) -1
        return


cv_slots = ['slices', 'slice_ids', 'cv_counter', 'isSlice', 'slice_slices']

class CrossValidation(Slotted_obj):
    __slots__ = cv_slots
    def __init__(self):
        self.slices = {}
        self.slice_ids = []
        self.cv_counter = 0
        self.isSlice = False
        self.slice_slices = None

    def getCurrentSlice(self):
        return self.slices[self.slice_ids[self.cv_counter]]

    def getNextSlice(self, config):
        self.cv_counter += 1
        if self.cv_counter >= len(self.slice_ids):
            config.logger.error('ERROR: cv_counter out of bounds')
            return None
        return ray.get(self.slices[self.slice_ids[self.cv_counter]])

    def reset(self):
        self.cv_counter = 0

    def reset_confusion_maps(self):
        for slice_id in self.slices:
            self.slices[slice_id].reset_confusion_maps()

    def serialize(self):
        l = []
        for slot in self.__slots__:
            l.append(self.__getattribute__(slot))
        return l

class Tag_nested_cv(CrossValidation):
    def __init__(self, sampleSpace, config):
        super().__init__()

        """
        if config.tag_based_crossValidation != None:
            if self.tag_based_crossValidation_counter == len(config.tag_based_crossValidation):
                return None
            test_tag = config.tag_based_crossValidation[self.tag_based_crossValidation_counter]
            self.tag_based_crossValidation_counter += 1
            if print_out:
                config.logger.info('Tag-based crossvailidation, test tag: ',test_tag)
            for sample_id in self.samples:
                sample = self.samples[sample_id]
                tags = sample.tags
                test_sample = False
                for tag in tags.split(','):
                    if tag == test_tag:
                        test_sample = True

                if test_sample:
                    self.samples[sample_id].testtrain = 'test'
                else:
                    self.samples[sample_id].testtrain = 'train'

            if config.balanceSubsampling != None:
                self.balanceSubSampleTrainSet(config)

            self.calcVectors(config,print_out=print_out,vector_store_id=test_tag)

            return self.getVectors(config),[]
        """

class FullSlice(CrossValidation):
    def __init__(self, sampleSpace, config, internal_cv = None, train_equal_test = False):
        super().__init__()

        if config.verbosity >= 1:
            config.logger.info(f'Init of FullSlice: {internal_cv is None} {train_equal_test}')
        
        self.slice_ids.append(0)

        full_slice_obj = CrossValidationSlice(train_ids = list(sampleSpace.samples.keys()), raw_feature_names = sampleSpace.feature_names, sample_dict = sampleSpace.samples, config = config, train_equal_test = train_equal_test)

        if internal_cv is not None:
            full_slice_obj.slice_slices = []
            for slice_id in internal_cv.slices:
                if config.verbosity >= 1:
                    config.logger.info(f'Adding a slice_slice to the FullSlice: {slice_id}')
                full_slice_obj.slice_slices.append(internal_cv.slices[slice_id])

        self.slices[0] = full_slice_obj

class X_fold_cv(CrossValidation):
    def __init__(self, sampleSpace, config):
        super().__init__()
        prot_map = {}
        for (u_ac,aac) in sampleSpace.samples:
            if u_ac not in prot_map:
                prot_map[u_ac] = 0
            prot_map[u_ac] += 1

        self.prot_map = prot_map
        self.prots = list(prot_map.keys())

        self.cross_slice_size = len(sampleSpace.samples)/config.crossValidation_fold

        self.sample_ids = list(sampleSpace.samples.keys())
        self.tag_based_separation_done = False

        self.tag_based_crossValidation_counter = 0

        n_of_assigned_prots = 0
        n_of_assigned_samples = 0

        for cv_counter in range(config.crossValidation_fold):
            self.slice_ids.append(cv_counter)
            test_prots = set()
            slice_size = 0
            test_samples = set()
            while slice_size < self.cross_slice_size:

                if config.prot_based_separation:
                    if n_of_assigned_prots == len(self.prots):
                        break
                    test_prot = self.prots[n_of_assigned_prots]
                    prot_size = self.prot_map[test_prot]
                    new_slice_size = slice_size + prot_size

                    if abs(new_slice_size-self.cross_slice_size) > abs(slice_size-self.cross_slice_size):
                        break
                    slice_size = new_slice_size
                    test_prots.add(test_prot)
                    n_of_assigned_prots += 1
                else:
                    if n_of_assigned_samples == len(self.sample_ids):
                        break
                    test_samples.add(self.sample_ids[n_of_assigned_samples])
                    n_of_assigned_samples += 1
                    slice_size += 1
            config.logger.info(f'Next crossvalidation slice, size: {slice_size}')

            if config.prot_based_separation:
                test_ids, train_ids = splitDataSet(config, sampleSpace.samples, specific_id=test_prots, protein_wise=True)
            else:
                test_ids, train_ids = splitDataSet(config, sampleSpace.samples, specific_id=test_samples, protein_wise=False)

            cv_slice = CrossValidationSlice(test_ids = test_ids, train_ids = train_ids, raw_feature_names = sampleSpace.feature_names, sample_dict = sampleSpace.samples, config = config)
            self.slices[cv_counter] = cv_slice

@ray.remote(max_calls = 1)
def init_lopo_slice(store, test_prots, train_prots, cv_counter = None, protein_wise=True):
    config, sample_dict, raw_feature_names, sample_list = store

    test_ids, train_ids = splitDataSet(config, sample_list, specific_id=test_prots, protein_wise=protein_wise)

    if cv_counter is None:
        for name in test_prots:
            break
    else:
        name = cv_counter

    if config.verbosity >= 2:
        config.logger.info(f'Init LOPO slice: {name}: Test set size: {len(test_ids)}, Train set size: {len(train_ids)}')

    cv_slice = CrossValidationSlice(test_ids = test_ids, train_ids = train_ids, raw_feature_names = raw_feature_names, sample_dict = sample_dict, config = config, name = name, train_prots = train_prots, test_prots = test_prots)
    if config.verbosity >= 5:
        #config.logger.info(f'Features of slice {cv_slice.name}:\n{cv_slice.features}')
        cv_slice.featureSanityCheck()
    
    return (name, cv_slice)
    #return (name, cv_slice)

class LOPO(CrossValidation):
    def __init__(self, sampleSpace: SampleSpace, config):
        t0 = time.time()
        config.logger.info('-- LOPO initialization --')
        super().__init__()
        prots = set([])
        for (u_ac,aac) in sampleSpace.samples:
            prots.add(u_ac)

        self.prots = list(prots)

        init_ids = []

        config.logger.info(f'Put sample space into store: {prots}')

        store = ray.put((config, sampleSpace.samples, sampleSpace.feature_names, prots))

        t1 = time.time()
        config.logger.info('Start paralell slice init, with number of processes:',config.proc_n,',time:',t1-t0)

        for prot in self.prots:
            init_ids.append(init_lopo_slice.remote(store, set([prot])))

        init_results = ray.get(init_ids)

        t2 = time.time()
        config.logger.info('Paralell slice init finished, time:',t2-t1)

        for prot, cv_slice in init_results:
            self.slices[prot] = unpack(cv_slice)
            self.slice_ids.append(prot)

def write_weight_map(weight_map, outfile):
    lines = []
    for prot_id in weight_map:
        lines.append(f'{prot_id}\t{weight_map[prot_id]}\n')
    f = open(outfile,'w')
    f.write(''.join(lines))
    f.close()

def parse_splits(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()

    test_id_dict = {}
    for line in lines:
        words = line[:-1].split('\t')
        dms_id = words[0]
        aac = words[1]
        test_bin = int(words[2])

        if test_bin not in test_id_dict:
            test_id_dict[test_bin] = set()
        test_id_dict[test_bin].add((dms_id, aac))

    return test_id_dict

class Given_split(CrossValidation):
    __slots__ = cv_slots
    def __init__(self, samplespace, config):
        super().__init__()
        
        test_id_dict = parse_splits(config.path_to_splits_file)
        for slice_id in test_id_dict:
            test_ids = test_id_dict[slice_id]
            train_ids = []
            for sample_id in samplespace.samples:
                if sample_id not in test_ids:
                    train_ids.append(sample_id)
            cv_slice = CrossValidationSlice(test_ids = test_ids, train_ids = train_ids, sample_dict= samplespace.samples, raw_feature_names = samplespace.feature_names, config = config)
            self.slices[slice_id] = cv_slice
            self.slice_ids.append(slice_id)

            slice_slices = []
            for sub_slice_id in test_id_dict:
                if sub_slice_id == slice_id:
                    continue
                sub_slice_test_ids = test_ids | test_id_dict[sub_slice_id]
                train_ids = []
                for sample_id in samplespace.samples:
                    if sample_id not in sub_slice_test_ids:
                        train_ids.append(sample_id)
                cv_slice_slice = CrossValidationSlice(test_ids = test_id_dict[sub_slice_id], train_ids = train_ids, raw_feature_names = samplespace.feature_names, sample_dict = samplespace.samples, config = config, name = f'{cv_slice.name}_subslice_{sub_slice_id}')
                slice_slices.append(cv_slice_slice)

            cv_slice.slice_slices = slice_slices
            self.slice_slices[slice_id] = slice_slices

def get_datasail_protein_split_assignment(config) -> dict[str, int]:
    # Sequence-identity aware clustering of the proteins in config.path_to_sequence_fasta into
    # config.crossValidation_fold folds, so that similar/related proteins don't end up split
    # across folds (which DataSAIL_cv itself relies on for the xgboost cross validation).
    datasail_test_size = 100 // config.crossValidation_fold
    splits = [datasail_test_size] * config.crossValidation_fold
    names = [f'split_{x}' for x in range(config.crossValidation_fold)]

    eps = 0.05
    if config.random_split:
        technique = ['R']
    else:
        technique = ['C1e']

    try:
        if config.verbosity >= 4:
            config.logger.info(f'Call of datasail with: e_data: {config.path_to_sequence_fasta}, splits: {splits}, names: {names}')
            raw_datasail_splits = datasail(
                e_data = config.path_to_sequence_fasta,
                splits = splits,
                techniques = technique,
                names = names,
                e_type = 'P',
                solver = 'SCIP',
                epsilon = eps,
                overflow = 'assign',
                e_sim = 'mmseqs',
                verbose = 'I')
        else:
            with contextlib.redirect_stdout(None):
                raw_datasail_splits = datasail(
                    e_data = config.path_to_sequence_fasta,
                    splits = splits,
                    techniques = technique,
                    names = names,
                    e_type = 'P',
                    solver = 'SCIP',
                    epsilon = eps,
                    overflow = 'assign',
                    e_sim = 'mmseqs')
    except ValueError:
        raw_datasail_splits = datasail(
                e_data = config.path_to_sequence_fasta,
                splits = splits,
                techniques = technique,
                names = names,
                e_type = 'P',
                solver = 'SCIP',
                epsilon = eps,
                overflow = 'assign'
                )

    datasail_splits = raw_datasail_splits[0][technique[0]][0]

    prot_fold = {}
    for prot_id in datasail_splits:
        split_name = datasail_splits[prot_id]
        prot_fold[prot_id] = int(split_name.split('_')[1])
    return prot_fold


class DataSAIL_cv(CrossValidation):
    __slots__ = cv_slots + ['prots', 'subslice_refs']
    def __init__(self, samples_store_id = None, sampleSpace: SampleSpace | None = None, config: Config = None, as_list = None):
        t0 = time.time()
        if as_list is not None:
            for slot_number, slot in enumerate(self.__slots__):
                self.__setattr__(slot, as_list[slot_number])
            return
        super().__init__()
        if sampleSpace is None:
            return

        prots = set()
        sample_list: list[tuple[str, str]] = list(sampleSpace.samples.keys())

        for (prot_id, aac) in sample_list:
            prots.add(prot_id)

        self.prots = list(prots)

        t1 = time.time()
        if config.verbosity >= 2:
            config.logger.info(f'Time for init DataSAIL_cv Part 1: {t1-t0}')

        if not config.random_split:
            prot_fold = get_datasail_protein_split_assignment(config)

        t2 = time.time()
        if config.verbosity >= 2:
            config.logger.info(f'Time for init DataSAIL_cv Part 2: {t2-t1}')

        train_test_pairs = {}
        if not config.random_split:
            
            for cv_counter in range(config.crossValidation_fold):
                train_test_pairs[cv_counter] = [[], [], {}] #train - test - subslices test

            for prot_id, cv_counter in prot_fold.items():
                for cv in train_test_pairs:
                    if cv == cv_counter:
                        train_test_pairs[cv][1].append(prot_id)
                    else:
                        train_test_pairs[cv][0].append(prot_id)
                        if cv_counter not in train_test_pairs[cv][2]:
                            train_test_pairs[cv][2][cv_counter] = set()
                        train_test_pairs[cv][2][cv_counter].add(prot_id)

        t3 = time.time()
        if config.verbosity >= 2:
            config.logger.info(f'Time for init DataSAIL_cv Part 3: {t3-t2}')

        if config.verbosity >= 4:
            config.logger.info(f'Init datasail:\nSplits: {train_test_pairs}\n')

        store = ray.put((config, sampleSpace.samples, sampleSpace.feature_names, sample_list))

        t4 = time.time()
        if config.verbosity >= 2:
            config.logger.info(f'Time for init DataSAIL_cv Part 4: {t4-t3}')

        init_ids = []
        if not config.random_split:
            for cv_counter in train_test_pairs:
                train_set, test_set, _ = train_test_pairs[cv_counter]

                if config.verbosity >= 2:
                    config.logger.info(f'Init datasail slice {cv_counter=}: {len(test_set)=}')

                init_ids.append(init_lopo_slice.remote(store, test_set, train_set, cv_counter = cv_counter))
        else:
            for cv_counter in range(config.crossValidation_fold):
                if config.verbosity >= 2:
                    config.logger.info(f'Init datasail slice {cv_counter=}')

                init_ids.append(init_lopo_slice.remote(store, None, None, cv_counter = cv_counter, protein_wise=False))

        init_results = ray.get(init_ids)

        t5 = time.time()
        if config.verbosity >= 2:
            config.logger.info(f'Time for init DataSAIL_cv Part 5: {t5-t4}')

        file_path_dict = {}
        self.subslice_refs = {}

        for cv_counter, cv_slice in init_results:

            
            cv_slice.test_slice_id = cv_counter
            cv_slice.train_slice_ids = []
            for train_cv_counter in range(config.crossValidation_fold):
                if train_cv_counter != cv_counter:
                    cv_slice.train_slice_ids.append(train_cv_counter)

            if config.verbosity >= 4:
                config.logger.info(f'Init CV slice {cv_counter=} {cv_slice.train_slice_ids=}')

            if config.verbosity >= 6:
                cv_slice.featureSanityCheck(verbose = True)

            self.slice_ids.append(cv_counter)
            if not config.random_split:
                if train_test_pairs[cv_counter][2] is not None:
                    cv_slice.subslices = {}
                    for subslice_counter in train_test_pairs[cv_counter][2]:
                        cv_slice.subslices[subslice_counter] = (train_test_pairs[cv_counter][2][subslice_counter])
            else:
                cv_slice.subslices = {}
                for subslice_counter in range(config.crossValidation_fold-1):
                    cv_slice.subslices[subslice_counter] = None


            detectBiasedFeaturesByMeanCorrelation(
                config, cv_slice, samples_store_id, samples=sampleSpace, dummy_call=True
            )

            file_paths, prot_vec_path = put_data_to_tmp_storage(cv_slice, sampleSpace, config)

            file_path_dict[cv_counter] = file_paths, prot_vec_path

            slice_slices = []

            for i, subslice_test_id in enumerate(cv_slice.subslices.keys()):
                subslice_test_proteins = cv_slice.subslices[subslice_test_id]
                try:
                    remaining_prots = set(cv_slice.train_prots) - subslice_test_proteins
                except TypeError:
                    remaining_prots = []

                if cv_slice.train_equal_test:
                    ignore_samples = None
                else:
                    ignore_samples = set(cv_slice.test_sample_ids)
                test_ids, train_ids = splitDataSet(config, sample_list, specific_id=subslice_test_proteins, protein_wise=(not config.random_split), ignore_samples=ignore_samples)

                cv_slice_slice = CrossValidationSlice(
                    test_ids,
                    train_ids,
                    raw_feature_names=sampleSpace.feature_names,
                    sample_dict=sampleSpace.samples,
                    config=config,
                    name=f"{cv_slice.name}_subslice_{i}",
                    train_prots=remaining_prots,
                    feature_names=cv_slice.feature_names,
                )
                
                cv_slice_slice.test_slice_id = subslice_test_id
                cv_slice_slice.train_slice_ids = []
                for train_cv_counter in range(config.crossValidation_fold):
                    if train_cv_counter == cv_counter:
                        continue
                    if train_cv_counter == subslice_test_id:
                        continue
                    cv_slice_slice.train_slice_ids.append(train_cv_counter)

                slice_slices.append(ray.put(pack(cv_slice_slice)))

            cv_slice.slice_slices = slice_slices

            self.subslice_refs[cv_counter] = cv_slice.slice_slices

            self.slices[cv_counter] = ray.put(pack(cv_slice))

        config.file_path_dict = file_path_dict

        t6 = time.time()
        if config.verbosity >= 2:
            config.logger.info(f'Time for init DataSAIL_cv Part 6: {t6-t5}')


def put_data_to_tmp_storage(cv_slice: CrossValidationSlice, samples: SampleSpace, config: Config):

    feat_matrix = get_raw_feat_matrix_from_sample_ids(cv_slice.test_sample_ids, samples.sample_pos_dict, samples.raw_feature_matrix)

    file_paths: list[tuple[str, str]] = []

    # These chunks only partition the host side storage. The batches that reach the GPU
    # are cut from them by support_classes.Iterator, sized to the pool of the training
    # process and the selected features (see extmem_batch_bytes).
    gmem = get_gpu_memory()[0]*config.vram_limit
    sub_share = 1 / (config.threads_per_gpu * config.crossValidation_fold * (config.crossValidation_fold-1))
    num_of_batches = max([1, int(1 * (len(feat_matrix)/ (600 * gmem*sub_share)))])

    batch_size = len(feat_matrix) // num_of_batches
    if len(feat_matrix) % num_of_batches != 0:
        batch_size += 1

    config.logger.info(f'{gmem=} {sub_share=} {num_of_batches=} {batch_size=} {sys.getsizeof(feat_matrix)//num_of_batches=} {len(feat_matrix)=}')

    prot_id_vec = cv_slice.get_encoded_test_prot_vec()
    #prot_vec_path = f'{dump_precursor}_prot_id_vec.npy'
    #prot_id_vec.dump(prot_vec_path)
    prot_vec_ref = ray.put(prot_id_vec)

    for batch_nr in range(num_of_batches):
        batch = np.array(feat_matrix[batch_nr*batch_size:(batch_nr+1)*batch_size], dtype=np.float32)
        if len(batch) == 0:
            continue
        
        y_batch = np.array(cv_slice.test_targets[batch_nr*batch_size:(batch_nr+1)*batch_size])

        #batch_path = f'{dump_precursor}_X_{batch_nr}_test.npy'
        #batch.dump(batch_path)
        batch_ref = ray.put(batch)
        
        #cp.save(batch_path, batch)

        #y_batch_path = f'{dump_precursor}_y_{batch_nr}_test.npy'
        #y_batch.dump(y_batch_path)

        y_batch_ref = ray.put(y_batch)

        #file_paths.append((batch_path, y_batch_path))
        file_paths.append((batch_ref, y_batch_ref))

    return file_paths, prot_vec_ref