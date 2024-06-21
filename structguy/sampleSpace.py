import os
import sys
import numpy as np
import math
import random
import ray
import time
import pickle
import copy

from scipy import stats

from datasail.sail import datasail

from structguy.sequence_util import parseFromFasta
from structguy.support_classes import CrossValidationSlice, Feature

from structman.base_utils.base_utils import pack, unpack


class Sample:
    def __init__(self,sample_id,nr):
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


def splitDataSet(config, sample_dict, specific_id=None, protein_wise=False, debug=0, skip_protein = None, ignore_samples = None):
    split_rate = config.split_rate
    total_size = len(sample_dict)
    test_size = max([1,int(total_size*split_rate)])
    train_size = total_size-test_size

    test_ids = []
    train_ids = []

    if not protein_wise:
        #simple random split
        if specific_id == None:
            test_nrs = set(np.random.choice(total_size,test_size,replace=False))
            for sample_id in sample_dict:
                sample = sample_dict[sample_id]
                if sample.nr in test_nrs:
                    test_ids.append(sample_id)
                else:
                    train_ids.append(sample_id)
        else:
            for sample_id in sample_dict:
                if sample_id in specific_id:
                    test_ids.append(sample_id)
                else:
                    train_ids.append(sample_id)

        if debug >= 1:
            print('Train size: ',train_size,' ,Test size: ',test_size)
    else:
        # Protein-nested split
        test_proteins = set()
        test_set_sum = 0
        if specific_id == None:
            protein_sizes = {}
            for u_ac,aac in sample_dict:
                if not u_ac in protein_sizes:
                    protein_sizes[u_ac] = 1
                else:
                    protein_sizes[u_ac] += 1
            while test_set_sum < (test_size - test_size*split_rate):
                random_protein = random.choice(list(protein_sizes.keys()))
                if not random_protein in test_proteins:
                    test_proteins.add(random_protein)
                    test_set_sum += protein_sizes[random_protein]

        else:
            test_proteins = specific_id

        for (u_ac,aac) in sample_dict:
            if skip_protein is not None:
                if u_ac == skip_protein:
                    continue
            if ignore_samples is not None:
                if (u_ac, aac) in ignore_samples:
                    continue

            if u_ac in test_proteins:
                test_ids.append((u_ac,aac))
            else:
                train_ids.append((u_ac,aac))

        #print 'Protein Id based sample splitting:\nTest set Ids: ',test_proteins,'\nTestset size: ',test_set_sum
        if debug >= 1:
            print('Train size: ',train_size,' ,Test size: ',test_size)
    if len(test_ids) == 0:
        if ignore_samples is not None:
            ignored_prots = set()
            for (prot_id, aac) in ignore_samples:
                ignored_prots.add(prot_id)
        else:
            ignored_prots = None
        print(f'Splitted into empty test set: {specific_id}, {ignored_prots}')
    return test_ids, train_ids

class SampleSpace:
    def __init__(self, config):
        self.samples = {}
        self.features = {}
        self.feature_matrix_dict = {}
        self.raw_feature_matrix = []
        self.sample_nr = 0
        self.feature_names = []
        self.feat_pos_dict = {}
        self.sample_pos_dict = {}

        self.vector_store = {}
        self.current_vector = None

        self.geometric_distance_map = {}
        self.current_geometric_exponent = None

        if config.addBias:
            self.addFeature('Protein bias','real',group='amino acid property',default_value='0.5')

        seq_map , _ = parseFromFasta(config.path_to_sequence_fasta)

        self.sequence_map = seq_map
        self.impute_map = {}

    def addFeature(self,name,f_type,group=None,default_value=None,mutation_specific=False):
        feat = Feature(name = name, f_type = f_type,group=group,default_value=default_value,mutation_specific=mutation_specific)
        self.features[name] = feat
        self.feature_names.append(name)

    def removeFeature(self, feat_name):
        del self.features[feat_name]
        self.feature_names.remove(feat_name)


    def print_feat_types(self):
        for featname in self.features:
            print(f'{featname} - {self.features[featname].f_type}')

    def print_stats(self):
        prot_ids = set()
        for sample_id in self.samples:
            prot_id, aac = sample_id
            prot_ids.add(prot_id)

        print(f'State of samples object: Proteins: {len(prot_ids)}, Samples: {len(self.samples)}, Features: {len(self.feature_names)}')

    def oneHotify(self,feat_name):
        feat = self.features[feat_name]
        f_type = feat.f_type
        if not f_type == 'categorical':
            print('Warning: Cannot oneHotify non-categorical features')
            return
        for category in feat.category_map:
            oh_name = 'oh_%s_%s' % (feat_name,category)
            self.addFeature(oh_name,'binary',group=feat.group,default_value=0)

        for sample_id in self.feature_matrix_dict:
            cat_value = self.feature_matrix_dict[sample_id][feat_name]
            category = feat.category_backmap[cat_value]
            oh_name = 'oh_%s_%s' % (feat_name,category)
            self.addValue(sample_id,1,oh_name)

    def oneHotifyAll(self):
        print('Convert all categorical features to 1-hot encodings')
        del_list = []
        for feat_name in self.features:
            if self.features[feat_name].f_type == 'categorical':
                del_list.append(feat_name)
        for feat_name in del_list:
            self.oneHotify(feat_name)
            del self.features[feat_name]
            for sample_id in self.feature_matrix_dict:
                del self.feature_matrix_dict[sample_id][feat_name]

        self.feature_names = list(self.features.keys())
        self.fillDefaultValues()

    def external_impute(self, external_impute):
        for feat_name in self.features:
            feat = self.features[feat_name]
            if feat.f_type == 'categorical':
                continue
            if feat_name in external_impute:
                impute_value = external_impute[feat_name]
            else:
                impute_value = 0.
            for sample_id in self.feature_matrix_dict:
                if not feat_name in self.feature_matrix_dict[sample_id]:
                    self.feature_matrix_dict[sample_id][feat_name] = impute_value
                elif self.feature_matrix_dict[sample_id][feat_name] is None:
                    self.feature_matrix_dict[sample_id][feat_name] = impute_value

    def get_bounds(self, feat_name):
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
        for feat_name in self.features:
            feat = self.features[feat_name]
            if feat.f_type == 'categorical':
                continue
            #print(feat_name)
            min_val, max_val = self.get_bounds(feat_name)
            if min_val is not None:
                impute_value = min_val - abs(max_val - min_val)
            else:
                impute_value = 0
            self.impute_map[feat_name] = impute_value
            for sample_id in self.feature_matrix_dict:
                if not feat_name in self.feature_matrix_dict[sample_id]:
                    self.feature_matrix_dict[sample_id][feat_name] = impute_value
                elif self.feature_matrix_dict[sample_id][feat_name] is None:
                    self.feature_matrix_dict[sample_id][feat_name] = impute_value

    def dump_impute_map(self, outfile):
        with open(outfile, 'wb') as output:
            pickle.dump(self.impute_map, output, pickle.HIGHEST_PROTOCOL)

    def fusePositions(self):
        pos_map = {}
        for sample_id in self.samples:
            (u_ac,aac) = sample_id
            target_value = self.samples[sample_id].targetValue
            aac_base = aac[:-1]
            if not (u_ac,aac_base) in pos_map:
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
        del_values = []
        for sample_id in self.feature_matrix_dict:
            if not sample_id in self.samples:
                del_values.append(sample_id)
        for sample_id in del_values:
            del self.feature_matrix_dict[sample_id]
        return

    def removeFeaturesByType(self,feat_type):
        print('Remove features of type: ',feat_type)
        del_list = []
        for feat_name in self.features:
            feat = self.features[feat_name]
            if feat.group == feat_type:
                del_list.append(feat_name)
        for feat_name in del_list:
            del self.features[feat_name]
        self.feature_names = list(self.features.keys())
        return

    def addValue(self,sample_id,value,feat_name):
        if not sample_id in self.samples:
            self.samples[sample_id] = Sample(sample_id,self.sample_nr)
            self.sample_nr += 1
            self.feature_matrix_dict[sample_id] = {}
        if not self.features[feat_name].f_type == 'categorical':
            self.feature_matrix_dict[sample_id][feat_name] = value
        else:
            if not value in self.features[feat_name].category_map:
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
        for sample_id in self.samples:
            for feat_name in self.features:
                if not feat_name in self.feature_matrix_dict[sample_id]:
                    self.feature_matrix_dict[sample_id][feat_name] = self.features[feat_name].default

    def write(self, outfile):

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
            if not class_name in class_out_lines:
                class_out_lines[class_name] = [header]
            class_out_lines[class_name].append('%s\t%s\t%s\t%s' % (u_ac,aac,target_value_str,'\t'.join(feature_vector)))


        f = open(outfile,'w')
        f.write('\n'.join(outlines))
        f.close()

        base_name,file_type = outfile.rsplit('.',1)
        if not os.path.isdir(base_name):
            os.mkdir(base_name)

        for class_name in class_out_lines:
            class_outfile = '%s_%s.%s' % (base_name,class_name,file_type)
            f = open(class_outfile,'w')
            f.write('\n'.join(class_out_lines[class_name]))
            f.close()

    def transform_matrix_dict(self):
        fixed_feat_names = []
        for sample_pos, sample_id in enumerate(self.feature_matrix_dict):
            self.sample_pos_dict[sample_id] = sample_pos
            self.raw_feature_matrix.append([])
            if sample_pos == 0:
                for feat_pos, feat_name in enumerate(self.feature_matrix_dict[sample_id]):
                    self.feat_pos_dict[feat_name] = feat_pos
                    fixed_feat_names.append(feat_name)
            for feat_name in fixed_feat_names:
                self.raw_feature_matrix[sample_pos].append(self.feature_matrix_dict[sample_id][feat_name])
        del self.feature_matrix_dict


    def get_feature_value(self, sample_id, feat_name):
        feat_value = self.raw_feature_matrix[self.sample_pos_dict[sample_id]][self.feat_pos_dict[feat_name]]
        return feat_value

    def get_feature_value_vector(self, sample_ids, feat_name):
        feature_id = self.feat_pos_dict[feat_name]
        feature_value_vector = []
        sample_positions = []
        for sample_id in sample_ids:
            sample_positions.append(self.sample_pos_dict[sample_id])
        for sample_pos in sample_positions:
            feature_value_vector.append(self.raw_feature_matrix[sample_pos][feature_id])

        return feature_value_vector


    def get_feat_matrix(self, feat_id_vec, sample_pos_vec):
        feat_matrix = []
        for sample_pos in sample_pos_vec:
            feat_vec = []
            for feat_id in feat_id_vec:
                try:
                    feat_vec.append(self.raw_feature_matrix[sample_pos][feat_id])
                except:
                    feat_vec.append(None)
            feat_matrix.append(feat_vec)
        return feat_matrix

    def get_feat_matrix_from_feat_names(self, feat_names):
        feat_id_vec = []
        for feat_name in feat_names:
            try:
                feat_id_vec.append(self.feat_pos_dict[feat_name])
            except:
                feat_id_vec.append(None)

        sample_pos_vec = list(range(len(self.raw_feature_matrix)))
        return self.get_feat_matrix(feat_id_vec, sample_pos_vec)
    
    def get_feat_matrix_from_ids(self, sample_ids, feat_names):
        feat_id_vec = []
        for feat_name in feat_names:
            feat_id_vec.append(self.feat_pos_dict[feat_name])
  
        sample_pos_vec = []
        for sample_id in sample_ids:
            sample_pos_vec.append(self.sample_pos_dict[sample_id])
        return self.get_feat_matrix(feat_id_vec, sample_pos_vec)

    def setGeometricDistanceMap(self,config):
        if self.geometric_distance_map != None and self.current_geometric_exponent == config.geometric_exponent:
            return
        t0 = time.time()
        print('Start of GDM calculation with gexp:', config.geometric_exponent)
        self.geometric_distance_map = {}
        bin_size = len(self.samples) // config.proc_n
        if len(self.samples) % config.proc_n != 0:
            bin_size += 1
        targets = [self.samples[sample_id].targetValue for sample_id in self.samples]
        sample_ids = list(self.samples.keys())
        store = ray.put((targets,sample_ids,bin_size,config))

        t1 = time.time()
        print('GDM store created, time:',t1-t0)

        block_ids = []

        for i in range(config.proc_n):
            block_ids.append(calc_gdm_submatrix.remote(store,i))

        block_results = ray.get(block_ids)

        t2 = time.time()
        print('GDM submatrices calculated, time:',t2-t1)

        for submatrix in block_results:
            self.geometric_distance_map.update(submatrix)

        self.current_geometric_exponent = config.geometric_exponent

        t3 = time.time()
        print('GDM calculation complete, time:',t3-t2)

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
            if not bin_lower in bins:
                bins[bin_lower] = []
            bins[bin_lower].append(sample_id)

        deletion_list = []
        for bin_lower in bins:
            if len(bins[bin_lower]) <= min_bin_size:
                for sample_id in bins[bin_lower]:
                    deletion_list.append(sample_id)
                    print('Outlier detection:',sample_id,self.samples[sample_id].targetValue)
        self.removeSamples(deletion_list)

    def printPureMixedProportion(self,config):
        pure_map = {}
        mixed_map = set()
        prot_set = set()
        target_value_map = {}
        for (u_ac,aac) in self.samples:
            prot_set.add(u_ac)
            targetValue = self.samples[(u_ac,aac)].targetValue
            if not targetValue in target_value_map:
                target_value_map[targetValue] = 0
            target_value_map[targetValue] += 1
            if u_ac in mixed_map:
                continue
            if not u_ac in pure_map:
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
            print('# of Proteins: ',n_prot,'# of Variants: ',len(self.samples),'target value balance: ',target_value_map)
        else:
            print('# of Proteins: ',n_prot,'# of Variants: ',len(self.samples))
        print('Pure protein proportion: ',ppr,'Mixed protein proportion: ',mpr)
        return

    def undoBalancing(self,config):
        for sample_id in self.filtered_samples:
            self.samples[sample_id].testtrain = 'train'
        self.calcVectors(config)
        return

    def getVectors(self,config,forceCalc=False):
        if self.test_feature_matrix == None or forceCalc:
            self.calcVectors(config)
        return self.test_feature_matrix,self.test_targets,self.train_feature_matrix,self.train_targets

    def get_test_data_for_feature_list(self, extern_feature_list):
        test_feature_matrix = self.get_feat_matrix_from_feat_names(extern_feature_list)
        test_targets = []
        sample_id_list = []
        for sample_id in self.samples:
            sample_id_list.append(sample_id)
            test_targets.append(self.samples[sample_id].targetValue)
        return test_feature_matrix, test_targets, sample_id_list

    def filterSamplesByMappedStructures(self,config):
        del_list = []
        for sample_id in self.samples:
            if self.samples[sample_id].amount_of_structures == None:
                self.samples[sample_id].amount_of_structures = 0
            if self.samples[sample_id].amount_of_structures < config.structure_threshold:
                del_list.append(sample_id)
        print('Filtered ',len(del_list),' samples in the structure filtering')
        self.removeSamples(del_list)

    def standardFilter(self, config):
        del_list = []
        num_tv_viol = 0
        num_ff = 0
        num_tf = 0
        num_pf = 0
        N = len(self.samples)
        for sample_id in self.samples:
            sample = self.samples[sample_id]
            u_ac,aac = sample_id

            for tag in sample.tags.split(','):
                if tag in config.tag_filter:
                    del_list.append(sample_id)
                    num_tf += 1
                    break

            if u_ac in config.protein_filter:
                del_list.append(sample_id)
                num_pf += 1
                continue

            if ((sample.targetValue == None or sample.targetValue == 'None') and not config.predict_mode) or sample.targetValue != sample.targetValue or sample.targetValue in config.targetFilter:
                if sample.targetValue in config.target_translator:
                    sample.targetValue = config.target_translator[sample.targetValue]
                else:
                    del_list.append(sample_id)
                    num_tv_viol += 1
                    continue
                
            
        print('Filtered ',len(del_list),'of',N,' samples in the standard filtering.',num_tv_viol,
                'due to target value violation,',num_ff,'due to feature filter',num_tf,'due to tag filter',num_pf,'due to protein filter')
        self.removeSamples(del_list)

    def adjustParameterRanges(self,config):
        if config.tvmb_rank_half_step[-1] == 'max':
            config.tvmb_rank_half_step[-1] = len(self.feature_names) -1
        if config.tvpmb_rank_half_step[-1] == 'max':
            config.tvpmb_rank_half_step[-1] = len(self.feature_names) -1
        if config.confusion_rank_threshold_bounds[-1] == 'max':
            config.confusion_rank_threshold_bounds[-1] = len(self.feature_names) -1
        return

cv_slots = ['slices', 'slice_ids', 'cv_counter', 'isSlice']

class CrossValidation:
    __slots__ = cv_slots
    def __init__(self):
        self.slices = {}
        self.slice_ids = []
        self.cv_counter = 0
        self.isSlice = False

    def getCurrentSlice(self):
        return self.slices[self.slice_ids[self.cv_counter]]

    def getNextSlice(self):
        self.cv_counter += 1
        if self.cv_counter >= len(self.slice_ids):
            print('ERROR: cv_counter out of bounds')
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
                print('Tag-based crossvailidation, test tag: ',test_tag)
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
            print(f'Init of FullSlice: {internal_cv is None} {train_equal_test}')
        
        self.slice_ids.append(0)

        full_slice_obj = CrossValidationSlice(train_ids = list(sampleSpace.samples.keys()), raw_feature_names = sampleSpace.feature_names, sample_dict = sampleSpace.samples, config = config, train_equal_test = train_equal_test)

        if internal_cv is not None:
            full_slice_obj.slice_slices = []
            for slice_id in internal_cv.slices:
                if config.verbosity >= 1:
                    print(f'Adding a slice_slice to the FullSlice: {slice_id}')
                full_slice_obj.slice_slices.append(internal_cv.slices[slice_id])

        self.slices[0] = full_slice_obj

class X_fold_cv(CrossValidation):
    def __init__(self, sampleSpace, config, x_fold):
        super().__init__()
        prot_map = {}
        for (u_ac,aac) in sampleSpace.samples:
            if not u_ac in prot_map:
                prot_map[u_ac] = 0
            prot_map[u_ac] += 1

        self.prot_map = prot_map
        self.prots = list(prot_map.keys())

        self.cross_slice_size = len(sampleSpace.samples)/x_fold

        self.sample_ids = list(sampleSpace.samples.keys())
        self.tag_based_separation_done = False

        self.tag_based_crossValidation_counter = 0

        n_of_assigned_prots = 0
        n_of_assigned_samples = 0

        for cv_counter in range(x_fold):
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
            print('Next crossvalidation slice, size: ',slice_size)

            if config.prot_based_separation:
                test_ids, train_ids = splitDataSet(config, sampleSpace.samples, specific_id=test_prots, protein_wise=True)
            else:
                test_ids, train_ids = splitDataSet(config, sampleSpace.samples, specific_id=test_samples, protein_wise=False)

            cv_slice = CrossValidationSlice(test_ids = test_ids, train_ids = train_ids, raw_feature_names = sampleSpace.feature_names, sample_dict = sampleSpace.samples, config = config)
            self.slices[cv_counter] = cv_slice

@ray.remote(max_calls = 1)
def init_lopo_slice(store, test_prots, cv_counter = None):
    config, sample_dict, raw_feature_names, all_prots = store

    train_prots = all_prots - test_prots

    test_ids, train_ids = splitDataSet(config, sample_dict, specific_id=test_prots, protein_wise=True)

    if cv_counter is None:
        for name in test_prots:
            break
    else:
        name = cv_counter

    if config.verbosity >= 2:
        print(f'Init LOPO slice: {name}: Test set size: {len(test_ids)}, Train set size: {len(train_ids)}')

    cv_slice = CrossValidationSlice(test_ids = test_ids, train_ids = train_ids, raw_feature_names = raw_feature_names, sample_dict = sample_dict, config = config, name = name, train_prots = train_prots, test_prots = test_prots)
    if config.verbosity >= 5:
        #print(f'Features of slice {cv_slice.name}:\n{cv_slice.features}')
        cv_slice.featureSanityCheck()
    cv_slice = pack(cv_slice)
    return (name, cv_slice)
    #return (name, cv_slice)

class LOPO(CrossValidation):
    def __init__(self, sampleSpace, config):
        t0 = time.time()
        print('-- LOPO initialization --')
        super().__init__()
        prots = set([])
        for (u_ac,aac) in sampleSpace.samples:
            prots.add(u_ac)

        self.prots = list(prots)

        init_ids = []

        print(f'Put sample space into store: {prots}')

        store = ray.put((config, sampleSpace.samples, sampleSpace.feature_names, prots))

        t1 = time.time()
        print('Start paralell slice init, with number of processes:',config.proc_n,',time:',t1-t0)

        for prot in self.prots:
            init_ids.append(init_lopo_slice.remote(store, set([prot])))

        init_results = ray.get(init_ids)

        t2 = time.time()
        print('Paralell slice init finished, time:',t2-t1)

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

class DataSAIL_cv(CrossValidation):
    __slots__ = cv_slots + ['prots']
    def __init__(self, sampleSpace = None, config = None, as_list = None):
        t0 = time.time()
        if as_list is not None:
            for slot_number, slot in enumerate(self.__slots__):
                self.__setattr__(slot, as_list[slot_number])
            return
        super().__init__()

        weight_map = {}
        prots = set()
        for (prot_id, aac) in sampleSpace.samples:
            if prot_id not in weight_map:
                weight_map[prot_id] = 0
                prots.add(prot_id)
            weight_map[prot_id] += 1

        self.prots = list(prots)

        datasail_test_size = 100 // config.crossValidation_fold
        datasail_train_size = 100 - datasail_test_size

        splits = [datasail_test_size] * config.crossValidation_fold

        names = [f'split_{x}' for x in range(config.crossValidation_fold)]

        t1 = time.time()
        if config.verbosity >= 2:
            print(f'Time for init DataSAIL_cv Part 1: {t1-t0}')

        if config.verbosity >= 3:
            print(f'Call of datasail with: e_data: {config.path_to_sequence_fasta}, e_weights: {weight_map} ({len(weight_map)}), splits: {splits}, names: {names}')

            write_weight_map(weight_map, 'weight_map_for_datasail.tsv')
            raw_datasail_splits = datasail(e_data = config.path_to_sequence_fasta, e_weights = weight_map, splits = splits, techniques = ['C1e'], names = names, e_type = 'P', solver = 'SCIP', verbose = 'I')

        else:

            raw_datasail_splits = datasail(e_data = config.path_to_sequence_fasta, e_weights = weight_map, splits = splits, techniques = ['C1e'], names = names, e_type = 'P', solver = 'SCIP')

        t2 = time.time()
        if config.verbosity >= 2:
            print(f'Time for init DataSAIL_cv Part 2: {t2-t1}')

        print(raw_datasail_splits)

        datasail_splits = raw_datasail_splits[0]['C1e'][0]
        print(datasail_splits)

        train_test_pairs = {}
        for cv_counter in range(config.crossValidation_fold):
            train_test_pairs[cv_counter] = [[], [], {}]

        for prot_id in datasail_splits:
            split_name = datasail_splits[prot_id]
            cv_counter = int(split_name.split('_')[1])

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
            print(f'Time for init DataSAIL_cv Part 3: {t3-t2}')

        if config.verbosity >= 2:
            print(f'Init datasail:\nSplits: {train_test_pairs}\n')

        store = ray.put((config, sampleSpace.samples, sampleSpace.feature_names, prots))

        t4 = time.time()
        if config.verbosity >= 2:
            print(f'Time for init DataSAIL_cv Part 4: {t4-t3}')

        init_ids = []
        for cv_counter in train_test_pairs:
            train_set, test_set, _ = train_test_pairs[cv_counter]

            if config.verbosity >= 2:
                print(f'Init datasail slice {cv_counter}: {test_set}')

            init_ids.append(init_lopo_slice.remote(store, set(test_set), cv_counter = cv_counter))

        init_results = ray.get(init_ids)

        t5 = time.time()
        if config.verbosity >= 2:
            print(f'Time for init DataSAIL_cv Part 5: {t5-t4}')

        for cv_counter, cv_slice in init_results:

            self.slices[cv_counter] = unpack(cv_slice)
            if config.verbosity >= 5:
                cv_slice = self.slices[cv_counter]
                #print(f'Features of slice {cv_slice.name}:\n{cv_slice.features}')
                cv_slice.featureSanityCheck(verbose = True)

            self.slice_ids.append(cv_counter)
            if train_test_pairs[cv_counter][2] is not None:
                self.slices[cv_counter].subslices = []
                for subslice_counter in train_test_pairs[cv_counter][2]:
                    self.slices[cv_counter].subslices.append(train_test_pairs[cv_counter][2][subslice_counter])

        t6 = time.time()
        if config.verbosity >= 2:
            print(f'Time for init DataSAIL_cv Part 6: {t6-t5}')


