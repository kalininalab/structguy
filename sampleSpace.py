import os
import sys
import numpy as np
import math
import random
import ray
import time
import util

from scipy import stats

from structman.base_utils.base_utils import calculate_chunksizes
#from scala.tree_split import tree
from datasail.sail import datasail


import dicts
possible_na_values = set(['-', 'None'])
class Feature:
    def __init__(self,name,f_type,group=None,default_value=None,mutation_specific=False):
        self.name = name
        self.f_type = f_type
        self.value_map = {}
        self.group = group
        self.default = default_value
        self.mutation_specific=mutation_specific

        if f_type == 'categorical':
            self.category_map = {}
            self.category_counter = 0
            self.category_backmap = {}

    def addValue(self,sample_id,value):
        if value == None:
            value = self.default
        if not self.f_type == 'categorical':
            self.value_map[sample_id] = value
        else:
            if not value in self.category_map:
                self.category_map[value] = self.category_counter
                self.category_backmap[self.category_counter] = value
                self.category_counter += 1
            self.value_map[sample_id] = self.category_map[value]

    def value_from_string(self, string):
        if self.name == 'Blosum62':
            try:
                value = value = float(string)
            except:
                try:
                    value = dicts.BLOSUM62[(string[0],string[-1])]
                except:
                    value = dicts.BLOSUM62[(string[-1],string[0])]
            return value

        try:
            if self.f_type == 'categorical':
                value = string
            elif self.f_type == 'real':
                value = float(string)
            elif self.f_type == 'integer' or self.f_type == 'binary':
                value = int(string)
            else:
                print(f'Error in value_from_string: {self.f_type} {self.name} {string}')
        except:
            if string in possible_na_values:
                value = self.default
            else:
                print(f'Error in value_from_string: {self.f_type} {self.name} {string}')
        return value

    def string_convert(self,value):
        if not self.f_type == 'categorical':
            return str(value)
        else:
            return str(self.category_backmap[value])

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

class SampleSpace:
    def __init__(self, config):
        self.samples = {}
        self.features = {}
        self.sample_nr = 0
        self.feature_names = []

        self.vector_store = {}
        self.current_vector = None

        self.geometric_distance_map = {}
        self.current_geometric_exponent = None

        if config.addBias:
            self.addFeature('Protein bias','real',group='amino acid property',default_value='0.5')

    def addFeature(self,name,f_type,group=None,default_value=None,mutation_specific=False):
        feat = Feature(name,f_type,group=group,default_value=default_value,mutation_specific=mutation_specific)
        self.features[name] = feat
        self.feature_names.append(name)

    def removeFeature(self, feat_name):
        del self.features[feat_name]
        self.feature_names.remove(feat_name)

    def cleanse_empty_features(self, verbosity = 0):
        n = 0
        for feat_name in list(self.features.keys()):
            if len(self.features[feat_name].value_map) == 0:
                self.removeFeature(feat_name)
                n += 1
                if verbosity >= 1:
                    print(f'Cleansed {feat_name}')
        print(f'Cleansed {n} empty features')

    def oneHotify(self,feat_name):
        feat = self.features[feat_name]
        f_type = feat.f_type
        if not f_type == 'categorical':
            print('Warning: Cannot oneHotify non-categorical features')
            return
        for category in feat.category_map:
            oh_name = 'oh_%s_%s' % (feat_name,category)
            self.addFeature(oh_name,'binary',group=feat.group,default_value=0)

        for sample_id in feat.value_map:
            cat_value = feat.value_map[sample_id]
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

        self.feature_names = list(self.features.keys())
        self.fillDefaultValues()

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

                    value = feat.value_map[(u_ac,'%s%s' % (aac_base,example_mut_aa))]
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
        for feat_name in self.features:
            feat = self.features[feat_name]
            del_values = []
            for sample_id in feat.value_map:
                if not sample_id in self.samples:
                    del_values.append(sample_id)
            for sample_id in del_values:
                del feat.value_map[sample_id]
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
        self.features[feat_name].addValue(sample_id,value)

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
                feat = self.features[feat_name]
                if not sample_id in feat.value_map:
                    feat.value_map[sample_id] = feat.default

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
                feature_value = feat.value_map[sample_id]
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

    def getSize(self):
        return len(self.samples)

    def getFeatureVector(self,sample_id):
        feature_vector = []
        for feat_name in self.feature_names:
            feat = self.features[feat_name]
            try:
                feature_vector.append(feat.value_map[sample_id])
            except:
                print(f'Value map (length: {len(feat.value_map)}) of {feat_name} did not contain: {sample_id}')
                sys.exit()
        return feature_vector

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
        test_feature_matrix = []
        test_targets = []
        sample_id_list = []
        for sample_id in self.samples:
            sample_id_list.append(sample_id)
            test_targets.append(self.samples[sample_id].targetValue)
            feature_value_vector = []
            for feat_name in extern_feature_list:
                if feat_name in self.features:
                    feature_value_vector.append(self.features[feat_name].value_map[sample_id])
                else:
                    feature_value_vector.append(0) #Need to put default values here
            test_feature_matrix.append(feature_value_vector)
        return test_feature_matrix, test_targets, sample_id_list


    def splitDataSet(self,config,specific_id=None,protein_wise=False,debug=0, skip_protein = None, ignore_samples = None):
        split_rate = config.split_rate
        total_size = self.getSize()
        test_size = max([1,int(total_size*split_rate)])
        train_size = total_size-test_size

        test_ids = []
        train_ids = []

        if not protein_wise:
            #simple random split
            if specific_id == None:
                test_nrs = set(np.random.choice(total_size,test_size,replace=False))
                for sample_id in self.samples:
                    sample = self.samples[sample_id]
                    if sample.nr in test_nrs:
                        test_ids.append(sample_id)
                    else:
                        train_ids.append(sample_id)
            else:
                for sample_id in self.samples:
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
                for u_ac,aac in self.samples:
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

            for (u_ac,aac) in self.samples:
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

    def filterSamplesByMappedStructures(self,config):
        del_list = []
        for sample_id in self.samples:
            if self.samples[sample_id].amount_of_structures == None:
                self.samples[sample_id].amount_of_structures = 0
            if self.samples[sample_id].amount_of_structures < config.structure_threshold:
                del_list.append(sample_id)
        print('Filtered ',len(del_list),' samples in the structure filtering')
        self.removeSamples(del_list)

    def standardFilter(self,config):
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

            if sample.targetValue == None or sample.targetValue == 'None' or sample.targetValue != sample.targetValue or sample.targetValue in config.targetFilter:
                if sample.targetValue in config.target_translator:
                    sample.targetValue = config.target_translator[sample.targetValue]
                else:
                    del_list.append(sample_id)
                    num_tv_viol += 1
                    continue
                
            for feat_name in config.standard_feature_filter:
                feat = self.features[feat_name]
                if not sample_id in feat.value_map:
                    feat.value_map[sample_id] = feat.default
                val = feat.value_map[sample_id]
                if val == config.standard_feature_filter[feat_name]:
                    del_list.append(sample_id)
                    num_ff += 1
                    break
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

class CrossValidation:

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
    def __init__(self, sampleSpace, config, train_equal_test = False):
        super().__init__()
        
        self.slice_ids.append(0)

        cv_slice = CrossValidationSlice([], list(sampleSpace.samples.keys()), sampleSpace, config, train_equal_test = train_equal_test)
        self.slices[0] = cv_slice

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
                test_ids, train_ids = sampleSpace.splitDataSet(config,specific_id=test_prots,protein_wise=True)
            else:
                test_ids, train_ids = sampleSpace.splitDataSet(config,specific_id=test_samples,protein_wise=False)

            cv_slice = CrossValidationSlice(test_ids, train_ids, sampleSpace, config)
            self.slices[cv_counter] = cv_slice

@ray.remote(max_calls = 1)
def init_lopo_slice(store, test_prots, cv_counter = None):
    config, sampleSpace, all_prots = store

    train_prots = all_prots - test_prots

    test_ids, train_ids = sampleSpace.splitDataSet(config, specific_id=test_prots, protein_wise=True)

    if cv_counter is None:
        for name in test_prots:
            break
    else:
        name = cv_counter

    if config.verbosity >= 2:
        print(f'Init LOPO slice: {name}: Test set size: {len(test_ids)}, Train set size: {len(train_ids)}')

    cv_slice = CrossValidationSlice(test_ids, train_ids, sampleSpace, config, name = name, train_prots = train_prots, test_prots = test_prots)
    return (name, cv_slice)

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

        store = ray.put((config, sampleSpace, prots))

        t1 = time.time()
        print('Start paralell slice init, with number of processes:',config.proc_n,',time:',t1-t0)

        for prot in self.prots:
            init_ids.append(init_lopo_slice.remote(store, set([prot])))

        init_results = ray.get(init_ids)

        t2 = time.time()
        print('Paralell slice init finished, time:',t2-t1)

        for prot, cv_slice in init_results:
            self.slices[prot] = cv_slice
            self.slice_ids.append(prot)

def write_weight_map(weight_map, outfile):
    lines = []
    for prot_id in weight_map:
        lines.append(f'{prot_id}\t{weight_map[prot_id]}\n')
    f = open(outfile,'w')
    f.write(''.join(lines))
    f.close()

class DataSAIL_cv(CrossValidation):
    def __init__(self, sampleSpace, config):
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

        if config.verbosity >= 3:
            print(f'Call of datasail with: e_data: {config.path_to_sequence_fasta}, e_weights: {weight_map} ({len(weight_map)}), splits: {splits}, names: {names}')

            write_weight_map(weight_map, 'weight_map_for_datasail.tsv')

        raw_datasail_splits = datasail(e_data = config.path_to_sequence_fasta, e_weights = weight_map, splits = splits, techniques = ['CCSe'], names = names, e_type = 'P', solver = 'SCIP')

        datasail_splits = raw_datasail_splits[0]['CCS'][0]
        print(datasail_splits)

        train_test_pairs = {}
        for cv_counter in range(config.crossValidation_fold):
            train_test_pairs[cv_counter] = [[], [], []]

        for prot_id in datasail_splits:
            split_name = datasail_splits[prot_id]
            cv_counter = int(split_name.split('_')[1])
            subslice_counter = cv_counter + 1
            if subslice_counter == config.crossValidation_fold:
                subslice_counter = 0
            for cv in train_test_pairs:
                if cv == cv_counter:
                    train_test_pairs[cv][1].append(prot_id)
                else:
                    train_test_pairs[cv][0].append(prot_id)
                    if cv == subslice_counter:
                        train_test_pairs[cv][2].append(prot_id)


        if config.verbosity >= 2:
            print(f'Init datasail:\nSplits: {train_test_pairs}\n')

        store = ray.put((config, sampleSpace, prots))

        init_ids = []
        for cv_counter in train_test_pairs:
            train_set, test_set, subslice = train_test_pairs[cv_counter]

            if config.verbosity >= 2:
                print(f'Init datasail slice {cv_counter}: {test_set}')

            init_ids.append(init_lopo_slice.remote(store, set(test_set), cv_counter = cv_counter))

        init_results = ray.get(init_ids)

        for cv_counter, cv_slice in init_results:

            self.slices[cv_counter] = cv_slice
            self.slice_ids.append(cv_counter)
            if train_test_pairs[cv_counter][2] is not None:
                self.slices[cv_counter].subslice = train_test_pairs[cv_counter][2]
            else:
                train_set = train_test_pairs[cv_counter][0]
                if len(train_set) == 1:
                    self.slices[cv_counter].subslice = []
                else:
                    self.slices[cv_counter].subslice = train_set[0]

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

class CrossValidationSlice:
    def __init__(self, test_ids, train_ids, sampleSpace, config, name = '', train_prots = None, test_prots = None, train_equal_test = False, para_number = None, feature_names = None):
        self.isSlice = True
        self.test_feature_matrix = []
        self.test_targets = []
        self.test_sample_ids = [x for x in test_ids]
        self.train_feature_matrix = []
        self.train_targets = []
        self.train_sample_ids = [x for x in train_ids]
        features_to_remove = []
        if feature_names is None:
            keep_features = None
        else:
            keep_features = set(feature_names)
        self.feature_names = []
        for feat_name in sampleSpace.feature_names:
            self.feature_names.append(feat_name)
            if keep_features is not None:
                if feat_name not in keep_features:
                    features_to_remove.append(feat_name)

        self.name = name
        if train_prots is None:
            self.train_prots =  set([])
            test_samples = set(self.test_sample_ids)
            for (prot_id, aac) in sampleSpace.samples:
                if (prot_id, aac) in test_samples:
                    continue
                self.train_prots.add(prot_id)
        else:
            self.train_prots = train_prots

        self.test_prots = test_prots

        self.train_equal_test = train_equal_test

        self.slice_slice = None
        self.slice_slices = None
        self.subslice = None
        self.subslices = None
        self.random_subslice = None

        self.features = {}
        for pos,feat in enumerate(self.feature_names):
            self.features[feat] = pos

        self.deactivated_features = {}
        self.slice_specific_features = {}
        self.slice_specific_feature_names = []
        self.slice_specific_feature_map = {}

        self.current_geometric_exponent = None
        self.weight_vector_store = {}
        self.geometric_distance_map = {}
        self.int_map = {}
        self.int_counter = 0

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

        self.confusion_map = None

        if config.verbosity >= 2:
            t0 = time.time()

        if not self.train_equal_test:
            self.checkCircularity(remove_t2=config.remove_t2,print_out=True)

        if config.verbosity >= 2:
            t1 = time.time()
            print(f'Init CV slice part 1: {t1-t0}')

        if config.verbosity >=3:
            print(f'In CVSLice init of {self.name}: train equal test: {train_equal_test}, train prots: {train_prots}, Test set: {len(self.test_sample_ids)}, Train set: {len(self.train_sample_ids)}')

        if not train_equal_test:
            for sample_id in self.test_sample_ids:
                sample = sampleSpace.samples[sample_id]

                self.test_targets.append(sample.targetValue)
                feature_vector = sampleSpace.getFeatureVector(sample_id)
                self.test_feature_matrix.append(feature_vector)
                self.slice_specific_features[sample_id] = {}

        if config.verbosity >= 2:
            t2 = time.time()
            print(f'Init CV slice part 2 {train_equal_test}: {t2-t1}')

        for sample_id in self.train_sample_ids:
            sample = sampleSpace.samples[sample_id]
            self.train_targets.append(sample.targetValue)
            feature_vector = sampleSpace.getFeatureVector(sample_id)

            if config.verbosity >= 4:
                util.sanity_check_value_list(feature_vector, label_vector = self.feature_names, datastructure_name = f'Feature vector of {sample_id}')

            self.train_feature_matrix.append(feature_vector)
            self.slice_specific_features[sample_id] = {}
            if train_equal_test:
                self.test_feature_matrix.append([x for x in feature_vector])
                self.test_targets.append(sample.targetValue)

        if config.verbosity >= 2:
            t3 = time.time()
            print(f'Init CV slice part 3: {t3-t2}')

        if train_equal_test:
            self.test_sample_ids = [x for x in self.train_sample_ids]

        if config.verbosity >= 2:
            t4 = time.time()
            print(f'Init CV slice part 4: {t4-t3}')

        if config.addBias:
            self.setProteinBias(config)

        if config.verbosity >= 2:
            t5 = time.time()
            print(f'Init CV slice part 5: {t5-t4}')

        if config.balanceSubsampling != None:
            self.balanceSubSampleTrainSet(config)

        if config.verbosity >= 2:
            t6 = time.time()
            print(f'Init CV slice part 6: {t6-t5}')

        self.printBalance(config)

        if config.verbosity >= 2:
            t7 = time.time()
            print(f'Init CV slice part 7: {t7-t6}')

        if config.regression:
            if config.geometric_weighting:
                self.calcSampleWeights(config, sampleSpace.geometric_distance_map)
            else:
                self.calcSubsampleDistanceWeights(config, para_number = para_number)

        for feat_name in features_to_remove:
            self.removeFeature(feat_name)

        if config.verbosity >= 2:
            t8 = time.time()
            print(f'Init CV slice part 8: {t8-t7}')

    def mirrorTrainSamples(self):
        self.test_feature_matrix = []
        self.test_targets = []
        self.test_sample_ids = []
        for pos, sample_id in enumerate(self.train_sample_ids):
            self.test_feature_matrix.append([x for x in self.train_feature_matrix[pos]])
            self.test_targets.append(self.train_targets[pos])
            self.test_sample_ids.append(sample_id)

    def write(self, outfile):

        class_out_lines = {}

        header = 'Uniprot Ac\tAAC\tTarget value\t%s' % '\t'.join(self.feature_names)
        outlines = [header]

        for sample_pos, sample_id in enumerate(self.train_sample_ids):
            u_ac, aac = sample_id
            target_value_str = str(self.train_targets[sample_pos])

            feature_vector = []
            for feat_pos,feature_name in enumerate(self.feature_names):
                feature_value = self.train_feature_matrix[sample_pos][feat_pos]
                feature_vector.append(str(feature_value))

            outlines.append('%s\t%s\t%s\t%s' % (u_ac, aac, target_value_str, '\t'.join(feature_vector)))

        f = open(outfile,'w')
        f.write('\n'.join(outlines))
        f.close()


    def featureSanityCheck(self):
        for pos_1,feat in enumerate(self.feature_names):
            pos_2 = self.features[feat]
            if pos_1 != pos_2:
                raise 'Features are not continuous anymore'
        for feat_vec in self.train_feature_matrix:
            if len(feat_vec) != len(self.features):
                raise 'The feature matrix is insane'
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
                if not u_ac in prot_bins:
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
                if not u_ac in balance_map:
                    balance_map[u_ac] = {}
                if not target in balance_map[u_ac]:
                    balance_map[u_ac][target] = set()
                balance_map[u_ac][target].add(aac)

            for u_ac in balance_map:
                min_label = None
                max_label = None
                min_n = None
                max_n = None
                for label in balance_map[u_ac]:
                    label_n = len(balance_map[u_ac][label])
                    if min_label == None or min_n > label_n:
                        min_label = label
                        min_n = label_n
                    if max_label == None or max_n < label_n:
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
            if not sample_id in train_sub_sample_ids:
                self.filtered_samples.append(sample_id)
            else:
                kept_pos.append(pos)

        train_feature_matrix = []
        train_targets = []
        train_sample_ids = []
        for pos in kept_pos:
            train_feature_matrix.append(self.train_feature_matrix[pos])
            train_targets.append(self.train_targets[pos])
            train_sample_ids.append(self.train_sample_ids[pos])

        self.train_feature_matrix = train_feature_matrix
        self.train_targets = train_targets
        self.train_sample_ids = train_sample_ids

        return

    def printBalance(self, config):
        if config.regression:
            print(f'{self.name} Mean target value: {sum(self.train_targets)/len(self.train_targets)}')
            print(f'{self.name} Test set size: {len(self.test_targets)}, Train set size: {len(self.train_targets)}')
            return
        balance_map = {}
        for ttv in self.train_targets:
            if not ttv in balance_map:
                balance_map[ttv] = 0
            balance_map[ttv] += 1
        print('Train set balance: ',balance_map)
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
            if not ttv in balance_map:
                balance_map[ttv] = 0
            balance_map[ttv] += 1
        print('Test set balance: ',balance_map)

    def getGeometricDistanceMap(self,config):
        if self.geometric_distance_map != None:
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
        if 'Protein bias' not in self.slice_specific_feature_map:
            self.slice_specific_feature_map['Protein bias'] = len(self.slice_specific_feature_names)
            self.slice_specific_feature_names.append('Protein bias')
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
            if not u_ac in bias_map:
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
            if not (u_ac,pos) in bias_map:
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

    def featureTargetCorr(self,feat_name,config):
        val_list_1 = []
        val_list_2 = []

        for pos,tv in enumerate(self.train_targets):
            if not feat_name in self.deactivated_features:
                value = self.train_feature_matrix[pos][self.features[feat_name]]
            else:
                value = self.deactivated_features[feat_name][self.train_sample_ids[pos]]
            val_list_1.append(value)
            if not config.regression:
                if not tv in self.int_map:
                    self.int_map[tv] = self.int_counter
                    self.int_counter += 1
                tv = self.int_map[tv]
            val_list_2.append(tv)
        corr,p_val = stats.spearmanr(val_list_1,val_list_2)
        return corr,p_val

    def featureCorr(self,feat_name_1,feat_name_2, slice_specific_feature = False):
        val_list_1 = []
        val_list_2 = []

        for pos,tv in enumerate(self.train_targets):
            if not feat_name_1 in self.deactivated_features:
                value_1 = self.train_feature_matrix[pos][self.features[feat_name_1]]
            else:
                value_1 = self.deactivated_features[feat_name_1][self.train_sample_ids[pos]]
            if not feat_name_2 in self.deactivated_features:
                if slice_specific_feature:
                    value_2 = self.slice_specific_features[self.train_sample_ids[pos]][feat_name_2]
                else:
                    value_2 = self.train_feature_matrix[pos][self.features[feat_name_2]]
            else:
                value_2 = self.deactivated_features[feat_name_2][self.train_sample_ids[pos]]
            val_list_1.append(value_1)
            val_list_2.append(value_2)
        corr,p_val = stats.spearmanr(val_list_1,val_list_2)
        return corr,p_val

    def filterFeatures(self,filtered_features, print_out = False):
        dont_reactivate = set(filtered_features)
        reacs = []
        for deac_feat in self.deactivated_features:
            if not deac_feat in dont_reactivate:
                reacs.append(deac_feat)

        for reac in reacs:
            self.reactivateFeature(reac)

        for ff in filtered_features:
            self.deactivateFeature(ff)

        self.featureSanityCheck()

        if print_out:
            print('======\n',self.name,'filter features:',len(filtered_features),'remaining features:',len(self.feature_names),'\n=====')

    def removeFeature(self,feat_name):
        feat_pos = self.features[feat_name]
        del self.features[feat_name]
        try:
            del self.feature_names[feat_pos]
        except:
            print(feat_name, '\n', feat_pos, '\n', self.feature_names, '\n', self.features)
            sys.exit()
        for feat_vec in self.train_feature_matrix:
            del feat_vec[feat_pos]
        for feat_vec in self.test_feature_matrix:
            del feat_vec[feat_pos]
        for other_feat in self.features:
            if self.features[other_feat] > feat_pos:
                self.features[other_feat] -= 1

    def deactivateFeature(self,feat_name, print_out = False):
        if print_out:
            print('Slice:',self.name,'Deactivate feature:',feat_name)
        if not feat_name in self.features:
            return
        feat_pos = self.features[feat_name]
        self.deactivated_features[feat_name] = {}
        for pos,sample_id in enumerate(self.train_sample_ids):
            self.deactivated_features[feat_name][sample_id] = self.train_feature_matrix[pos][feat_pos]
        for pos,sample_id in enumerate(self.test_sample_ids):
            self.deactivated_features[feat_name][sample_id] = self.test_feature_matrix[pos][feat_pos]
        self.removeFeature(feat_name)

    def reactivateFeature(self,feat_name, print_out = False):
        if print_out:
            print('Slice:',self.name,'Reactivate feature:',feat_name)
        if not feat_name in self.deactivated_features:
            return
        self.features[feat_name] = len(self.feature_names)
        self.feature_names.append(feat_name)
        for pos,sample_id in enumerate(self.train_sample_ids):
            self.train_feature_matrix[pos].append(self.deactivated_features[feat_name][sample_id])
        for pos,sample_id in enumerate(self.test_sample_ids):
            self.test_feature_matrix[pos].append(self.deactivated_features[feat_name][sample_id])
        del self.deactivated_features[feat_name]

    def add_empty_feature(self, feat_name):
        self.features[feat_name] = len(self.feature_names)
        self.feature_names.append(feat_name)
        for pos,sample_id in enumerate(self.train_sample_ids):
            self.train_feature_matrix[pos].append(0)
        for pos,sample_id in enumerate(self.test_sample_ids):
            self.test_feature_matrix[pos].append(0)

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

    def reorderByNames(self, feature_names):
        snap_shot = list(self.feature_names)
        for feat_name in snap_shot:
            self.deactivateFeature(feat_name)
        for feat_name in feature_names:
            if feat_name in self.deactivated_features:
                self.reactivateFeature(feat_name)
            else:
                self.add_empty_feature(feat_name)

    def addToTvmbMap(self,feat_name,config):
        try:
            target_corr,target_p_val = self.featureTargetCorr(feat_name,config)
        except:
            self.removeFeature(feat_name)
            print('Error: cant get feature target corr for:',feat_name)
            return
        mean_value_corr,p_val = self.featureCorr(feat_name,'Protein bias',slice_specific_feature = True)
        if mean_value_corr != mean_value_corr:
            self.removeFeature(feat_name)
            print(self.name,'Removed feature:',feat_name,'due to nan mean_value_corr')
            return
        tvmb_score = abs(mean_value_corr)-abs(target_corr) #target value mean bias score
        if tvmb_score != tvmb_score: #test for 'nan'
            self.removeFeature(feat_name)
            print(self.name,'Removed feature:',feat_name,'due to nan tvmb score')
            return
        else:
            self.tvmb_map[feat_name] = tvmb_score,target_p_val

    def rank_tvmb(self,config):
        self.ranked_tvmb = []
        for feat_name in list(self.feature_names):
            if not feat_name in self.tvmb_map:
                self.addToTvmbMap(feat_name,config)
                if not feat_name in self.tvmb_map:
                    continue
            if not feat_name in self.tvmb_map:
                print('Error debug out:',self.name,len(self.feature_names),len(self.features),len(self.test_feature_matrix[0]),len(self.train_feature_matrix[0]))
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
            if not feat_name in self.tvpmb_map:
                self.addToTvpmbMap(feat_name,config)
            tvpmb_score = self.tvpmb_map[feat_name]
            self.ranked_tvpmb.append((feat_name,tvpmb_score))
        self.ranked_tvpmb.sort(key=lambda x:x[1],reverse=True)

    def classToInt(self,data):
        int_data = []
        for class_name in data:
            if not class_name in self.int_map:
                self.int_map[class_name] = self.int_counter
                self.int_counter += 1
            int_data.append(self.int_map[class_name])
        return int_data

