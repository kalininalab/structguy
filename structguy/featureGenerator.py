import random
import sys
import traceback
import time
import ray

from structguy import sampleSpace, consts
from structguy import structural_feature_generation as strfg
from structguy import sequence_feature_generation as seqfg

from structman.base_utils.base_utils import calculate_chunksizes

def expand_structural_feature_table(config):
    samples = sampleSpace.SampleSpace(config)
    strfg.initFeatures(samples)
    seqfg.initFeatures(samples)

    parse_structural_features(samples, config)

    seqfg.getSequenceFeatures(config, samples, n_of_processes=config.seq_feat_processes)

    outfile = f'{config.outfolder}/{config.dataset_name}_structguy_features.tsv'

    samples.write(outfile)

    config.add_entry_to_project_file('path_to_features_file', outfile)
    return

@ray.remote(max_calls = 1)
def parseLines(store, left, right):

    config, lines, primary_protein_id_col, amount_of_struct_col, effect_col, aac_col_s, tags_col, non_feature_cols, features, feature_names = store

    output = []

    for line in lines[left:right]:

        if line == '':

            continue
        words = line.split('\t')
        if len(words) == 1:
            print(list(line))
            words = line.split('    ')
        #if effectRegressor != None:
        #    words = words[:-1]

        """
        if (len(words) - len(non_feature_cols)) < len(feature_names):
            if config.verbosity >= 4:
                print(f'Line reject, count incorrect: {len(words)}, {len(non_feature_cols)}, {len(feature_names)}')
            continue
        """

        primary_protein_id = words[primary_protein_id_col]
        if len(aac_col_s) == 1:
            aac = words[aac_col_s[0]]
        else:
            aac = f'{words[aac_col_s[0]]}{words[aac_col_s[1]]}{words[aac_col_s[2]]}'
        tags = words[tags_col]
        try:
            amount_of_structures = int(words[amount_of_struct_col])
        except:
            amount_of_structures = 0

        sample_id = (primary_protein_id, aac)

        if config.verbosity >= 5:
            print(f'Parsing sample: {sample_id}')


        if primary_protein_id in config.blacklist:
            if config.verbosity >= 4:
                print(f'Sample was blacklisted {sample_id}')

            continue

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
                    continue
                target_values.append(tag_value)
            if len(target_values) == 0:
                target_value = None
            else:
                target_value = sum(target_values)/len(target_values)
        else:
            if effect_col is not None:
                if words[effect_col] == 'None':
                    if config.verbosity >= 4:
                        print(f'Sample rejected: effect is None, while effect_col is not')

                    continue

                if not config.regression:
                    target_value = words[effect_col]
                else:
                    target_value = float(words[effect_col])
            else:
                target_value = None

        '''
        if add_file_to_samples:
            if target_value in config.target_translator:
                target_value = config.target_translator[target_value]
            else:
                target_value = None
        '''

        feat_out = []

        for pos,x in enumerate(words):
            if pos in non_feature_cols:
                continue
            feat_name = feature_names[pos]
            if feat_name in consts.FEAT_NAME_SYNONYMS:
                feat_name = consts.FEAT_NAME_SYNONYMS[feat_name]
            if feat_name == 'Classification confidence':
                continue
            feat = features[feat_name]
            try:
                value = feat.value_from_string(x)
            except:
                [e, f, g] = sys.exc_info()
                g = traceback.format_exc()
                print(f'Feature parse error, sample will be filtered: {primary_protein_id} {aac} {tags} {feat_name} {x}\n{e}\n{f}\n{g}')
                target_value = None
                continue

            feat_out.append((value, feat_name))

        output.append((sample_id, target_value, amount_of_structures, tags, feat_out))

    return output


def parse_structural_features(samples, config, non_feature_cols = [0,1,2,4,7,19], primary_protein_id_col = 1, aac_col_s = [3,4,5], tags_col = 7, amount_of_struct_col = 19, effect_col = None):
    file_path = config.path_structural_feature_table
    parse_feature_table(file_path, samples, config,
            non_feature_cols = non_feature_cols, primary_protein_id_col = primary_protein_id_col, aac_col_s = aac_col_s,
            tags_col = tags_col, amount_of_struct_col = amount_of_struct_col, effect_col = effect_col
            )


def parse_feature_table(file_path, samples, config, non_feature_cols = [0,1,2,3,4], primary_protein_id_col = 0, aac_col_s = [1], tags_col = 3, amount_of_struct_col = 4, effect_col = 2):
    print(f'Reading feature file: {file_path}, non_feature_cols: {non_feature_cols}, primary_protein_id_col: {primary_protein_id_col}, aac_col_s: {aac_col_s}, effect_col: {effect_col}')

    f = open(file_path,'r')
    lines = f.read().split('\n')
    f.close()

    if config.verbosity >= 1:
        t0 = time.time()

    non_feature_cols = set(non_feature_cols)

    feature_names = lines[0].split('\t')
    for pos, feat_name in enumerate(feature_names):
        if pos in non_feature_cols:
            continue
        if feat_name in consts.FEAT_NAME_SYNONYMS:
            feat_name = consts.FEAT_NAME_SYNONYMS[feat_name]
        if feat_name not in samples.features:
            samples.addFeature(feat_name,'binary',group='structural',default_value=0)

    if config.verbosity >= 1:
        t1 = time.time()
        print(f'parse_feature_table, part 1: {t1-t0}')

    headless_lines = lines[1:]
    random.shuffle(headless_lines)

    if config.verbosity >= 1:
        t2 = time.time()
        print(f'parse_feature_table, part 2: {t2-t1}')

    small_chunksize, big_chunksize, n_of_small_chunks, n_of_big_chunks = calculate_chunksizes(config.proc_n, len(headless_lines))

    store = ray.put((config, headless_lines, primary_protein_id_col, amount_of_struct_col, effect_col, aac_col_s, tags_col, non_feature_cols, samples.features, feature_names))

    if config.verbosity >= 1:
        t3 = time.time()
        print(f'parse_feature_table, part 3: {t3-t2}')

    parse_subroutine_results = []

    for i in range(n_of_big_chunks):
        parse_subroutine_results.append(parseLines.remote(store, i*big_chunksize, (i+1)*big_chunksize))

    border = (i+1)*big_chunksize

    for i in range(n_of_small_chunks):
        parse_subroutine_results.append(parseLines.remote(store, border + i*small_chunksize, border + (i+1)*small_chunksize))

    if config.verbosity >= 1:
        t4 = time.time()
        print(f'parse_feature_table, part 4: {t4-t3}')

    line_parse_out = ray.get(parse_subroutine_results)

    n_f = 0
    n_s = 0
    for parse_out in line_parse_out:
        for sample_id, target_value, amount_of_structures, tags, feat_out in parse_out:
            for value, feat_name in feat_out:
                samples.addValue(sample_id,value,feat_name)
                n_f += 1
            samples.addTargetValue(sample_id,target_value)
            samples.samples[sample_id].amount_of_structures = amount_of_structures
            samples.samples[sample_id].tags = tags
            n_s += 1

    samples.cleanse_empty_features(verbosity = config.verbosity)

    if config.verbosity >= 1:
        t5 = time.time()
        print(f'parse_feature_table, part 5: {t5-t4}')
        print(f'Total samples: {n_s}, total feature values: {n_f}')
    print('Finihsehd parsing of feature file')

def createTrainingSet(config):

    samples = sampleSpace.SampleSpace(config)

    msa_db = config.msa_db

    strfg.initFeatures(samples)
    seqfg.initFeatures(samples)

    if config.path_to_processed_features_file == None:

        parse_feature_table(config.path_to_features_file, samples, config)

        for additonal_infile in config.add_more_sample_files:
            parse_feature_table(additonal_infile, samples, config)

        if config.fusePositions:
            config.regression = False
            #print(samples.samples.keys())
            samples.fusePositions()
            #print(samples.samples.keys())
            config.target_values = ['all neutral','possibly damaging']

        samples.standardFilter(config)

        if config.structure_threshold != None:
            samples.filterSamplesByMappedStructures(config)

        if config.transform:
            samples.targetTransformation(config)
        if config.regression:
            samples.detectOutliers(config)

        samples.printPureMixedProportion(config)

        samples.adjustParameterRanges(config)

        if config.filterStructuralFeatures:
            samples.removeFeaturesByType('structural')

        samples.oneHotifyAll()

        config.path_to_processed_features_file = f'{config.outfolder}/{config.dataset_name}_structguy_features_processed.tsv'

        samples.write(config.path_to_processed_features_file)

        config.add_entry_to_project_file('path_to_processed_features_file', config.path_to_processed_features_file)

    else:
        parse_feature_table(config.path_to_processed_features_file, samples, config)
        #Propably call some stuff here, TODO
    config.n_of_features = len(samples.feature_names)
    return samples
