import sampleSpace
import random
import structural_feature_generation as strfg
import sequence_feature_generation as seqfg
import consts

def expand_structural_feature_table(path_structural_feature_table, config, path_to_outfile, path_to_sequence_fasta = None):
    samples = sampleSpace.SampleSpace(config)
    strfg.initFeatures(samples)
    seqfg.initFeatures(samples,['ref50','ref90'])

    parse_feature_table(samples, path_structural_feature_table, config, non_feature_cols = [0,1,2,4,7,19], primary_protein_id_col = 1, aac_col_s = [3,4,5], tags_col = 7, amount_of_struct_col = 19, effect_col = None)

    seqfg.getSequenceFeatures(config, samples, n_of_processes=config.seq_feat_processes, seqs_from_fasta = path_to_sequence_fasta)

    samples.write(path_to_outfile)
    return

def parse_feature_table(samples, infile, config, non_feature_cols = [0,1,2,3,4], primary_protein_id_col = 0, aac_col_s = [1], tags_col = 3, amount_of_struct_col = 4, effect_col = 2):
    f = open(infile,'r')
    lines = f.read().split('\n')
    f.close()

    print('Reading feature file: ',infile)

    non_feature_cols = set(non_feature_cols)

    feature_names = lines[0].split('\t')
    for pos, feat_name in enumerate(feature_names):
        if pos in non_feature_cols:
            continue
        if feat_name in consts.FEAT_NAME_SYNONYMS:
            feat_name = consts.FEAT_NAME_SYNONYMS[feat_name]
        if feat_name not in samples.features:
            print('Warning: Parsed unknown feature:', feat_name)
            samples.addFeature(feat_name,'binary',group='structural',default_value=0)

    headless_lines = lines[1:]
    random.shuffle(headless_lines)

    for line in headless_lines:
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

        if config.verbosity >= 4:
            print(f'Parsing sample: {sample_id}')


        if primary_protein_id in config.blacklist:
            if config.verbosity >= 4:
                print(f'Sample was blacklisted')
            continue

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

        for pos,x in enumerate(words):
            if pos in non_feature_cols:
                continue
            feat_name = feature_names[pos]
            if feat_name in consts.FEAT_NAME_SYNONYMS:
                feat_name = consts.FEAT_NAME_SYNONYMS[feat_name]
            if feat_name == 'Classification confidence':
                continue
            feat = samples.features[feat_name]
            try:
                if feat.f_type == 'categorical':
                    value = x
                elif feat.f_type == 'real':
                    value = float(x)
                elif feat.f_type == 'integer' or feat.f_type == 'binary':
                    value = int(x)
            except:
                print('Feature parse error, sample will be filtered:', primary_protein_id, aac, tags, feat_name, x)
                target_value = None
                continue

            samples.addValue(sample_id,value,feat_name)

        samples.addTargetValue(sample_id,target_value)
        samples.samples[sample_id].amount_of_structures = amount_of_structures
        samples.samples[sample_id].tags = tags

def createTrainingSet(config, session, effectRegressor=None, outfile=None, infile=None, debug=0, samples=None):

    add_file_to_samples = True
    if samples == None:
        samples = sampleSpace.SampleSpace(config)
        add_file_to_samples = False

    msa_db = config.msa_db
    if infile != None:
        if not add_file_to_samples:
            strfg.initFeatures(samples)
            seqfg.initFeatures(samples,['ref50','ref90','ref100'])

        parse_feature_table(samples, infile, config)

        if config.fusePositions:
            config.regression = False
            #print(samples.samples.keys())
            samples.fusePositions()
            #print(samples.samples.keys())
            config.target_values = ['all neutral','possibly damaging']

    else:

        #generates structural and amino acid property features based on a structman session
        #also adds the target values to sample space
        strfg.generateStructuralFeatures(config,session,samples,debug=debug)

        #generates the sequence based features, requires generateStructuralFeatures to be processed previously
        seqfg.getSequenceFeatures(config, samples, n_of_processes=config.seq_feat_processes)

        if outfile != None:
            samples.write(outfile)

    samples.standardFilter(config)

    if config.structure_threshold != None:
        samples.filterSamplesByMappedStructures(config)

    if config.transform:
        samples.targetTransformation(config)
    if config.regression:
        samples.detectOutliers(config)

    samples.printPureMixedProportion(config)

    samples.adjustParameterRanges(config)

    return samples
