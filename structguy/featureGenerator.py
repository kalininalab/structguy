import random
import sys
import traceback
import time
import ray

from structguy import sampleSpace, consts
from structguy import structural_feature_generation as strfg
from structguy import sequence_feature_generation as seqfg

from structman.base_utils.base_utils import calculate_chunksizes, pack, unpack


def expand_structural_feature_table(config):
    samples = sampleSpace.SampleSpace(config)
    # strfg.initFeatures(samples)
    seqfg.initFeatures(config, samples)

    parse_structural_features(samples, config)

    seqfg.getSequenceFeatures(config, samples, n_of_processes=config.proc_n)

    outfile = f"{config.outfolder}/{config.dataset_name}_structguy_features.tsv"

    samples.write(outfile)

    config.add_entry_to_project_file("path_to_features_file", outfile)
    return


@ray.remote(max_calls=1)
def parseLines_remote_wrapper(store, left, right):
    config, features, package = store
    lines, primary_protein_id_col, amount_of_struct_col, effect_col, aac_col_s, tags_col, non_feature_cols, feature_names = unpack(package)

    output = parseLines(config, left, right, features, lines, primary_protein_id_col, amount_of_struct_col, effect_col, aac_col_s, tags_col, non_feature_cols, feature_names)

    return output


def parseLines(config, left, right, features, lines, primary_protein_id_col, amount_of_struct_col, effect_col, aac_col_s, tags_col, non_feature_cols, feature_names):
    output = []
    if config.verbosity >= 1:
        print(
            f"parseLines: non_feature_cols: {non_feature_cols}, primary_protein_id_col: {primary_protein_id_col}, aac_col_s: {aac_col_s}, effect_col: {effect_col}, tags_col: {tags_col}, config.target_values: {config.target_values}, lines: {left}-{right} of {len(lines)}"
        )

    max_print = 10
    print_n = 0

    for line in lines[left:right]:
        if line == "":
            continue
        force_filter = False
        words = line.split("\t")
        if len(words) == 1:
            print(list(line))
            words = line.split("    ")
        # if effectRegressor != None:
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
            aac = f"{words[aac_col_s[0]]}{words[aac_col_s[1]]}{words[aac_col_s[2]]}"

        if config.filter_synonymous:
            if aac[0] == aac[-1]:
                continue

        tags = words[tags_col]
        try:
            amount_of_structures = int(words[amount_of_struct_col])
        except:
            amount_of_structures = 0

        sample_id = (primary_protein_id, aac)

        if config.verbosity >= 6:
            print(f"Parsing sample: {sample_id}")

        if primary_protein_id in config.blacklist:
            if config.verbosity >= 4:
                print(f"Sample was blacklisted {sample_id}")

            continue

        if config.target_values is not None:
            target_values = []
            for tag in tags.split(","):
                if tag == "":
                    continue
                if tag[0] != "#":
                    continue
                try:
                    tag_id, tag_value = tag.split(":")
                except:
                    try:
                        tag_id, tag_value = tag.split("=")
                    except:
                        continue
                if tag_id != config.target_values:
                    continue
                try:
                    tag_value = float(tag_value)
                except:
                    print(f"Given effect tag had non-float tag value: {tag_value} for {sample_id}")
                    continue
                target_values.append(tag_value)
            if len(target_values) == 0:
                target_value = None
            else:
                target_value = sum(target_values) / len(target_values)
        else:
            if effect_col is not None:
                if words[effect_col] == "None" and config.target_values is not None:
                    if config.verbosity >= 4:
                        print(f"Sample rejected: effect is None, while effect_col is not")

                    continue

                if not config.regression:
                    target_value = words[effect_col]
                else:
                    try:
                        target_value = float(words[effect_col])
                    except:
                        target_value = None
            else:
                target_value = None
                for tag in tags.split(","):
                    if tag[:6] == "label:":
                        target_value = tag[6:]

        """
        if add_file_to_samples:
            if target_value in config.target_translator:
                target_value = config.target_translator[target_value]
            else:
                target_value = None
        """

        feat_out = []

        for pos, x in enumerate(words):
            if pos in non_feature_cols:
                continue
            try:
                feat_name = feature_names[pos]
            except:
                continue
            if feat_name in consts.FEAT_NAME_SYNONYMS:
                feat_name = consts.FEAT_NAME_SYNONYMS[feat_name]
            if feat_name == "Classification confidence":
                continue
            feat = features[feat_name]

            try:
                value = feat.value_from_string(x)
            except:
                [e, f, g] = sys.exc_info()
                g = traceback.format_exc()
                print(f"Feature parse error, sample will be filtered: {primary_protein_id} {aac} {tags} {feat_name} {x}\n{e}\n{f}\n{g}")
                # force_filter = True
                continue

            feat_out.append((value, feat_name))

        if target_value is None and print_n < max_print and config.verbosity >= 1:
            print(f"In parseLines - Target value is None for {sample_id}")
            print_n += 1
        if force_filter:
            continue

        output.append((sample_id, target_value, amount_of_structures, tags, feat_out))

    return output


def parse_structural_features(
    samples, config, non_feature_cols=[0, 1, 2, 4, 7, 19], primary_protein_id_col=1, aac_col_s=[3, 4, 5], tags_col=7, amount_of_struct_col=19, effect_col=None, filter_none_tv=False
):
    file_path = config.path_structural_feature_table
    parse_feature_table(
        file_path,
        samples,
        config,
        filter_none_tv,
        non_feature_cols=non_feature_cols,
        primary_protein_id_col=primary_protein_id_col,
        aac_col_s=aac_col_s,
        tags_col=tags_col,
        amount_of_struct_col=amount_of_struct_col,
        effect_col=effect_col,
    )


def parse_feature_table(file_path, samples, config, filter_none_tv, non_feature_cols=[0, 1, 2, 3, 4], primary_protein_id_col=0, aac_col_s=[1], tags_col=3, amount_of_struct_col=4, effect_col=2):
    if config.verbosity >= 1:
        print(
            f"Reading feature file: {file_path}, non_feature_cols: {non_feature_cols}, primary_protein_id_col: {primary_protein_id_col}, aac_col_s: {aac_col_s}, effect_col: {effect_col}, tags_col: {tags_col}, filter none TV: {filter_none_tv}"
        )

    f = open(file_path, "r")
    lines = f.read().split("\n")
    f.close()

    if config.verbosity >= 1:
        t0 = time.time()

    non_feature_cols = set(non_feature_cols)

    feature_names = lines[0].split("\t")
    for pos, feat_name in enumerate(feature_names):
        if pos in non_feature_cols:
            continue
        if feat_name in consts.FEAT_NAME_SYNONYMS:
            feat_name = consts.FEAT_NAME_SYNONYMS[feat_name]
        if feat_name not in samples.features:
            samples.addFeature(feat_name, "unknown", group="structural")

    if config.verbosity >= 1:
        t1 = time.time()
        print(f"parse_feature_table, part 1: {t1 - t0}")

    headless_lines = lines[1:]
    random.shuffle(headless_lines)

    if config.verbosity >= 1:
        t2 = time.time()
        print(f"parse_feature_table, part 2: {t2 - t1}")

    if config.verbosity >= 4:
        samples.print_feat_types()

    output = parseLines(
        config, 0, len(headless_lines), samples.features, headless_lines, primary_protein_id_col, amount_of_struct_col, effect_col, aac_col_s, tags_col, non_feature_cols, feature_names
    )

    if config.verbosity >= 1:
        t5 = time.time()
        print(f"parse_feature_table, part 5: {t5 - t2}, {len(output)}")

    n_f = 0
    n_s = 0
    max_print = 10
    n_print = 0
    for sample_id, target_value, amount_of_structures, tags, feat_out in output:
        if filter_none_tv and target_value is None:
            continue
        for value, feat_name in feat_out:
            samples.addValue(sample_id, value, feat_name)
            n_f += 1
        samples.addTargetValue(sample_id, target_value)
        if target_value is None and n_print < max_print:
            print(f"TV is None for {sample_id}")
            n_print += 1
        samples.samples[sample_id].amount_of_structures = amount_of_structures
        samples.samples[sample_id].tags = tags
        n_s += 1

    if config.verbosity >= 3:
        samples.print_stats()

    if config.verbosity >= 4:
        samples.print_feat_types()

    if config.verbosity >= 1:
        t6 = time.time()
        print(f"parse_feature_table, part 6: {t6 - t5}")
        print(f"Total samples: {n_s}, total feature values: {n_f}")
    if config.verbosity >= 1:
        print("Finished parsing of feature file")


def createTrainingSet(
        config,
        external_impute=None,
        for_prediction=False,
        stop_matrix_transformation=False,
        other_features_path=None,
        filter_none_tv=False,
        filter_synon=False
        ):
    if config.verbosity >= 2:
        print(f"Call of createTrainingSet: {external_impute is None=} {for_prediction=} {config.path_to_processed_features_file=}")

    samples = sampleSpace.SampleSpace(config)

    if config.path_to_processed_features_file is not None and not config.overwrite:
        if config.path_to_processed_features_file[-5:] == '.dump':
            f = open(config.path_to_processed_features_file, 'rb')
            packed_samples = f.read()
            f.close()
            bu_slotmask = samples.deactivate_slot_mask()
            samples = unpack(packed_samples)
            samples.reactivate_slot_mask(bu_slotmask)
            config.n_of_features = len(samples.feature_names)
            return samples 

    # strfg.initFeatures(samples)
    seqfg.initFeatures(config, samples)

    if (config.path_to_processed_features_file is None and config.path_to_imputed_features_file is None and other_features_path is None) or config.overwrite:
        parse_feature_table(config.path_to_features_file, samples, config, filter_none_tv)

        for additonal_infile in config.add_more_sample_files:
            parse_feature_table(additonal_infile, samples, config, filter_none_tv)

        if config.fusePositions:
            config.regression = False
            # print(samples.samples.keys())
            samples.fusePositions()
            # print(samples.samples.keys())
            config.target_values = ["all neutral", "possibly damaging"]

        samples.standardFilter(config, filter_synon=filter_synon)

        if config.structure_threshold is not None:
            samples.filterSamplesByMappedStructures(config)

        if config.transform:
            samples.targetTransformation(config)
        if config.regression and not for_prediction:
            samples.detectOutliers(config)

        if config.verbosity >= 1:
            samples.printPureMixedProportion(config)

        samples.adjustParameterRanges(config)

        if config.filterStructuralFeatures:
            samples.removeFeaturesByType("structural")

        samples.oneHotifyAll()

        if external_impute is None:
            if config.impute_missing_values:
                samples.impute_all()
                path_to_impute_map_dump = f"{config.outfolder}/{config.dataset_name}_structguy_trained_impute_map.dump"
                samples.dump_impute_map(path_to_impute_map_dump)
                config.add_entry_to_project_file("path_to_impute_map", path_to_impute_map_dump)

            #config.path_to_processed_features_file = f"{config.outfolder}/{config.dataset_name}_structguy_features_processed.tsv"
            #samples.write(config.path_to_processed_features_file)
            config.path_to_processed_features_file = f"{config.outfolder}/{config.dataset_name}_structguy_features_processed.dump"
            if not stop_matrix_transformation:
                t_0 = time.time()
                samples.transform_matrix_dict()
                samples.calc_subsamples_feat_corr_matrix(config)
                t_1 = time.time()
                print(f'Time for calculating feat_corr_matrix: {t_1-t_0}')
                stop_matrix_transformation = True
            samples.dump(config.path_to_processed_features_file)

            config.add_entry_to_project_file("path_to_processed_features_file", config.path_to_processed_features_file)
        else:
            if config.impute_missing_values:
                samples.external_impute(external_impute)

                config.path_to_imputed_features_file = f"{config.outfolder}/{config.dataset_name}_structguy_features_imputed.tsv"

                samples.write(config.path_to_imputed_features_file)

                config.add_entry_to_project_file("path_to_imputed_features_file", config.path_to_imputed_features_file)
            else:
                #config.path_to_processed_features_file = f"{config.outfolder}/{config.dataset_name}_structguy_features_processed.tsv"
                #samples.write(config.path_to_processed_features_file)
                config.path_to_processed_features_file = f"{config.outfolder}/{config.dataset_name}_structguy_features_processed.dump"
                if not stop_matrix_transformation:
                    t_0 = time.time()
                    samples.transform_matrix_dict()
                    samples.calc_subsamples_feat_corr_matrix(config)
                    t_1 = time.time()
                    print(f'Time for calculating feat_corr_matrix: {t_1-t_0}')
                    stop_matrix_transformation = True
                samples.dump(config.path_to_processed_features_file)

                config.add_entry_to_project_file("path_to_processed_features_file", config.path_to_processed_features_file)

    elif other_features_path is not None:
        parse_feature_table(other_features_path, samples, config, filter_none_tv)

    elif external_impute is not None:
        parse_feature_table(config.path_to_imputed_features_file, samples, config, filter_none_tv)
    else:
        parse_feature_table(config.path_to_processed_features_file, samples, config, filter_none_tv)
        if config.verbosity >= 4:
            samples.print_feat_types()
        # Propably call some stuff here, TODO
    config.n_of_features = len(samples.feature_names)
    if not stop_matrix_transformation:
        t_0 = time.time()
        samples.transform_matrix_dict()
        samples.calc_subsamples_feat_corr_matrix(config)
        t_1 = time.time()
        print(f'Time for calculating feat_corr_matrix: {t_1-t_0}')
    return samples
