import time
import numpy as np
from structguy.sampleSpace import SampleSpace, load_samples_dump
from structguy import sequence_feature_generation as seqfg
from structguy.util import Config, log_timing
from structguy.feature_table_parsing import parse_feature_table_to_matrix
from structguy.feature_interactions import combination_names_from_config, parse_combination_name
from structguy.measurements import DatasetMeasurements


def expand_structural_feature_table(config):
    samples = SampleSpace(config)
    # strfg.initFeatures(samples)
    # seqfg.initFeatures(config, samples)

    parse_structural_features(samples, config)

    seqfg.getSequenceFeatures(config, samples, n_of_processes=config.proc_n)

    outfile = f"{config.outfolder}/{config.dataset_name}_structguy_features.tsv"

    samples.write(outfile)

    config.add_entry_to_project_file("path_to_features_file", outfile)
    # Processed from the previous feature table; createTrainingSet processes the new one again
    config.remove_entries_from_project_file({"path_to_processed_features_file", "path_to_imputed_features_file"})
    return

def msa_bench(config):
    samples = SampleSpace(config)
    # strfg.initFeatures(samples)
    # seqfg.initFeatures(config, samples)

    parse_structural_features(samples, config)

    seqfg.msa_bench(config, samples)

def parse_structural_features(
    samples: SampleSpace,
    config: Config, 
    non_feature_cols=[0, 1, 2, 4, 7, 19], 
    primary_protein_id_col=1, 
    aac_col_s=[3, 4, 5], 
    tags_col=7, 
    amount_of_struct_col=19, 
    effect_col=None, 
    filter_none_tv=False
):
    # The expanded table (the same rows and leading columns, plus the features of each
    # position's interacting positions) replaces the plain one when the project names it.
    file_path = getattr(config, 'path_expanded_feature_table', None) or config.path_structural_feature_table
    if file_path is None:
        config.logger.info(f'Critical error in parse_structural_features: {config.path_structural_feature_table=}')
        raise TypeError(f'{config.path_structural_feature_table=}')
    if config.verbosity >= 1 and file_path != config.path_structural_feature_table:
        config.logger.info(f'parse_structural_features: using the expanded feature table {file_path}')
    
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
        effect_col=effect_col
    )
    # The sequence features get added to the feature matrix afterwards (seqfg.getSequenceFeatures)


def parse_feature_table(
        file_path,
        samples: SampleSpace, 
        config, 
        filter_none_tv, 
        non_feature_cols=[0, 1, 2, 3, 4], 
        primary_protein_id_col=0, 
        aac_col_s=[1], 
        tags_col=3, 
        amount_of_struct_col=4, 
        effect_col=2,
        ablation_structural_feature_filter=False,
        no_evo_structural_feature_filter=False
        ):
    if config.verbosity >= 1:
        config.logger.info(
            f"Reading feature file: {file_path}, non_feature_cols: {non_feature_cols}, primary_protein_id_col: {primary_protein_id_col}, aac_col_s: {aac_col_s}, effect_col: {effect_col}, tags_col: {tags_col}, filter none TV: {filter_none_tv}"
        )

    if samples.feature_matrix_dict:
        samples.transform_matrix_dict()

    with log_timing(config, f'parse_feature_table total ({file_path})'):
        parse_feature_table_to_matrix(
            file_path, samples, config, filter_none_tv,
            non_feature_cols=non_feature_cols,
            primary_protein_id_col=primary_protein_id_col,
            aac_col_s=aac_col_s,
            tags_col=tags_col,
            amount_of_struct_col=amount_of_struct_col,
            effect_col=effect_col,
            ablation_structural_feature_filter=ablation_structural_feature_filter,
            no_evo_structural_feature_filter=no_evo_structural_feature_filter,
        )

    if config.verbosity >= 3:
        samples.print_stats(config)

    if config.verbosity >= 4:
        samples.print_feat_types(config)

    if config.verbosity >= 1:
        config.logger.info("Finished parsing of feature file")


def createTrainingSet(
        config,
        external_impute=None,
        for_prediction=False,
        stop_matrix_transformation=False,
        other_features_path=None,
        filter_none_tv=False,
        filter_synon=False,
        combination_names=None
        ):
    if config.verbosity >= 2:
        config.logger.info(f"Call of createTrainingSet: {external_impute is None=} {for_prediction=} {config.path_to_processed_features_file=}")

    samples: SampleSpace = SampleSpace(config)

    if config.path_to_processed_features_file is not None and not config.overwrite:
        if config.path_to_processed_features_file[-5:] == '.dump':
            with log_timing(config, f'createTrainingSet: loading {config.path_to_processed_features_file}'):
                empty_samples = samples
                bu_slotmask = empty_samples.deactivate_slot_mask()
                try:
                    samples = load_samples_dump(config.path_to_processed_features_file)
                finally:
                    empty_samples.reactivate_slot_mask(bu_slotmask)
                config.n_of_features = len(samples.feature_names)
                if isinstance(samples.raw_feature_matrix, tuple):
                    samples.raw_feature_matrix = np.array(samples.raw_feature_matrix)
                # Dumps of older versions hold an object array with None for missing values
                if samples.raw_feature_matrix.dtype == object:
                    samples.raw_feature_matrix = np.array(samples.raw_feature_matrix, dtype=np.float64)
            # A dump written without the configured combination features: add them, the stored feature
            # correlations then miss their columns and are computed again where needed
            all_combination_names = combination_names_from_config(config) + list(combination_names or [])
            if all_combination_names and samples.add_combination_features(all_combination_names, config) > 0:
                samples.feat_corr_matrix = None
                config.n_of_features = len(samples.feature_names)
            return samples

    # strfg.initFeatures(samples)
    seqfg.initFeatures(config, samples)

    ablation_structural_feature_filter = config.ablation
    no_evo_structural_feature_filter = config.no_evo

    if (config.path_to_processed_features_file is None and config.path_to_imputed_features_file is None and other_features_path is None) or config.overwrite:
        # StructMAn writes the project file anew, dropping the entries of earlier StructGuy steps
        if config.path_to_features_file is None:
            raise ValueError(f'No path_to_features_file in {config.path_to_project_file}: run "structguy generate_features" on this project file first')
        parse_feature_table(config.path_to_features_file, samples, config, filter_none_tv, ablation_structural_feature_filter=ablation_structural_feature_filter, no_evo_structural_feature_filter=no_evo_structural_feature_filter)

        # The feature table has a row per SAV of a protein, with the measurements of all its input entries mixed:
        # a sample per function type, each with the mean of its measurements. Before the target transformation
        # and the outlier detection, which work on these values.
        if config.regression:
            dataset_measurements = DatasetMeasurements.load(config)
            if dataset_measurements is not None:
                dataset_measurements.apply_to_samples(samples, config)

        for additonal_infile in config.add_more_sample_files:
            parse_feature_table(additonal_infile, samples, config, filter_none_tv, ablation_structural_feature_filter=ablation_structural_feature_filter, no_evo_structural_feature_filter=no_evo_structural_feature_filter)

        if config.fusePositions:
            config.regression = False
            # config.logger.info(samples.samples.keys())
            samples.fusePositions()
            # config.logger.info(samples.samples.keys())
            config.target_values = ["all neutral", "possibly damaging"]

        with log_timing(config, 'createTrainingSet: sample filters'):
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
            samples.removeFeaturesByType("structural", config)

        # Before the imputation, so a combination of a missing value is missing (and gets imputed itself)
        all_combination_names = combination_names_from_config(config) + list(combination_names or [])
        if all_combination_names:
            with log_timing(config, 'createTrainingSet: combination features'):
                samples.add_combination_features(all_combination_names, config)

        #samples.oneHotifyAll(config)

        if external_impute is None:
            if config.impute_missing_values:
                with log_timing(config, 'createTrainingSet: imputation'):
                    samples.impute_all()
                path_to_impute_map_dump = f"{config.outfolder}/{config.dataset_name}_structguy_trained_impute_map.dump"
                samples.dump_impute_map(path_to_impute_map_dump)
                config.add_entry_to_project_file("path_to_impute_map", path_to_impute_map_dump)

            write_processed_tsv(samples, config)
            config.path_to_processed_features_file = f"{config.outfolder}/{config.dataset_name}_structguy_features_processed.dump"
            if not stop_matrix_transformation:
                transform_and_calc_feat_corr(samples, config, calc_corr=not for_prediction)
                stop_matrix_transformation = True
            with log_timing(config, 'createTrainingSet: writing the .dump'):
                samples.dump(config.path_to_processed_features_file)

            config.add_entry_to_project_file("path_to_processed_features_file", config.path_to_processed_features_file)
        else:
            if config.impute_missing_values:
                samples.external_impute(external_impute)

                config.path_to_imputed_features_file = f"{config.outfolder}/{config.dataset_name}_structguy_features_imputed.tsv"

                # Later runs parse this file again
                with log_timing(config, 'createTrainingSet: writing the imputed feature table'):
                    samples.write(config.path_to_imputed_features_file)

                config.add_entry_to_project_file("path_to_imputed_features_file", config.path_to_imputed_features_file)
            else:
                write_processed_tsv(samples, config)
                config.path_to_processed_features_file = f"{config.outfolder}/{config.dataset_name}_structguy_features_processed.dump"
                if not stop_matrix_transformation:
                    transform_and_calc_feat_corr(samples, config, calc_corr=not for_prediction)
                    stop_matrix_transformation = True
                with log_timing(config, 'createTrainingSet: writing the .dump'):
                    samples.dump(config.path_to_processed_features_file)

                config.add_entry_to_project_file("path_to_processed_features_file", config.path_to_processed_features_file)

    elif other_features_path is not None:
        parse_feature_table(other_features_path, samples, config, filter_none_tv, ablation_structural_feature_filter=ablation_structural_feature_filter, no_evo_structural_feature_filter=no_evo_structural_feature_filter)

    elif external_impute is not None:
        parse_feature_table(config.path_to_imputed_features_file, samples, config, filter_none_tv, ablation_structural_feature_filter=ablation_structural_feature_filter, no_evo_structural_feature_filter=no_evo_structural_feature_filter)
    else:
        parse_feature_table(config.path_to_processed_features_file, samples, config, filter_none_tv, ablation_structural_feature_filter=ablation_structural_feature_filter, no_evo_structural_feature_filter=no_evo_structural_feature_filter)
        if config.verbosity >= 4:
            samples.print_feat_types(config)
        # Propably call some stuff here, TODO
    config.n_of_features = len(samples.feature_names)
    if not stop_matrix_transformation:
        transform_and_calc_feat_corr(samples, config, calc_corr=not for_prediction)
    return samples


def write_processed_tsv(samples: SampleSpace, config: Config):
    # build_model and predict reload the .dump, so the TSV is only written on request
    if not config.write_processed_tsv:
        return
    path = f"{config.outfolder}/{config.dataset_name}_structguy_features_processed.tsv"
    with log_timing(config, f'createTrainingSet: writing {path}'):
        samples.write(path)


def transform_and_calc_feat_corr(samples: SampleSpace, config: Config, calc_corr: bool = True):
    with log_timing(config, 'createTrainingSet: transform_matrix_dict'):
        samples.transform_matrix_dict(config = config)
    # Predictions take feat_stats and the features from the model, the correlations are only needed for training
    if calc_corr:
        with log_timing(config, 'createTrainingSet: calc_subsamples_feat_corr_matrix'):
            samples.calc_subsamples_feat_corr_matrix(config)


def load_data_for_pred(
    config: Config,
    impute_map,
    booster_list,
    extern_features
):
    t0 = time.time()
    # Combination features the model was trained with are rebuilt from their names
    model_combination_names = [feat_name for _, feature_names in booster_list for feat_name in feature_names if parse_combination_name(feat_name) is not None]
    samples = createTrainingSet(config, external_impute=impute_map, for_prediction=True, filter_synon=config.trace_decisions, combination_names=model_combination_names)
    # A cached or re-parsed feature table may lack them
    if model_combination_names:
        samples.add_combination_features(model_combination_names, config)
    t1 = time.time()

    config.logger.info(f"Time for loading dataset: {t1 - t0}")

    for sample_id in samples.samples:
        samples.samples[sample_id].testtrain = "test"

    booster_specific_data = []
    for _, feature_names in booster_list:
        test_feature_matrix, test_targets, sample_id_list, feat_id_vec, cat_vec = samples.get_test_data_for_feature_list(feature_names, config, extern_features)
        booster_specific_data.append((test_feature_matrix, test_targets, sample_id_list, feat_id_vec, cat_vec))

    return samples, booster_specific_data