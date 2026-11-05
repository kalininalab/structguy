"""The measurements of the dataset, each with the input entry it comes from.

StructMAn fuses input entries with identical sequences into one protein, and its feature table
has a row per SNV of it with the tags of all entries mixed. Several entries can measure the same
SAV, e.g. the MAVEs of different labs or of different function types on one protein. StructMAn
keeps them apart in its session tables and writes every entry's variants to
<output>.input_measurements.tsv, which is read here. For StructMAn output without that table,
the measurements are taken from the dataset fasta and mapped onto the proteins of the feature
table through StructMAn's Input Protein ID column.

- Measurements of different function types are different kinds of experiments and always stay
  apart: a protein whose entries have several function types becomes a sample per function type
  for each of its SAVs (protein id <protein>|function_type=<type>, see function_type_protein_id),
  with the same structural features but its own Function Type feature and target value.
- Several measurements of a SAV within one function type are averaged into its target value.
- For evaluation, every input entry keeps its own measurements, so each entry gets its own
  performance metrics.
"""
import os

import numpy as np
from structman.lib.sdsc.consts import residues as residue_consts

FUNCTION_TYPE_TAG = 'function_type'
FUNCTION_TYPE_FEATURE = 'Function Type'
# StructMAn's Function Type of an input entry without a function_type tag
DEFAULT_FUNCTION_TYPE = 'OrganismalFitness'
FUNCTION_TYPE_PROTEIN_SEPARATOR = '|function_type='

MAX_MESSAGES = 10


def normalize_sav(sav: str) -> str | None:
    """A SAV in the one letter notation of the feature table (K551Q), or None if it is no plain SAV.

    Accepts K551Q and p.Lys551Gln. Notations naming another sequence (ENSP...:p.Leu12Val) are no
    measurement of the entry's own sequence and give None.
    """
    if sav.startswith('p.'):
        body = sav[2:]
        if len(body) < 7:
            return None
        wt_aa = residue_consts.THREE_TO_ONE.get(body[:3].upper())
        mut_aa = residue_consts.THREE_TO_ONE.get(body[-3:].upper())
        position = body[3:-3]
        if wt_aa is None or mut_aa is None or len(wt_aa) != 1 or len(mut_aa) != 1 or not position.isdigit():
            return None
        return f'{wt_aa}{position}{mut_aa}'
    if len(sav) >= 3 and sav[0].isalpha() and sav[-1].isalpha() and sav[1:-1].isdigit():
        return sav
    return None


def value_from_tags(tags: str, target_tag: str) -> float | None:
    """The value of the target tag (#effect:0.5 or #effect=0.5) in a comma separated tag string."""
    for tag in tags.split(','):
        if not tag.startswith(target_tag):
            continue
        rest = tag[len(target_tag):]
        if rest[:1] not in (':', '='):
            continue
        try:
            return float(rest[1:])
        except ValueError:
            return None
    return None


def function_type_from_tags(tags: str) -> str:
    for tag in tags.split(','):
        key, sep, value = tag.partition('=')
        if sep and key == FUNCTION_TYPE_TAG:
            return value
    return DEFAULT_FUNCTION_TYPE


def function_type_protein_id(protein_id: str, function_type: str) -> str:
    """The sample protein id of one function type of a protein measured under several."""
    return f'{protein_id}{FUNCTION_TYPE_PROTEIN_SEPARATOR}{function_type}'


def base_protein_id(protein_id: str) -> str:
    """The protein of a sample protein id: all function types of a protein belong to the same protein,
    e.g. for the cross validation folds."""
    return protein_id.split(FUNCTION_TYPE_PROTEIN_SEPARATOR, 1)[0]


def input_measurement_table_path(config) -> str | None:
    """StructMAn's input measurement table next to its feature table, if it was written."""
    feature_table = getattr(config, 'path_structural_feature_table', None)
    if feature_table is None or not feature_table.endswith('.features.tsv'):
        return None
    path = f"{feature_table[:-len('.features.tsv')]}.input_measurements.tsv"
    return path if os.path.isfile(path) else None


def split_input_ids(joined: str, known_ids) -> list[str]:
    """Splits StructMAn's comma joined Input Protein ID, whose entry ids can contain commas themselves."""
    input_ids = []
    current = None
    for token in joined.split(','):
        current = token if current is None else f'{current},{token}'
        if current in known_ids:
            input_ids.append(current)
            current = None
    return input_ids


class DatasetMeasurements:

    def __init__(self, target_tag: str):
        self.target_tag = target_tag
        # input entry -> function type
        self.function_types: dict[str, str] = {}
        # input entry -> SAV -> values (a list: an entry can name a SAV more than once)
        self.singles: dict[str, dict[str, list[float]]] = {}
        # input entry -> [(SAVs, value)]
        self.multis: dict[str, list[tuple[tuple[str, ...], float]]] = {}
        # protein of the feature table -> input entries fused into it
        self.members: dict[str, list[str]] = {}
        self.n_unparsed_savs = 0

    @classmethod
    def load(cls, config):
        """The measurements of the configured dataset, or None if there are none or no numeric target tag.

        From StructMAn's input measurement table if there is one, else from the dataset fasta.
        """
        target_tag = config.target_values
        if not isinstance(target_tag, str):
            return None
        measurements = cls(target_tag)

        input_measurement_table = input_measurement_table_path(config)
        path_to_dataset = getattr(config, 'path_to_dataset', None)
        if input_measurement_table is not None:
            source = input_measurement_table
            measurements.parse_input_measurement_table(input_measurement_table)
        elif path_to_dataset is not None and os.path.isfile(path_to_dataset):
            source = path_to_dataset
            measurements.parse_dataset_fasta(path_to_dataset)
            measurements.read_members(config)
        else:
            return None

        if config.verbosity >= 1:
            n_singles = sum(len(values) for savs in measurements.singles.values() for values in savs.values())
            n_multis = sum(len(multis) for multis in measurements.multis.values())
            n_fused = sum(1 for entries in measurements.members.values() if len(entries) > 1)
            config.logger.info(
                f'Dataset measurements of {source}: {len(measurements.function_types)} input entries, '
                f'{n_singles} single SAV and {n_multis} multi SAV measurements, {n_fused} proteins with fused entries, '
                f'{measurements.n_unparsed_savs} variants of unsupported notation left out'
            )
        return measurements

    def parse_input_measurement_table(self, path: str) -> None:
        """StructMAn's <output>.input_measurements.tsv: a line per input entry and variant, with the entry's
        protein tags and its own variant tags, and the protein of the feature table it was fused into."""
        members: dict[str, set[str]] = {}
        with open(path, 'r') as f:
            f.readline()
            for line in f:
                words = line.rstrip('\n').split('\t')
                if len(words) < 6:
                    continue
                entry_id, protein_id, _, protein_tags, variant, tags = words[:6]
                if entry_id not in self.function_types:
                    self.function_types[entry_id] = function_type_from_tags(protein_tags)
                    self.singles[entry_id] = {}
                    self.multis[entry_id] = []
                if protein_id not in members:
                    members[protein_id] = set()
                members[protein_id].add(entry_id)
                self.add_measurement(entry_id, variant.replace(':', ','), tags)
        self.members = {protein_id: sorted(entries) for protein_id, entries in members.items()}

    def parse_dataset_fasta(self, path_to_dataset: str) -> None:
        entry_id = None
        with open(path_to_dataset, 'r') as f:
            for line in f:
                line = line.strip()
                if len(line) == 0:
                    continue
                if line[0] == '>':
                    words = line[1:].split(None, 1)
                    entry_id = words[0]
                    self.function_types[entry_id] = DEFAULT_FUNCTION_TYPE
                    self.singles[entry_id] = {}
                    self.multis[entry_id] = []
                    if len(words) > 1:
                        # Same format as StructMAn's parseFasta: SAV<tags;SAV<tags, an empty SAV for the protein tags
                        for sav_tag_tuple in words[1].split()[0].split(';'):
                            savs, _, tags = sav_tag_tuple.partition('<')
                            if savs == '':
                                self.function_types[entry_id] = function_type_from_tags(tags)
                            else:
                                self.add_measurement(entry_id, savs, tags)
                elif line[0] == '<' and entry_id is not None:
                    words = line[1:].split()
                    if len(words) == 0:
                        continue
                    self.add_measurement(entry_id, words[0], words[1] if len(words) > 1 else '')

    def add_measurement(self, entry_id: str, savs: str, tags: str) -> None:
        value = value_from_tags(tags, self.target_tag)
        if value is None:
            return
        normalized = [normalize_sav(sav) for sav in savs.split(',')]
        if None in normalized:
            self.n_unparsed_savs += 1
            return
        if len(normalized) == 1:
            sav = normalized[0]
            if sav not in self.singles[entry_id]:
                self.singles[entry_id][sav] = []
            self.singles[entry_id][sav].append(value)
        else:
            self.multis[entry_id].append((tuple(normalized), value))

    def read_members(self, config) -> None:
        """Maps the proteins of StructMAn's output onto the input entries fused into them.

        Reads the Input Protein ID (column 1) and Primary Protein ID (column 2) of the classification
        table, which has them for every position, or of the feature table if there is no classification table.
        """
        table_path = None
        feature_table = getattr(config, 'path_structural_feature_table', None)
        if feature_table is not None:
            classification_table = feature_table.replace('.features.tsv', '.classification.tsv')
            if classification_table != feature_table and os.path.isfile(classification_table):
                table_path = classification_table
            elif os.path.isfile(feature_table):
                table_path = feature_table

        members: dict[str, set[str]] = {}
        unknown_input_ids: dict[str, str] = {}
        if table_path is not None:
            seen = set()
            with open(table_path, 'r') as f:
                f.readline()
                for line in f:
                    words = line.rstrip('\n').split('\t', 2)
                    if len(words) < 2 or (words[0], words[1]) in seen:
                        continue
                    seen.add((words[0], words[1]))
                    joined_input_ids, protein_id = words[0], words[1]
                    if protein_id not in members:
                        members[protein_id] = set()
                    input_ids = split_input_ids(joined_input_ids, self.function_types)
                    if len(input_ids) == 0 and joined_input_ids not in self.function_types:
                        unknown_input_ids[protein_id] = joined_input_ids
                    members[protein_id].update(input_ids)

        # Without any table, or for proteins it does not name, a protein stands for the entry of its own id
        for protein_id, entries in members.items():
            if len(entries) == 0 and protein_id in self.function_types:
                entries.add(protein_id)
        self.members = {protein_id: sorted(entries) for protein_id, entries in members.items()}

        if len(unknown_input_ids) > 0:
            examples = list(unknown_input_ids.items())[:MAX_MESSAGES]
            config.logger.info(f'{len(unknown_input_ids)} proteins with an Input Protein ID that is no entry of the dataset fasta, e.g.: {examples}')

    def members_of(self, protein_id: str) -> list[str]:
        if protein_id in self.members:
            return self.members[protein_id]
        if protein_id in self.function_types:
            return [protein_id]
        return []

    def function_types_of(self, protein_id: str) -> list[str]:
        return sorted(set(self.function_types[entry_id] for entry_id in self.members_of(protein_id)))

    def training_target(self, protein_id: str, sav: str, function_type: str) -> float | None:
        """The mean of the measurements of a SAV by the entries of one function type, None if there is none."""
        values = []
        for entry_id in self.members_of(protein_id):
            if self.function_types[entry_id] == function_type:
                values += self.singles[entry_id].get(sav, [])
        if len(values) == 0:
            return None
        return sum(values) / len(values)

    def apply_to_samples(self, samples, config) -> None:
        """Sets the target values of the samples of the proteins known to the dataset from its measurements.

        A protein measured under one function type keeps its samples, each SAV gets the mean of its measurements.
        A protein measured under several gets a sample per function type for each SAV instead (function_type_protein_id),
        a copy of the SAV's features with the Function Type feature set to that type, and with the mean of that type's
        measurements as target. Measurements of different function types are never mixed.
        """
        n_targets = 0
        n_averaged = 0
        removed_ids = []
        new_meta = []
        new_rows = []
        function_type_col = samples.feat_pos_dict.get(FUNCTION_TYPE_FEATURE)
        function_type_feat = samples.features.get(FUNCTION_TYPE_FEATURE)
        split_proteins: dict[str, list[str]] = {}

        for sample_id, sample in samples.samples.items():
            protein_id, sav = sample_id
            if len(self.members_of(protein_id)) == 0:
                continue
            function_types = self.function_types_of(protein_id)
            if len(function_types) == 1:
                sample.targetValue = self.training_target(protein_id, sav, function_types[0])
                targets = [(function_types[0], sample.targetValue)]
            else:
                split_proteins[protein_id] = function_types
                row = samples.raw_feature_matrix[samples.sample_pos_dict[sample_id]]
                targets = []
                for function_type in function_types:
                    target_value = self.training_target(protein_id, sav, function_type)
                    targets.append((function_type, target_value))
                    new_meta.append(((function_type_protein_id(protein_id, function_type), sav), target_value, sample.amount_of_structures, sample.tags))
                    new_row = row.copy()
                    if function_type_col is not None and function_type_feat is not None and function_type_feat.f_type == 'categorical':
                        new_row[function_type_col] = function_type_feat.encode(function_type)
                    new_rows.append(new_row)
                removed_ids.append(sample_id)

            for function_type, target_value in targets:
                if target_value is None:
                    continue
                n_targets += 1
                n_measurements = sum(len(self.singles[entry_id].get(sav, ())) for entry_id in self.members_of(protein_id)
                                     if self.function_types[entry_id] == function_type)
                if n_measurements > 1:
                    n_averaged += 1

        if len(removed_ids) > 0:
            for sample_id in removed_ids:
                del samples.samples[sample_id]
            samples.compact_matrix_rows()
            block = np.array(new_rows, dtype=np.float64)
            samples.add_parsed_rows(new_meta, block, np.arange(block.shape[1], dtype=np.int64))
            if function_type_col is None or function_type_feat is None or function_type_feat.f_type != 'categorical':
                config.logger.info(f'No categorical {FUNCTION_TYPE_FEATURE} feature: the function type samples of a protein differ in their targets only')

        if config.verbosity >= 1:
            config.logger.info(f'Target values from the dataset measurements: {n_targets} samples, {n_averaged} of them averaged over several measurements of one function type')
            if len(split_proteins) > 0:
                examples = list(split_proteins.items())[:MAX_MESSAGES]
                config.logger.info(f'{len(split_proteins)} proteins measured under several function types got a sample per function type, e.g.: {examples}')

    def prediction_protein_id(self, protein_id: str, entry_id: str, predicted_proteins: set[str]) -> str:
        """The sample protein id that holds the predictions for an input entry: its function type's, if the protein got split."""
        function_type_id = function_type_protein_id(protein_id, self.function_types[entry_id])
        if function_type_id in predicted_proteins:
            return function_type_id
        return protein_id

    def evaluation_records(self, predictions: dict[tuple[str, str], float], combine_effects):
        """Per input entry records (entry id, variant, true value, predicted value, sample protein id) of the predicted proteins.

        Returns the single SAV and the multi SAV records. A multi SAV gets the combination of the predictions of its
        SAVs (combine_effects) and is left out if one of them was not predicted.
        """
        single_records = []
        multi_records = []
        predicted_proteins = set(protein_id for protein_id, _ in predictions)
        for protein_id in sorted(set(base_protein_id(protein_id) for protein_id in predicted_proteins)):
            for entry_id in self.members_of(protein_id):
                sample_protein_id = self.prediction_protein_id(protein_id, entry_id, predicted_proteins)
                for sav, values in self.singles[entry_id].items():
                    if (sample_protein_id, sav) not in predictions:
                        continue
                    single_records.append((entry_id, sav, sum(values) / len(values), predictions[(sample_protein_id, sav)], sample_protein_id))
                for savs, value in self.multis[entry_id]:
                    sav_predictions = [predictions.get((sample_protein_id, sav)) for sav in savs]
                    if None in sav_predictions:
                        continue
                    multi_records.append((entry_id, ':'.join(savs), value, combine_effects(sav_predictions), sample_protein_id))
        return single_records, multi_records
