"""Parallel parsing of a feature table straight into the numeric feature matrix of a SampleSpace.

Every ray task reads one byte range of the table and converts its feature cells column-wise with numpy.
Missing values become NaN; categorical features are returned as strings and encoded by the driver.
"""
import os
import random
from operator import itemgetter

import numpy as np
import ray

from structman.lib.sdsc.consts import residues as residue_consts

from structguy import consts
from structguy.base_classes import possible_na_values
from structguy.consts import feature_categories
from structguy.util import log_timing, categorize_feat_by_name

NA_STRINGS = tuple(possible_na_values)
# Fills up lines with less columns than the header, these cells become missing values
PAD_STRING = 'None'
SUB_BLOCK_ROWS = 128
CHUNKS_PER_PROCESS = 4
N_TYPE_SAMPLE_LINES = 2000
MAX_MESSAGES = 10
STANDARD_AAS = 'ACDEFGHIKLMNPQRSTVWY'
# The mutation specific columns of StructMAn's feature table, recomputed for every substitution of a position-only table
SUBSTITUTION_FEATURES = (
    'KD mean', 'Volume mean', 'Chemical distance', 'Blosum62',
    'Aliphatic change', 'Hydrophobic change', 'Aromatic change', 'Positive charged change', 'Polar change',
    'Negative charge change', 'Charged change', 'Small change', 'Tiny change', 'Total change',
)


def tuple_getter(cols):
    if len(cols) == 0:
        return lambda words: ()
    if len(cols) == 1:
        col = cols[0]
        return lambda words: (words[col],)
    return itemgetter(*cols)


def decode_lines(data: bytes) -> list[str]:
    # Universal newlines, as in text mode
    text = data.decode('utf-8')
    if '\r' in text:
        text = text.replace('\r\n', '\n').replace('\r', '\n')
    return text.split('\n')


def split_line(line: str) -> list[str]:
    words = line.split("\t")
    if len(words) == 1:
        words = line.split("    ")
    return words


def split_feature_line(line: str, first_feature_col: int) -> list[str]:
    """split_line, with every NA cell from column first_feature_col on replaced by 'nan'.

    numpy then converts whole blocks of cells to floats at once; the replacement runs on the line string and is much
    cheaper than comparing every cell with the NA strings.
    """
    if '\t' not in line:
        return line.split("    ")
    words = line.split('\t', first_feature_col)
    if len(words) <= first_feature_col:
        return words
    tail = f'\t{words[-1]}\t'
    for na_string in NA_STRINGS:
        cell = f'\t{na_string}\t'
        if na_string != 'nan' and cell in tail:
            # Twice, as adjacent NA cells share their tab
            tail = tail.replace(cell, '\tnan\t').replace(cell, '\tnan\t')
    words[-1:] = tail[1:-1].split('\t')
    return words


def parse_line_meta(config, words, primary_protein_id_col, amount_of_struct_col, effect_col, aac_col_s, tags_col, messages):
    """Returns (sample_id, target_value, amount_of_structures, tags) of a feature table line, or None if the line gets filtered."""
    primary_protein_id = words[primary_protein_id_col]
    if len(aac_col_s) == 1:
        aac = words[aac_col_s[0]]
    else:
        aac = f"{words[aac_col_s[0]]}{words[aac_col_s[1]]}{words[aac_col_s[2]]}"

    if config.filter_synonymous and aac[0] == aac[-1]:
        return None

    tags = words[tags_col]
    try:
        amount_of_structures = int(words[amount_of_struct_col])
    except Exception:
        amount_of_structures = 0

    sample_id = (primary_protein_id, aac)

    if primary_protein_id in config.blacklist:
        return None

    if config.target_values is not None:
        target_values = []
        for tag in tags.split(","):
            if tag == "":
                continue
            if tag[0] != "#":
                continue
            try:
                tag_id, tag_value = tag.split(":")
            except ValueError:
                try:
                    tag_id, tag_value = tag.split("=")
                except ValueError:
                    continue
            if tag_id != config.target_values:
                continue
            try:
                tag_value = float(tag_value)
            except ValueError:
                if len(messages) < MAX_MESSAGES:
                    messages.append(f"Given effect tag had non-float tag value: {tag_value} for {sample_id}")
                continue
            target_values.append(tag_value)
        if len(target_values) == 0:
            target_value = None
        else:
            target_value = sum(target_values) / len(target_values)
    elif effect_col is not None:
        if not config.regression:
            target_value = words[effect_col]
        else:
            try:
                target_value = float(words[effect_col])
            except ValueError:
                target_value = None
    else:
        target_value = None
        for tag in tags.split(","):
            if tag[:6] == "label:":
                target_value = tag[6:]

    return sample_id, target_value, amount_of_structures, tags


def substitution_values(wt_aa: str, mut_aa: str) -> dict[str, object]:
    """The values of SUBSTITUTION_FEATURES as StructMAn writes them (structman lib/output/classification.py).

    Kept identical to StructMAn, including its 'Charged change' (new AA tested against the negative AAs),
    so the substitutions get the values the models saw for real mutations.
    """
    try:
        kd_mean = abs(residue_consts.HYDROPATHY[wt_aa] - residue_consts.HYDROPATHY[mut_aa])
    except KeyError:
        kd_mean = 0
    try:
        d_vol = abs(residue_consts.VOLUME[wt_aa] - residue_consts.VOLUME[mut_aa])
    except KeyError:
        d_vol = 0
    try:
        chemical_distance = residue_consts.CHEM_DIST_MATRIX[wt_aa][mut_aa]
    except KeyError:
        chemical_distance = None
    blosum_value = residue_consts.BLOSUM62.get((wt_aa, mut_aa), residue_consts.BLOSUM62.get((mut_aa, wt_aa), 0.0))

    def change(aa_map, mut_map=None):
        return int((wt_aa in aa_map) != (mut_aa in (aa_map if mut_map is None else mut_map)))

    changes = {
        'Aliphatic change': change(residue_consts.AA_MAP_ALIPHATIC),
        'Hydrophobic change': change(residue_consts.AA_MAP_HYDROPHOBIC),
        'Aromatic change': change(residue_consts.AA_MAP_AROMATIC),
        'Positive charged change': change(residue_consts.AA_MAP_POSITIVE),
        'Polar change': change(residue_consts.AA_MAP_POLAR),
        'Negative charge change': change(residue_consts.AA_MAP_NEGATIVE),
        'Charged change': change(residue_consts.AA_MAP_CHARGED, residue_consts.AA_MAP_NEGATIVE),
        'Small change': change(residue_consts.AA_MAP_SMALL),
        'Tiny change': change(residue_consts.AA_MAP_TINY),
    }
    return {
        'KD mean': kd_mean,
        'Volume mean': d_vol,
        'Chemical distance': chemical_distance,
        'Blosum62': blosum_value,
        **changes,
        'Total change': sum(changes.values()),
    }


def build_expansion(header_words, wt_col, mut_col) -> dict:
    """The cells to overwrite when a position row of a position-only table becomes its 19 substitutions."""
    cols = [mut_col]
    names = [None]
    if 'AA change' in header_words:
        cols.append(header_words.index('AA change'))
        names.append('AA change')
    for feat_name in SUBSTITUTION_FEATURES:
        if feat_name in header_words:
            cols.append(header_words.index(feat_name))
            names.append(feat_name)

    values = {}
    for wt_aa in STANDARD_AAS:
        for mut_aa in STANDARD_AAS:
            if mut_aa == wt_aa:
                continue
            feat_values = substitution_values(wt_aa, mut_aa)
            feat_values['AA change'] = f'{wt_aa}{mut_aa}'
            values[(wt_aa, mut_aa)] = [mut_aa] + [PAD_STRING if feat_values[name] is None else str(feat_values[name]) for name in names[1:]]
    return {'wt_col': wt_col, 'mut_col': mut_col, 'cols': cols, 'values': values}


def expand_position_row(words, expansion, messages) -> list[list[str]]:
    """The 19 substitution rows of a position row (WT AA == mutant AA), or [words] for any other row."""
    if expansion is None:
        return [words]
    wt_aa = words[expansion['wt_col']]
    if wt_aa != words[expansion['mut_col']]:
        return [words]
    if wt_aa not in STANDARD_AAS:
        if len(messages) < MAX_MESSAGES:
            messages.append(f'Position row with non-standard wildtype AA {wt_aa} is not expanded: {words[:expansion["mut_col"] + 1]}')
        return [words]
    cols = expansion['cols']
    rows = []
    for mut_aa in STANDARD_AAS:
        if mut_aa == wt_aa:
            continue
        row = words.copy()
        for col, value in zip(cols, expansion['values'][(wt_aa, mut_aa)]):
            row[col] = value
        rows.append(row)
    return rows


def chunk_has_mutations(file_path, start, end, wt_col, mut_col) -> bool:
    """True at the first line of the byte range [start, end) whose wildtype AA differs from its mutant AA."""
    with open(file_path, 'rb') as f:
        f.seek(start)
        while f.tell() < end:
            line = decode_lines(f.readline())[0]
            if line == '':
                continue
            words = line.split('\t', mut_col + 1)
            if len(words) > mut_col and words[wt_col] != words[mut_col]:
                return True
    return False


@ray.remote
def remote_chunk_has_mutations(file_path, start, end, wt_col, mut_col):
    return chunk_has_mutations(file_path, start, end, wt_col, mut_col)


def table_has_mutations(file_path, chunks, wt_col, mut_col) -> bool:
    """False for a position-only StructMAn table, where every line has the wildtype AA as mutant AA."""
    if not ray.is_initialized():
        return any(chunk_has_mutations(file_path, a, b, wt_col, mut_col) for a, b in chunks)
    return any(ray.get([remote_chunk_has_mutations.remote(file_path, a, b, wt_col, mut_col) for a, b in chunks]))


def cells_to_floats(cells: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Converts a 2D object array of strings: returns the floats (NaN for missing), the NA mask and the per column count of unparsable strings.

    Object arrays, because a fixed-width string array is as wide as the longest cell of the table.
    """
    n_fail = np.zeros(cells.shape[1], dtype=np.int64)
    # Fast path for cells without NA strings other than 'nan' (see split_feature_line); 'inf' is an NA string too
    try:
        values = cells.astype(np.float64)
        if not np.isposinf(values).any():
            return values, np.isnan(values), n_fail
    except ValueError:
        pass

    na = np.zeros(cells.shape, dtype=bool)
    for na_string in NA_STRINGS:
        na |= cells == na_string
    filled = cells.copy()
    filled[na] = 'nan'
    try:
        return filled.astype(np.float64), na, n_fail
    except ValueError:
        pass
    values = np.empty(cells.shape, dtype=np.float64)
    for j in range(cells.shape[1]):
        try:
            values[:, j] = filled[:, j].astype(np.float64)
        except ValueError:
            column = values[:, j]
            for i, string in enumerate(filled[:, j]):
                try:
                    column[i] = float(string)
                except ValueError:
                    column[i] = np.nan
                    n_fail[j] += 1
    return values, na, n_fail


def parse_feature_chunk(config, file_path, start, end, meta_args, num_cols, str_cols, special_cols, width, filter_none_tv, first_feature_col, expansion=None):
    """Parses the lines in the byte range [start, end) of the feature table.

    num_cols / str_cols are the table columns returned as floats / strings. special_cols maps positions in num_cols
    to their Feature, for values that need Feature.value_from_string (Blosum62). Columns from first_feature_col on
    are feature columns only. With expansion (build_expansion), every position row becomes its 19 substitutions.
    """
    with open(file_path, 'rb') as f:
        f.seek(start)
        lines = decode_lines(f.read(end - start))

    get_num = tuple_getter(num_cols)
    get_str = tuple_getter(str_cols)

    n_max = len(lines) if expansion is None else len(lines) * (len(STANDARD_AAS) - 1)
    num_block = np.empty((n_max, len(num_cols)), dtype=np.float64)
    str_block = np.empty((n_max, len(str_cols)), dtype=object)
    non_na = np.zeros(len(num_cols), dtype=np.int64)
    n_fail = np.zeros(len(num_cols), dtype=np.int64)
    non_integral = np.zeros(len(num_cols), dtype=bool)
    meta = []
    messages = []
    pending = []

    def flush():
        row = len(meta) - len(pending)
        k = len(pending)
        if len(num_cols) > 0:
            cells = np.array([get_num(words) for words in pending], dtype=object)
            values, na, fails = cells_to_floats(cells)
            for pos, feat in special_cols:
                for i, string in enumerate(cells[:, pos]):
                    value = feat.value_from_string(str(string))
                    values[i, pos] = np.nan if value is None else value
                fails[pos] = 0
            num_block[row:row + k] = values
            non_na[:] += (~na).sum(axis=0)
            n_fail[:] += fails
            with np.errstate(invalid='ignore'):
                non_integral[:] |= (~np.isnan(values) & (values != np.floor(values))).any(axis=0)
        if len(str_cols) > 0:
            str_block[row:row + k] = [get_str(words) for words in pending]
        pending.clear()

    for line in lines:
        if line == "":
            continue
        words = split_feature_line(line, first_feature_col)
        if len(words) < width:
            words += [PAD_STRING] * (width - len(words))
        for row in expand_position_row(words, expansion, messages):
            sample_meta = parse_line_meta(config, row, *meta_args, messages)
            if sample_meta is None:
                continue
            if filter_none_tv and sample_meta[1] is None:
                continue
            meta.append(sample_meta)
            pending.append(row)
            if len(pending) >= SUB_BLOCK_ROWS:
                flush()
    if len(pending) > 0:
        flush()

    n = len(meta)
    return {
        'meta': meta,
        'num_block': num_block[:n],
        'str_block': str_block[:n],
        'non_na': non_na,
        'n_fail': n_fail,
        'non_integral': non_integral,
        'messages': messages,
    }


@ray.remote
def remote_parse_feature_chunk(store, file_path, start, end):
    config, meta_args, num_cols, str_cols, special_cols, (width, first_feature_col), filter_none_tv, expansion = store
    return parse_feature_chunk(config, file_path, start, end, meta_args, num_cols, str_cols, special_cols, width, filter_none_tv, first_feature_col, expansion)


def run_chunks(config, file_path, chunks, meta_args, num_cols, str_cols, special_cols, line_layout, filter_none_tv, expansion=None) -> list[dict]:
    """Parses all chunks, as ray tasks if ray is running; line_layout is (width, first_feature_col)."""
    if not ray.is_initialized():
        width, first_feature_col = line_layout
        return [parse_feature_chunk(config, file_path, a, b, meta_args, num_cols, str_cols, special_cols, width, filter_none_tv, first_feature_col, expansion) for a, b in chunks]
    store = ray.put((config, meta_args, num_cols, str_cols, special_cols, line_layout, filter_none_tv, expansion))
    return ray.get([remote_parse_feature_chunk.remote(store, file_path, a, b) for a, b in chunks])


def compute_byte_chunks(file_path, n_chunks) -> tuple[str, list[tuple[int, int]]]:
    """Returns the header line and byte ranges covering the remaining lines, each range starting at a line start."""
    size = os.path.getsize(file_path)
    with open(file_path, 'rb') as f:
        header = decode_lines(f.readline())[0]
        first = f.tell()
        bounds = [first]
        for i in range(1, n_chunks):
            f.seek(first + ((size - first) * i) // n_chunks)
            f.readline()
            pos = f.tell()
            if bounds[-1] < pos < size:
                bounds.append(pos)
        bounds.append(size)
    return header, [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


def read_sample_lines(file_path, chunks, n_lines) -> list[list[str]]:
    """Reads lines spread over the whole table, from the start of every chunk."""
    per_chunk = max(1, n_lines // max(1, len(chunks)))
    sample = []
    with open(file_path, 'rb') as f:
        for start, end in chunks:
            f.seek(start)
            for _ in range(per_chunk):
                if f.tell() >= end:
                    break
                line = decode_lines(f.readline())[0]
                if line != '':
                    sample.append(split_line(line))
    return sample


def register_features(file_path, header_words, non_feature_cols, samples):
    new_features = []
    uncategorized = []
    for pos, feat_name in enumerate(header_words):
        if pos in non_feature_cols:
            continue
        feat_name = consts.FEAT_NAME_SYNONYMS.get(feat_name, feat_name)
        if feat_name not in samples.features:
            feat_cat_pos, _ = categorize_feat_by_name(feat_name)
            if feat_cat_pos is None:
                uncategorized.append(feat_name)
            else:
                new_features.append((feat_name, feature_categories[feat_cat_pos]))
    # All at once, so a changed feature table shows every name the categorization lacks.
    if uncategorized:
        raise ValueError(f'{len(uncategorized)} feature(s) of {file_path} without a category '
                         f'(util.categorize_feat_by_name): {uncategorized}')
    for feat_name, feat_cat in new_features:
        samples.addFeature(feat_name, "unknown", group=feat_cat)


def build_column_plan(header_words, non_feature_cols, features, ablation_structural_feature_filter, no_evo_structural_feature_filter) -> list[tuple[str, int]]:
    """Maps the parsed feature names to their table column; a later column with the same name wins."""
    plan: dict[str, int] = {}
    for pos, feat_name in enumerate(header_words):
        if pos in non_feature_cols:
            continue
        feat_name = consts.FEAT_NAME_SYNONYMS.get(feat_name, feat_name)
        if feat_name == "Classification confidence":
            continue
        feat_cat = features[feat_name].group
        if ablation_structural_feature_filter and feat_cat != 'evolutionary' and feat_cat != 'amino acid properties':
            continue
        if no_evo_structural_feature_filter and feat_cat == 'evolutionary':
            continue
        plan[feat_name] = pos
    return list(plan.items())


def is_categorical_by_counts(non_na, n_fail) -> bool:
    # Mostly unparsable as float: a categorical feature. A few unparsable values in a numeric feature become missing values.
    return n_fail * 2 > non_na


def guess_categorical_features(sample_lines, unknown_plan, width) -> set[str]:
    """Guesses which features of unknown type are categorical, from a sample of the table lines."""
    if len(unknown_plan) == 0 or len(sample_lines) == 0:
        return set()
    get_cells = tuple_getter([col for _, col in unknown_plan])
    padded = [words + [PAD_STRING] * (width - len(words)) if len(words) < width else words for words in sample_lines]
    cells = np.array([get_cells(words) for words in padded], dtype=object)
    _, na, n_fail = cells_to_floats(cells)
    non_na = (~na).sum(axis=0)
    return {feat_name for j, (feat_name, _) in enumerate(unknown_plan) if is_categorical_by_counts(non_na[j], n_fail[j])}


def strings_to_floats(values: np.ndarray) -> tuple[np.ndarray, int, int]:
    floats, na, n_fail = cells_to_floats(np.array([str(value) for value in values.tolist()], dtype=object).reshape(-1, 1))
    return floats[:, 0], int((~na).sum()), int(n_fail[0])


def has_decimal_strings(strings) -> bool:
    # Feature.value_from_string makes a feature real at the first value that int() cannot parse, like '0.0' or '1e-05'
    joined = '\t'.join(strings)
    return '.' in joined or 'E' in joined or 'e' in joined.replace('None', '')


def find_decimal_features(sample_lines, plan, width) -> set[str]:
    """Features written with decimal strings in the sample lines: real, even if all values are integral."""
    padded = [words + [PAD_STRING] * (width - len(words)) if len(words) < width else words for words in sample_lines]
    return {feat_name for feat_name, col in plan if has_decimal_strings(words[col] for words in padded)}


def numeric_type(f_type: str, non_integral: bool) -> str:
    if f_type == 'unknown':
        return 'real' if non_integral else 'integer'
    if f_type in ('integer', 'binary') and non_integral:
        return 'real'
    return f_type


def parse_feature_table_to_matrix(
        file_path,
        samples,
        config,
        filter_none_tv,
        non_feature_cols,
        primary_protein_id_col,
        aac_col_s,
        tags_col,
        amount_of_struct_col,
        effect_col,
        ablation_structural_feature_filter=False,
        no_evo_structural_feature_filter=False):
    """Parses the feature table into samples.raw_feature_matrix (float64, NaN for missing values)."""
    non_feature_cols = set(non_feature_cols)
    n_chunks = max(1, CHUNKS_PER_PROCESS * config.proc_n)

    with log_timing(config, 'parse_feature_table: header, feature registration and column plan'):
        header, chunks = compute_byte_chunks(file_path, n_chunks)
        header_words = header.split("\t")
        register_features(file_path, header_words, non_feature_cols, samples)
        plan = build_column_plan(header_words, non_feature_cols, samples.features, ablation_structural_feature_filter, no_evo_structural_feature_filter)
        meta_cols = [primary_protein_id_col, tags_col, *aac_col_s] + [col for col in (amount_of_struct_col, effect_col) if col is not None]
        width = max([col for _, col in plan] + meta_cols) + 1
        # The NA replacement of split_feature_line must not touch the non-feature columns (ids, tags, ...)
        line_layout = (width, max(non_feature_cols | set(meta_cols)) + 1)

    # A StructMAn table of positions only (every mutant AA is the wildtype AA): each position becomes its 19 substitutions
    expansion = None
    if len(aac_col_s) == 3:
        wt_col, mut_col = aac_col_s[0], aac_col_s[2]
        with log_timing(config, 'parse_feature_table: checking for a position-only table'):
            if not table_has_mutations(file_path, chunks, wt_col, mut_col):
                expansion = build_expansion(header_words, wt_col, mut_col)
        if expansion is not None and config.verbosity >= 1:
            config.logger.info(f'{file_path} lists positions without mutations: expanding each position to its 19 substitutions')

    with log_timing(config, 'parse_feature_table: guessing the feature types from a sample of lines'):
        sample_lines = read_sample_lines(file_path, chunks, N_TYPE_SAMPLE_LINES)
        unknown_plan = [(feat_name, col) for feat_name, col in plan if samples.features[feat_name].f_type == 'unknown']
        guessed_categorical = guess_categorical_features(sample_lines, unknown_plan, width)
        decimal_feats = find_decimal_features(sample_lines, [(feat_name, col) for feat_name, col in plan if feat_name not in guessed_categorical], width)
        del sample_lines

    str_plan = [(feat_name, col) for feat_name, col in plan if samples.features[feat_name].f_type == 'categorical' or feat_name in guessed_categorical]
    str_names = {feat_name for feat_name, _ in str_plan}
    num_plan = [(feat_name, col) for feat_name, col in plan if feat_name not in str_names]
    special_cols = [(j, samples.features[feat_name]) for j, (feat_name, _) in enumerate(num_plan) if feat_name == 'Blosum62']
    meta_args = (primary_protein_id_col, amount_of_struct_col, effect_col, aac_col_s, tags_col)

    with log_timing(config, f'parse_feature_table: parsing {len(chunks)} chunks ({len(num_plan)} numeric, {len(str_plan)} string columns)'):
        results = run_chunks(config, file_path, chunks, meta_args, [col for _, col in num_plan], [col for _, col in str_plan], special_cols, line_layout, filter_none_tv, expansion)

    if num_plan:
        non_na = np.sum([r['non_na'] for r in results], axis=0)
        n_fail = np.sum([r['n_fail'] for r in results], axis=0)
        non_integral = np.any([r['non_integral'] for r in results], axis=0)
    else:
        non_na = n_fail = np.zeros(0, dtype=np.int64)
        non_integral = np.zeros(0, dtype=bool)

    # Unknown features guessed as numeric, but mostly unparsable over the whole table: parse them again as strings
    reparse_plan = [(feat_name, col) for j, (feat_name, col) in enumerate(num_plan)
                    if samples.features[feat_name].f_type == 'unknown' and is_categorical_by_counts(non_na[j], n_fail[j])]
    reparsed_results = None
    if reparse_plan:
        with log_timing(config, f'parse_feature_table: parsing {len(reparse_plan)} categorical columns again'):
            reparsed_results = run_chunks(config, file_path, chunks, meta_args, [], [col for _, col in reparse_plan], [], line_layout, filter_none_tv, expansion)

    for r in results:
        for message in r['messages']:
            config.logger.info(message)

    with log_timing(config, 'parse_feature_table: assembling the feature matrix'):
        add_results_to_samples(config, samples, results, reparsed_results, num_plan, str_plan, reparse_plan, non_na, n_fail, non_integral, decimal_feats)

    if config.verbosity >= 1:
        config.logger.info(f"Total samples: {len(samples.samples)}, features: {len(samples.feat_pos_dict)}, matrix shape: {samples.raw_feature_matrix.shape}")


def add_results_to_samples(config, samples, results, reparsed_results, num_plan, str_plan, reparse_plan, non_na, n_fail, non_integral, decimal_feats):
    n_rows = sum(len(r['meta']) for r in results)

    # The old parser shuffled the lines, which defines the order of the samples
    order = list(range(n_rows))
    random.shuffle(order)
    dest = np.empty(n_rows, dtype=np.int64)
    dest[np.array(order, dtype=np.int64)] = np.arange(n_rows)

    sample_meta = [None] * n_rows
    chunk_dests = []
    offset = 0
    for r in results:
        n = len(r['meta'])
        chunk_dest = dest[offset:offset + n]
        chunk_dests.append(chunk_dest)
        for i, meta in enumerate(r['meta']):
            sample_meta[chunk_dest[i]] = meta
        offset += n

    # Lines with the same sample id: the later line wins, as with the old value-wise overwrite
    keep = np.ones(n_rows, dtype=bool)
    last_pos = {}
    for pos, meta in enumerate(sample_meta):
        if meta[0] in last_pos:
            keep[last_pos[meta[0]]] = False
        last_pos[meta[0]] = pos
    n_duplicates = n_rows - int(keep.sum())
    if n_duplicates > 0:
        config.logger.info(f'{n_duplicates} lines of the feature table repeat an earlier sample id; the later line is used')

    samples.feature_matrix_dict = None
    samples.ensure_feature_columns()
    feat_pos_dict = samples.feat_pos_dict
    block = np.empty((n_rows, len(feat_pos_dict)), dtype=np.float64)
    block[:] = samples.default_value_vector(list(feat_pos_dict.keys()))

    reparsed = {feat_name for feat_name, _ in reparse_plan}
    for j, (feat_name, _) in enumerate(num_plan):
        if feat_name in reparsed or non_na[j] == 0:
            continue
        feat = samples.features[feat_name]
        # Blosum62 keeps its type, as in Feature.value_from_string
        if feat_name != 'Blosum62':
            feat.f_type = numeric_type(feat.f_type, non_integral[j] or feat_name in decimal_feats)
        if n_fail[j] > 0 and config.verbosity >= 1:
            config.logger.info(f'Feature {feat_name}: {n_fail[j]} values are not numeric and are treated as missing')

    keep_num = np.array([feat_name not in reparsed for feat_name, _ in num_plan], dtype=bool)
    num_matrix_cols = np.array([feat_pos_dict[feat_name] for feat_name, _ in num_plan], dtype=np.int64)[keep_num]
    if len(num_matrix_cols) > 0:
        for r, chunk_dest in zip(results, chunk_dests):
            if len(chunk_dest) > 0:
                block[np.ix_(chunk_dest, num_matrix_cols)] = r['num_block'][:, keep_num]

    def gather_strings(chunk_results, k):
        values = np.empty(n_rows, dtype=object)
        for r, chunk_dest in zip(chunk_results, chunk_dests):
            values[chunk_dest] = r['str_block'][:, k]
        return values

    # Categorical features, plus unknown ones whose type guess has to be checked
    string_columns = [(feat_name, gather_strings(results, k)) for k, (feat_name, _) in enumerate(str_plan)]
    if reparsed_results is not None:
        string_columns += [(feat_name, gather_strings(reparsed_results, k)) for k, (feat_name, _) in enumerate(reparse_plan)]

    for feat_name, values in string_columns:
        feat = samples.features[feat_name]
        col = feat_pos_dict[feat_name]
        if feat.f_type != 'categorical':
            floats, n_non_na, n_values_failed = strings_to_floats(values)
            if not is_categorical_by_counts(n_non_na, n_values_failed):
                block[:, col] = floats
                if n_non_na > 0:
                    with np.errstate(invalid='ignore'):
                        non_integral_values = bool(np.any(~np.isnan(floats) & (floats != np.floor(floats))))
                    feat.f_type = numeric_type(feat.f_type, non_integral_values or has_decimal_strings(str(value) for value in values.tolist()))
                continue
            feat.set_categorical()
        block[:, col] = samples.encode_categories(feat, values)

    if n_duplicates > 0:
        block = block[keep]
        sample_meta = [meta for pos, meta in enumerate(sample_meta) if keep[pos]]

    file_cols = np.array(sorted(feat_pos_dict[feat_name] for feat_name, _ in num_plan + str_plan), dtype=np.int64)
    samples.add_parsed_rows(sample_meta, block, file_cols)

    if config.verbosity >= 1:
        n_print = 0
        for sample_id, target_value, _, _ in sample_meta:
            if target_value is None:
                config.logger.info(f"TV is None for {sample_id}")
                n_print += 1
                if n_print >= MAX_MESSAGES:
                    break
