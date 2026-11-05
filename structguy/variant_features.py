"""Feature tables of all single mutations in the background of sequence variants, adapted from the wildtype table.

A variant differs from the wildtype sequence of a protein by substitutions only. Its rows are the rows of the
wildtype structguy feature table, with the mutation specific columns recomputed:
- at substituted positions, the wildtype AA becomes the variant's AA: identity and AA property columns, and the
  MSA based features from the position's AA profile, as read from the 19 wildtype rows of the position (the same
  values seqfg.fill_prot_mut_map and msa.calcPsicProfiles compute with the variant as seed on the same MSA);
- the window and protein median dPSIC everywhere, as they depend on the positional medians of the whole sequence.

Approximations: the structural features stay those of the wildtype structure, the variant is not part of the MSA
and the GEMME scores of a substituted position are differences, gemme(V->a) = gemme(wt->a) - gemme(wt->V).
"""
import math
import os

from structguy.feature_interactions import parse_combination_name
from structguy.feature_table_parsing import STANDARD_AAS, substitution_values
from structguy.msa import median, window_bounds
from structguy.util import loadModel, parse_conf

META_COLS = ['Uniprot Ac', 'AAC', 'Target value', 'Tags', 'Amount of mapped structures']
NA_CELL = 'None'
GEMME_TYPES = ('evolInd', 'evolEpi', 'evolCombi')
# Project file entries of the wildtype dataset that predict must not use for the variants
DROPPED_PROJECT_ENTRIES = ('path_to_processed_features_file', 'path_to_imputed_features_file', 'Path_to_multi_savs_table')


def parse_variants_file(path) -> list[str]:
    with open(path) as f:
        return [line.strip() for line in f if line.strip() != '']


def to_float(cell: str) -> float:
    try:
        return float(cell)
    except ValueError:
        return math.nan


def to_cell(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return NA_CELL
    return str(value)


def msa_suffixes(header) -> list[str]:
    """The MSA tags of the sequence features, like 'MSA gpw_distant'."""
    prefix = 'Wildtype AA rate '
    return [name[len(prefix):] for name in header if name.startswith(prefix) and not name.startswith(f'{prefix}gapless ')]


def computation_columns(header) -> set[str]:
    """The columns read to build the AA profiles."""
    names = {'Wildtype AA', 'Mutant AA'}
    for suffix in msa_suffixes(header):
        names.update(f'{base} {suffix}' for base in (
            'Wildtype AA rate', 'Mutant AA rate', 'Wildtype AA rate gapless', 'Mutant AA rate gapless',
            'MSA allel freq', 'PSIC wildtype AA', 'PSIC mutant AA', 'Positional median dPSIC',
        ))
        names.update(f'gemme_{gemme_type} {suffix}' for gemme_type in GEMME_TYPES)
    return names & set(header)


def needed_columns(model_feature_names, header, logger) -> set[str]:
    """The table columns the model's features are made of: plain columns, one-hot bases and combination parts."""
    columns = set(header)
    needed = set()
    unknown = []

    def resolve(name):
        if name in columns:
            needed.add(name)
            return True
        parsed = parse_combination_name(name)
        if parsed is not None:
            _, feat_a, feat_b = parsed
            return resolve(feat_a) & resolve(feat_b)
        if name.startswith('oh_'):
            bases = [col for col in columns if name.startswith(f'oh_{col}_')]
            if bases:
                needed.add(max(bases, key=len))
                return True
        return False

    for name in dict.fromkeys(model_feature_names):
        if not resolve(name):
            unknown.append(name)
    if unknown:
        logger.info(f'{len(unknown)} model features are not columns of the feature table and come from the prediction pipeline or stay missing: {unknown[:20]}')
    return needed


def read_header(path) -> list[str]:
    with open(path) as f:
        return f.readline().rstrip('\n').split('\t')


def read_wildtype_sequences(path) -> dict[str, str]:
    """The wildtype sequences of all proteins of the table, reconstructed from their AACs."""
    residues: dict[str, dict[int, str]] = {}
    with open(path) as f:
        f.readline()
        for line in f:
            words = line.split('\t', 2)
            if len(words) < 2:
                continue
            aac = words[1]
            residues.setdefault(words[0], {})[int(aac[1:-1])] = aac[0]
    sequences = {}
    for prot_id, pos_map in residues.items():
        if sorted(pos_map) == list(range(1, len(pos_map) + 1)):
            sequences[prot_id] = ''.join(pos_map[pos] for pos in range(1, len(pos_map) + 1))
    return sequences


def pick_protein(variants, wt_seqs, prot_id=None) -> str:
    """The protein whose wildtype sequence the variants are substitution variants of."""
    if prot_id is not None:
        if prot_id not in wt_seqs:
            raise ValueError(f'{prot_id} has no complete wildtype sequence in the feature table: {sorted(wt_seqs)}')
        candidates = [prot_id]
    else:
        candidates = [p for p, seq in wt_seqs.items() if len(seq) == len(variants[0])]
        if not candidates:
            raise ValueError(f'No protein in the feature table has the length of the variants ({len(variants[0])})')
    identity = {p: sum(a == b for a, b in zip(variants[0], wt_seqs[p])) for p in candidates}
    best = max(candidates, key=lambda p: identity[p])
    for i, variant in enumerate(variants):
        if len(variant) != len(wt_seqs[best]):
            raise ValueError(f'Variant {i + 1} has length {len(variant)}, the wildtype sequence of {best} {len(wt_seqs[best])}')
    return best


def load_wildtype_rows(path, header, prot_id, columns) -> dict[tuple[int, str], dict[str, str]]:
    """rows[(position index, mutant AA)] = {column: cell} of the protein's rows."""
    col_index = [(header.index(name), name) for name in columns]
    rows = {}
    prefix = f'{prot_id}\t'
    with open(path) as f:
        f.readline()
        for line in f:
            if not line.startswith(prefix):
                continue
            words = line.rstrip('\n').split('\t')
            aac = words[1]
            rows[(int(aac[1:-1]) - 1, aac[-1])] = {name: words[col] for col, name in col_index}
    return rows


def build_profiles(rows, wt_seq, suffixes) -> list[dict]:
    """Per position: per MSA suffix the 20-AA profiles (rates, PSIC, GEMME), the allel frequency and the positional median dPSIC."""
    profiles = []
    for pos, wt_aa in enumerate(wt_seq):
        pos_rows = {aa: rows[(pos, aa)] for aa in STANDARD_AAS if (pos, aa) in rows}
        any_row = next(iter(pos_rows.values()))
        pos_profile = {}
        for suffix in suffixes:
            def column(base_wt, base_mut):
                profile = {aa: to_float(row.get(f'{base_mut} {suffix}', NA_CELL)) for aa, row in pos_rows.items()}
                profile[wt_aa] = to_float(any_row.get(f'{base_wt} {suffix}', NA_CELL))
                return profile

            gemme = {}
            for gemme_type in GEMME_TYPES:
                name = f'gemme_{gemme_type} {suffix}'
                gemme[gemme_type] = {aa: to_float(row.get(name, NA_CELL)) for aa, row in pos_rows.items()}
                gemme[gemme_type][wt_aa] = 0.0
            pos_profile[suffix] = {
                'rate': column('Wildtype AA rate', 'Mutant AA rate'),
                'gl_rate': column('Wildtype AA rate gapless', 'Mutant AA rate gapless'),
                'psic': column('PSIC wildtype AA', 'PSIC mutant AA'),
                'gemme': gemme,
                'allel_freq': to_float(any_row.get(f'MSA allel freq {suffix}', NA_CELL)),
                'positional_median': to_float(any_row.get(f'Positional median dPSIC {suffix}', NA_CELL)),
            }
        profiles.append(pos_profile)
    return profiles


def positional_median_dpsic(psic, wt_aa) -> float:
    dpsics = [psic[wt_aa] - value for aa, value in psic.items() if aa != wt_aa and not math.isnan(value)]
    if math.isnan(psic[wt_aa]) or not dpsics:
        return math.nan
    return median(dpsics)


def finite_median(values) -> float:
    values = [value for value in values if not math.isnan(value)]
    return median(values) if values else math.nan


def sequence_medians(variant, wt_seq, profiles, suffix) -> tuple[list[float], list[float], float]:
    """Positional, window and protein median dPSIC of the variant, as msa.calcPsicProfiles computes them."""
    positional = []
    for pos, aa in enumerate(variant):
        profile = profiles[pos][suffix]
        if aa == wt_seq[pos]:
            positional.append(profile['positional_median'])
        else:
            positional.append(positional_median_dpsic(profile['psic'], aa))
    window = []
    for pos in range(len(variant)):
        window_a, window_b = window_bounds(pos, len(variant))
        window.append(finite_median(positional[window_a:window_b + 1]))
    return positional, window, finite_median(positional)


def substituted_values(pos_profile, suffixes, wt_aa, mut_aa) -> dict[str, object]:
    """The mutation specific values of wt_aa -> mut_aa at a position whose wildtype AA in the variant is wt_aa."""
    values = {'Wildtype AA': wt_aa, 'Mutant AA': mut_aa, 'AA change': f'{wt_aa}{mut_aa}'}
    values.update(substitution_values(wt_aa, mut_aa))
    for suffix in suffixes:
        profile = pos_profile[suffix]
        rate, gl_rate, psic = profile['rate'], profile['gl_rate'], profile['psic']
        values[f'Wildtype AA rate {suffix}'] = rate[wt_aa]
        values[f'Mutant AA rate {suffix}'] = rate[mut_aa]
        values[f'Wildtype AA rate gapless {suffix}'] = gl_rate[wt_aa]
        values[f'Mutant AA rate gapless {suffix}'] = gl_rate[mut_aa]
        # The MSA sequences without a gap have the wildtype, the mutant or another AA (seqfg.fill_prot_mut_map)
        values[f'Other mutant AA rate {suffix}'] = profile['allel_freq'] - rate[wt_aa] - rate[mut_aa]
        values[f'Other mutant AA rate gapless {suffix}'] = 1.0 - gl_rate[wt_aa] - gl_rate[mut_aa]
        values[f'PSIC wildtype AA {suffix}'] = psic[wt_aa]
        values[f'PSIC mutant AA {suffix}'] = psic[mut_aa]
        values[f'dPSIC {suffix}'] = psic[wt_aa] - psic[mut_aa]
        values[f'Positional median dPSIC {suffix}'] = positional_median_dpsic(psic, wt_aa)
        for gemme_type, gemme in profile['gemme'].items():
            values[f'gemme_{gemme_type} {suffix}'] = gemme[mut_aa] - gemme[wt_aa]
    return values


def variant_lines(variant_id, variant, wt_seq, rows, profiles, suffixes, out_columns) -> list[str]:
    """The table lines of all single mutations in the background of the variant."""
    out_set = set(out_columns)
    medians = {suffix: sequence_medians(variant, wt_seq, profiles, suffix) for suffix in suffixes}
    lines = []
    for pos, aa in enumerate(variant):
        wt_aa = wt_seq[pos]
        base_row = rows.get((pos, aa)) or next(row for (p, _), row in rows.items() if p == pos)
        median_values = {}
        for suffix, (_, window, protein_median) in medians.items():
            median_values[f'Window median dPSIC {suffix}'] = window[pos]
            median_values[f'Protein median dPSIC {suffix}'] = protein_median
        for mut_aa in STANDARD_AAS:
            if mut_aa == aa:
                continue
            if aa == wt_aa:
                cells = rows[(pos, mut_aa)].copy()
                overrides = median_values
            else:
                cells = base_row.copy()
                overrides = substituted_values(profiles[pos], suffixes, aa, mut_aa)
                overrides.update(median_values)
            for name, value in overrides.items():
                if name in out_set:
                    cells[name] = to_cell(value)
            cells['Uniprot Ac'] = variant_id
            cells['AAC'] = f'{aa}{pos + 1}{mut_aa}'
            lines.append('\t'.join(cells[name] for name in out_columns))
    return lines


def substitutions(wt_seq, variant) -> str:
    return ':'.join(f'{a}{pos + 1}{b}' for pos, (a, b) in enumerate(zip(wt_seq, variant)) if a != b)


def write_project_file(config, project_file, outfolder, dataset_name, features_file):
    replaced = {'outfolder': outfolder, 'dataset_name': dataset_name, 'path_to_features_file': features_file}
    lines = []
    for opt, arg in parse_conf(config.path_to_project_file):
        if opt in DROPPED_PROJECT_ENTRIES or opt in replaced:
            continue
        lines.append(f'{opt} = {arg}\n')
    lines += [f'{opt} = {arg}\n' for opt, arg in replaced.items()]
    with open(project_file, 'w') as f:
        f.write(''.join(lines))


def write_variant_table(config, path_to_model=None):
    """Writes the feature table of all single mutations of every variant in config.path_to_variants_file and a project file for predict."""
    logger = config.logger
    variants = parse_variants_file(config.path_to_variants_file)
    if not variants:
        raise ValueError(f'No sequences in {config.path_to_variants_file}')
    wt_table = config.path_to_features_file
    header = read_header(wt_table)
    if header[:len(META_COLS)] != META_COLS:
        raise ValueError(f'{wt_table} is not a structguy feature table, its columns start with {header[:len(META_COLS)]}')

    if path_to_model is not None:
        booster_list, feature_names, _, _, _, _ = loadModel(path_to_model)
        model_feature_names = list(feature_names) + [name for _, names in booster_list for name in names]
        needed = needed_columns(model_feature_names, header, logger)
    else:
        logger.info('variant_features without a model (-m): writing all columns, the table gets very large')
        needed = set(header)
    out_columns = META_COLS + [name for name in header[len(META_COLS):] if name in needed]

    wt_seqs = read_wildtype_sequences(wt_table)
    prot_id = pick_protein(variants, wt_seqs, getattr(config, 'variant_protein', None))
    wt_seq = wt_seqs[prot_id]
    suffixes = msa_suffixes(header)
    rows = load_wildtype_rows(wt_table, header, prot_id, set(out_columns) | computation_columns(header))
    profiles = build_profiles(rows, wt_seq, suffixes)
    logger.info(f'variant_features: {len(variants)} variants of {prot_id} ({len(wt_seq)} aa, {len(rows)} wildtype rows), {len(out_columns) - len(META_COLS)} feature columns, MSA tags {suffixes}')

    stem = os.path.basename(config.path_to_variants_file).rsplit('.', 1)[0]
    outfolder = f'{config.outfolder}/{stem}'
    os.makedirs(outfolder, exist_ok=True)
    features_file = f'{outfolder}/{stem}_variant_features.tsv'
    variants_file = f'{outfolder}/{stem}_variants.tsv'
    project_file = f'{outfolder}/{stem}.structguy_project.conf'

    n_digits = max(4, len(str(len(variants))))
    with open(features_file, 'w') as f_features, open(variants_file, 'w') as f_variants:
        f_features.write('\t'.join(out_columns) + '\n')
        f_variants.write('Variant ID\tSequence\tSubstitutions vs wildtype\n')
        for i, variant in enumerate(variants, start=1):
            variant_id = f'{prot_id}_v{i:0{n_digits}d}'
            f_variants.write(f'{variant_id}\t{variant}\t{substitutions(wt_seq, variant)}\n')
            f_features.write('\n'.join(variant_lines(variant_id, variant, wt_seq, rows, profiles, suffixes, out_columns)) + '\n')

    write_project_file(config, project_file, outfolder, stem, features_file)
    logger.info(f'variant_features: wrote {features_file}, {variants_file} and the project file for predict {project_file}')
    return project_file
