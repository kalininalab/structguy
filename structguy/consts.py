FEAT_NAME_SYNONYMS = {
    'WT Amino Acid' : 'Wildtype AA',
    'Mut Amino Acid': 'Mutant AA',
    'Structure Location' : 'Structure location',
    'Mainchain Location' : 'Mainchain location',
    'Sidechain Location' : 'Sidechain location'
}

n_of_unifref_splits = 240

refseq_datasets = ['ref50','ref90']

feature_categories = [
    'evolutionary',
    'intra contacts',
    'solvent access',
    'RIN centrality',
    'amino acid properties',
    'PPI contacts',
    'structural conservation',
    'geometric',
    'non-PPI contacts',
    'flexibility',
    'function type',
    'protein attributes',
    'structural classification'
]

structural_feature_sub_categories = [
    'Tertiary',
    'Interaction',
    'Structural conservation',
    'Aggregation'
]

feat_name_category_dict = {
    'chain_dist' : (5, 1),
    'phi' : (7, 0),
    'psi' : (7, 0),
    'dna_dist' : (8, 1),
    'ion_dist' : (8, 1),
    'b_factor' : (9, 0),
    'metal_dist' : (8, 1),
    'oh_simple_class_Protein interaction' : (5, 3),
    'Blosum62' : (4, None),
    'Chemical distance' : (4, None),
    'KD mean' : (4, None),
    'Protein Size' : (11, None),
    'Relative Sequence Position' : (4, None),
    'Sequence Position Number' : (11, None),
    'Volume mean' : (4, None),
    'Mutant AA' : (4, None),
    'Wildtype AA' : (4, None),
    'simple_class' : (12, 3),
    'structural_classification' : (12, 3),
    'ssa' : (7, 0),
    'link_length': (7, 0)
}