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
    'structural classification',
    # The positional features of a position's ranked interacting positions ('IP<k> <feature>'
    # columns of StructMAn's expanded feature table, --local_features).
    'interacting positions',
]
INTERACTING_POSITIONS_CATEGORY = feature_categories.index('interacting positions')

structural_feature_sub_categories = [
    'Tertiary',
    'Interaction',
    'Structural conservation',
    'Aggregation'
]

evolutionary_sub_categories = [
    'statistical close',
    'statistical distant',
    'gemme close',
    'gemme distant'
]

aa_sub_categories = [
    'physico-chemical',
    'AA ID',
    'statistical'
]

feat_name_category_dict = {
    'chain_dist' : (5, 1),
    'phi' : (7, 0),
    'psi' : (7, 0),
    'tco' : (7, 0),
    'kappa' : (7, 0),
    'alpha' : (7, 0),
    'modres' : (7, 3),
    'dna_dist' : (8, 1),
    'ion_dist' : (8, 1),
    'b_factor' : (9, 0),
    'metal_dist' : (8, 1),
    'oh_simple_class_Protein interaction' : (5, 3),
    'Blosum62' : (4, 2),
    'Chemical distance' : (4, 0),
    'KD mean' : (4, 0),
    'Protein Size' : (4, 1),
    'Relative Sequence Position' : (4, 1),
    'Sequence Position Number' : (4, 1),
    'Volume mean' : (4, 0),
    'Mutant AA' : (4, 1),
    'Wildtype AA' : (4, 1),
    'simple_class' : (12, 3),
    'rin_class' : (12, 3),
    'rin_simple_class' : (12, 3),
    'structural_classification' : (12, 3),
    'ssa' : (7, 0),
    'intra_link' : (7, 0),
    'inter_link' : (7, 0),
    'link_length': (7, 0),
    'Degree': (3, 3),
    'WeightSum': (3, 3)
}