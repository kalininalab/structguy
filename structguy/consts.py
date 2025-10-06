FEAT_NAME_SYNONYMS = {
    'WT Amino Acid' : 'Wildtype AA',
    'Mut Amino Acid': 'Mutant AA',
    'Structure Location' : 'Structure location',
    'Mainchain Location' : 'Mainchain location',
    'Sidechain Location' : 'Sidechain location'
}

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

feat_name_category_dict = {
    'chain_dist' : 5,
    'phi' : 7,
    'psi' : 7,
    'dna_dist' : 8,
    'ion_dist' : 8,
    'b_factor' : 9,
    'metal_dist' : 8,
    'oh_simple_class_Protein interaction' : 5,
    'Blosum62' : 4,
    'Chemical distance' : 4,
    'KD mean' : 4,
    'Protein Size' : 11,
    'Relative Sequence Position' : 4,
    'Sequence Position Number' : 11,
    'Volume mean' : 4,
    'Mutant AA' : 4,
    'Wildtype AA' : 4,
    'simple_class' : 12,
    'structural_classification' : 12,
    'ssa' : 7
}