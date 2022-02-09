
random_features = set(['Random float',
                        'Random bool'])
#protein_bias_features = set(['Classification confidence','Surface/Core confidence','Protein size','Mutation position','Maximal oligo chain number','Maximal chain number','Raw Centrality'])
protein_bias_features = set(['Protein size','Mutation position','Maximal oligo chain number','Maximal chain number','Raw Centrality','Iupred'])#,'Classification confidence','Surface/Core confidence'])
#protein_bias_features = set([])
polyphen_features = set(['Phat','d-PSIC','PSIC aa1','PSIC aa2','msav','nobs','idpmax','idpsnp','idqmin'])
structural_features = set([#'Neighborhood  1 distance','Neighborhood  1 RSA','Neighborhood  1 hydropathy','Neighborhood  1 class',
                           # 'Neighborhood  2 distance','Neighborhood  2 RSA','Neighborhood  2 hydropathy','Neighborhood  2 class',
                           # 'Neighborhood  3 distance','Neighborhood  3 RSA','Neighborhood  3 hydropathy','Neighborhood  3 class',
                           # 'Neighborhood  4 distance','Neighborhood  4 RSA','Neighborhood  4 hydropathy','Neighborhood  4 class',
                           # 'Neighborhood  5 distance','Neighborhood  5 RSA','Neighborhood  5 hydropathy','Neighborhood  5 class',
                           # 'Neighborhood  6 distance','Neighborhood  6 RSA','Neighborhood  6 hydropathy','Neighborhood  6 class',
                           # 'Neighborhood  7 distance','Neighborhood  7 RSA','Neighborhood  7 hydropathy','Neighborhood  7 class',
                           # 'Neighborhood  8 distance','Neighborhood  8 RSA','Neighborhood  8 hydropathy','Neighborhood  8 class',
                           # 'Neighborhood  9 distance','Neighborhood  9 RSA','Neighborhood  9 hydropathy','Neighborhood  9 class',
                           # 'Neighborhood  10 distance','Neighborhood  10 RSA','Neighborhood  10 hydropathy','Neighborhood  10 class',
                            'Maximal oligo chain number','Maximal chain number',
                            'Ligand_Interaction_Degree','Ligand_Interaction_Score',
                            'Metal_Interaction_Degree',
                            'Metal_Interaction_Score',
                            'Chain_Interaction_Degree','Chain_Interaction_Score',
                            'Short_Interaction_Degree','Short_Interaction_Score',
                            'Medium_Interaction_Degree','Medium_Interaction_Score',
                            'Long_Interaction_Degree','Long_Interaction_Score','Metal_Interaction_Average',
                            'Ligand_Interaction_Average',
                            'Chain_Interaction_Average',
                            'Short_Interaction_Average',
                            'Medium_Interaction_Average',
                            'Long_Interaction_Average',

                            'Long_Medium_Interaction_Score',
                            'Long_Medium_Interaction_Degree',
                            'Long_Medium_Interaction_Average',
                            'Weighted chain distance','Weighted DNA distance','Weighted RNA distance','Weighted ligand distance','Weighted metal distance','Weighted homomer distance',
                            'Class','Classification confidence','weighted Surface/Core','Surface/Core confidence','RSA in recommended structure','Secondary structure assignment',
                            'Raw Centrality','Size normalized Centrality','0-1 normalized Centrality',
                            'Weighted rsa neighborhood','Weighted kd neighborhood','kd mean neighborhood',
                            'Interacting metal ion',
                            'Ligand degree',
                            'Ligand score',
                            'Ligand score portion',
                            'Metal degree',
                            'Metal score',
                            'Metal score portion',
                            'Prot degree',
                            'Prot score',
                            'Prot score portion',
                            'Rna degree',
                            'Rna score',
                            'Rna score portion',
                            'Dna degree',
                            'Dna score',
                            'Dna score portion'
                        ])


prior_features = set(['Wildtype Amino Acid','New Amino Acid','Mutation position','Protein size','Relative position','Chemical distance','Blosum62','KDmean','Volume change','KD new amino acid','total_change',
                        'aliphatic_change',
                        'hydrophobic_change',
                        'aromatic_change',
                        'positive_change',
                        'polar_change',
                        'negative_change',
                        'charged_change',
                        'small_change',
                        'tiny_change',])

sequence_features = set(['Iupred','IUpred_glob'])

position_specific_features = set([
                    'Ligand_Interaction_Degree','Ligand_Interaction_Score',
                    'Metal_Interaction_Degree',
                    'Metal_Interaction_Score',
                    'Chain_Interaction_Degree','Chain_Interaction_Score',
                    'Short_Interaction_Degree','Short_Interaction_Score',
                    'Medium_Interaction_Degree','Medium_Interaction_Score',
                    'Long_Interaction_Degree','Long_Interaction_Score','Metal_Interaction_Average',
                    'Ligand_Interaction_Average',
                    'Chain_Interaction_Average',
                    'Short_Interaction_Average',
                    'Medium_Interaction_Average',
                    'Long_Interaction_Average',

                    'Long_Medium_Interaction_Score',
                    'Long_Medium_Interaction_Degree',
                    'Long_Medium_Interaction_Average',
                    'Weighted chain distance','Weighted DNA distance','Weighted RNA distance','Weighted ligand distance','Weighted metal distance','Weighted homomer distance',
                    'Class',
                    'Classification confidence',
                    'weighted Surface/Core',
                    'Surface/Core confidence',
                'RSA in recommended structure','Secondary structure assignment',
                    'Size normalized Centrality','0-1 normalized Centrality',
                    'Weighted rsa neighborhood','Weighted kd neighborhood','kd mean neighborhood',
                    'Interacting metal ion',
                    'Ligand degree',
                    'Ligand score',
                    'Ligand score portion',
                    'Metal degree',
                    'Metal score',
                    'Metal score portion',
                    'Prot degree',
                    'Prot score',
                    'Prot score portion',
                    'Rna degree',
                    'Rna score',
                    'Rna score portion',
                    'Dna degree',
                    'Dna score',
                    'Dna score portion',
                    'Wildtype Amino Acid'
                    ,'Relative position',
])

mutation_specific_features = set(['Wildtype Amino Acid','New Amino Acid','Chemical distance','Blosum62','KDmean','Volume change','KD new amino acid','total_change',
                        'aliphatic_change',
                        'hydrophobic_change',
                        'aromatic_change',
                        'positive_change',
                        'polar_change',
                        'negative_change',
                        'charged_change',
                        'small_change',
                        'tiny_change'
])

top_features = set(['ref50 msa dPSIC',
                    'Class',
                    'New Amino Acid',
                    'Wildtype Amino Acid',
                    'ref50 gpw dPSIC',
                    'Weighted homomer distance',
                    'ref50 msa gapless WT rate',
                    'ref50 gpw coverage',
                    'Size normalized Centrality',
                    'ref50 msa coverage',
                    'ref50 gpw gapless WT rate',
                    'Iupred',
                    'ref90 gpw dPSIC',
                    'Short_Interaction_Score',
                    #'ref50 gpw Average MI',
                    'Raw Centrality',
                    'ref50 msa gapless dif rate',
                    'ref90 msa dPSIC',
                    'ref90 msa coverage',
                    'KD new amino acid',
                    'Relative position',
                    'Volume change',
                    'Chemical distance',
                    'Blosum62',
                    'Raw Centrality',
                    'KDmean',
                    'RSA in recommended structure'])

top_ref50_gpw = set(['Class',
                    'New Amino Acid',
                    'Wildtype Amino Acid',
                    'ref50 gpw dPSIC',
                    'Weighted homomer distance',
                    'ref50 gpw coverage',
                    'Size normalized Centrality',
                    'ref50 gpw gapless WT rate',
                    'Iupred',
                    'Short_Interaction_Score',
                    'Raw Centrality',
                    'ref50 gpw gapless dif rate',
                    'KD new amino acid',
                    'Relative position',
                    'Volume change',
                    'Chemical distance',
                    'Blosum62',
                    'Raw Centrality',
                    'KDmean',
                    'RSA in recommended structure'])

top_ref90_gpw = set(['Class',
                    'New Amino Acid',
                    'Wildtype Amino Acid',
                    'ref90 gpw dPSIC',
                    'Weighted homomer distance',
                    'ref90 gpw coverage',
                    'Size normalized Centrality',
                    'ref90 gpw gapless WT rate',
                    'Iupred',
                    'Short_Interaction_Score',
                    'Raw Centrality',
                    'ref90 gpw gapless dif rate',
                    'KD new amino acid',
                    'Relative position',
                    'Volume change',
                    'Chemical distance',
                    'Blosum62',
                    'Raw Centrality',
                    'KDmean',
                    'RSA in recommended structure'])

top_gpw = set(['Class',
                    'New Amino Acid',
                    'Wildtype Amino Acid',
                    'ref90 gpw dPSIC',
                    'Weighted homomer distance',
                    'ref90 gpw coverage',
                    'Size normalized Centrality',
                    'ref90 gpw gapless WT rate',
                    'Iupred',
                    'Short_Interaction_Score',
                    'Raw Centrality',
                    'ref90 gpw gapless dif rate',
                    'KD new amino acid',
                    'Relative position',
                    'Volume change',
                    'Chemical distance',
                    'Blosum62',
                    'Raw Centrality',
                    'KDmean',
                    'RSA in recommended structure',
                    'ref50 gpw dPSIC',
                    'ref50 gpw coverage',
                    'ref50 gpw gapless WT rate',
                    'ref50 gpw gapless dif rate'
])

top_ref50_msa = set(['Class',
                    'New Amino Acid',
                    'Wildtype Amino Acid',
                    'ref50 msa dPSIC',
                    'Weighted homomer distance',
                    'ref50 msa coverage',
                    'Size normalized Centrality',
                    'ref50 msa gapless WT rate',
                    'Iupred',
                    'Short_Interaction_Score',
                    'Raw Centrality',
                    'ref50 msa gapless dif rate',
                    'KD new amino acid',
                    'Relative position',
                    'Volume change',
                    'Chemical distance',
                    'Blosum62',
                    'Raw Centrality',
                    'KDmean',
                    'RSA in recommended structure'])

top_ref90_msa = set(['Class',
                    'New Amino Acid',
                    'Wildtype Amino Acid',
                    'ref90 msa dPSIC',
                    'Weighted homomer distance',
                    'ref90 msa coverage',
                    'Size normalized Centrality',
                    'ref90 msa gapless WT rate',
                    'Iupred',
                    'Short_Interaction_Score',
                    'Raw Centrality',
                    'ref90 msa gapless dif rate',
                    'KD new amino acid',
                    'Relative position',
                    'Volume change',
                    'Chemical distance',
                    'Blosum62',
                    'Raw Centrality',
                    'KDmean',
                    'RSA in recommended structure'])


aa_map = {'C':0, 
                    'N':1,
                    'S':2,
                    'V':3,
                    'Q':4,
                    'K':5,
                    'P':6,
                    'T':7,
                    'F':8,
                    'A':9,
                    'H':10,
                    'G':11,
                    'I':12,
                    'L':13,
                    'R':14,
                    'W':15,
                    'Y':16,
                    'M':17,
                    'E':18,
                    'D':19,
                    'X':20,
                    'B':20}
aa_order = {y:x for x,y in aa_map.items()}

class_map = {'Disorder': 0,
            'Surface':1,
            'Core':2,
            'Ligand Interaction':3,
            'Ligand far Interaction':4,
            'Protein Interaction':5,
            'Protein far Interaction':6,
            'Metal Interaction':7,
            'Metal far Interaction':8,
            'DNA Interaction':9,
            'DNA far Interaction':10,
            'RNA Interaction':11,
            'RNA far Interaction':12,
            'Double Interaction: Protein and Ligand':13,
            'Double Interaction: Protein far and Ligand':14,
            'Double Interaction: DNA and Ligand':15,
            'Double Interaction: DNA far and Ligand':16,
            'Double Interaction: RNA and Ligand':17,
            'Double Interaction: RNA far and Ligand':18,
            'Double Interaction: Protein and Ligand far':19,
            'Double Interaction: Protein far and Ligand far':20,
            'Double Interaction: DNA and Ligand far':21,
            'Double Interaction: DNA far and Ligand far':22,
            'Double Interaction: RNA and Ligand far':23,
            'Double Interaction: RNA far and Ligand far':24,
            'Double Interaction: Protein and Metal':25,
            'Double Interaction: Protein and Metal far':26,
            'Double Interaction: Protein and DNA':27,
            'Double Interaction: Protein and DNA far':28,
            'Double Interaction: Protein and RNA':29,
            'Double Interaction: Protein and RNA far':30,
            'Double Interaction: Protein far and Metal':31,
            'Double Interaction: Protein far and Metal far':32,
            'Double Interaction: Protein far and DNA':33,
            'Double Interaction: Protein far and DNA far':34,
            'Double Interaction: Protein far and RNA':35,
            'Double Interaction: Protein far and RNA far':36,
            'Double Interaction: DNA and Metal':37,
            'Double Interaction: DNA far and Metal':38,
            'Double Interaction: RNA and Metal':39,
            'Double Interaction: RNA far and Metal':40,
            'Double Interaction: DNA and Metal far':41,
            'Double Interaction: DNA far and Metal far':42,
            'Double Interaction: RNA and Metal far':43,
            'Double Interaction: RNA far and Metal far':44,
            'Double Interaction: Metal and Ligand':45,
            'Double Interaction: Metal far and Ligand':46,
            'Double Interaction: Metal far and Ligand far':47,
            'Double Interaction: Metal and Ligand far':48,
            'Triple Interaction: Protein and Ligand and Metal':49,
            'Triple Interaction: DNA and Ligand and Metal':50,
            'Triple Interaction: RNA and Ligand and Metal':51,
            'Double Interaction: Ligand and Ion':52,
            'Double Interaction: Metal and Ion':54,
            'Double Interaction: Protein and Ion':53,
            'Triple Interaction: Ligand and Metal and Ion':56,
            'Ion Interaction':57,
            'None':55
        }
class_map_counter = 53

class_order = {y:x for x,y in class_map.items()}

ion_values = {None:-1}
ion_order = {-1:None}
if hasattr(database, 'ions'):
    for pos,ion in enumerate(database.ions):
        ion_values[ion] = pos
        ion_order[pos] = ion

dbs = ['ref50','ref90']
for db in dbs:
    for al_type in ['msa','gpw']:
        sequence_features.add('%s %s WT rate' % (db,al_type))
        sequence_features.add('%s %s Mut rate' % (db,al_type))
        sequence_features.add('%s %s weighted Joint score' % (db,al_type))
        sequence_features.add('%s %s Joint score' % (db,al_type))
        sequence_features.add('%s %s gapless WT rate' % (db,al_type))
        sequence_features.add('%s %s gapless Mut rate' % (db,al_type))
        sequence_features.add('%s %s coverage' % (db,al_type))
        sequence_features.add('%s %s Dif rate' % (db,al_type))
        sequence_features.add('%s %s gapless dif rate' % (db,al_type))
        sequence_features.add('%s %s Max MI' % (db,al_type))
        sequence_features.add('%s %s Average MI' % (db,al_type))
        sequence_features.add('%s %s WT Joint Average' % (db,al_type))
        sequence_features.add('%s %s Mut Joint Average' % (db,al_type))
        sequence_features.add('%s %s Avergae Joint difference' % (db,al_type))
        sequence_features.add('%s %s Max Joint difference' % (db,al_type))
        
        sequence_features.add('%s %s gapless WT Joint Average' % (db,al_type))
        sequence_features.add('%s %s gapless Mut Joint Average' % (db,al_type))
        sequence_features.add('%s %s weighted WT Joint' % (db,al_type))
        sequence_features.add('%s %s weighted Mut Joint' % (db,al_type))

        sequence_features.add('%s %s PSIC WT' % (db,al_type))
        sequence_features.add('%s %s PSIC Mut' % (db,al_type))
        sequence_features.add('%s %s dPSIC' % (db,al_type))

        position_specific_features.add('%s %s WT rate' % (db,al_type))
        position_specific_features.add('%s %s gapless WT rate' % (db,al_type))
        position_specific_features.add('%s %s PSIC WT' % (db,al_type))
        
        mutation_specific_features.add('%s %s WT rate' % (db,al_type))
        mutation_specific_features.add('%s %s Mut rate' % (db,al_type))
        mutation_specific_features.add('%s %s gapless WT rate' % (db,al_type))
        mutation_specific_features.add('%s %s gapless Mut rate' % (db,al_type))
        mutation_specific_features.add('%s %s coverage' % (db,al_type))
        mutation_specific_features.add('%s %s Dif rate' % (db,al_type))
        mutation_specific_features.add('%s %s gapless dif rate' % (db,al_type))
        mutation_specific_features.add('%s %s PSIC WT' % (db,al_type))
        mutation_specific_features.add('%s %s PSIC Mut' % (db,al_type))
        mutation_specific_features.add('%s %s dPSIC' % (db,al_type))



    struct_and_prior = prior_features | structural_features# | random_features
    sequence_and_prior = prior_features | sequence_features# | random_features
    wo_bias = (prior_features | sequence_features | structural_features) - protein_bias_features
    
    str_pr_wo_bias = (prior_features | structural_features) - protein_bias_features
    seq_pr_wo_bias = (prior_features | sequence_features) - protein_bias_features
    

    fssn = {'Structural features':structural_features,
            'Structural and prior features':struct_and_prior,
            'Sequence features':sequence_features,
            'Sequence and prior':sequence_and_prior,
            'Prior features':prior_features,
            'Without protein level features':wo_bias,
            'Seq prior wo bias':seq_pr_wo_bias,
            'Struct prior wo bias':str_pr_wo_bias}
    #fssn = {'Without protein level features':wo_bias}
    fssn = {'Top features':top_features,'Top features wo bias':(top_features - protein_bias_features)}
    fssn = {'Top features wo bias':(top_features - protein_bias_features),'Without protein level features':wo_bias}
    fssn = {'Top features wo bias':(top_features - protein_bias_features),'Seq prior wo bias':seq_pr_wo_bias,'Sequence and prior':sequence_and_prior,}
    #fssn = {'Top features':top_features,'Structural and prior features':struct_and_prior,}
    #fssn = {'Sequence and prior':sequence_and_prior}
    fssn = {'Top features wo bias':(top_features - protein_bias_features)}
    fssn = {'Top gpw wo bias': (top_gpw - protein_bias_features),'Without protein level features':wo_bias}
    #fssn = {'Top ref50 gpw wo bias': (top_ref50_gpw - protein_bias_features),'Top ref90 gpw wo bias': (top_ref90_gpw - protein_bias_features),'Top ref50 msa wo bias': (top_ref50_msa - protein_bias_features),'Top ref90 msa wo bias': (top_ref90_msa - protein_bias_features)}
    fssn = {'Without protein level features':wo_bias}#,'Mutation specific samples':mutation_specific_features}
    #fssn = {}
    include_all = False


