import dicts
import sys
import pymysql as MySQLdb
import ast
import structman.lib as sml

def initFeatures(samples):
    samples.addFeature('Distance-based classification','categorical',group='structural',default_value='Not mapped')
    samples.addFeature('Distance-based simple classification','categorical',group='structural',default_value='Not mapped')
    samples.addFeature('RIN-based classification','categorical',group='structural',default_value='Not mapped')
    samples.addFeature('RIN-based simple classification','categorical',group='structural',default_value='Not mapped')
    samples.addFeature('Classification confidence','real',group='structural',default_value=0.)
    samples.addFeature('Structure location','categorical',group='structural',default_value='Not mapped')
    samples.addFeature('Mainchain location','categorical',group='structural',default_value='Not mapped')
    samples.addFeature('Sidechain location','categorical',group='structural',default_value='Not mapped')
    samples.addFeature('RSA','real',group='structural',default_value=None)
    samples.addFeature('Mainchain RSA','real',group='structural',default_value=None)
    samples.addFeature('Sidechain RSA','real',group='structural',default_value=None)
    samples.addFeature('Amount of mapped structures','integer',group='structural',default_value=0)
    samples.addFeature('Secondary structure assignment','categorical',group='structural',default_value='Not mapped')
    samples.addFeature('IUPred value','real',group='amino acid property',default_value=0.)
    samples.addFeature('Region structure type','categorical',group='amino acid property',default_value=None)
    samples.addFeature('Modres score','real',group='structural',default_value=0.)
    samples.addFeature('Modres probability','real',group='structural',default_value='0.')
    samples.addFeature('Phi','real',group='structural',default_value='0.')
    samples.addFeature('Psi','real',group='structural',default_value='0.')

    for chaintype in ['mc','sc']:
        for interaction_type in ['neighbor','short','long','ligand','ion','metal','Protein','DNA','RNA','Peptide']:
            feature_name = '%s %s score' % (chaintype,interaction_type)
            samples.addFeature(feature_name,'real',group='structural',default_value=0.)
            feature_name = '%s %s degree' % (chaintype,interaction_type)
            samples.addFeature(feature_name,'integer',group='structural',default_value=0)
            feature_name = '%s %s H-bond score' % (chaintype,interaction_type)
            samples.addFeature(feature_name,'real',group='structural',default_value=0.)

    samples.addFeature('KD mean','real',group='amino acid property',default_value='0.',mutation_specific=True)
    samples.addFeature('Volume mean','real',group='amino acid property',default_value='0.',mutation_specific=True)
    samples.addFeature('Chemical distance','real',group='amino acid property',default_value='0.',mutation_specific=True)
    samples.addFeature('Blosum62','real',group='amino acid property',default_value='0.',mutation_specific=True)

    samples.addFeature('Aliphatic change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Hydrophobic change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Aromatic change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Positive charged change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Polar change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Negative charge change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Charged change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Small change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Tiny change','binary',group='amino acid property',mutation_specific=True)
    samples.addFeature('Total change','binary',group='amino acid property',mutation_specific=True)

    samples.addFeature('Wildtype AA','categorical',group = 'amino acid property')
    samples.addFeature('Mutant AA','categorical',group = 'amino acid property',mutation_specific=True)
    samples.addFeature('AA change','categorical',group = 'amino acid property',mutation_specific=True)

    samples.addFeature('B Factor','real',group='structural',default_value=0.)

    samples.addFeature('AbsoluteCentrality','real',group='structural',default_value=0.)
    samples.addFeature('LengthNormalizedCentrality','real',group='structural',default_value=0.)
    samples.addFeature('MinMaxNormalizedCentrality','real',group='structural',default_value=0.)
    samples.addFeature('AbsoluteCentralityWithNegative','real',group='structural',default_value=0.)
    samples.addFeature('LengthNormalizedCentralityWithNegative','real',group='structural',default_value=0.)
    samples.addFeature('MinMaxNormalizedCentralityWithNegative','real',group='structural',default_value=0.)
    samples.addFeature('AbsoluteComplexCentrality','real',group='structural',default_value=0.)
    samples.addFeature('LengthNormalizedComplexCentrality','real',group='structural',default_value=0.)
    samples.addFeature('MinMaxNormalizedComplexCentrality','real',group='structural',default_value=0.)
    samples.addFeature('AbsoluteComplexCentralityWithNegative','real',group='structural',default_value=0.)
    samples.addFeature('LengthNormalizedComplexCentralityWithNegative','real',group='structural',default_value=0.)
    samples.addFeature('MinMaxNormalizedComplexCentralityWithNegative','real',group='structural',default_value=0.)

    samples.addFeature('Intra_SSBOND_Propensity','real',group='structural',default_value=0.)
    samples.addFeature('Inter_SSBOND_Propensity','real',group='structural',default_value=0.)
    samples.addFeature('Intra_Link_Propensity','real',group='structural',default_value=0.)
    samples.addFeature('Inter_Link_Propensity','real',group='structural',default_value=0.)
    samples.addFeature('CIS_Conformation_Propensity','real',group='structural',default_value=0.)
    samples.addFeature('CIS_Follower_Propensity','real',group='structural',default_value=0.)

    samples.addFeature('Inter Chain Median KD','real',group = 'structural',default_value=0.)
    samples.addFeature('Inter Chain Distance Weighted KD','real',group = 'structural',default_value=0.)
    samples.addFeature('Inter Chain Median RSA','real',group = 'structural',default_value=0.)
    samples.addFeature('Inter Chain Distance Weighted RSA','real',group = 'structural',default_value=0.)

    samples.addFeature('Intra Chain Median KD','real',group = 'structural',default_value=0.)
    samples.addFeature('Intra Chain Distance Weighted KD','real',group = 'structural',default_value=0.)
    samples.addFeature('Intra Chain Median RSA','real',group = 'structural',default_value=0.)
    samples.addFeature('Intra Chain Distance Weighted RSA','real',group = 'structural',default_value=0.)

    samples.addFeature('Inter Chain Interactions Median','real', group = 'structural',default_value=0.)
    samples.addFeature('Inter Chain Interactions Distance Weighted','real',group = 'structural',default_value=0.)
    samples.addFeature('Intra Chain Interactions Median','real', group = 'structural',default_value=0.)
    samples.addFeature('Intra Chain Interactions Distance Weighted','real',group = 'structural',default_value=0.)

def generateStructuralFeatures(config,session,samples,debug=0,effectRegressor=None):

    structman_config = config.structman_config

    initFeatures(samples)

    if effectRegressor != None:
        effectRegressor,substitution_list = joblib.load(effectRegressor)

    columns = ['SNV','Tag']
    table = 'RS_SNV_Session'
    equals_cols = {'Session':session}

    results = sml.database.select(structman_config,columns,table,equals_rows=equals_cols)

    tag_map = {}
    mut_ids = set()
    for row in results:
        tag_map[row[0]] = row[1]
        mut_ids.add(row[0])

    columns = ['SNV_Id','Position','New_AA']
    table = 'SNV'
    results = sml.database.binningSelect(mut_ids,columns,table,structman_config)

    pos_snv_map = {}
    for row in results:
        pos_db_id = row[1]
        if not pos_db_id in pos_snv_map:
            pos_snv_map[pos_db_id] = {}
        pos_snv_map[pos_db_id][row[0]] = row[2]

    columns = ['Position_Id', 'Amino_Acid_Change', 'Protein', 'Location', 'Mainchain_Location', 'Sidechain_Location',
               'Weighted_Surface_Access', 'Weighted_Surface_Access_Main_Chain', 'Weighted_Surface_Access_Side_Chain',
               'Class', 'Simple_Class', 'RIN_Class', 'RIN_Simple_Class', 'Interactions', 'Confidence',
               'Secondary_Structure', 'Mapped_Structures', 'RIN_Profile', 'IUPRED', 'IUPRED_Glob',
               'Modres_Score', 'Modres_Propensity',
               'B_Factor', 'Weighted_Centrality_Scores', 'Weighted_Phi', 'Weighted_Psi', 'Intra_SSBOND_Propensity',
               'Inter_SSBOND_Propensity', 'Intra_Link_Propensity', 'Inter_Link_Propensity', 'CIS_Conformation_Propensity', 'CIS_Follower_Propensity',
               'Weighted_Inter_Chain_Median_KD', 'Weighted_Inter_Chain_Dist_Weighted_KD', 'Weighted_Inter_Chain_Median_RSA',
               'Weighted_Inter_Chain_Dist_Weighted_RSA', 'Weighted_Intra_Chain_Median_KD', 'Weighted_Intra_Chain_Dist_Weighted_KD',
               'Weighted_Intra_Chain_Median_RSA', 'Weighted_Intra_Chain_Dist_Weighted_RSA',
               'Weighted_Inter_Chain_Interactions_Median', 'Weighted_Inter_Chain_Interactions_Dist_Weighted',
               'Weighted_Intra_Chain_Interactions_Median', 'Weighted_Intra_Chain_Interactions_Dist_Weighted'
              ]
    table = 'Position'
    results = sml.database.binningSelect(pos_snv_map.keys(),columns,table,structman_config)
    
    prot_db_ids = set()
    structman_map = {}
    for row in results:
        pos_db_id = row[0]
        prot_db_id = row[2]
        prot_db_ids.add(prot_db_id)
        structman_map[pos_db_id] = row[1:]

    columns = ['Protein_Id','Primary_Protein_Id']
    table = 'Protein'
    results = sml.database.binningSelect(prot_db_ids,columns,table,structman_config)

    protein_id_map = {}
    for row in results:
        prot_db_id = row[0]
        protein_primary_id = row[1]
        protein_id_map[prot_db_id] = protein_primary_id

    print('Total mutations (before tag filtering): ',len(tag_map))

    init_set = set()

    for pos_db_id in pos_snv_map:
        (aa1_pos,prot_db_id,location, main_chain_location, side_chain_location, rsa, main_chain_rsa, side_chain_rsa,
         d_class,d_s_class,r_class,r_s_class,interactions,conf,ssa,amount_of_struct,rin_profile_str,
         iupred,iupred_glob,modres_score,modres_prob,b_factor,centrality_scores_str,phi,psi,intra_ssbond_prop,
         inter_ssbond_prop,intra_link_prop,inter_link_prop,cis_prop,cis_follower_prop,weighted_inter_chain_median_kd,
         weighted_inter_chain_dist_weighted_kd, weighted_inter_chain_median_rsa, weighted_inter_chain_dist_weighted_rsa,
         weighted_intra_chain_median_kd, weighted_intra_chain_dist_weighted_kd, weighted_intra_chain_median_rsa,
         weighted_intra_chain_dist_weighted_rsa, weighted_inter_chain_interactions_median,
         weighted_inter_chain_interactions_dist_weighted, weighted_intra_chain_interactions_median,
         weighted_intra_chain_interactions_dist_weighted
        ) = structman_map[pos_db_id]

        primary_protein_id = protein_id_map[prot_db_id]
        aa1 = aa1_pos[0]

        for snv_db_id in pos_snv_map[pos_db_id]:
            new_aa = pos_snv_map[pos_db_id][snv_db_id]
            aac = "%s%s" % (aa1_pos,new_aa)
            sample_id = (primary_protein_id,aac)

            samples.addValue(sample_id,aa1,'Wildtype AA')
            samples.addValue(sample_id,new_aa,'Mutant AA')
            samples.addValue(sample_id,'%s%s' % (aa1,new_aa),'AA change')

            samples.addValue(sample_id,location,'Structure location')
            samples.addValue(sample_id,main_chain_location,'Mainchain location')
            samples.addValue(sample_id,side_chain_location,'Sidechain location')
            samples.addValue(sample_id,rsa,'RSA')
            samples.addValue(sample_id,main_chain_rsa,'Mainchain RSA')
            samples.addValue(sample_id,side_chain_rsa,'Sidechain RSA')

            samples.addValue(sample_id,d_class,'Distance-based classification')
            samples.addValue(sample_id,d_s_class,'Distance-based simple classification')
            samples.addValue(sample_id,r_class,'RIN-based classification')
            samples.addValue(sample_id,r_s_class,'RIN-based simple classification')

            samples.addValue(sample_id,conf,'Classification confidence')
            samples.addValue(sample_id,ssa,'Secondary structure assignment')

            samples.samples[sample_id].amount_of_structures = amount_of_struct

            samples.addValue(sample_id,amount_of_struct,'Amount of mapped structures')

            rin_profile = sml.rin.Interaction_profile(profile_str=rin_profile_str)
            for chaintype in ['mc','sc']:
                for interaction_type in ['neighbor','short','long','ligand','ion','metal','Protein','DNA','RNA','Peptide']:
                    feature_name = '%s %s score' % (chaintype,interaction_type)
                    value = rin_profile.getChainSpecificCombiScore(chaintype,interaction_type)
                    samples.addValue(sample_id,value,feature_name)

                    feature_name = '%s %s degree' % (chaintype,interaction_type)
                    value = rin_profile.getChainSpecificCombiDegree(chaintype,interaction_type)
                    samples.addValue(sample_id,value,feature_name)

                    feature_name = '%s %s H-bond score' % (chaintype,interaction_type)
                    value = rin_profile.getScore(chaintype,'hbond',interaction_type)
                    samples.addValue(sample_id,value,feature_name)

            samples.addValue(sample_id,iupred,'IUPred value')
            samples.addValue(sample_id,iupred_glob,'Region structure type')
            samples.addValue(sample_id,modres_score,'Modres score')
            samples.addValue(sample_id,modres_prob,'Modres probability')

            samples.addValue(sample_id,phi,'Phi')
            samples.addValue(sample_id,psi,'Psi')

            samples.addValue(sample_id,b_factor,'B Factor')

            centrality_scores = sml.rin.Centrality_scores(code_str = centrality_scores_str)

            samples.addValue(sample_id,centrality_scores.AbsoluteCentrality,'AbsoluteCentrality')
            samples.addValue(sample_id,centrality_scores.LengthNormalizedCentrality,'LengthNormalizedCentrality')
            samples.addValue(sample_id,centrality_scores.MinMaxNormalizedCentrality,'MinMaxNormalizedCentrality')
            samples.addValue(sample_id,centrality_scores.AbsoluteCentralityWithNegative,'AbsoluteCentralityWithNegative')
            samples.addValue(sample_id,centrality_scores.LengthNormalizedCentralityWithNegative,'LengthNormalizedCentralityWithNegative')
            samples.addValue(sample_id,centrality_scores.MinMaxNormalizedCentralityWithNegative,'MinMaxNormalizedCentralityWithNegative')
            samples.addValue(sample_id,centrality_scores.AbsoluteComplexCentrality,'AbsoluteComplexCentrality')
            samples.addValue(sample_id,centrality_scores.LengthNormalizedComplexCentrality,'LengthNormalizedComplexCentrality')
            samples.addValue(sample_id,centrality_scores.MinMaxNormalizedComplexCentrality,'MinMaxNormalizedComplexCentrality')
            samples.addValue(sample_id,centrality_scores.AbsoluteComplexCentralityWithNegative,'AbsoluteComplexCentralityWithNegative')
            samples.addValue(sample_id,centrality_scores.LengthNormalizedComplexCentralityWithNegative,'LengthNormalizedComplexCentralityWithNegative')
            samples.addValue(sample_id,centrality_scores.MinMaxNormalizedComplexCentralityWithNegative,'MinMaxNormalizedComplexCentralityWithNegative')

            samples.addValue(sample_id,intra_ssbond_prop,'Intra_SSBOND_Propensity')
            samples.addValue(sample_id,inter_ssbond_prop,'Inter_SSBOND_Propensity')
            samples.addValue(sample_id,intra_link_prop,'Intra_Link_Propensity')
            samples.addValue(sample_id,inter_link_prop,'Inter_Link_Propensity')
            samples.addValue(sample_id,cis_prop,'CIS_Conformation_Propensity')
            samples.addValue(sample_id,cis_follower_prop,'CIS_Follower_Propensity')

            samples.addValue(sample_id,weighted_inter_chain_median_kd,'Inter Chain Median KD')
            samples.addValue(sample_id,weighted_inter_chain_dist_weighted_kd,'Inter Chain Distance Weighted KD')
            samples.addValue(sample_id,weighted_inter_chain_median_rsa,'Inter Chain Median RSA')
            samples.addValue(sample_id,weighted_inter_chain_dist_weighted_rsa,'Inter Chain Distance Weighted RSA')

            samples.addValue(sample_id,weighted_intra_chain_median_kd,'Intra Chain Median KD')
            samples.addValue(sample_id,weighted_intra_chain_dist_weighted_kd,'Intra Chain Distance Weighted KD')
            samples.addValue(sample_id,weighted_intra_chain_median_rsa,'Intra Chain Median RSA')
            samples.addValue(sample_id,weighted_intra_chain_dist_weighted_rsa,'Intra Chain Distance Weighted RSA')

            samples.addValue(sample_id,weighted_intra_chain_interactions_median,'Intra Chain Interactions Median')
            samples.addValue(sample_id,weighted_intra_chain_interactions_dist_weighted,'Intra Chain Interactions Distance Weighted')
            samples.addValue(sample_id,weighted_inter_chain_interactions_median,'Inter Chain Interactions Median')
            samples.addValue(sample_id,weighted_inter_chain_interactions_dist_weighted,'Inter Chain Interactions Distance Weighted')

            KDmean = abs(dicts.hydropathy[aa1]-dicts.hydropathy[new_aa])
            samples.addValue(sample_id,KDmean,'KD mean')

            d_vol = abs(dicts.volume[aa1]-dicts.volume[new_aa])
            samples.addValue(sample_id,d_vol,'Volume mean')

            if aa1 != 'X' and new_aa != 'X':
                chemical_distance = sml.sdsc.consts.residues.CHEM_DIST_MATRIX[aa1][new_aa]
            else:
                chemical_distance = 0.
            samples.addValue(sample_id,chemical_distance,'Chemical distance')

            try:
                blosum_value = sml.sdsc.consts.residues.BLOSUM62[(aa1,new_aa)]
            except:
                blosum_value = sml.sdsc.consts.residues.BLOSUM62[(new_aa,aa1)]
            samples.addValue(sample_id,blosum_value,'Blosum62')

            aliphatic_change = int((aa1 in dicts.aa_map_aliphatic) != (new_aa in dicts.aa_map_aliphatic))
            samples.addValue(sample_id,aliphatic_change,'Aliphatic change')

            hydrophobic_change = int((aa1 in dicts.aa_map_hydrophobic) != (new_aa in dicts.aa_map_hydrophobic))
            samples.addValue(sample_id,hydrophobic_change,'Hydrophobic change')

            aromatic_change = int((aa1 in dicts.aa_map_aromatic) != (new_aa in dicts.aa_map_aromatic))
            samples.addValue(sample_id,aromatic_change,'Aromatic change')

            positive_change = int((aa1 in dicts.aa_map_positive) != (new_aa in dicts.aa_map_positive))
            samples.addValue(sample_id,positive_change,'Positive charged change')

            polar_change = int((aa1 in dicts.aa_map_polar) != (new_aa in dicts.aa_map_polar))
            samples.addValue(sample_id,polar_change,'Polar change')

            negative_change = int((aa1 in dicts.aa_map_negative) != (new_aa in dicts.aa_map_negative))
            samples.addValue(sample_id,negative_change,'Negative charge change')

            charged_change = int((aa1 in dicts.aa_map_charged) != (new_aa in dicts.aa_map_charged))
            samples.addValue(sample_id,charged_change,'Charged change')

            small_change = int((aa1 in dicts.aa_map_small) != (new_aa in dicts.aa_map_small))
            samples.addValue(sample_id,small_change,'Small change')

            tiny_change = int((aa1 in dicts.aa_map_tiny) != (new_aa in dicts.aa_map_tiny))
            samples.addValue(sample_id,tiny_change,'Tiny change')

            total_change = aliphatic_change + hydrophobic_change + aromatic_change + positive_change + polar_change + negative_change + charged_change + small_change + tiny_change
            samples.addValue(sample_id,total_change,'Total change')

            if interactions != None:
                interaction_dict = ast.literal_eval(interactions)
                for interaction in interaction_dict:
                    if not interaction in init_set:
                        samples.addFeature(interaction,'binary',group='structural',default_value=0)
                        init_set.add(interaction)
                    samples.addValue(sample_id,1,interaction)

            tags = tag_map[snv_db_id]
            #print(u_ac,aac,tags)
            for tag in tags.split(','):
                if config.regression:
                    parts = tag.split(':')
                    if len(parts) < 2:
                        parts = tag.split('=')
                        if len(parts) < 2:
                            continue
                    if parts[0] != config.target_values:
                        continue
                    if parts[1] == '':
                        continue
                    target_value = float(parts[1])
                    #print(sample_id,target_value)
                    samples.addTargetValue(sample_id,target_value)
                else:
                    if not tag in config.target_values:
                        if tag in config.target_translator:
                            tag = config.target_translator[tag]
                        else:
                            continue
                    target_value = tag
                    samples.addTargetValue(sample_id,target_value)

                samples.samples[sample_id].tags = tags


    if effectRegressor != None:
        pass
        """ TODO
        for (f_keys,substitution_map) in substitution_list:
            feature_matrix,sub_list,empty_train_set = substitute(feature_matrix,[],substitution_map,f_keys=f_keys)[0]

        #print feature_matrix[:2]

        r_less_f_mat = []
        for f  in feature_matrix:
            r_less_f_mat.append(f[2:])

        pred_effects = effectRegressor.predict(r_less_f_mat)
        for pos,pred_effect in enumerate(pred_effects):
            feature_matrix[pos].append(pred_effect)
        """


