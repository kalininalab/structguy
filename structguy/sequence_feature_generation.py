import time
import subprocess
import multiprocessing
import string
import random
import os
import sys
import traceback
import ray
import gzip

from structguy import msa, consts, util
from structguy.sequence_util import parseFromFasta
from structman.base_utils.base_utils import pack, unpack

def initFeatures(config, samples):
    dbs = []
    for db_id in config.msa_dbs:
        dbs.append((db_id, 'MSA'))

    for db_id in config.gpw_dbs:
        dbs.append((db_id, "GPW"))

    for db_name, feature_name_tag in dbs:
        samples.addFeature(f'Wildtype AA rate {feature_name_tag} {db_name}','real',group='sequence',default_value=0.)
        samples.addFeature(f'Mutant AA rate {feature_name_tag} {db_name}','real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature(f'Wildtype AA rate gapless {feature_name_tag} {db_name}','real',group='sequence',default_value=0.)
        samples.addFeature(f'Mutant AA rate gapless {feature_name_tag} {db_name}','real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature(f'MSA allel freq {feature_name_tag} {db_name}','real',group='sequence',default_value=0.)

        samples.addFeature(f'Other mutant AA rate {feature_name_tag} {db_name}','real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature(f'Other mutant AA rate gapless {feature_name_tag} {db_name}','real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature(f'PSIC wildtype AA {feature_name_tag} {db_name}','real',group='sequence',default_value=0.)
        samples.addFeature(f'PSIC mutant AA {feature_name_tag} {db_name}','real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature(f'dPSIC {feature_name_tag} {db_name}','real',group='sequence',default_value=0.,mutation_specific=True)

        samples.addFeature(f'Positional median dPSIC {feature_name_tag} {db_name}','real',group='sequence',default_value=0.)
        samples.addFeature(f'Window median dPSIC {feature_name_tag} {db_name}','real',group='sequence',default_value=0.)
        samples.addFeature(f'Protein median dPSIC {feature_name_tag} {db_name}','real',group='sequence',default_value=0.)

    samples.addFeature('Sequence Position Number','integer',group = 'amino acid property')
    samples.addFeature('Relative Sequence Position','real',group = 'amino acid property')
    samples.addFeature('Protein Size','integer',group = 'amino acid property')

def geneSeqMapToFasta(prot_seq_map, outfile, verbosity = 0):
    lines = []
    n = 0
    m = 0

    if verbosity >= 2:
        print('Size of prot_seq_map converted into a fasta file:',len(prot_seq_map))

    for u_ac in prot_seq_map:
        seq = prot_seq_map[u_ac][0]

        if seq == 0 or seq == 1 or seq == '':
            continue

        lines.append(f'>{u_ac}\n')
        lines.append(f'{seq}\n')
        m += 1

    if len(lines) > 0:
        print('Filtered ',n,' Proteins before mmseqs2')
        print(m,' sequences going into mmseqs2')
        f = open(outfile,'w')
        page = ''.join(lines)
        f.write(page)
        f.close()
        return None
    else: 
        return 'Empty fasta file'


def estimate_cost(config, prot_id, msa_ref_dbs, gpw_ref_dbs, seq_len, n_of_mapped_seqs):
    total_cost = 0
    out_directory = msa.get_out_directory(prot_id, config)
    for msa_ref_db in msa_ref_dbs:
        filename = util.get_msa_path(out_directory, prot_id, msa_ref_db)
        if not os.path.isfile(filename):
            total_cost += ((seq_len**2) * n_of_mapped_seqs[msa_ref_db]) + (seq_len**2 * (n_of_mapped_seqs[msa_ref_db]**2))

        psic_name = util.get_msa_path(out_directory, prot_id, msa_ref_db, psic = True)
        if not os.path.isfile(psic_name):
            total_cost += seq_len * n_of_mapped_seqs[msa_ref_db]

    for gpw_ref_db in gpw_ref_dbs:
        filename = util.get_msa_path(out_directory, prot_id, gpw_ref_db, gpw = True)
        if os.path.isfile(filename):
            total_cost += (seq_len**2) * n_of_mapped_seqs[gpw_ref_db]

        psic_name = util.get_msa_path(out_directory, prot_id, gpw_ref_db,  gpw = True, psic = True)
        if not os.path.isfile(psic_name):
            total_cost += seq_len * n_of_mapped_seqs[gpw_ref_db]

    return total_cost
        

def getSequenceFeatures(config, samples, n_of_processes = 6, update_mode=False):

    if config.verbosity >= 2:
        t0 = time.time()
        print(f'Call of getSequenceFeatures with MSA DB: {config.msa_db}')

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    inqueue = manager.Queue()
    outqueue = manager.Queue()

    msa_dbs = config.msa_dbs
    gpw_dbs = config.gpw_dbs
    msa_map = {}
    gpw_map = {}

    initFeatures(config, samples)

    n = 0
    u_acs = set([])
    pdb_ids = set()

    for (u_ac, aac) in samples.samples:
        if u_ac.count(':') > 0:
            pdb_ids.add(u_ac)
        else:
            u_acs.add(u_ac)
            #Additionally add the canonical sequences
            if u_ac.count('-') > 0:
                u_acs.add(u_ac.split('-')[0])

    dbs = consts.refseq_datasets
    sequence_maps = {}
    for db in dbs:
        sequence_maps[db] = {}

    gene_seq_map = {}

    n_mapped_sequences = 0

    
    gene_seq_map, in_db = parseFromFasta(config.path_to_sequence_fasta, config = config, dbs = dbs)
    if config.verbosity >= 2:
        print(f'Parsed protein sequences from: {config.path_to_sequence_fasta}\nLength of the map:{len(gene_seq_map)}, Length of in_db: {len(in_db)}')
    
    if config.verbosity >= 2:
        print(f'Query proteins that are in_db:\n{in_db}')

    N = 0
    mmseq_searchs = []
    mmseq_search = {}
    msa_to_process = []
    for primary_protein_id in gene_seq_map:
        msa_to_process.append(primary_protein_id)
        if not primary_protein_id in in_db:
            mmseq_search[primary_protein_id] = gene_seq_map[primary_protein_id]

            N += 1
            if N == 5000:
                mmseq_searchs.append(mmseq_search)
                mmseq_search = {}
                N = 0

    if N > 0:
        mmseq_searchs.append(mmseq_search)

    if config.verbosity >= 2:
        t1 = time.time()
        print('getSequenceFeature part 1: ',t1-t0)

    mmseqs2_search_dbs = {'ref50':config.mmseqs_search_db_ref50,'ref90':config.mmseqs_search_db_ref90,'ref100':config.mmseqs_search_db_ref100}

    mmseqs_tmp_folder = config.mmseqs_tmp_folder

    mmseqs2_path = config.mmseqs_path

    if config.verbosity >= 2:
        t2 = time.time()
        print('getSequenceFeature part 2: ',t2-t1)
    M = 0
    for mmseq_search in mmseq_searchs:

        temp_fasta = '%s/tmp_%s.fasta' % (mmseqs_tmp_folder,randomString())
        returncode = geneSeqMapToFasta(mmseq_search, temp_fasta, verbosity = config.verbosity)

        if returncode == None:

            for db in dbs:

                mmseqs2_search_db = mmseqs2_search_dbs[db]

                if config.verbosity >= 1:
                    M += 1
                    print('Starting sequence search with MMseqs2 on database: ',mmseqs2_search_db,M)
                    t20 = time.time()

                temp_outfile = '%s/tmp_outfile_%s.fasta' % (mmseqs_tmp_folder,randomString())
                if not config.verbosity >= 3:
                    FNULL = open(os.devnull, 'w')
                    p = subprocess.Popen([mmseqs2_path,'easy-search',temp_fasta,mmseqs2_search_db,temp_outfile,mmseqs_tmp_folder,'--max-seqs','999999','--format-output','query,target,tseq','--max-seq-len','999999'],stdout=FNULL)
                else:
                    p = subprocess.Popen([mmseqs2_path,'easy-search',temp_fasta,mmseqs2_search_db,temp_outfile,mmseqs_tmp_folder,'--max-seqs','999999','--format-output','query,target,tseq','--max-seq-len','999999'])
                #p = subprocess.Popen([mmseqs2_path,'easy-linsearch',temp_fasta,mmseqs2_search_db,temp_outfile,mmseqs_tmp_folder,'--format-output','query,target,tseq'],stdout=FNULL)
                p.wait()

                f = open(temp_outfile,'r')
                lines = f.read().split('\n')
                f.close()

                if len(lines) == 0:
                    return

                for line in lines:
                    if line == '':
                        continue
                    words = line.split()
                    #print line
                    gene = words[0]
                    hit = words[1]
                    tseq = words[2]

                    if not gene in sequence_maps[db]:
                        sequence_maps[db][gene] = {}
                    sequence_maps[db][gene][hit] = tseq

                    n_mapped_sequences += 1

                os.remove(temp_outfile)

                if config.verbosity >= 2:
                    t21 = time.time()
                    print('Time for sequence search: ',t21-t20)
            os.remove(temp_fasta)

        else:
            print('Skipped mmseqs2 because of: ',returncode)
            for db in dbs:
                sequence_maps[db] = {}

    if config.verbosity >= 2:
        t3 = time.time()
        print('getSequenceFeature part 3: ',t3-t2)

    cost_map = {}
    cost_tuples = []
    total_cost = 0
    for primary_protein_id in msa_to_process:

        n_of_mapped_seqs = {}
        for db in dbs:
            if primary_protein_id in sequence_maps[db]:
                n_of_mapped_seqs[db] = len(sequence_maps[db][primary_protein_id])
            else:
                n_of_mapped_seqs[db] = 0
        seq_len = len(gene_seq_map[primary_protein_id])
        cost = estimate_cost(config, primary_protein_id, msa_dbs, gpw_dbs, seq_len, n_of_mapped_seqs)
        total_cost += cost
        cost_map[primary_protein_id] = cost
        cost_tuples.append((primary_protein_id, cost))

        #for db in dbs:
        #    n += 1
        #    inqueue.put((primary_protein_id, db, n))

    prots_sorted_by_cost = sorted(cost_tuples, key=lambda x:x[1], reverse=True)

    optimal_cost = total_cost / n_of_processes

    chunks = []
    current_chunk = 0
    for prot_id, cost in prots_sorted_by_cost:
        if len(chunks) <= current_chunk:
            chunks.append([0, [], {}, {}])
        assigned = False
        start_chunk = current_chunk
        while not assigned:
            if chunks[current_chunk][0] <= optimal_cost:
                chunks[current_chunk][0] += cost
                chunks[current_chunk][1].append(prot_id)
                chunks[current_chunk][2][prot_id] = gene_seq_map[prot_id]
                if not prot_id in chunks[current_chunk][3]:
                    chunks[current_chunk][3][prot_id] = {}
                
                for db in dbs:
                    if not db in chunks[current_chunk][3][prot_id]:
                        chunks[current_chunk][3][prot_id][db] = {}
                    if prot_id in sequence_maps[db]:
                        for hit_id in sequence_maps[db][prot_id]:
                            chunks[current_chunk][3][prot_id][db][hit_id] = sequence_maps[db][prot_id][hit_id]
                assigned = True
            current_chunk += 1
            if current_chunk == n_of_processes:
                current_chunk = 0
            if not assigned and current_chunk == start_chunk:
                print('Chunk assigned failed')
                sys.exit(1)

    store = ray.put((config, update_mode, in_db, msa_dbs, gpw_dbs))

    if config.verbosity >= 2:
        print(f'Going into ray_paraMSA, msa_dbs: {msa_dbs}, gpw_dbs: {gpw_dbs}, optimal cost: {optimal_cost}, number of chunks: {len(chunks)}')

    ray_process_ids = []
    for chunk in chunks:
        chunk_cost = chunk[0]
        if optimal_cost > 0:
            sub_threads = max([chunk_cost // optimal_cost, 1])
        else:
            sub_threads = 1
        if config.verbosity >= 2:
            print(f'Starting a ray_paraMSA thread, chunk cost: {chunk_cost}, sub_threads: {sub_threads}')
        ray_process_ids.append(ray_paraMSA.remote(store, pack((chunk[1], chunk[2], chunk[3], sub_threads))))

    #for pdb_tuple in pdb_ids:
    #    for db in dbs:
    #        n += 1
    #        inqueue.put((pdb_tuple,db,n))


    if config.verbosity >= 1:
        print('Amount of total mapped sequences: ',n_mapped_sequences)
        print('Amount of alignments: ',n)
    """
    processes = {}
    for i in range(1,n_of_processes + 1):
        p = multiprocessing.Process(target=paraMSA, args=(config,lock,inqueue,outqueue,config.verbosity,n,gene_seq_map,sequence_maps,update_mode,in_db))
        processes[i] = p
        p.start()
    for i in processes:
        processes[i].join()

    outqueue.put(None)

    while True:
        outs = outqueue.get()
        if outs == None:
            break
        u_ac,msas,gpws = outs

        if not u_ac in msa_map:
            msa_map[u_ac] = {}
            gpw_map[u_ac] = {}
        for db in msas:
            msa_map[u_ac][db] = msas[db]
        for db in gpws:
            gpw_map[u_ac][db] = gpws[db]

    """

    para_results = ray.get(ray_process_ids)
    for results in para_results:
        for prot_id, msas, gpws in unpack(results):
            if not prot_id in msa_map:
                msa_map[prot_id] = {}
                gpw_map[prot_id] = {}
            for db in msas:
                msa_map[prot_id][db] = msas[db]
            for db in gpws:
                gpw_map[prot_id][db] = gpws[db]

    if config.verbosity >= 2:
        t4 = time.time()
        print('getSequenceFeature part 4: ',t4-t3)


    inqueue = manager.Queue()
    outqueue = manager.Queue()

    prot_mut_map = {}
    for u_ac, aac in samples.samples:
        if not u_ac in prot_mut_map:
            prot_mut_map[u_ac] = []
        prot_mut_map[u_ac].append(aac)

    for u_ac in prot_mut_map:
        aacs = prot_mut_map[u_ac]

        if config.verbosity >= 6:
            print('Added to CalcSeqFeat queue:',u_ac,aacs)

        inqueue.put((u_ac,aacs))

    n_of_processes = min([40,len(prot_mut_map)])

    processes = {}
    for i in range(1,n_of_processes + 1):
        p = multiprocessing.Process(target=paraCalcSeqFeat, args=(config, lock, inqueue, outqueue, config.verbosity, msa_map, gpw_map))
        processes[i] = p
        p.start()
    for i in processes:
        processes[i].join()

    outqueue.put(None)
    if config.verbosity >= 1:
        print('after paraCalcSeqFeat')
    while True:
        outs = outqueue.get()
        if outs == None:
            break
        u_ac,prot_mut_map = outs
        for aac in prot_mut_map:
            for value,feat_name in prot_mut_map[aac]:
                samples.addValue((u_ac,aac),value,feat_name)

    if config.verbosity >= 2:
        t5 = time.time()
        print('getSequenceFeature part 5: ',t5-t4)

    return 

@ray.remote(max_calls = 1)
def ray_paraMSA(store, package):
    config, update_mode, in_db, msa_dbs, gpw_dbs = store
    prot_ids, prot_seq_map, hit_seq_maps, sub_threads = unpack(package)
    
    results = []

    for prot_id in prot_ids:
        if not prot_id in prot_seq_map:
            if prot_id.split('-')[0] in prot_seq_map:
                seq = prot_seq_map[prot_id.split('-')[0]][0]
            elif prot_id in in_db:
                seq = None
            else:
                if debug >= 1:
                    print('Skipped getMSA for:',prot_id,'It was not in the gene_seq_map')
                continue
        else:
             seq = prot_seq_map[prot_id][0]


        msas, gpws = msa.getMSA(config, prot_id, sequence_maps=hit_seq_maps[prot_id], sequence=seq, ref_db_ids=msa_dbs, gpw_ref_db_ids=gpw_dbs, update_mode=update_mode, sub_threads = sub_threads)
        
        results.append((prot_id, msas, gpws))

    return pack(results)



def paraMSA(config, lock, inqueue, outqueue, debug, N, gene_seq_map, sequence_maps, update_mode, in_db):

    with lock:
        inqueue.put(None)

    while True:
        intuple = inqueue.get()
        if intuple == None:
            break

        (prot_id, db, n) = intuple
        if prot_id in sequence_maps[db]:
            sequence_map = sequence_maps[db][prot_id]
        else:
            sequence_map = None
            if debug >= 1:
                print('prot_id not in sequence_map:', prot_id, db)
        if debug >= 2:
            print('Processing Alignment number: ',n,' out of ',N)

        if not prot_id in gene_seq_map:
            if prot_id.split('-')[0] in gene_seq_map:
                seq = gene_seq_map[prot_id.split('-')[0]][0]
            elif prot_id in in_db:
                seq = None
            else:
                if debug >= 1:
                    print('Skipped getMSA for:',prot_id,'It was not in the gene_seq_map')
                continue
        else:
            seq = gene_seq_map[prot_id][0]

        msas, gpws = msa.getMSA(config, prot_id, sequence_map=sequence_map, sequence=seq, ref_db_ids=[], gpw_ref_db_ids=[db], debug=debug, update_mode=update_mode)

        if len(msas) == 0 and len(gpws) == 0:
            if debug >= 1:
                print('Results of getMSA were empty for:', prot_id)
            continue
        if debug >= 2:
            print('Done Alignment number: ',n,' out of ',N)

        with lock:
            outqueue.put((prot_id, msas, gpws))
    return

def paraCalcSeqFeat(config, lock, inqueue, outqueue, debug, msa_map, gpw_map,):
    dbs = []
    for db_id in config.msa_dbs:
        dbs.append((db_id, False))

    for db_id in config.gpw_dbs:
        dbs.append((db_id, True))

    with lock:
        inqueue.put(None)

    while True:
        intuple = inqueue.get()
        if intuple == None:
            break

        (u_ac,aacs) = intuple

        prot_mut_map = {}
        for aac in aacs:
            prot_mut_map[aac] = []
        first_db = True
        for db_name, is_gpw in dbs:
            if is_gpw:
                results_map = gpw_map
                feature_name_tag = "GPW"
            else:
                results_map = msa_map
                feature_name_tag = 'MSA'
            
            if not u_ac in results_map:
                if debug >= 1:
                    print(f'Filtered {u_ac}, since it was not in the results_map: {db_name} ({is_gpw})')
                continue
            if not db_name in results_map[u_ac]:
                if debug >= 1:
                    print(f'Filtered {u_ac} since db_name {db_name} was not in the results_map_map[u_ac], {is_gpw}')
                continue

            gpw_file_path = results_map[u_ac][db_name]
            #print(gpw_file_path)

            try:
                f = gzip.open(gpw_file_path, 'r')
                gpw_fasta = f.read()
                f.close()
            except:
                [e,f,g] = sys.exc_info()
                g = traceback.format_exc()
                print(f'Error with reading file, path: {gpw_file_path}, protein: {u_ac}, db_name: {db_name}, is_gpw: {is_gpw}\n{e}\n{f}\n{g}')

            if gpw_fasta == None:
                if debug >= 1:
                    print('Filtered',u_ac,',since gpw_fasta was None',db_name)

                continue
            if is_gpw:
                gpw_ds, seed = msa.parseGpwFasta(gpw_fasta, dict_out = True)

                if len(gpw_ds) == 0:
                    print('Error, gpw_ds is empty: ',u_ac)

                seed_seq = gpw_ds[list(gpw_ds.keys())[0]][0].replace('-','')
                try:
                    pos_wise_map = msa.getPosWiseGPW(gpw_ds)
                except:
                    [e,f,g] = sys.exc_info()
                    g = traceback.format_exc()
                    print(f'Error in getPosWiseGPW: {u_ac} {db_name} {seed} {gpw_file_path}\n{e}\n{f}\n{g}')
                    continue
            else:
                seq_map = msa.parseMsaFasta(gpw_fasta)
                seed = u_ac
                seed_seq = seq_map[seed].replace('-','')
                pos_wise_map = msa.getPosWiseMSA(seq_map, seed)
            
            (psic_wt_map,psic_mut_map,dpsic_map,
                positional_dpsic_map,window_dpsic_map,protein_median_dpsic) = msa.calcPsicProfiles(config, u_ac, aacs, seed_seq, db_name, gpw = is_gpw, debug=debug)

            for aac in aacs:
                aa1 = aac[0]
                aa2 = aac[-1]
                pos = int(aac[1:-1])-1

                if pos >= len(seed_seq):
                    if debug >= 1:
                        print('Filtered',u_ac,pos,',since it was outside of the seed_seq, len:',len(seed_seq))
                    continue

                n_wt_aa = 0.
                n_mut_aa = 0.
                n_gap = 0.
                n_dif_aa = 0.
                s = float(len(pos_wise_map[0]))
                for al_seq_aa in pos_wise_map[pos]:
                    if al_seq_aa == aa1:
                        n_wt_aa += 1.
                    elif al_seq_aa == aa2:
                        n_mut_aa += 1.
                    elif al_seq_aa == '-':
                        n_gap += 1.
                    else:
                        n_dif_aa += 1.
                wt_rate = n_wt_aa/s
                mut_rate = n_mut_aa/s
                gl_wt_rate = n_wt_aa/(s-n_gap)
                gl_mut_rate = n_mut_aa/(s-n_gap)
                coverage = (s-n_gap)/s
                dif_rate = n_dif_aa/s
                gl_dif_rate = n_dif_aa/(s-n_gap)

                prot_mut_map[aac].append((wt_rate, f'Wildtype AA rate {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((mut_rate,f'Mutant AA rate {feature_name_tag} {db_name}'))

                prot_mut_map[aac].append((gl_wt_rate,f'Wildtype AA rate gapless {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((gl_mut_rate,f'Mutant AA rate gapless {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((coverage,f'MSA allel freq {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((dif_rate,f'Other mutant AA rate {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((gl_dif_rate,f'Other mutant AA rate gapless {feature_name_tag} {db_name}'))

                prot_mut_map[aac].append((psic_wt_map[aac],f'PSIC wildtype AA {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((psic_mut_map[aac],f'PSIC mutant AA {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((dpsic_map[aac],f'dPSIC {feature_name_tag} {db_name}'))

                prot_mut_map[aac].append((positional_dpsic_map[aac],f'Positional median dPSIC {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((window_dpsic_map[aac],f'Window median dPSIC {feature_name_tag} {db_name}'))
                prot_mut_map[aac].append((protein_median_dpsic,f'Protein median dPSIC {feature_name_tag} {db_name}'))
                if first_db:
                    prot_mut_map[aac].append((pos,'Sequence Position Number'))
                    prot_mut_map[aac].append((pos/len(seed_seq),'Relative Sequence Position'))
                    prot_mut_map[aac].append((len(seed_seq),'Protein Size'))
            first_db = False
        with lock:
            outqueue.put((u_ac,prot_mut_map))

def getPosMap(seq):
    #print seq
    pos_map = []
    for pos,char in enumerate(seq):
        if char != '-':
            pos_map.append(pos)
    #print pos_map
    return pos_map


def randomString(stringLength=10):
    """Generate a random string of fixed length """
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(stringLength))
