import time
import subprocess
import multiprocessing
import sys
import string
import random
import pymysql as MySQLdb
import os
import msa

def initFeatures(samples,dbs):
    for db in dbs:
        samples.addFeature('Wildtype AA rate %s' % db,'real',group='sequence',default_value=0.)
        samples.addFeature('Mutant AA rate %s' % db,'real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature('Wildtype AA rate gapless %s' % db,'real',group='sequence',default_value=0.)
        samples.addFeature('Mutant AA rate gapless %s' % db,'real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature('MSA allel freq %s' % db,'real',group='sequence',default_value=0.)

        samples.addFeature('Other mutant AA rate %s' % db,'real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature('Other mutant AA rate gapless %s' % db,'real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature('PSIC wildtype AA %s' % db,'real',group='sequence',default_value=0.)
        samples.addFeature('PSIC mutant AA %s' % db,'real',group='sequence',default_value=0.,mutation_specific=True)
        samples.addFeature('dPSIC %s' % db,'real',group='sequence',default_value=0.,mutation_specific=True)

        samples.addFeature('Positional median dPSIC %s' % db,'real',group='sequence',default_value=0.)
        samples.addFeature('Window median dPSIC %s' % db,'real',group='sequence',default_value=0.)
        samples.addFeature('Protein median dPSIC %s' % db,'real',group='sequence',default_value=0.)

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

def parseFromFasta(seqs_from_fasta, config, dbs):

    fasta_file_name_base = seqs_from_fasta.split('/')[-1].rsplit('.',1)[0]
    config.custom_msa_db = f'{config.msa_db}/{fasta_file_name_base}'

    config.fasta_mode = True

    if not os.path.exists(config.custom_msa_db):
        os.mkdir(config.custom_msa_db)

    f = open(seqs_from_fasta, 'r')
    lines = f.readlines()
    f.close()

    seq_map = {}
    in_db = set()

    for line in lines:
        line = line[:-1]
        if len(line) == 0:
            continue
        if line[0] == '>':
            words = line[1:].split()
            entry_id = words[0]
            if entry_id.count('|') > 1:
                entry_id = entry_id.split('|')[1]
            seq_map[entry_id] = ['']
            inside_all = True
            for db_name in dbs:
                filename = f'{config.custom_msa_db}/{entry_id}_{db_name}_gpw.fasta.gz'
                if not os.path.isfile(filename):
                    inside_all = False
            if inside_all:
                in_db.add(entry_id)
        else:
            seq_map[entry_id][0] += line.replace('\n', '').replace('/','').replace('*','').upper()
    return seq_map, in_db 

def getSequenceFeatures(config, samples, n_of_processes = 6, update_mode=False, seqs_from_fasta = None):
    msa_db = config.msa_db
    pdb_path = config.pdb_path

    import structman.lib as sml

    if config.verbosity >= 2:
        t0 = time.time()

    manager = multiprocessing.Manager()
    lock = manager.Lock()

    inqueue = manager.Queue()
    outqueue = manager.Queue()

    msa_map = {}
    gpw_map = {}
    dbs = config.search_dbs#['ref50','ref90','ref100']

    initFeatures(samples, dbs)

    n = 0
    u_acs = set([])
    pdb_ids = set()

    for u_ac, aac in samples.samples:
        if u_ac.count(':') > 0:
            pdb_ids.add(u_ac)
        else:
            u_acs.add(u_ac)
            #Additionally add the canonical sequences
            if u_ac.count('-') > 0:
                u_acs.add(u_ac.split('-')[0])

    sequence_maps = {}
    for db in dbs:
        sequence_maps[db] = {}

    gene_seq_map = {}

    n_mapped_sequences = 0

    if seqs_from_fasta is not None:
        gene_seq_map, in_db = parseFromFasta(seqs_from_fasta, config, dbs)
        if config.verbosity >= 2:
            print(f'Parsed protein sequences from: {seqs_from_fasta}\nLength of the map:{len(gene_seq_map)}, Length of in_db: {len(in_db)}')
    elif len(u_acs) > 0:
        gene_seq_map,in_db = sml.uniprot.getSequencesPlain(u_acs, config.structman_config, filtering_db=(msa_db,dbs))
    else:
        gene_seq_map,pdb_pos_map,in_db = sml.pdbParser.getSequences(pdb_ids,pdb_path,filtering_db=(msa_db,dbs))

    N = 0
    mmseq_searchs = []
    mmseq_search = {}
    msa_to_process = []
    for primary_protein_id in gene_seq_map:
        if not primary_protein_id in in_db:
            mmseq_search[primary_protein_id] = gene_seq_map[primary_protein_id]
            msa_to_process.append(primary_protein_id)
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
                    return(entries)

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

    for primary_protein_id in msa_to_process:
        for db in dbs:
            n += 1
            inqueue.put((primary_protein_id, db, n))

    for pdb_tuple in pdb_ids:
        for db in dbs:
            n += 1
            inqueue.put((pdb_tuple,db,n))

    if config.verbosity >= 1:
        print('Amount of total mapped sequences: ',n_mapped_sequences)
        print('Amount of alignments: ',n)

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

        if config.verbosity >= 3:
            print('Added to CalcSeqFeat queue:',u_ac,aacs)

        inqueue.put((u_ac,aacs))

    n_of_processes = min([40,len(prot_mut_map)])

    processes = {}
    for i in range(1,n_of_processes + 1):
        p = multiprocessing.Process(target=paraCalcSeqFeat, args=(config,lock,inqueue,outqueue,config.verbosity,dbs,msa_map,gpw_map))
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

def paraMSA(config,lock,inqueue,outqueue,debug,N,gene_seq_map,sequence_maps,update_mode,in_db):

    with lock:
        inqueue.put(None)

    while True:
        intuple = inqueue.get()
        if intuple == None:
            break

        (u_ac,db,n) = intuple
        if u_ac in sequence_maps[db]:
            sequence_map = sequence_maps[db][u_ac]
        else:
            sequence_map = None
            if debug >= 1:
                print('U_ac not in sequence_map:',u_ac,db)
        if debug >= 2:
            print('Processing Alignment number: ',n,' out of ',N)

        if not u_ac in gene_seq_map:
            if u_ac.split('-')[0] in gene_seq_map:
                seq = gene_seq_map[u_ac.split('-')[0]][0]
            elif u_ac in in_db:
                seq = None
            else:
                if debug >= 1:
                    print('Skipped getMSA for:',u_ac,'It was not in the gene_seq_map')
                continue
        else:
            seq = gene_seq_map[u_ac][0]

        msas,gpws = msa.getMSA(config,u_ac,sequence_map=sequence_map,sequence=seq,msa_names=[],gpw_names=[db],debug=debug,update_mode=update_mode)

        if len(msas) == 0 and len(gpws) == 0:
            if debug >= 1:
                print('Results of getMSA were empty for:',u_ac)
            continue
        if debug >= 2:
            print('Done Alignment number: ',n,' out of ',N)

        with lock:
            outqueue.put((u_ac,msas,gpws))
    return

def paraCalcSeqFeat(config,lock,inqueue,outqueue,debug,dbs,msa_map,gpw_map,):
    msa_db = config.msa_db
    sys.path.append(config.msa_source)
    import msa
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
        for db_name in dbs:
            
            if not u_ac in gpw_map:
                if debug >= 1:
                    print('Filtered',u_ac,',since it was not in the gpw_map',db_name)
                continue
            if not db_name in gpw_map[u_ac]:
                if debug >= 1:
                    print('Filtered',u_ac,',since db_name was not in the gpw_map[u_ac]',db_name)
                continue
            gpw_fasta = gpw_map[u_ac][db_name]
            if gpw_fasta == None:
                if debug >= 1:
                    print('Filtered',u_ac,',since gpw_fasta was None',db_name)
                continue
            gpw_ds = msa.parseGpwFasta(gpw_fasta)

            if len(gpw_ds) == 0:
                print('Error, gpw_ds is empty: ',u_ac)

            seed_seq = gpw_ds[0][1].replace('-','')
            pos_wise_map = msa.getPosWiseGPW(gpw_ds)
            
            (psic_wt_map,psic_mut_map,dpsic_map,
                positional_dpsic_map,window_dpsic_map,protein_median_dpsic) = msa.calcPsicProfiles(config,u_ac,aacs,seed_seq,db_name,gpw=True,debug=debug)

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

                prot_mut_map[aac].append((wt_rate,'Wildtype AA rate %s' % db_name))
                prot_mut_map[aac].append((mut_rate,'Mutant AA rate %s' % db_name))

                prot_mut_map[aac].append((gl_wt_rate,'Wildtype AA rate gapless %s' % db_name))
                prot_mut_map[aac].append((gl_mut_rate,'Mutant AA rate gapless %s' % db_name))
                prot_mut_map[aac].append((coverage,'MSA allel freq %s' % db_name))
                prot_mut_map[aac].append((dif_rate,'Other mutant AA rate %s' % db_name))
                prot_mut_map[aac].append((gl_dif_rate,'Other mutant AA rate gapless %s' % db_name))

                prot_mut_map[aac].append((psic_wt_map[aac],'PSIC wildtype AA %s' % db_name))
                prot_mut_map[aac].append((psic_mut_map[aac],'PSIC mutant AA %s' % db_name))
                prot_mut_map[aac].append((dpsic_map[aac],'dPSIC %s' % db_name))

                prot_mut_map[aac].append((positional_dpsic_map[aac],'Positional median dPSIC %s' % db_name))
                prot_mut_map[aac].append((window_dpsic_map[aac],'Window median dPSIC %s' % db_name))
                prot_mut_map[aac].append((protein_median_dpsic,'Protein median dPSIC %s' % db_name))
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
