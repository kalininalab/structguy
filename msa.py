import os
import sys
import traceback
import gzip
import subprocess
import pymysql as MySQLdb
from Bio.Align.Applications import MafftCommandline

import xml.etree.ElementTree as ET
from multiprocessing import Process, Queue, Manager, Value, Lock
from sklearn.metrics import mutual_info_score
from Bio import pairwise2
from Bio.SubsMat import MatrixInfo as matlist
matrix = matlist.blosum62
import time

import structman.lib.uniprot as uniprot
import structman.lib.pdbParser as pdbParser

import psic_wrapper as psic

search_db_sequences = {}

def median(l):
    n = len(l)
    l = sorted(l)
    if n % 2 == 0:
        med = (l[(n//2)-1]+l[n//2])/2.0
    else:
        med = l[(n-1)//2]
    return med

def lookup(config,u_ac,msa_names=['ref50','ref90'],gpw_names=['ref50','ref90'],debug=0,update_mode=False,sequence=None,sequence_map=None,pdb_tuple=None):

    out_directory = get_out_directory(u_ac, config, pdb_tuple = pdb_tuple)


    files = {}
    msas = {}
    gpws = {}
    for msa_name in msa_names:
        filename = f'{out_directory}/{u_ac}_{msa_name}.fasta.gz'
        if os.path.isfile(filename):
            files[msa_name] = filename
            if debug >= 2:
                print('Look-up found a msa: ',filename)

    gpw_files = {}
    for gpw_name in gpw_names:
        filename = f'{out_directory}/{u_ac}_{gpw_name}_gpw.fasta.gz'
        if os.path.isfile(filename):
            gpw_files[gpw_name] = filename
            if debug >= 2:
                print('Look-up found a gpw: ',filename)

    for msa_name in files:
        filename = files[msa_name]
        f = gzip.open(filename,'r')
        msa = f.read()
        f.close()
        if msa == '':
            continue

        psic_name = f'{out_directory}/{u_ac}_{msa_name}.psic.gz'
        if not os.path.isfile(psic_name):
            if debug >= 1:
                print('Calc psic profiles from lookup',msa_name)
            psic.psicFromFasta(msa,psic_name[:-3])
            
            if os.path.isfile(psic_name):
                os.remove(psic_name)
            try:
                os.system("gzip %s" % psic_name[:-3])
            except:
                pass

        msas[msa_name] = msa

    for gpw_name in gpw_files:

        filename = gpw_files[gpw_name]

        if update_mode:
            gpw = updateGPW(filename,gpw_name,sequence,sequence_map,u_ac)

        else:
            f = gzip.open(filename,'r')
            gpw = f.read()
            f.close()
            if gpw == '':
                continue

        psic_name = f'{out_directory}/{u_ac}_{gpw_name}_gpw.psic.gz'
        if (not os.path.isfile(psic_name)) or update_mode:
            if debug >= 1:
                print('Calc psic profiles from lookup',u_ac,gpw_name)
            psic.psicFromGPW(gpw,psic_name[:-3],config,debug=debug)
            
            if os.path.isfile(psic_name):
                os.remove(psic_name)
                if debug >= 1:
                    print('removed old psic file: ',psic_name)
            try:
                os.system("gzip %s" % psic_name[:-3])
            except:
                pass
        gpws[gpw_name] = gpw

    return msas,gpws

def getSequence(config,u_ac,pdb_tuple):
    db_adress = config.db_adress
    db_user_name = config.db_user_name
    db_password = config.password
    try:
        db = MySQLdb.connect(db_adress,db_user_name,db_password,'struct_man_db_uniprot')
        cursor = db.cursor()
    except:
        db = None
        cursor = None
    if pdb_tuple == None:
        gene_sequence_map = uniprot.getSequencesPlain([u_ac],db,cursor)
    else:
        pdb,chain = pdb_tuple.split(':')
        return pdbParser.getSequencePlain(pdb,chain,pdb_path)
        
    if db != None:
        db.close()
    return gene_sequence_map[u_ac]

def parseFasta(path,lines=None):
    if lines == None:
        f = open(path,'rb')
        lines = f.read().split(b'\n')
        f.close()

    seq_map = {}
    for line in lines:
        if len(line) == 0:
            continue
        if line[0:1] == b'>':
            entry_id = line[1:].split()[0].decode('ascii')
            seq_map[entry_id] = ''
        else:
            seq_map[entry_id] += line.decode('ascii')

    return seq_map

def blast(config,seq,name,search_db,search_db_path,search_db_sequences={}):
    blast_path = config.blast_path

    blast_db_path = search_db_path

    if not search_db in search_db_sequences:
        search_db_sequences[search_db] = parseFasta(blast_db_path)

    sequences = search_db_sequences[search_db]

    FNULL = open(os.devnull, 'w')
    cwd = os.getcwd()
    seq = seq.replace(" ","")
    target_length = len(seq)
    
    blast_in = "%s/blast_in_%s_%s.tmp.fasta" % (cwd,name,search_db)
    blast_out = "%s/blast_out_%s_%s.tmp" % (cwd,name,search_db)

    f = open(blast_in, "wb")
    f.write(b">" + name.encode('ascii') + b"\n" + seq.encode('ascii'))
    f.close()

    if os.path.isfile(blast_out):
        os.remove(blast_out)

    path = blast_path + "blastp"
    arg1 = "-db"
    arg2 = blast_db_path
    arg3 = "-evalue"
    arg4 = "1e-10"
    arg5 = "-outfmt"
    arg6 = "5"
    arg7 = "-query"
    arg8 = blast_in
    arg9 = "-out"
    arg10 = blast_out
    p = subprocess.Popen([path,arg1,arg2,arg3,arg4,arg5,arg6,arg7,arg8,arg9,arg10])#,stdout=FNULL)
    p.wait()

    if not os.path.isfile(blast_out):
        print(name,search_db,'Blast error')
        return None

    f = open(blast_out, "r")
    lines = f.read()
    f.close()
    #print name,lines
    
    os.remove(blast_in)
    os.remove(blast_out)

    uniref_ids = []

    if len(lines) == 0:
        return ''

    root = ET.fromstring(lines)
    blast_iter_hits = root[8][0][4]    
 
    for child in blast_iter_hits:
        uniref_id = child[2].text.split()[0]
        uniref_ids.append(uniref_id)
    
    #print uniref_ids
    fasta_lines = ['>%s' % name,seq.replace('U','C').replace('O','K').replace('J','I')]
    for uniref_id in uniref_ids:
        fasta_lines.append('>%s' % uniref_id)
        fasta_lines.append(sequences[uniref_id].replace('U','C').replace('O','K').replace('J','I'))

    page = '\n'.join(fasta_lines)
    return page,search_db_sequences

def computeMSA(config,seq,u_ac,search_db='ref50',search_db_path='',debug=0,search_db_sequences={}):
    mafft_exe = config.mafft_path
    print('Compute MSA: ',u_ac,search_db)

    #Blast against search database
    fasta_page,search_db_sequences = blast(seq,'%s_%s' % (u_ac,search_db),search_db,search_db_path,search_db_sequences=search_db_sequences)

    if fasta_page == None:
        return None,None

    #Write the Blast results into a fasta file
    cwd = os.getcwd()

    temp_fasta = '%s/temp_fasta_%s_%s.fasta' % (cwd,u_ac,search_db)
    f = open(temp_fasta,'w')
    f.write(fasta_page)
    f.close()

    stderr = None
    #Run mafft
    try:
        mafft_cline = MafftCommandline(mafft_exe, input=temp_fasta,thread=4,amino=True)
        stdout, stderr = mafft_cline()
    except:
        [e,f,g] = sys.exc_info()
        g = traceback.format_exc(g)
        print(u_ac,search_db,e,f,g,stderr)
        return None,None
    

    #delete temporary files
    os.remove(temp_fasta)

    print('Done computing MSA: ',u_ac,search_db)

    return stdout,fasta_page,search_db_sequences

def updateGPW(filename,search_db,seq,sequence_map,u_ac):
    print("Update GPW: ",filename)
    f = gzip.open(filename,'rb')
    gpw = f.read()
    f.close()
    gpw_ds = parseGpwFasta(gpw,dict_out=True)

    #print gpw_ds

    out_fasta_lines = []

    new_fasta_lines = []

    n = 0
    m = 0
    target_seq = seq.replace('U','C').replace('O','K').replace('J','I')
    for seq_id in sequence_map:
        #print seq_id
        if seq_id in gpw_ds:
            n += 1
            out_fasta_lines.append('>%s_%s' % (u_ac,seq_id))
            out_fasta_lines.append(gpw_ds[seq_id][0])
            out_fasta_lines.append('>%s' % seq_id)
            out_fasta_lines.append(gpw_ds[seq_id][1])
            continue
        m += 1
        template_seq = sequence_map[seq_id]
        
        try:
            (target_aligned_sequence,template_aligned_sequence,a,b,c) = pairwise2.align.globalds(target_seq, template_seq,matrix,-10.0,-0.5,one_alignment_only=True)[0]
        except:
            print('GPW error: ',u_ac,seq_id)
            print(target_seq[:10],template_seq[:10])
            continue
        new_fasta_lines.append('>%s_%s' % (u_ac,seq_id))
        new_fasta_lines.append(target_aligned_sequence)
        new_fasta_lines.append('>%s' % seq_id)
        new_fasta_lines.append(template_aligned_sequence)

    print('Done updating GPW: ',u_ac,search_db)
    print('Used ',n,' old alignments and computed ',m,' new alignments')

    gpw = '%s\n%s' % ('\n'.join(out_fasta_lines),'\n'.join(new_fasta_lines))

    if m>0:
        new_total_gpw = '%s\n%s' % (gpw,'\n'.join(new_fasta_lines))
        f = gzip.open(filename,'w')
        f.write(new_total_gpw)
        f.close()

    return gpw

def computeGPW(config,seq,u_ac,search_db='ref50',search_db_path={},debug=0,fasta_page=None,sequence_map=None,search_db_sequences={}):
    print('Compute GPW: ',u_ac,search_db)

    if sequence_map == None:
        if fasta_page == None:
            pass
            #not supported at the moment
            """
            #Blast against search database
            if debug >= 1:
                print('In computeGPW sequence_map is None and fasta_page is None, try BLAST:',u_ac,search_db)
            fasta_page,search_db_sequences = blast(seq,'%s_%s' % (u_ac,search_db),search_db,search_db_path,search_db_sequences=search_db_sequences)
            """
        if fasta_page == None:
            return None,search_db_sequences

        if debug >= 1:
            print('Blast result size: ',len(fasta_page))

        seq_map = parseFasta('',lines=fasta_page.split('\n'))
        
    else:
        seq_map = sequence_map

    if seq == 0 or seq == 1:
        if debug >= 1:
            print('Sequence error in computeGPW:',u_ac,search_db)
        return '',search_db_sequences

    out_fasta_lines = []
    target_seq = seq.replace('U','C').replace('O','K').replace('J','I')
    for seq_id in seq_map:
        template_seq = seq_map[seq_id]
        if debug >= 3:
            print('Aligning:',u_ac,seq_id)
            print(target_seq)
            print(template_seq)
        try:
            (target_aligned_sequence,template_aligned_sequence,a,b,c) = pairwise2.align.globalds(target_seq, template_seq,matrix,-10.0,-0.5,one_alignment_only=True)[0]
        except:
            if debug >= 1:
                print('GPW error: ', u_ac, seq_id)
                print(target_seq[:10],template_seq[:10])
            continue
        out_fasta_lines.append('>%s_%s' % (u_ac,seq_id))
        out_fasta_lines.append(target_aligned_sequence)
        out_fasta_lines.append('>%s' % seq_id)
        out_fasta_lines.append(template_aligned_sequence)
    if debug >= 1:
        print('Done computing GPW: ',u_ac,search_db)

    return '\n'.join(out_fasta_lines),search_db_sequences

def saveMSA(config, msa, u_ac, db_name, pdb_tuple):

    if msa == None:
        return

    out_directory = get_out_directory(u_ac, config, pdb_tuple = pdb_tuple)

    files = {}

    filename = f'{out_directory}/{u_ac}_{db_name}.fasta'

    f = open(filename,'w')
    f.write(msa)
    f.close()

    if os.path.isfile('%s.gz' % filename):
        os.remove('%s.gz' % filename)
    os.system("gzip %s" % filename)

    return

def saveGpw(config,gpw,u_ac,db_name,pdb_tuple):

    if gpw == None:
        return

    out_directory = get_out_directory(u_ac, config, pdb_tuple = pdb_tuple)

    filename = f'{out_directory}/{u_ac}_{db_name}_gpw.fasta'

    f = open(filename,'w')
    f.write(gpw)
    f.close()

    if os.path.isfile('%s.gz' % filename):
        os.remove('%s.gz' % filename)
    os.system("gzip %s" % filename)

    return

def parsePsicFile(infile,debug=0):
    if not os.path.isfile(infile):
        if debug >= 1:
            print('Did not found psic-file: ',infile)
        return {}

    f = gzip.open(infile,'rb')
    lines = f.read().split(b'\n')
    f.close()

    aa_key = lines[1].split()[1:-1]

    psic_profiles = {}

    for pos,line in enumerate(lines[2:]):
        if line == b'':
            continue
        data = line.split()[1:-1]
        psic_profiles[pos] = {}
        for aa_pos,aa in enumerate(aa_key):
            psic_profiles[pos][aa.decode('ascii')] = float(data[aa_pos].decode('ascii'))

    return psic_profiles

#called by sequence_feature_generation
def calcPsicProfiles(config, u_ac, aacs, seq, msa_name, gpw=False, debug=0):

    out_directory = get_out_directory(u_ac, config)

    if not gpw:
        psic_name = f'{out_directory}/{u_ac}_{msa_name}.psic.gz'
    else:
        psic_name = f'{out_directory}/{u_ac}_{msa_name}_gpw.psic.gz'

    psic_profiles = parsePsicFile(psic_name,debug=debug)

    psic_wt_map = {}
    psic_mut_map = {}
    dpsic_map = {}
    window_dpsic_map = {}
    positional_dpsic_map = {}

    if psic_profiles == {}:
        print('Empty psic profiles: ',u_ac,msa_name)
        return psic_wt_map,psic_mut_map,dpsic_map

    positional_median_dpsics = []
    for pos,wt in enumerate(seq):
        if not pos in psic_profiles:
            if debug >= 1:
                print('pos not in psic_profiles:',u_ac,pos)
            continue
        if not wt in psic_profiles[pos]:
            if debug >= 1:
                print('wt not in psic_profiles[pos]:',u_ac,pos,wt)
            continue
        psic_wt = psic_profiles[pos][wt]
        positional_dpsics = []
        for mut_aa in psic_profiles[pos]:
            if mut_aa == wt:
                continue
            psic_mut = psic_profiles[pos][mut_aa]
            dpsic = psic_wt-psic_mut
            positional_dpsics.append(dpsic)
        positional_median_dpsic = median(positional_dpsics)
        positional_median_dpsics.append(positional_median_dpsic)

    protein_median_dpsic = median(positional_median_dpsics)

    for aac in aacs:
        aa_wt = aac[0]
        aa_mut = aac[-1]
        pos = int(aac[1:-1]) -1

        window_a = pos-10
        window_b = pos+10

        if window_a < 0 and window_b >= len(seq):
            window_a = 0
            window_b = len(seq) - 1
        if window_a < 0:
            window_b -= window_a
            window_a = 0
        if window_b >= len(seq):
            window_a -= window_b - (len(seq)-1)
            window_b = len(seq) - 1
            if window_a < 0:
                window_a = 0

        if not pos in psic_profiles:
            if debug >= 1:
                print('psic error ',u_ac,aac,msa_name,gpw)
            positional_dpsic_map[aac] = 0.
            psic_wt_map[aac] = 0.
            psic_mut_map[aac] = 0.
            dpsic_map[aac] = 0.
            continue

        if not aa_wt in psic_profiles[pos]:
            if debug >= 1:
                print('psic error 2',u_ac,aac,msa_name,gpw)
            positional_dpsic_map[aac] = 0.
            psic_wt_map[aac] = 0.
            psic_mut_map[aac] = 0.
            dpsic_map[aac] = 0.
            continue

        window_median_position_dpsics = positional_median_dpsics[window_a:(window_b+1)]
        window_median_dpsic = median(window_median_position_dpsics)

        window_dpsic_map[aac] = window_median_dpsic

        if pos >= len(positional_median_dpsics):
            if debug >= 1:
                print('psic error 3',u_ac,aac,msa_name,gpw)
            positional_dpsic_map[aac] = 0.
            psic_wt_map[aac] = 0.
            psic_mut_map[aac] = 0.
            dpsic_map[aac] = 0.
            continue

        positional_median_dpsic = positional_median_dpsics[pos]
        positional_dpsic_map[aac] = positional_median_dpsic

        psic_wt = psic_profiles[pos][aa_wt]
        if aa_mut in psic_profiles[pos]:
            psic_mut = psic_profiles[pos][aa_mut]
        else:
            psic_mut = -1.

        dpsic = psic_wt-psic_mut

        psic_wt_map[aac] = psic_wt
        psic_mut_map[aac] = psic_mut
        dpsic_map[aac] = dpsic

    return psic_wt_map,psic_mut_map,dpsic_map,positional_dpsic_map,window_dpsic_map,protein_median_dpsic

#called by structural_feature_generation
def getPosWiseGPW(gpw):
    pos_wise_map = {}
    target = True
    first_target = True
    for [seq_id,seq] in gpw:
        if target:
            pos_map = getPosMap(seq)
            target = False
            if first_target:
                for gl_pos,pos in enumerate(pos_map):
                    pos_wise_map[gl_pos] = [seq[pos]]
            first_target = False
        else:
            target = True
            for gl_pos,pos in enumerate(pos_map):
                pos_wise_map[gl_pos].append(seq[pos])

    return pos_wise_map

def getPosMap(seq):
    #print seq
    pos_map = []
    for pos,char in enumerate(seq):
        if char != '-':
            pos_map.append(pos)
    #print pos_map
    return pos_map

def get_out_directory(protein_id, config, pdb_tuple = None):
    if not config.fasta_mode:
        if pdb_tuple == None:
            folder_key = protein_id.split('-')[0][-2:]
        else:
            folder_key = pdb_tuple[1:3]
        out_directory = f'{config.msa_db}/{folder_key}'

    else:
        out_directory = f'{config.custom_msa_db}' #gets set in parse_from_fasta in sequence_feature_generation

    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)

    return out_directory

def getMSA(config,u_ac,sequence_map=None,sequence=None,msa_names=['ref50','ref90'],gpw_names=['ref50','ref90'],debug=0,update_mode=False):

    if u_ac.count(':') > 0:
        pdb_tuple = u_ac
    else:
        pdb_tuple = None

    search_dbs = {'ref50':config.mmseqs_search_db_ref50,'ref90':config.mmseqs_search_db_ref90,'ref100':config.mmseqs_search_db_ref100}

    if debug >= 1:
        print('getMSA', u_ac,msa_names,gpw_names,update_mode)

    #If the sequence is not given, get it
    if update_mode and sequence == None:
        sequence = getSequence(config,u_ac,pdb_tuple)

        if sequence == 0 or sequence == 1 or sequence == 2:
            print('Sequence error: ',u_ac)
            return {},{}

    #Check if the protein is in the database
    msas,gpws = lookup(config,u_ac,msa_names=msa_names,gpw_names=gpw_names,debug=debug,update_mode=update_mode,sequence=sequence,sequence_map=sequence_map,pdb_tuple=pdb_tuple)

    if debug >= 2:
        print('Lookup results: ',u_ac,list(msas.keys()),list(gpws.keys()))

    #If the msa is not in the database, compute it
    if len(msas) == len(msa_names) and len(gpws) == len(gpw_names):
        if debug >= 1:
            print('found msas in the db')
        return msas,gpws

    fasta_results = {}

    out_directory = get_out_directory(u_ac, config, pdb_tuple = pdb_tuple)

    #compute the msa's
    search_db_sequences = {}
    for msa_name in msa_names:
        if msa_name in msas:
            continue
        msa,fasta_page,search_db_sequences = computeMSA(config,sequence,u_ac,search_db=msa_name,search_db_path=search_dbs[msa_name],debug=debug,sequence_map=sequence_map,search_db_sequences=search_db_sequences)
        msas[msa_name] = msa

        psic_name = f'{out_directory}/{u_ac}_{msa_name}.psic.gz'
        if not os.path.isfile(psic_name):
            if debug >= 1:
                print('Calc psic profiles from getMSA',msa_name)
            psic.psicFromFasta(msa,outfile=psic_name[:-3])
            
            if os.path.isfile(psic_name):
                os.remove(psic_name)
            try:
                os.system("gzip %s" % psic_name[:-3])
            except:
                pass

        #save them into the database
        saveMSA(config,msa,u_ac,msa_name,pdb_tuple)

    for gpw_name in gpw_names:
        if gpw_name in gpws:
            continue
        if gpw_name in fasta_results:
            fasta_page = fasta_results[gpw_name]
        else:
            fasta_page = None
        gpw,search_db_sequences = computeGPW(config,sequence,u_ac,search_db=gpw_name,search_db_path=search_dbs[gpw_name],debug=debug,
                                                fasta_page=fasta_page,sequence_map=sequence_map,search_db_sequences=search_db_sequences)
        if gpw == '':
            if debug >= 1:
                print('computeGPW returned empty result:',u_ac,gpw_name)
            continue
        gpws[gpw_name] = gpw

        #save them into the database
        saveGpw(config,gpw,u_ac,gpw_name,pdb_tuple)

        psic_name = f'{out_directory}/{u_ac}_{gpw_name}_gpw.psic.gz'
        if not os.path.isfile(psic_name):
            if debug >= 1:
                print('Calc psic profiles from getMSA',gpw_name)
            psic.psicFromGPW(gpw,psic_name[:-3],config,debug=debug)
            
            if os.path.isfile(psic_name):
                os.remove(psic_name)
            try:
                os.system("gzip %s" % psic_name[:-3])
            except:
                pass

    return msas,gpws

def parseGpwFasta(page,dict_out=False):

    try:
        lines = page.split(b'\n')
    except:
        page = page.encode('ascii')
        lines = page.split(b'\n')

    if dict_out:
        seq_map = {}
        second_seq = False
        for line in lines:
            if len(line) == 0:
                continue
            if line[0:1] == b'>':
                entry_id = line[1:].split()[0]
                if entry_id.count(b'_') == 2:
                    qid = entry_id.split(b'_')[0]
                    t_id = (b'_'.join(entry_id.split(b'_')[1:])).decode('ascii')
                    seq_map[t_id] = ['','']
                    second_seq = False
                else:
                    t_id = entry_id.decode('ascii')
                    second_seq = True
            else:
                if second_seq:
                    seq_map[t_id][1] += line.decode('ascii')
                else:
                    seq_map[t_id][0] += line.decode('ascii')
    else:

        seq_map = []
        for line in lines:
            if len(line) == 0:
                continue
            if line[0:1] == b'>':
                entry_id = line[1:].split()[0].decode('ascii')
                
                
                seq_map.append([entry_id,''])
            else:
                seq_map[-1][1] += line.decode('ascii')
    return seq_map

def parseMsaFasta(page):
    lines = page.split(b'\n')

    seed = None

    seq_map = {}
    for line in lines:
        if len(line) == 0:
            continue
        if line[0:1] == b'>':
            entry_id = line[1:].split()[0].decode('ascii')
            seq_map[entry_id] = ''
            if seed == None:
                seed = entry_id
        else:
            seq_map[entry_id] += line.decode('ascii')
    return seq_map,seed


