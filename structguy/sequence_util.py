import os
import sys
import gzip
import subprocess
from structguy.util import get_msa_path
from structguy.consts import n_of_unifref_splits
def parseFromFasta(seqs_from_fasta, config = None, dbs = []):

    print(f'Call parseFromFasta: {seqs_from_fasta=} {config=} {dbs=}')

    if config is not None:
        
        fasta_file_name_base = seqs_from_fasta.split('/')[-1].rsplit('.',1)[0]
        config.custom_msa_db = f'{config.msa_db}'
        if config.verbosity >= 2:
            print(f'Call if parseFromFasta {seqs_from_fasta=} {config.msa_db} {fasta_file_name_base=}')
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
            for ref_db_id in dbs:
                msa_db_filename = get_msa_path(config.custom_msa_db, entry_id, ref_db_id, gpw = True)
                if config.verbosity >=4:
                    print(f'{entry_id=} {ref_db_id=} {msa_db_filename=}')
                if not os.path.isfile(msa_db_filename):
                    inside_all = False
            if inside_all:
                in_db.add(entry_id)
        else:
            seq_map[entry_id][0] += line.replace('\n', '').replace('/','').replace('*','').upper()
    return seq_map, in_db

def parseFasta(seqs_from_fasta):

    f = open(seqs_from_fasta, 'r')
    lines = f.readlines()
    f.close()

    seq_map = {}

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
            
        else:
            seq_map[entry_id][0] += line.replace('\n', '').replace('/','').replace('*','').upper()
    return seq_map

def split_fasta_db():
    infile = sys.argv[1]

    trunk = infile[:-9]

    n_splits = n_of_unifref_splits

    f = gzip.open(infile, 'rb')
    lines = f.readlines()
    f.close()

    n_lines = len(lines)

    lines_per_split = (n_lines // n_splits) + 1

    outlines = []
    out_files = []

    print(f'{n_lines=} {lines_per_split=}')

    count = 0
    current_split = 0
    for line in lines:
        if line == b'':
            continue
        count += 1
            
        if count >= lines_per_split:
            if line[0:1] == b'>':
                out_f = f'{trunk}_{current_split}.fasta.gz'
                f = open(out_f, 'wb')
                f.write(b''.join(outlines))
                f.close()
                out_files.append(out_f)
                outlines = []
                current_split += 1
                count = 0

        outlines.append(line)

    out_f = f'{trunk}_{current_split}.fasta.gz'
    f = gzip.open(out_f, 'wb')
    f.write(b''.join(outlines))
    f.close()
    out_files.append(out_f)

    stem = trunk.split('/')[-1]
    
    for index, out_f in enumerate(out_files):
        search_db = f'{stem}_{index}_search_db'
        cmds = ['mmseqs', 'createdb', out_f, search_db]
        p = subprocess.Popen(cmds)
        p.wait()
        cmds = ['mmseqs', 'createindex', search_db, sys.argv[2], '-s', '7.5']
        p = subprocess.Popen(cmds)
        p.wait()

def check_psic_file(infile, target_len = None):
    if not os.path.isfile(infile):
        return
    
    f = open(infile, 'r')
    lines = f.readlines()
    f.close()

    if len(lines) < 2:
        os.remove(infile)

    if target_len is not None:
        if len(lines) < target_len + 2:
            os.remove(infile)

    return