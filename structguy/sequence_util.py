import os
from structguy.util import get_msa_path
def parseFromFasta(seqs_from_fasta, config = None, dbs = []):

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