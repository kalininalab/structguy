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
import shutil
import requests
import json
import numpy as np
#import xml.etree.ElementTree as ET

from pathlib import Path
from ray.util.queue import Queue

from structguy import msa, consts, util
from structguy.msa import computeMSA, calcGPW
from structguy.sequence_util import parseFromFasta, parseFasta, check_psic_file
from structman.base_utils.base_utils import pack, unpack
from structman.base_utils.ray_utils import ray_init
from structman.lib.lib_utils import clean_prot_id, check_msa_file
from structguy.sampleSpace import SampleSpace
from structguy.psic_wrapper import psicFromFasta
from structguy.consts import n_of_unifref_splits
import stat
from scipy import stats

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def initFeatures(config, samples):
    dbs = []
    for db_id in config.msa_dbs:
        dbs.append((db_id, "MSA"))

    for db_id in config.gpw_dbs:
        dbs.append((db_id, "GPW"))

    for db_name, feature_name_tag in dbs:
        samples.addFeature(f"Wildtype AA rate {feature_name_tag} {db_name}", "real", group="sequence")
        samples.addFeature(f"Mutant AA rate {feature_name_tag} {db_name}", "real", group="sequence", mutation_specific=True)
        samples.addFeature(f"Wildtype AA rate gapless {feature_name_tag} {db_name}", "real", group="sequence")
        samples.addFeature(f"Mutant AA rate gapless {feature_name_tag} {db_name}", "real", group="sequence", mutation_specific=True)
        samples.addFeature(f"MSA allel freq {feature_name_tag} {db_name}", "real", group="sequence")

        samples.addFeature(f"Other mutant AA rate {feature_name_tag} {db_name}", "real", group="sequence", mutation_specific=True)
        samples.addFeature(f"Other mutant AA rate gapless {feature_name_tag} {db_name}", "real", group="sequence", mutation_specific=True)
        samples.addFeature(f"PSIC wildtype AA {feature_name_tag} {db_name}", "real", group="sequence")
        samples.addFeature(f"PSIC mutant AA {feature_name_tag} {db_name}", "real", group="sequence", mutation_specific=True)
        samples.addFeature(f"dPSIC {feature_name_tag} {db_name}", "real", group="sequence", mutation_specific=True)

        samples.addFeature(f"Positional median dPSIC {feature_name_tag} {db_name}", "real", group="sequence")
        samples.addFeature(f"Window median dPSIC {feature_name_tag} {db_name}", "real", group="sequence")
        samples.addFeature(f"Protein median dPSIC {feature_name_tag} {db_name}", "real", group="sequence")

        samples.addFeature(f"gemme_evolInd {feature_name_tag} {db_name}", "real", group="sequence")
        samples.addFeature(f"gemme_evolEpi {feature_name_tag} {db_name}", "real", group="sequence")
        samples.addFeature(f"gemme_evolCombi {feature_name_tag} {db_name}", "real", group="sequence")

        samples.addFeature(f"MSA depth {feature_name_tag} {db_name}", "integer", group="sequence")

    samples.addFeature("Sequence Position Number", "integer", group="amino acid property")
    samples.addFeature("Relative Sequence Position", "real", group="amino acid property")
    samples.addFeature("Protein Size", "integer", group="amino acid property")

    

def prepare_gemme(config: util.Config):
    gene_seq_map = parseFasta(config.path_to_sequence_fasta)
    prot_ids = list(gene_seq_map.keys())
    msa_file_dict = parse_msa_folder(prot_ids, gene_seq_map, config)
    config.logger.info(f'{len(msa_file_dict)=}')
    build_gemme_cmd_script(config.msa_folder_path, msa_file_dict)

    config.logger.info('- Gemme is prepared -')
    config.logger.info('Call in msas folder [docker run -ti --rm --mount type=bind,source=$PWD,target=/project elodielaine/gemme:gemme]')
    config.logger.info('Then inside: [./call_gemme_inside_container.sh]')
    config.logger.info('Finally: [exit]')

def parse_msa_folder(prot_ids, seq_map, config):
    config.logger.info(f'{len(prot_ids)=}')
    msa_file_dict = {}
    

    for fn in os.listdir(config.msa_folder_path):
        subfolder = f'{config.msa_folder_path}/{fn}'
        if not os.path.isdir(subfolder):
            continue
        for prot_id in prot_ids:
            if prot_id in msa_file_dict:
                continue
            name_len = len(prot_id)
            if fn[:name_len] == clean_prot_id(prot_id):
                done = False
                for sfn in os.listdir(subfolder):
                    if sfn[-13:] == 'evolCombi.txt' and not config.overwrite:
                        done = True
                if not done:
                    to_remove = []
                    for sfn in os.listdir(subfolder):
                        if sfn[-10:] != '_msa.fasta':
                            continue
                        msa_file = f'{subfolder}/{sfn}'

                        checked = check_msa_file(msa_file, seq_map[prot_id][0])

                        if checked is None:
                            config.logger.info(f'Found invalid msa file: {msa_file}')
                            continue

                        if checked:
                            msa_file_dict[prot_id] = f'{fn}/{sfn}'
                        else:
                            to_remove.append(msa_file)

                    for msa_file in to_remove:
                        os.remove(msa_file)
    
    return msa_file_dict


def build_gemme_cmd_script(msa_folder_path, msa_file_dict):
    lines = []
    for prot_id in msa_file_dict:
        msa_path = msa_file_dict[prot_id]

        f = open(f'{msa_folder_path}/{msa_path}', 'r')
        page = f.read()
        f.close()

        msa = clean_msa(page)

        f = open(f'{msa_folder_path}/{msa_path}', 'w')
        f.write(msa)
        f.close()

        subfolder, msaf = msa_path.split('/')
        lines.append(f'cd "{subfolder}"\n')
        lines.append(f'echo "{msaf}"\n')
        #lines.append(f'head -n 1 "{msaf}"\n')
        line = f'python2.7 $GEMME_PATH/gemme.py "{msaf}" -r input -f "{msaf}"\n'
        lines.append(line)

        lines.append('cd ..\n')

    f = open(f'{msa_folder_path}/call_gemme_inside_container.sh', 'w')
    f.write(''.join(lines))
    f.close()

    f = Path(f'{msa_folder_path}/call_gemme_inside_container.sh')
    f.chmod(f.stat().st_mode | stat.S_IEXEC)
    #os.chmod(f'{msa_folder_path}/call_gemme_inside_container.sh', stat.st_mode | stat.S_IEXEC)


def geneSeqMapToFasta(prot_seq_map, outfile, config):
    lines = []
    n = 0
    m = 0

    if config.verbosity >= 2:
        config.logger.info(f"Size of prot_seq_map converted into a fasta file: {len(prot_seq_map)=}")

    for u_ac in prot_seq_map:
        seq = prot_seq_map[u_ac][0]

        if seq == 0 or seq == 1 or seq == "":
            continue

        lines.append(f">{u_ac}\n")
        lines.append(f"{seq}\n")
        m += 1

    if len(lines) > 0:
        config.logger.info(f"Filtered {n} Proteins before mmseqs2")
        config.logger.info(f"{m} sequences going into mmseqs2")
        f = open(outfile, "w")
        page = "".join(lines)
        f.write(page)
        f.close()
        return None
    else:
        return "Empty fasta file"


def estimate_cost(config, prot_id, msa_ref_dbs, gpw_ref_dbs, seq_len, n_of_mapped_seqs):
    total_cost = 0
    out_directory = msa.get_out_directory(prot_id, config)
    for msa_ref_db in msa_ref_dbs:
        filename = util.get_msa_path(out_directory, prot_id, msa_ref_db)
        if not os.path.isfile(filename):
            total_cost += ((seq_len**2) * n_of_mapped_seqs[msa_ref_db]) + (seq_len**2 * (n_of_mapped_seqs[msa_ref_db] ** 2))

        psic_name = util.get_msa_path(out_directory, prot_id, msa_ref_db, psic=True)
        if not os.path.isfile(psic_name):
            total_cost += seq_len * n_of_mapped_seqs[msa_ref_db]

    for gpw_ref_db in gpw_ref_dbs:
        filename = util.get_msa_path(out_directory, prot_id, gpw_ref_db, gpw=True)
        if os.path.isfile(filename):
            total_cost += (seq_len**2) * n_of_mapped_seqs[gpw_ref_db]

        psic_name = util.get_msa_path(out_directory, prot_id, gpw_ref_db, gpw=True, psic=True)
        if not os.path.isfile(psic_name):
            total_cost += seq_len * n_of_mapped_seqs[gpw_ref_db]

    return total_cost

@ray.remote
def para_mmseqs(store, db, indeces, temp_fasta, n_threads):
    config, mmseqs2_search_dbs, max_seqs = store
    util.reset_logger_for_remotes(config)
    sequence_maps: dict[str , dict[str, str]] = {}
    n_mapped_sequences = 0

    coverage_cut_off = 0.7

    for index in indeces:
        mmseqs2_search_db = mmseqs2_search_dbs[db].replace('_search_db', f'_{index}_search_db')

        config.logger.info(f'Para mmseqs2: {mmseqs2_search_db=}, {indeces=} in this process')

        temp_subfolder = f'{config.mmseqs_tmp_folder}/{index}'
        if not os.path.isdir(temp_subfolder):
            os.makedirs(temp_subfolder)

        temp_fasta_trunk = os.path.abspath(temp_fasta)[:-6]
        cp_temp_fasta = f'{temp_fasta_trunk}_{index}.fasta'

        shutil.copy(temp_fasta, cp_temp_fasta)


        temp_outfile = f"{temp_subfolder}/tmp_outfile_{randomString()}.fasta"
        FNULL = open(os.devnull, 'w')
        p = subprocess.Popen(
            [
                config.mmseqs_path,
                "easy-search",
                temp_fasta,
                mmseqs2_search_db,
                temp_outfile,
                temp_subfolder,
                "--max-seqs",
                max_seqs,
                "--format-output",
                "query,target,tseq,qcov,pident",
                "--max-seq-len",
                "200000",
                '--min-aln-len', '30',
                '--threads', str(n_threads)
            ], stdout=FNULL
        )
        p.wait()

        config.logger.info(f'finished Para mmseqs2: {mmseqs2_search_db=}')

        f = open(temp_outfile, "r")
        lines = f.read().split("\n")
        f.close()

        if len(lines) == 0:
            return

        for line in lines:
            if line == "":
                continue
            words = line.split()

            #config.logger.info(f'{mmseqs2_search_db=} {line=}')

            gene = words[0]
            hit = words[1]
            tseq = words[2]
            coverage = float(words[3])
            seq_id = float(words[4])

            if coverage < coverage_cut_off:
                continue

            if gene not in sequence_maps:
                sequence_maps[gene] = []
            sequence_maps[gene].append((hit,tseq, coverage, seq_id))

            n_mapped_sequences += 1

        os.remove(temp_outfile)
        os.remove(cp_temp_fasta)

    return sequence_maps, n_mapped_sequences, db


def do_mmseqs_search(
        config: util.Config,
        mmseq_searchs:list[dict[str, list[str]]],
        dbs,
        mmseqs2_search_dbs,
        n_splits = n_of_unifref_splits
        ):
    sequence_maps: dict[str, dict[str, list[tuple[str, str, float, float]]]] = {}
    for db in dbs:
        sequence_maps[db] = {}
    
    n_mapped_sequences = 0
    M = 0
    max_seqs = '5000'

    store = ray.put((config, mmseqs2_search_dbs, max_seqs))

    for mmseq_search in mmseq_searchs:
        mmseq_search: dict[str, list[str]]
        temp_fasta = "%s/tmp_%s.fasta" % (config.mmseqs_tmp_folder, randomString())
        returncode = geneSeqMapToFasta(mmseq_search, temp_fasta, config)

        number_of_targets = len(mmseq_search)
        distribution_factor = max([4, (4*config.proc_n) // number_of_targets])
        n_procs = min([config.proc_n, n_splits])

        splits_per_proc = n_splits // n_procs
        if n_splits % n_procs != 0:
            splits_per_proc += 1

        config.logger.info(f'Starting para mmseqs2: {config.proc_n=} {number_of_targets=} {distribution_factor=} {n_procs=} {splits_per_proc=}')

        if returncode is None:
            for db in dbs:
                #mmseqs2_search_db = mmseqs2_search_dbs[db]

                procs = []
                current_index = 0
                for _ in range(n_procs):
                    indeces = []
                    for _ in range(splits_per_proc):
                        if current_index < n_splits:
                            indeces.append(current_index)
                        current_index += 1
                    procs.append(para_mmseqs.remote(store, db, indeces, temp_fasta, distribution_factor))

                results = ray.get(procs)
                remote_sequence_maps: dict[str, list[tuple[str, str, float, float]]]
                for remote_sequence_maps, remote_n_mapped_sequences, remote_db in results:
                    for gene in remote_sequence_maps:
                        if gene in sequence_maps[remote_db]:
                            sequence_maps[remote_db][gene] += remote_sequence_maps[gene]
                        else:
                            sequence_maps[remote_db][gene] = remote_sequence_maps[gene]
                    n_mapped_sequences += remote_n_mapped_sequences

            os.remove(temp_fasta)

        else:
            config.logger.info(f"Skipped mmseqs2 because of: {returncode}")
            for db in dbs:
                sequence_maps[db] = {}
    return sequence_maps, n_mapped_sequences


def local_ali_pipeline(config, samples, n_of_processes=6, update_mode=False):
    if config.verbosity >= 2:
        t0 = time.time()
        config.logger.info(f"Call of getSequenceFeatures with MSA DB: {config.msa_db}")

    msa_dbs = config.msa_dbs
    gpw_dbs = config.gpw_dbs
    msa_map = {}
    gpw_map = {}

    u_acs = set([])
    pdb_ids = set()

    for u_ac, aac in samples.samples:
        if u_ac.count(":") > 0:
            pdb_ids.add(u_ac)
        else:
            u_acs.add(u_ac)
            # Additionally add the canonical sequences
            if u_ac.count("-") > 0:
                u_acs.add(u_ac.split("-")[0])

    dbs = consts.refseq_datasets

    gene_seq_map: dict[str, list[str]]

    gene_seq_map, in_db = parseFromFasta(config.path_to_sequence_fasta, config=config, dbs=dbs)
    if config.verbosity >= 2:
        config.logger.info(f"Parsed protein sequences from: {config.path_to_sequence_fasta}\nLength of the map:{len(gene_seq_map)}, Length of in_db: {len(in_db)}")

    if config.verbosity >= 2:
        config.logger.info(f"Query proteins that are in_db:\n{in_db=}")

    N = 0
    mmseq_searchs: list[dict[str, list[str]]] = []
    mmseq_search = {}
    msa_to_process = []
    for primary_protein_id in gene_seq_map:
        msa_to_process.append(primary_protein_id)
        if primary_protein_id not in in_db:
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
        config.logger.info(f"getSequenceFeature part 1: {t1 - t0}")

    mmseqs2_search_dbs = {"ref50": config.mmseqs_search_db_ref50, "ref90": config.mmseqs_search_db_ref90, "ref100": config.mmseqs_search_db_ref100}

    sequence_maps, n_mapped_sequences = do_mmseqs_search(config, mmseq_searchs, dbs, mmseqs2_search_dbs)

    if config.verbosity >= 2:
        t2 = time.time()
        config.logger.info(f"getSequenceFeature part 2: {t2 - t1}")
    

    if config.verbosity >= 2:
        t3 = time.time()
        config.logger.info(f"getSequenceFeature part 3: {t3 - t2}")

    logging_level = 0
    if config.verbosity >= 4:
        logging_level = 20
    ray_init(config, overwrite_logging_level=logging_level)

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

        if config.verbosity >= 3:
            config.logger.info(f"Adding {primary_protein_id=} to cost_map  with {cost=} {seq_len=} {n_of_mapped_seqs=}")

    prots_sorted_by_cost = sorted(cost_tuples, key=lambda x: x[1], reverse=True)

    optimal_cost = total_cost / (10 * n_of_processes)

    n = 0

    chunks = []
    current_chunk = 0

    sequence_store = {}

    for prot_id, cost in prots_sorted_by_cost:
        if len(chunks) <= current_chunk:
            chunks.append([0, [], {}, {}])
        assigned = False
        while not assigned:
            if chunks[current_chunk][0] <= optimal_cost:
                chunks[current_chunk][0] += cost
                chunks[current_chunk][1].append(prot_id)
                chunks[current_chunk][2][prot_id] = gene_seq_map[prot_id]
                if prot_id not in chunks[current_chunk][3]:
                    chunks[current_chunk][3][prot_id] = {}

                for db in dbs:
                    if db not in chunks[current_chunk][3][prot_id]:
                        chunks[current_chunk][3][prot_id][db] = []
                    if prot_id in sequence_maps[db]:
                        for hit_id in sequence_maps[db][prot_id]:
                            if hit_id not in sequence_store:
                                sequence_store[hit_id] = ray.put(sequence_maps[db][prot_id][hit_id])
                            chunks[current_chunk][3][prot_id][db].append(hit_id)
                            n += 1
                assigned = True
            current_chunk += 1
            
            if current_chunk >= len(chunks):
                chunks.append([0, [], {}, {}])


    store = ray.put((config, update_mode, in_db, msa_dbs, gpw_dbs))

    com_queue = Queue()

    if config.verbosity >= 2:
        config.logger.info(f"Going into ray_paraMSA, msa_dbs: {msa_dbs}, gpw_dbs: {gpw_dbs}, optimal cost: {optimal_cost}, number of chunks: {len(chunks)}")

    ray_process_ids = []
    started_procs: int = 0
    for chunk_nr, chunk in enumerate(chunks):
        chunk_cost = chunk[0]
        if optimal_cost > 0:
            sub_threads = min([n_of_processes, max([(chunk_cost // optimal_cost), 1])])
        else:
            sub_threads = 1
        if config.verbosity >= 2:
            config.logger.info(f"Starting a ray_paraMSA thread, chunk cost: {chunk_cost}, sub_threads: {sub_threads}")
        ray_process_ids.append(ray_paraMSA.remote(store, sequence_store, chunk[1], chunk[2], chunk[3], sub_threads, com_queue))
        started_procs += sub_threads
        if started_procs >= n_of_processes:
            break

    if config.verbosity >= 1:
        config.logger.info(f"Amount of total mapped sequences: {n_mapped_sequences}")
        config.logger.info(f"Amount of alignments: {n}")

    while True:
        ready, not_ready = ray.wait(ray_process_ids, timeout=0.1)
        new_ray_process_ids = []

        freed_processes = 0

        while not com_queue.empty():
            freed_processes += com_queue.get()

        if len(ready) > 0:
            para_results = ray.get(ready)
            for packed_results, sub_threads in para_results:
                for prot_id, msas, gpws in unpack(packed_results):
                    if prot_id not in msa_map:
                        msa_map[prot_id] = {}
                        gpw_map[prot_id] = {}
                    for db in msas:
                        msa_map[prot_id][db] = msas[db]
                    for db in gpws:
                        gpw_map[prot_id][db] = gpws[db]
                freed_processes += 1
        
        started_procs -= freed_processes

        if freed_processes > 0 and chunk_nr < (len(chunks) - 1):

            cni = chunk_nr + 1
            for chunk in chunks[cni:]:
                chunk_nr += 1
                chunk_cost = chunk[0]
                try:
                    sub_threads = min([freed_processes, max([(chunk_cost // optimal_cost), 1])])
                except ZeroDivisionError:
                    sub_threads = freed_processes
                if config.verbosity >= 2:
                    config.logger.info(f"Starting2 a ray_paraMSA thread, chunk cost: {chunk_cost}, sub_threads: {sub_threads}")
                new_ray_process_ids.append(ray_paraMSA.remote(store, sequence_store, chunk[1], chunk[2], chunk[3], sub_threads, com_queue))
                started_procs += sub_threads
                if started_procs >= n_of_processes:
                    break

        ray_process_ids = new_ray_process_ids + not_ready

        if len(ray_process_ids) == 0:
            break

    if config.verbosity >= 2:
        t4 = time.time()
        config.logger.info(f"getSequenceFeature part 4: {t4 - t3}")

    return msa_map, gpw_map

def parseGemmeFileToMap(infile, gemme_pred_type, seq, config, prot_id, prot_mut_map, feature_name_tag, db_name, none_value=None):
    f = open(infile, 'r')
    lines = f.readlines()
    f.close()

    feat_name = f'gemme_{gemme_pred_type} {db_name} {feature_name_tag}'

    config.logger.info(f'Call of parseGemmeFileMap: {infile=} {feat_name=}')

    for line in lines[1:]:
        words = line[:-1].split()
        mut_aa = words[0][1:-1].upper()
        for seq_pos, gemme_val_str in enumerate(words[1:]):
            
            try:
                wt_aa = seq[seq_pos]
            except IndexError as e:
                config.logger.info(f'IndexError found:\n{prot_id=} {len(seq)=} {infile=} {line=}')
                os.remove(infile)
                return None
                #raise e
            aac = f'{wt_aa}{seq_pos+1}{mut_aa}'
            
            if gemme_val_str == 'NA':
                gemme_val = none_value
            else:
                gemme_val = float(gemme_val_str)

            if aac in prot_mut_map:
                prot_mut_map[aac].append((gemme_val, feat_name))

def parseGemmeFile(config, infile, prot_id: str, samples: SampleSpace, gemme_pred_type, ori_prot_id):
    seq = samples.sequence_map[ori_prot_id][0]

    f = open(infile, 'r')
    lines = f.readlines()
    f.close()

    feat_name = f'gemme_{gemme_pred_type}'
    value_map = {}

    for line in lines[1:]:
        words = line[:-1].split()
        mut_aa = words[0][1:-1].upper()
        for seq_pos, gemme_val_str in enumerate(words[1:]):
            
            try:
                wt_aa = seq[seq_pos]
            except IndexError as e:
                config.logger.info(f'IndexError found:\n{prot_id=} {len(seq)=} {infile=} {line=}')
                os.remove(infile)
                return None
                #raise e
            aac = f'{wt_aa}{seq_pos+1}{mut_aa}'
            if (ori_prot_id, aac) not in samples.samples:
                continue

            if gemme_val_str == 'NA':
                gemme_val = None
            else:
                gemme_val = float(gemme_val_str)

            value_map[aac] = gemme_val

            samples.addValue((ori_prot_id, aac), gemme_val, feat_name)
    return value_map

@ray.remote
def para_psic(package, config):
    util.reset_logger_for_remotes(config)
    for msa_path, psic_name in package:
        f = open(msa_path, "r")
        try:
            msa = f.read()
        except UnicodeDecodeError as e:
            print(f'Error reading msa: {msa_path=}')
            config.logger.info(f'Error reading msa: {msa_path=}')
            raise e
        f.close()
        psicFromFasta(msa, psic_name, config)

def clean_msa(msa_page, overwrite = False):
    outlines = []
    first_entry = True
    first_seq = []
    current_seq = []
    gap_mask = []
    for line in msa_page.splitlines(True):
        if first_entry:
            if line[0] == '>':
                if len(outlines) > 0:
                    masked_seq = []
                    first_seq = ''.join(first_seq)
                    for char in first_seq:
                        if char == '-':
                            gap_mask.append(True)
                        else:
                            gap_mask.append(False)
                            masked_seq.append(char)
                    masked_seq = ''.join(masked_seq)
                    if len(first_seq) == len(masked_seq) and not overwrite:
                        return msa_page

                    if masked_seq == len(masked_seq) * masked_seq[0]:
                        del outlines[-1]
                    else:
                        outlines.append(f'{masked_seq}\n')
                    first_entry = False
                outlines.append(line)
            else:
                first_seq.append(line[:-1])
        else:
            if line[0] == '>':
                masked_seq = []
                current_seq = ''.join(current_seq)
                for pos, char in enumerate(current_seq):
                    if not gap_mask[pos]:
                        masked_seq.append(char)
                masked_seq = ''.join(masked_seq)
                if masked_seq == len(masked_seq) * masked_seq[0]:
                    del outlines[-1]
                else:
                    outlines.append(f'{masked_seq}\n')
                current_seq = []
                outlines.append(line)
            else:
                current_seq.append(line[:-1])

    if len(current_seq) > 0:
        masked_seq = []
        current_seq = ''.join(current_seq)
        for pos, char in enumerate(current_seq):
            if not gap_mask[pos]:
                masked_seq.append(char)
        masked_seq = ''.join(masked_seq)
        if masked_seq == len(masked_seq) * masked_seq[0]:
            del outlines[-1]
        else:
            outlines.append(f'{masked_seq}\n')

    return ''.join(outlines)

def call_gemme(chunk, msa_folder_path, proc_id, overwrite, clean = True, scope = 1):
    lines = []
    for msa_path in chunk:
        if clean:
            f = open(f'{msa_folder_path}/{msa_path}', 'r')
            page = f.read()
            f.close()

            msa = clean_msa(page, overwrite=overwrite)

            f = open(f'{msa_folder_path}/{msa_path}', 'w')
            f.write(msa)
            f.close()

        subfolder, msaf = msa_path.rsplit('/',1)
        lines.append(f'cd "{subfolder}"\n')
        lines.append(f'echo "{msaf}"\n')
        #lines.append(f'head -n 1 "{msaf}"\n')
        line = f'python2.7 $GEMME_PATH/gemme.py "{msaf}" -r input -f "{msaf}"\n'
        lines.append(line)

        if scope == 1:
            lines.append('cd ..\n')
        elif scope == 2:
            lines.append('cd ../..\n')

    f = open(f'{msa_folder_path}/call_gemme_inside_container_{proc_id}.sh', 'w')
    f.write(''.join(lines))
    f.close()

    f = Path(f'{msa_folder_path}/call_gemme_inside_container_{proc_id}.sh')
    f.chmod(f.stat().st_mode | stat.S_IEXEC)

    cmds = ' '.join([
        "docker", "run", "--rm", "--mount", "type=bind,source=$PWD,target=/project",
        "elodielaine/gemme:gemme",
        "bash", f"./call_gemme_inside_container_{proc_id}.sh", "exit"
        ])

    p = subprocess.Popen(cmds, shell=True, cwd=msa_folder_path)
    p.wait()

@ray.remote
def para_gemme(chunk, msa_folder_path, proc_id, overwrite, clean, scope):
    call_gemme(chunk, msa_folder_path, proc_id, overwrite, clean = clean, scope = scope)
    

def calc_gemme_feats(config, samples, prot_id_back_map):
    gene_seq_map = parseFasta(config.path_to_sequence_fasta)
    prot_ids = list(gene_seq_map.keys())
    msa_file_dict = parse_msa_folder(prot_ids, gene_seq_map, config)
    config.logger.info(f'{len(msa_file_dict)=}')

    chunks = []
    current_chunk = 0
    for prot_id in msa_file_dict:
        msa_path = msa_file_dict[prot_id]

        if len(chunks) == current_chunk:
            chunks.append([])
        chunks[current_chunk].append(msa_path)

        current_chunk += 1
        if current_chunk >= config.proc_n:
            current_chunk = 0

    process_ids = []
    for proc_id, chunk in enumerate(chunks):
        process_ids.append(para_gemme.remote(chunk, config.msa_folder_path, proc_id, config.overwrite, True, 1))

    ray.get(process_ids)        

    for prot_id in os.listdir(config.msa_folder_path):
        subfolder = f'{config.msa_folder_path}/{prot_id}'
        if not os.path.isdir(subfolder):
            continue
        for sfn in os.listdir(subfolder):
            if sfn[-10:] == '_msa.fasta':
                continue
            else:
                tokens = sfn.split('_')
                if len(tokens) > 2:
                    if tokens[1] == 'normPred' and sfn[-4:] == '.txt':
                        gemme_pred_type = tokens[2].split('.')[0][4:]
                        try:
                            mapped_id = prot_id_back_map[prot_id]
                        except KeyError:
                            continue
                        _ = parseGemmeFile(config, f'{subfolder}/{sfn}', prot_id, samples, gemme_pred_type, mapped_id)

@ray.remote
def para_mafft_remote_wrapper(chunk, config):
    para_mafft(chunk, config)

def para_mafft(chunk, config):
    util.reset_logger_for_remotes(config)
    for prot_id, tag, sequence_map, seq, target_file in chunk:
        if target_file is None:
            target_folder = f'{config.msa_folder_path}/{clean_prot_id(prot_id)}'
            if not os.path.isdir(target_folder):
                os.makedirs(target_folder)
            sfn = f'{clean_prot_id(prot_id)}_msa.fasta'
            target_file = f'{target_folder}/{sfn}'

        single_mafft(config, prot_id, tag, sequence_map, target_file, seq)
        

def single_mafft(config, prot_id, tag, seq_list, target_file, seq=None):
    config.logger.info(f'Calc MSA for {prot_id} {tag}')

    msa, _, _ = computeMSA(
        seq,
        prot_id,
        search_db=tag,
        keep_raw_seq_map=True,
        sequence_map=seq_list,
        sub_threads=config.proc_n,
        target_file = target_file
    )

    if msa is None:
        return

    #config.logger.info(f'MSA for {prot_id} {seq_list_id} {msa=}')

    if len(seq_list) > 1:
        msa = clean_msa(msa)

    f = open(target_file, 'w')
    f.write(msa)
    f.close()

def gpw_pipeline(config, samples: SampleSpace):
    gpw_map = {}
    psic_jobs = []
    current_job = 0
    config_store = ray.put(config)

    prot_id_back_map = {}
    for prot_id in samples.sequence_map:
        cl_pr_id = clean_prot_id(prot_id)
        prot_id_back_map[cl_pr_id] = prot_id

    if not os.path.isdir(config.msa_folder_path):
        os.makedirs(config.msa_folder_path)

    for prot_id in os.listdir(config.msa_folder_path):
        subfolder = f'{config.msa_folder_path}/{prot_id}'
        if not os.path.isdir(subfolder):
            continue
        
        distant_folder = f'{subfolder}/gpw_distant'

        if not os.path.isdir(distant_folder):
            os.makedirs(distant_folder)

        for sfn in os.listdir(distant_folder):
            if sfn.endswith('_gpw_distant.fasta'):
                gpw_distant_file = f'{distant_folder}/{sfn}'

                gpw_map[prot_id] = {'gpw_distant': gpw_distant_file}
                psic_distant_file = f'{gpw_distant_file[:-6]}.psic'
                check_psic_file(psic_distant_file)

                if not os.path.isfile(psic_distant_file):
                    if config.verbosity >= 3:
                        config.logger.info(f"Calc psic profiles from gpw_pipeline {sfn}")
                    if len(psic_jobs) == current_job:
                        psic_jobs.append([])
                    psic_jobs[current_job].append((gpw_distant_file, psic_distant_file))
                    current_job += 1
                    if current_job >= config.proc_n:
                        current_job = 0

        close_folder = f'{subfolder}/gpw_close'

        if not os.path.isdir(close_folder):
            os.makedirs(close_folder)

        for sfn in os.listdir(close_folder):
            if sfn.endswith('_gpw_close.fasta'):
                gpw_close_file = f'{close_folder}/{sfn}'

                if prot_id not in gpw_map:
                    gpw_map[prot_id] = {}

                gpw_map[prot_id]['gpw_close'] = gpw_close_file
                
                psic_close_file = f'{gpw_close_file[:-6]}.psic'
                check_psic_file(psic_close_file)
                if not os.path.isfile(psic_close_file):
                    if config.verbosity >= 3:
                        config.logger.info(f"Calc psic profiles from gpw_pipeline {sfn}")
                    if len(psic_jobs) == current_job:
                        psic_jobs.append([])
                    psic_jobs[current_job].append((gpw_close_file, psic_close_file))
                    current_job += 1
                    if current_job >= config.proc_n:
                        current_job = 0

    N = 0
    mmseq_searchs = []
    mmseq_search = {}

    for prot_id in samples.sequence_map:
        if not config.overwrite:
            if clean_prot_id(prot_id) in gpw_map or prot_id in gpw_map:
                continue
        mmseq_search[prot_id] = samples.sequence_map[prot_id]

        N += 1
        if N == 5000:
            mmseq_searchs.append(mmseq_search)
            mmseq_search = {}
            N = 0

    if N > 0:
        mmseq_searchs.append(mmseq_search)

    db_id = 'ref90'
    mmseqs2_search_dbs = {db_id: config.mmseqs_search_db_ref90}

    sequence_maps, n_mapped_sequences = do_mmseqs_search(config, mmseq_searchs, [db_id], mmseqs2_search_dbs)


    seq_lists = []
    for prot_id in sequence_maps[db_id]:

        if config.verbosity >= 3:
            config.logger.info(f'Subset slicing: {prot_id=}')

        seq_list = sequence_maps[db_id][prot_id]
        seq_list = msa.rank_and_cut(seq_list, max_number=None)

        try:
            seq = samples.sequence_map[prot_id][0]
        except KeyError:
            try:
                seq = samples.sequence_map[prot_id_back_map[prot_id]][0]
            except KeyError:
                config.logger.info(f'Warning in gpw_pipeline - {prot_id=} not in sampleSpace')
                continue

        slice_subset(seq_lists, seq_list, 0.85, 25., 80., prot_id, seq, db_id, seq_list_id='gpw_distant')
        slice_subset(seq_lists, seq_list, 1., 50., 100., prot_id, seq, db_id, seq_list_id='gpw_close')

    chunks = []
    current_chunk = 0

    for prot_id, seq_list_id, seq_list in seq_lists:
        target_folder = f'{config.msa_folder_path}/{clean_prot_id(prot_id)}'
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)

        sub_folder = f'{target_folder}/{seq_list_id}'

        if not os.path.isdir(sub_folder):
            os.makedirs(sub_folder)

        psic_file = f'{sub_folder}/{clean_prot_id(prot_id)}_{seq_list_id}.psic'
        target_file = f'{sub_folder}/{clean_prot_id(prot_id)}_{seq_list_id}.fasta'

        if not os.path.exists(target_file) or config.overwrite:
            calcGPW(seq_list, target_file, sub_threads = config.proc_n)

        check_psic_file(psic_file)
        if not os.path.isfile(psic_file) or config.overwrite:
            if config.verbosity >= 3:
                config.logger.info(f"Calc psic profiles from gpw_pipeline {target_file=}")
            if len(psic_jobs) == current_job:
                psic_jobs.append([])
            psic_jobs[current_job].append((target_file, psic_file))
            current_job += 1
            if current_job >= config.proc_n:
                current_job = 0        

        if prot_id not in gpw_map:
            gpw_map[prot_id] = {}
        gpw_map[prot_id][seq_list_id] = target_file

        gemme_files = []

        for fi in os.listdir(sub_folder):
            if fi[-4:] != '.txt':
                continue
            if fi.count('normPred') > 0:
                gemme_files.append(fi)

        if len(gemme_files) == 0:
            path_pieces = target_file.split('/')
            path_part = '/'.join(path_pieces[-3:])

            if len(chunks) == current_chunk:
                chunks.append([])
            chunks[current_chunk].append(path_part)

            current_chunk += 1
            if current_chunk >= config.proc_n:
                current_chunk = 0

    process_ids = []
    for proc_id, chunk in enumerate(chunks):
        process_ids.append(para_gemme.remote(chunk, config.msa_folder_path, proc_id, config.overwrite, False, 2))

    proc_ids = []
    for package in psic_jobs:
        proc_ids.append(para_psic.remote(package, config_store))
    ray.get(proc_ids)

    return gpw_map, prot_id_back_map

def afdb_msa_pipeline(config, samples: SampleSpace):
    msa_map = {}
    psic_jobs = []
    current_job = 0
    config_store = ray.put(config)

    prot_id_back_map = {}
    for prot_id in samples.sequence_map:
        cl_pr_id = clean_prot_id(prot_id)
        prot_id_back_map[cl_pr_id] = prot_id

    if not os.path.isdir(config.msa_folder_path):
        os.makedirs(config.msa_folder_path)

    for prot_id in os.listdir(config.msa_folder_path):
        subfolder = f'{config.msa_folder_path}/{prot_id}'
        if not os.path.isdir(subfolder):
            continue
        to_remove = []
        for sfn in os.listdir(subfolder):
            if sfn[-10:] == '_msa.fasta':
                if prot_id in msa_map:
                    continue
                msa_file = f'{subfolder}/{sfn}'
                try:
                    seq = samples.sequence_map[prot_id][0]
                except KeyError:
                    try:
                        seq = samples.sequence_map[prot_id_back_map[prot_id]][0]
                    except KeyError:
                        config.logger.info(f'Warning in afdb_msa_pipeline - {prot_id=} not in sampleSpace {msa_file=}')
                        continue
                checked = check_msa_file(msa_file, seq)
                if checked:
                    msa_map[prot_id] = {'smsa' : msa_file}
                    psic_name = f'{subfolder}/{sfn[:-6]}.psic'
                    check_psic_file(psic_name)
                    if not os.path.isfile(psic_name) or config.overwrite:
                        if config.verbosity >= 3:
                            config.logger.info(f"Calc psic profiles from afdb_msa_pipeline {sfn}")
                        if len(psic_jobs) == current_job:
                            psic_jobs.append([])
                        psic_jobs[current_job].append((msa_file, psic_name))
                        current_job += 1
                        if current_job >= config.proc_n:
                            current_job = 0
                else:
                    config.logger.info(f'Need to remove: {msa_file=}')
                    to_remove.append(msa_file)
            """
            else:
                tokens = sfn.split('_')
                if len(tokens) > 2:
                    if tokens[1] == 'normPred' and sfn[-4:] == '.txt':
                        gemme_pred_type = tokens[2].split('.')[0][4:]
                        
                        value_map = parseGemmeFile(config, f'{subfolder}/{sfn}', prot_id, samples, gemme_pred_type, prot_id_back_map[prot_id])
                        if value_map is None:
                            continue
                        if prot_id not in gemme_predictions:
                            gemme_predictions[prot_id] = {}
                        gemme_predictions[prot_id][gemme_pred_type] = value_map
            """
        #config.logger.info(f'Removing invalid msa files: {to_remove}')
        for msa_file in to_remove:
            os.remove(msa_file)


    N = 0
    mmseq_searchs = []
    mmseq_search = {}

    for prot_id in samples.sequence_map:
        if clean_prot_id(prot_id) in msa_map or prot_id in msa_map:
            continue
        mmseq_search[prot_id] = samples.sequence_map[prot_id]

        N += 1
        if N == 5000:
            mmseq_searchs.append(mmseq_search)
            mmseq_search = {}
            N = 0

    if N > 0:
        mmseq_searchs.append(mmseq_search)

    dbs = ['ref90']
    mmseqs2_search_dbs = {"ref90": config.mmseqs_search_db_ref90}

    sequence_maps, n_mapped_sequences = do_mmseqs_search(config, mmseq_searchs, dbs, mmseqs2_search_dbs)

    n_of_para_maffts = 1 #min([2, config.proc_n])

    current_chunk = 0
    chunks = []
    for prot_id in sequence_maps['ref90']:
        if len(chunks) == current_chunk:
            chunks.append([])

        chunks[current_chunk].append((prot_id, 'ref90', sequence_maps['ref90'][prot_id], samples.sequence_map[prot_id], None))
        current_chunk += 1
        if current_chunk >= n_of_para_maffts:
            current_chunk = 0

    process_ids = []
    for chunk in chunks:
        if n_of_para_maffts > 1:
            process_ids.append(para_mafft_remote_wrapper.remote(chunk, config_store))
        else:
            para_mafft(chunk, config)

    if n_of_para_maffts > 1:
        ray.get(process_ids)
        
    for msa_index, prot_id in enumerate(sequence_maps['ref90']):
        target_folder = f'{config.msa_folder_path}/{clean_prot_id(prot_id)}'
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
        for sfn in os.listdir(target_folder):
            if sfn[-10:] == '_msa.fasta':
                psic_name = f'{target_folder}/{sfn[:-6]}.psic'
                check_psic_file(psic_name)
                if not os.path.isfile(psic_name) or config.overwrite:
                    if config.verbosity >= 3:
                        config.logger.info(f"Calc psic profiles from afdb_msa_pipeline {sfn}")
                    if len(psic_jobs) == current_job:
                        psic_jobs.append([])
                    psic_jobs[current_job].append((f'{target_folder}/{sfn}', psic_name))
                    current_job += 1
                    if current_job >= config.proc_n:
                        current_job = 0        

        msa_map[prot_id] = {'smsa' : f'{target_folder}/{sfn}'}

    proc_ids = []
    for package in psic_jobs:
        proc_ids.append(para_psic.remote(package, config_store))
    ray.get(proc_ids)

    calc_gemme_feats(config, samples, prot_id_back_map)

    return msa_map, prot_id_back_map


def get_uniref_members_from_upi(upi):

    params = {
        "id": upi
    }
    headers = {
        "accept": "application/json"
    }
    base_url = "https://rest.uniprot.org/uniref/%7Bid%7D/members"

    response = requests.get(base_url, headers=headers, params=params)
    if not response.ok:
        #response.raise_for_status()
        return []

    data = response.json()

    u_ids = []
    u100s = set()
    for entry in data['results']:
        u100_id = entry['uniref100Id']
        if u100_id in u100s:
            continue
        u100s.add(u100_id)
        u_ids.append(entry['memberId'])

    return u_ids

def get_uniprot_sequences(id_list):

    max_retrieve = 50
    seq_list = []

    iter_index = 0
    while iter_index*max_retrieve < len(id_list):
        id_list_chunk = id_list[iter_index*max_retrieve:(iter_index+1)*max_retrieve]
        iter_index += 1

        params = {
            "query": ' OR '.join(id_list_chunk),
            "fields": [
                "id",
                "sequence"
        ]
        }
        headers = {
            "accept": "application/json"
        }
        base_url = "https://rest.uniprot.org/uniprotkb/stream"

        response = requests.get(base_url, headers=headers, params=params)
        if not response.ok:
            response.raise_for_status()
            return None

        data = response.json()

        
        id_set = set(id_list_chunk)

        for entry in data['results']:
            u_id = entry['uniProtkbId']
            if u_id not in id_set:
                continue
            seq = entry['sequence']['value']

            seq_list.append((u_id, seq, 1.0, 100.0))
        
    return seq_list

def parse_gpw_fasta(gpw_fasta_file):
    seq_list = []
    f = open(gpw_fasta_file, 'r')
    lines = f.readlines()
    f.close()

    target_prot_id = lines[0][1:-1]
    target_seq = lines[1][:-1]

    seq_list.append((target_prot_id, target_seq, 1.0, 100.0))

    for line in lines[2:]:
        if line[0] == '>':
            entry_id = line[1:-1]
        else:
            aligned_seq = line[:-1]

            n_matched = 0
            n_gaps = 0
            seq = []
            for al_index, al_char in enumerate(aligned_seq):
                if al_char == '-':
                    n_gaps += 1
                elif al_char == target_seq[al_index]:
                    n_matched += 1
                if al_char != '-':
                    seq.append(al_char)
            seq_id = (100*n_matched)/len(aligned_seq)
            cov = (len(aligned_seq) - n_gaps)/(len(aligned_seq))

            seq_list.append((entry_id, ''.join(seq), cov, seq_id))

    return seq_list

def slice_subset(seq_lists, ordered_seq_list, cov_thresh, seq_id_min, seq_id_max, prot_id, target_seq, id_tag, seq_list_id = None):
    seq_id_capped_list = [entry for entry in ordered_seq_list if entry[3] > seq_id_min and entry[3] < seq_id_max and entry[2] >= cov_thresh]
    if len(seq_id_capped_list) == 0:
        return
    if seq_id_capped_list[0][3] < 100. and seq_id_capped_list[0][2] < 1.0:
        seq_id_capped_list = [(clean_prot_id(prot_id), target_seq, 1.0, 100.0)] + seq_id_capped_list
    else:
        seq_id_capped_list[0] = (clean_prot_id(prot_id), target_seq, 1.0, 100.0)

    if seq_list_id is None:
        seq_list_id = f'seq_id_{int(seq_id_min)}-{int(seq_id_max)}_cov_{int(cov_thresh*100)}_{id_tag}'
    seq_lists.append((clean_prot_id(prot_id), seq_list_id, seq_id_capped_list))

def prepare_seq_lists(config, samples: SampleSpace):
    #xml_file = f'{os.path.dirname(config.mmseqs_search_db_ref50)}/uniref90.xml.gz'
    
    mmseqs2_search_dbs = {"ref50": config.mmseqs_search_db_ref50, "ref90": config.mmseqs_search_db_ref90}#, "ref100": config.mmseqs_search_db_ref100}

    #mmseqs2_search_dbs = {"ref90": config.mmseqs_search_db_ref90}

    dbs = mmseqs2_search_dbs.keys()

    mmseq_searchs = []
    mmseq_search = {}

    for prot_id in samples.sequence_map:
        mmseq_search[prot_id] = samples.sequence_map[prot_id]

    mmseq_searchs.append(mmseq_search)

    sequence_maps, n_mapped_sequences = do_mmseqs_search(config, mmseq_searchs, dbs, mmseqs2_search_dbs)

    #print(f'{sequence_maps=}')

    seq_lists = []

    for db_id in sequence_maps:
        for prot_id in sequence_maps[db_id]:
            seq_list = sequence_maps[db_id][prot_id]
            #for entry in seq_list:
            #    config.logger.info((f'{db_id=} {prot_id=} {entry=} '))
            ordered_seq_list = msa.rank_and_cut(seq_list)

            target_cluster_head = ordered_seq_list[0][0]
            target_seq = ordered_seq_list[0][1]

            #all_members = []
            #for entry in ordered_seq_list:
            #    all_members += get_uniref_members_from_upi(entry[0])

            ordered_seq_list[0] = (prot_id, ordered_seq_list[0][1], ordered_seq_list[0][2], ordered_seq_list[0][3])

            seq_lists.append((prot_id, f'raw_{db_id}', ordered_seq_list))

            min_seq_ids = [0., 25., 30., 35., 40., 50., 70., 80., 90.]
            max_seq_ids = [70., 80., 90., 95., 99., 100.]
            cov_threshs = [0., 0.8, 0.85, 0.9, 0.95, 1.]

            thresh_combinations = []

            for min_seq_id in min_seq_ids:
                for max_seq_id in max_seq_ids:
                    if min_seq_id >= max_seq_id:
                        continue
                    for cov_thresh in cov_threshs:
                        thresh_combinations.append((min_seq_id, max_seq_id, cov_thresh))

            for (min_seq_id, max_seq_id, cov_thresh) in thresh_combinations:
                slice_subset(seq_lists, ordered_seq_list, cov_thresh, min_seq_id, max_seq_id, prot_id, target_seq, db_id)

           
            #member_seqs = get_uniprot_sequences(all_members)

            target_folder = f'{config.msa_folder_path}/{clean_prot_id(prot_id)}'
            all_member_folder = f'{target_folder}/gpw_all_members_{db_id}/'
            all_member_gpw_fasta_file = f'{all_member_folder}/{prot_id}_gpw_all_members_{db_id}.fasta'

            member_seqs = parse_gpw_fasta(all_member_gpw_fasta_file)

            for (min_seq_id, max_seq_id, cov_thresh) in thresh_combinations:
                slice_subset(seq_lists, member_seqs, cov_thresh, min_seq_id, max_seq_id, prot_id, target_seq, f'member_{db_id}')


    return seq_lists

def make_msas(seq_lists, config, overwrite = False, make_plots=False):
    msa_lists = []

    n_of_para_maffts = 10 #min([2, config.proc_n])

    current_chunk = 0
    chunks = []

    for prot_id, seq_list_id, seq_list in seq_lists:
        target_folder = f'{config.msa_folder_path}/{clean_prot_id(prot_id)}'
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
        sfn = f'{clean_prot_id(prot_id)}_msa_{seq_list_id}.fasta'
        
        if not os.path.isdir(f'{target_folder}/{seq_list_id}'):
            os.makedirs(f'{target_folder}/{seq_list_id}')

        if make_plots:
            seq_id_list = sorted([x[3] for x in seq_list[1:]], reverse=True)

            plt.plot([i for i in range(1,len(seq_id_list)+1)], seq_id_list, color = 'blue')

            cov_list = sorted([100.*x[2] for x in seq_list[1:]], reverse=True)

            plt.plot([i for i in range(1,len(cov_list)+1)], cov_list, color = 'red')

            plt.savefig(f'{target_folder}/{seq_list_id}/{clean_prot_id(prot_id)}_{seq_list_id}_seq_id_distribution.png')
            plt.clf()
            plt.cla()
            plt.close()

        target_file = f'{target_folder}/{seq_list_id}/{sfn}'
        if not os.path.exists(target_file) or overwrite:

            if len(chunks) == current_chunk:
                chunks.append([])

            chunks[current_chunk].append((prot_id, seq_list_id, seq_list, None, target_file))
            current_chunk += 1
            if current_chunk >= n_of_para_maffts:
                current_chunk = 0

            #single_mafft(config, prot_id, seq_list_id, seq_list, target_file)

    config_store = ray.put(config)

    process_ids = []
    for chunk in chunks:
        if n_of_para_maffts > 1:
            process_ids.append(para_mafft_remote_wrapper.remote(chunk, config_store))
        else:
            para_mafft(chunk, config)

    if n_of_para_maffts > 1:
        ray.get(process_ids)

    for prot_id, seq_list_id, seq_list in seq_lists:
        target_folder = f'{config.msa_folder_path}/{clean_prot_id(prot_id)}'
        if not os.path.isdir(target_folder):
            os.makedirs(target_folder)
        sfn = f'{clean_prot_id(prot_id)}_msa_{seq_list_id}.fasta'
        psic_file = f'{target_folder}/{seq_list_id}/{clean_prot_id(prot_id)}_msa_{seq_list_id}.psic'
        target_file = f'{target_folder}/{seq_list_id}/{sfn}'

        if not os.path.exists(psic_file) or overwrite:
            f = open(target_file, "r")
            try:
                msa = f.read()
            except UnicodeDecodeError as e:
                print(f'Error reading msa: {target_file=}')
                config.logger.info(f'Error reading msa: {target_file=}')
                raise e
            f.close()
            psicFromFasta(msa, psic_file, config)

        msa_lists.append((prot_id, target_file, psic_file, seq_list_id))

        if not os.path.isdir(f'{target_folder}/gpw_{seq_list_id}'):
            os.makedirs(f'{target_folder}/gpw_{seq_list_id}')

        gpw_path = f'{target_folder}/gpw_{seq_list_id}/{clean_prot_id(prot_id)}_gpw_{seq_list_id}.fasta'

        if not os.path.exists(gpw_path) or overwrite:
            calcGPW(seq_list, gpw_path, sub_threads = config.proc_n)

        gpw_psic_path = f'{target_folder}/gpw_{seq_list_id}/{clean_prot_id(prot_id)}_gpw_{seq_list_id}.psic'

        check_psic_file(gpw_psic_path, target_len=len(seq_list[0][1]))

        if not os.path.exists(gpw_psic_path) or overwrite:
            f = open(gpw_path, "r")
            try:
                gpw = f.read()
            except UnicodeDecodeError as e:
                print(f'Error reading msa: {gpw_path=}')
                config.logger.info(f'Error reading msa: {gpw_path=}')
                raise e
            f.close()
            psicFromFasta(gpw, gpw_psic_path, config)

        msa_lists.append((prot_id, gpw_path, gpw_psic_path, f'gpw_{seq_list_id}'))

    for prot_id in os.listdir(config.msa_folder_path):
        subfolder = f'{config.msa_folder_path}/{prot_id}'
        if not os.path.isdir(subfolder):
            continue
        for sfn in os.listdir(subfolder):
            if sfn[-10:] == '_msa.fasta':
                msa_file = f'{subfolder}/{sfn}'
                
                psic_file = f'{subfolder}/{sfn[:-6]}.psic'
                check_psic_file(psic_file, target_len=len(seq_list[0][1]))
                if not os.path.isfile(psic_file):
                    f = open(msa_file, "r")
                    try:
                        msa = f.read()
                    except UnicodeDecodeError as e:
                        print(f'Error reading msa: {msa_file=}')
                        config.logger.info(f'Error reading msa: {msa_file=}')
                        raise e
                    f.close()
                    psicFromFasta(msa, psic_file, config)

                #af_msa_id = sfn.split('_')[-2] + '_msa'
                af_msa_id = 'af_msa'

                target_msa_file = f'{subfolder}/{af_msa_id}/{sfn}'
                target_psic_file = f'{subfolder}/{af_msa_id}/{sfn[:-6]}.psic'

                if not os.path.isdir(f'{subfolder}/{af_msa_id}/'):
                    os.makedirs(f'{subfolder}/{af_msa_id}/')

                shutil.copy(msa_file, target_msa_file)
                shutil.copy(psic_file, target_psic_file)

                msa_lists.append((prot_id, target_msa_file, target_psic_file, af_msa_id))

    return msa_lists

def featurize(msa_lists: list[tuple[str, str, str, str]], samples: SampleSpace, config):

    #chunk = []
    chunks = []
    current_chunk = 0
    for prot_id, msa_file, psic_file, seq_list_id in msa_lists:
        path_pieces = msa_file.split('/')

        subfolder = '/'.join(path_pieces[:-1])

        gemme_files = []

        for fi in os.listdir(subfolder):
            if fi[-4:] != '.txt':
                continue
            if fi.count('normPred') > 0:
                gemme_files.append(fi)

        if len(gemme_files) == 0:
            path_part = '/'.join(path_pieces[-3:])

            if len(chunks) == current_chunk:
                chunks.append([])
            chunks[current_chunk].append(path_part)

            current_chunk += 1
            if current_chunk >= config.proc_n:
                current_chunk = 0

            #chunk.append(path_part)
        

    process_ids = []
    for proc_id, chunk in enumerate(chunks):
        process_ids.append(para_gemme.remote(chunk, config.msa_folder_path, proc_id, config.overwrite, False, 2))

    #call_gemme(chunk, config.msa_folder_path, 0, None, clean=False, scope=2)

    aac_map = {}
    for u_ac, aac in samples.samples:
        if u_ac not in aac_map:
            aac_map[u_ac] = []
        aac_map[u_ac].append(aac)

    results = []

    for prot_id, msa_file, psic_file, seq_list_id in msa_lists:
        aac_list = aac_map[prot_id]
        prot_mut_map = {}
        for aac in aac_list:
            prot_mut_map[aac] = []

        if msa_file[-3:] == '.gz':
            f = gzip.open(msa_file, "r")
        else:
            f = open(msa_file, 'rb')
        msa_page = f.read()
        f.close()

        if len(msa_page) == 0:
            continue

        seq_map, seed = msa.parseMsaFasta(msa_page)
        try:
            seed_seq = seq_map[seed].replace("-", "")
        except KeyError as err:
            raise KeyError(f'{err=} {prot_id=} {seq_list_id=} {msa_file=} {msa_page=}')
        pos_wise_map = msa.getPosWiseMSA(seq_map, seed)

        feats = fill_prot_mut_map(
            config,
            prot_id,
            aac_list,
            pos_wise_map,
            seed_seq,
            prot_mut_map,
            seq_list_id,
            '',
            True,
            False,
            psic_file
            )
        
        path_pieces = msa_file.split('/')
        subfolder = '/'.join(path_pieces[:-1])

        #print(f'{(prot_id, msa_file, psic_file, seq_list_id)=} {subfolder=}')

        for fi in os.listdir(subfolder):
            if fi[-4:] != '.txt':
                continue
            if fi.count('normPred') > 0:
                gemme_pred_type = fi[:-4].split('_')[-1]
                feat_name = f'gemme_{gemme_pred_type}'
                feats.append(feat_name)
                #print(f'{fi=}')
                parseGemmeFileToMap(f'{subfolder}/{fi}', gemme_pred_type, seed_seq, config, prot_id, prot_mut_map, '', seq_list_id, none_value=0.)

        results.append((prot_id, seq_list_id, prot_mut_map, feats))

    return results

def bench(featurized_lists, samples: SampleSpace, config):
    prot_wise_lists = {}
    for sample_id in samples.samples:
        u_ac,aac = sample_id
        target_value = samples.samples[sample_id].targetValue
        if target_value is None:
            continue

        if u_ac not in prot_wise_lists:
            prot_wise_lists[u_ac] = ([], [])

        prot_wise_lists[u_ac][0].append(aac)
        prot_wise_lists[u_ac][1].append(target_value)

    max_list = {}
    msa_type_max = {}
    gemme_combi_vals = {}
    prot_ids = set()

    spear_lines = ['Protein\tMSA ID\tMSA depth\tFeat name\tSpear\n']

    for (prot_id, seq_list_id, prot_mut_map, feats) in featurized_lists:
        prot_ids.add(prot_id)
        feat_value_lists = {}
        for aac in prot_wise_lists[prot_id][0]:
            for feat_index, feat_name in enumerate(feats):
            #for value, feat_name in prot_mut_map[aac]:
                try:
                    value = prot_mut_map[aac][feat_index][0]
                except IndexError as err:
                    #value = np.nan
                    value = 0.
                    #raise IndexError(f'{err=} {prot_id=} {seq_list_id=} {aac=} {feat_name=} {prot_mut_map[aac]=}')
                if value is None:
                    value = 0.
                if feat_name not in feat_value_lists:
                    feat_value_lists[feat_name] = []
                feat_value_lists[feat_name].append(value)

        msa_depth = feat_value_lists['MSA depth'][0]

        spear_tuples = []

        for feat_name in feat_value_lists:
            feat_vec = feat_value_lists[feat_name]

            try:
                spear, _= stats.spearmanr(prot_wise_lists[prot_id][1] ,feat_vec)
            except (TypeError, ValueError) as err:
                print(f'Catched {err=} {prot_id=} {seq_list_id=} {feat_name=} {len(feat_vec)=} {len(prot_wise_lists[prot_id][1])=} {len(prot_wise_lists[prot_id][0])=}')
                spear = None

            if spear is not None and not np.isnan(spear):
                spear_tuples.append((feat_name, spear))

                if prot_id not in max_list:
                    max_list[prot_id] = (seq_list_id, spear, msa_depth, feat_name)
                elif abs(spear) > abs(max_list[prot_id][1]):
                    max_list[prot_id] = (seq_list_id, spear, msa_depth, feat_name)

                if seq_list_id not in msa_type_max:
                    msa_type_max[seq_list_id] = {}
                    gemme_combi_vals[seq_list_id] = [], {}
                if prot_id not in msa_type_max[seq_list_id]:
                    msa_type_max[seq_list_id][prot_id] = (spear, msa_depth, feat_name)
                elif abs(spear) > msa_type_max[seq_list_id][prot_id][0]:
                    msa_type_max[seq_list_id][prot_id] = (spear, msa_depth, feat_name)

                if feat_name == 'gemme_evolCombi':
                    gemme_combi_vals[seq_list_id][0].append(spear)
                    gemme_combi_vals[seq_list_id][1][prot_id] = feat_vec

            spear_lines.append(f'{prot_id}\t{seq_list_id}\t{msa_depth}\t{feat_name}\t{spear}\n')

            #print(f'{seq_list_id=} {prot_id=} {msa_depth=} {feat_name=} {spear=}')

        spear_tuples = sorted(spear_tuples, key=lambda x:abs(x[1]), reverse=True)

        config.logger.info(f'{seq_list_id=} {prot_id=} {msa_depth=} {spear_tuples[0:3]=}')
    prot_ids = list(prot_ids)

    config.logger.info(f'{max_list=}')

    all_spears_file = f'{config.msa_folder_path}/all_spears.tsv'

    f = open(all_spears_file, 'w')
    f.write(''.join(spear_lines))
    f.close()

    prot_wise_max_file = f'{config.msa_folder_path}/prot_wise_max.tsv'
    lines = ['Prot\tMSA ID\tMSA depth\tFeat name\tSpear\n']
    for prot_id in max_list:
        (seq_list_id, spear, msa_depth, feat_name) = max_list[prot_id]
        lines.append(f'{prot_id}\t{seq_list_id}\t{msa_depth}\t{feat_name}\t{spear}\n')

    f = open(prot_wise_max_file, 'w')
    f.write(''.join(lines))
    f.close()

    avg_msa_type_spears = []

    for seq_list_id in msa_type_max:
        spears = []
        feat_names = set()
        depths = []
        for prot_id in msa_type_max[seq_list_id]:
            spears.append(abs(msa_type_max[seq_list_id][prot_id][0]))
            feat_names.add(msa_type_max[seq_list_id][prot_id][2])
            depths.append(msa_type_max[seq_list_id][prot_id][1])
        avg_spear = sum(spears)/len(spears)
        avg_depth = sum(depths)/len(depths)

        avg_msa_type_spears.append((seq_list_id, avg_spear, avg_depth, feat_names))

    avg_msa_type_spears = sorted(avg_msa_type_spears, key= lambda x:x[1], reverse=True)

    config.logger.info(f'{avg_msa_type_spears=}')

    avg_msa_spears_file = f'{config.msa_folder_path}/avg_msa_spears.tsv'
    lines = ['MSA ID\tAvg MSA depth\tFeat names\tAvg Spear\n']
    for seq_list_id, avg_spear, avg_depth, feat_names in avg_msa_type_spears:
        lines.append(f'{seq_list_id}\t{avg_depth}\t{feat_names}\t{avg_spear}\n')

    f = open(avg_msa_spears_file, 'w')
    f.write(''.join(lines))
    f.close()

    avg_gemme_combi_vals = []
    for seq_list_id in gemme_combi_vals:
        if len(gemme_combi_vals[seq_list_id][0]) == 0:
            continue
        avg = sum(gemme_combi_vals[seq_list_id][0])/len(gemme_combi_vals[seq_list_id][0])
        feat_vecs = []
        try:
            for prot_id in prot_ids:
                feat_vecs.append(gemme_combi_vals[seq_list_id][1][prot_id])
        except KeyError:
            continue
        avg_gemme_combi_vals.append((seq_list_id, avg, feat_vecs))

    combination = []

    for seq_list_id_a, val_a, feat_vecs_a in avg_gemme_combi_vals:
        for seq_list_id_b, val_b, feat_vecs_b in avg_gemme_combi_vals:
            if seq_list_id_a == seq_list_id_b:
                continue
            spears = []
            for index, feat_vec_a in enumerate(feat_vecs_a):
                feat_vec_b = feat_vecs_b[index]
                spear, _= stats.spearmanr(feat_vec_a, feat_vec_b)
                spears.append(spear)

            avg_spear = sum(spears)/len(spears)

            combi_score = abs(val_a) + abs(val_b) - abs(avg_spear)

            combination.append((seq_list_id_a, seq_list_id_b, val_a, val_b, avg_spear, combi_score))

    combination = sorted(combination, key=lambda x:x[5], reverse=True)

    combi_file = f'{config.msa_folder_path}/msa_combinations.tsv'
    lines = ['MSA ID A\tMSA ID B\tAvg Spear A\tAvg Spear B\tAvg Gemme Combi Spear\tCombi score\n']
    for seq_list_id_a, seq_list_id_b, val_a, val_b, avg_spear, combi_score in combination:
        lines.append(f'{seq_list_id_a}\t{seq_list_id_b}\t{val_a}\t{val_b}\t{avg_spear}\t{combi_score}\n')

    f = open(combi_file, 'w')
    f.write(''.join(lines))
    f.close()

    config.logger.info(f'{combination[:100]=}')

def msa_bench(config, samples: SampleSpace):
    t0 = time.time()

    seq_lists = prepare_seq_lists(config, samples)

    t1 = time.time()
    print(f'prepare_seq_lists done: {t1-t0}')

    msa_lists = make_msas(seq_lists, config)

    t2 = time.time()
    print(f'make_msas done: {t2-t1}')

    featurized_lists = featurize(msa_lists, samples, config)

    t3 = time.time()
    print(f'featurize done: {t3-t2}')

    bench(featurized_lists, samples, config)

    t4 = time.time()
    print(f'bench done: {t4-t3}')

def getSequenceFeatures(config, samples, n_of_processes=6, update_mode=False):
    
    
    #config.msa_dbs = ['smsa']
    config.msa_dbs = ['gpw_distant', 'gpw_close']
    config.gpw_dbs = []
    initFeatures(config, samples)
    #msa_map, gpw_map = local_ali_pipeline(config, samples, n_of_processes=n_of_processes, update_mode=update_mode)

    #msa_map, prot_id_back_map = afdb_msa_pipeline(config, samples)
    msa_map, prot_id_back_map = gpw_pipeline(config, samples)
    
    gpw_map = {}

    packages = []
    current_package = 0

    prot_mut_map = {}
    for u_ac, aac in samples.samples:
        if u_ac not in prot_mut_map:
            prot_mut_map[u_ac] = []
        prot_mut_map[u_ac].append(aac)

    for u_ac in prot_mut_map:
        aacs = prot_mut_map[u_ac]

        if config.verbosity >= 6:
            config.logger.info(f"Added to CalcSeqFeat queue: {u_ac=}, {aacs=}")

        if len(packages) == current_package:
            packages.append([])
        packages[current_package].append((u_ac, aacs))
        current_package += 1
        if current_package >= config.proc_n:
            current_package = 0
    
    if len(packages) > 1:
        msa_store = ray.put((msa_map, gpw_map, config))
        processes = []
        for package in packages:
            processes.append(paraCalcSeqFeat.remote(package, msa_store))
        
        results = ray.get(processes)
    elif len(packages) == 1:
        results = [calcSeqFeat(packages[0], msa_map, gpw_map, config)]
    else:
        results = []

    if config.verbosity >= 1:
        config.logger.info("after paraCalcSeqFeat")

    for result in results:
        for u_ac, prot_mut_map in result:
            if u_ac in prot_id_back_map:
                prot_id = prot_id_back_map[u_ac]
            else:
                prot_id = u_ac
            

            target_folder = f'{config.msa_folder_path}/{clean_prot_id(prot_id)}'
            
            seq = samples.sequence_map[prot_id][0]

            for seq_list_id in config.msa_dbs:
                sub_folder = f'{target_folder}/{seq_list_id}'
                target_file = f'{sub_folder}/{clean_prot_id(prot_id)}_{seq_list_id}.fasta'

                path_pieces = target_file.split('/')
                subfolder = '/'.join(path_pieces[:-1])

                if not os.path.exists(subfolder):
                    continue
                    #raise FileNotFoundError(f'{subfolder=} {u_ac=} {prot_id=} {target_folder=} {seq_list_id=} {target_file=}')

                for fi in os.listdir(subfolder):
                    if fi[-4:] != '.txt':
                        continue
                    if fi.count('normPred') > 0:
                        gemme_pred_type = fi[:-4].split('_')[-1]
                        feat_name = f'gemme_{gemme_pred_type}'

                        parseGemmeFileToMap(f'{subfolder}/{fi}', gemme_pred_type, seq, config, prot_id, prot_mut_map, seq_list_id, 'MSA')

            for aac in prot_mut_map:
                for value, feat_name in prot_mut_map[aac]:
                    samples.addValue((prot_id, aac), value, feat_name, config=config)

    return


@ray.remote(max_calls=1)
def ray_paraMSA(store, sequence_store, prot_ids, prot_seq_map, hit_seq_maps, sub_threads, com_queue):
    config, update_mode, in_db, msa_dbs, gpw_dbs = store
    util.reset_logger_for_remotes(config)
    results = []

    for prot_id in prot_ids:
        if prot_id not in prot_seq_map:
            if prot_id.split("-")[0] in prot_seq_map:
                seq = prot_seq_map[prot_id.split("-")[0]][0]
            elif prot_id in in_db:
                seq = None
            else:
                config.logger.info(f"Skipped getMSA for: {prot_id}, It was not in the gene_seq_map")
                continue
        else:
            seq = prot_seq_map[prot_id][0]

        msas, gpws = msa.getMSA(config, prot_id, sequence_store, com_queue, sequence_maps=hit_seq_maps[prot_id], sequence=seq, ref_db_ids=msa_dbs, gpw_ref_db_ids=gpw_dbs, update_mode=update_mode, sub_threads=sub_threads)

        results.append((prot_id, msas, gpws))

    return (pack(results), sub_threads)

@ray.remote
def paraCalcSeqFeat(package, store):
    msa_map, gpw_map, config = store
    util.reset_logger_for_remotes(config)
    return calcSeqFeat(package, msa_map, gpw_map, config)

def calcSeqFeat(package, msa_map, gpw_map, config):
    dbs = []
    for db_id in config.msa_dbs:
        dbs.append((db_id, False))

    for db_id in config.gpw_dbs:
        dbs.append((db_id, True))


    results = []
    for intuple in package:

        (u_ac, aacs) = intuple

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
                feature_name_tag = "MSA"

            if u_ac not in results_map:
                if clean_prot_id(u_ac) in results_map:
                    u_ac = clean_prot_id(u_ac)
                else:
                    if config.verbosity >= 1:
                        config.logger.info(f"Filtered {u_ac}, since it was not in the results_map: {db_name} ({is_gpw=})")
                    continue
            if db_name not in results_map[u_ac]:
                if config.verbosity >= 1:
                    config.logger.info(f"Filtered {u_ac} since db_name {db_name} was not in the results_map[u_ac], {is_gpw=}")
                continue

            gpw_file_path = results_map[u_ac][db_name]
            # config.logger.info(gpw_file_path)

            try:
                if gpw_file_path[-3:] == '.gz':
                    f = gzip.open(gpw_file_path, "r")
                else:
                    f = open(gpw_file_path, 'rb')
                gpw_fasta = f.read()
                f.close()
            except:
                [e, f, g] = sys.exc_info()
                g = traceback.format_exc()
                config.logger.info(f"Error with reading file, path: {gpw_file_path}, protein: {u_ac}, db_name: {db_name}, is_gpw: {is_gpw}\n{e}\n{f}\n{g}")

            if gpw_fasta is None:
                if config.verbosity >= 1:
                    config.logger.info(f"Filtered {u_ac}, since gpw_fasta was None {db_name=}")

                continue
            if is_gpw:
                gpw_ds, seed = msa.parseGpwFasta(gpw_fasta, dict_out=True)

                if len(gpw_ds) == 0:
                    config.logger.info(f"Error, gpw_ds is empty: {u_ac}")

                seed_seq = gpw_ds[list(gpw_ds.keys())[0]][0].replace("-", "")
                try:
                    pos_wise_map = msa.getPosWiseGPW(gpw_ds)
                except:
                    [e, f, g] = sys.exc_info()
                    g = traceback.format_exc()
                    config.logger.info(f"Error in getPosWiseGPW: {u_ac} {db_name} {seed} {gpw_file_path}\n{e}\n{f}\n{g}")
                    continue
                psic_name = None
            else:
                seq_map, seed = msa.parseMsaFasta(gpw_fasta)
                try:
                    seed_seq = seq_map[seed].replace("-", "")
                except KeyError as e:
                    config.logger.info(f'Error: invalid seed: {seed=} {gpw_file_path=}')
                    raise e
                pos_wise_map = msa.getPosWiseMSA(seq_map, seed)
                psic_name = f'{gpw_file_path[:-6]}.psic'

            fill_prot_mut_map(
                config,
                u_ac,
                aacs,
                pos_wise_map,
                seed_seq,
                prot_mut_map,
                feature_name_tag,
                db_name,
                first_db,
                is_gpw,
                psic_name
                )

            first_db = False
        config.logger.info(f'Adding to results: {u_ac=} {len(prot_mut_map)=}')
        results.append((u_ac, prot_mut_map))
    return results

def fill_prot_mut_map(
        config,
        u_ac,
        aacs,
        pos_wise_map,
        seed_seq,
        prot_mut_map,
        feature_name_tag,
        db_name,
        first_db,
        is_gpw,
        psic_name
        ):
    (psic_wt_map, psic_mut_map, dpsic_map, positional_dpsic_map, window_dpsic_map, protein_median_dpsic) = msa.calcPsicProfiles(config, u_ac, aacs, seed_seq, db_name, gpw=is_gpw, psic_name=psic_name)

    config.logger.info(f'After calcPSicProfiles: {u_ac=} {len(aacs)=} {len(psic_wt_map)=} {len(dpsic_map)=}')
    
    err_count = 0
    for aac in aacs:
        aa1 = aac[0]
        aa2 = aac[-1]
        pos = int(aac[1:-1]) - 1
        
        if pos >= len(pos_wise_map):
            if config.verbosity >= 1:
                config.logger.info(f"Filtered {u_ac=}, {pos=} {aac=}, since it was outside of the seed_seq, {len(seed_seq)=} {len(pos_wise_map)=}")
            continue

        if aac not in psic_wt_map:
            if config.verbosity >= 1 and err_count < 5:
                config.logger.info(f"Filtered {u_ac=} {aac=} not in {len(psic_wt_map)=}")
                err_count += 1
            continue

        n_wt_aa = 0.0
        n_mut_aa = 0.0
        n_gap = 0.0
        n_dif_aa = 0.0
        s = float(len(pos_wise_map[0]))
        for al_seq_aa in pos_wise_map[pos]:
            if al_seq_aa == aa1:
                n_wt_aa += 1.0
            elif al_seq_aa == aa2:
                n_mut_aa += 1.0
            elif al_seq_aa == "-":
                n_gap += 1.0
            else:
                n_dif_aa += 1.0
        wt_rate = n_wt_aa / s
        mut_rate = n_mut_aa / s
        gl_wt_rate = n_wt_aa / (s - n_gap)
        gl_mut_rate = n_mut_aa / (s - n_gap)
        coverage = (s - n_gap) / s
        dif_rate = n_dif_aa / s
        gl_dif_rate = n_dif_aa / (s - n_gap)

        prot_mut_map[aac].append((wt_rate, f"Wildtype AA rate {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((mut_rate, f"Mutant AA rate {feature_name_tag} {db_name}"))

        prot_mut_map[aac].append((gl_wt_rate, f"Wildtype AA rate gapless {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((gl_mut_rate, f"Mutant AA rate gapless {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((coverage, f"MSA allel freq {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((dif_rate, f"Other mutant AA rate {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((gl_dif_rate, f"Other mutant AA rate gapless {feature_name_tag} {db_name}"))

        prot_mut_map[aac].append((psic_wt_map[aac], f"PSIC wildtype AA {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((psic_mut_map[aac], f"PSIC mutant AA {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((dpsic_map[aac], f"dPSIC {feature_name_tag} {db_name}"))

        prot_mut_map[aac].append((positional_dpsic_map[aac], f"Positional median dPSIC {feature_name_tag} {db_name}"))
        try:
            wdpsic = window_dpsic_map[aac]
        except KeyError:
            wdpsic = None
        prot_mut_map[aac].append((wdpsic, f"Window median dPSIC {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((protein_median_dpsic, f"Protein median dPSIC {feature_name_tag} {db_name}"))
        prot_mut_map[aac].append((s, f"MSA depth {feature_name_tag} {db_name}"))
        if first_db:
            
            prot_mut_map[aac].append((pos, "Sequence Position Number"))
            prot_mut_map[aac].append((pos / len(seed_seq), "Relative Sequence Position"))
            prot_mut_map[aac].append((len(seed_seq), "Protein Size"))

    feats = [
        f"Wildtype AA rate {feature_name_tag} {db_name}",
        f"Mutant AA rate {feature_name_tag} {db_name}",
        f"Wildtype AA rate gapless {feature_name_tag} {db_name}",
        f"Mutant AA rate gapless {feature_name_tag} {db_name}",
        f"MSA allel freq {feature_name_tag} {db_name}",
        f"Other mutant AA rate {feature_name_tag} {db_name}",
        f"Other mutant AA rate gapless {feature_name_tag} {db_name}",
        f"PSIC wildtype AA {feature_name_tag} {db_name}",
        f"PSIC mutant AA {feature_name_tag} {db_name}",
        f"dPSIC {feature_name_tag} {db_name}",
        f"Positional median dPSIC {feature_name_tag} {db_name}",
        f"Window median dPSIC {feature_name_tag} {db_name}",
        f"Protein median dPSIC {feature_name_tag} {db_name}",
        f"MSA depth {feature_name_tag} {db_name}",
        "Sequence Position Number",
        "Relative Sequence Position",
        "Protein Size",
    ]
    return feats


def getPosMap(seq):
    # print seq
    pos_map = []
    for pos, char in enumerate(seq):
        if char != "-":
            pos_map.append(pos)
    # print pos_map
    return pos_map


def randomString(stringLength=10):
    """Generate a random string of fixed length"""
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(stringLength))
