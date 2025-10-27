import os
import sys
import traceback
import gzip
import subprocess
import ray

from Bio.Align.Applications import MafftCommandline

import xml.etree.ElementTree as ET


from structman.lib.globalAlignment import (
    init_bp_aligner_class,
    call_biopython_alignment,
)

from structman.lib.sdsc.consts import residues as residue_consts
from structman.base_utils.base_utils import pack, unpack

from structguy import psic_wrapper as psic
from structguy import util

search_db_sequences = {}


def median(l):
    n = len(l)
    l = sorted(l)
    if n % 2 == 0:
        med = (l[(n // 2) - 1] + l[n // 2]) / 2.0
    else:
        med = l[(n - 1) // 2]
    return med


def lookup(
    config,
    prot_id,
    ref_db_ids=["ref50", "ref90"],
    gpw_ref_db_ids=["ref50", "ref90"],
    debug=0,
    update_mode=False,
    sequence=None,
    sequence_maps=None,
    pdb_tuple=None,
):
    out_directory = get_out_directory(prot_id, config, pdb_tuple=pdb_tuple)

    msa_files = {}

    for ref_db_id in ref_db_ids:
        filename = util.get_msa_path(out_directory, prot_id, ref_db_id)
        if os.path.isfile(filename):
            msa_files[ref_db_id] = filename
            if debug >= 2:
                print("Look-up found a msa: ", filename)

    gpw_files = {}
    for ref_db_id in gpw_ref_db_ids:
        filename = util.get_msa_path(out_directory, prot_id, ref_db_id, gpw=True)
        if os.path.isfile(filename):
            gpw_files[ref_db_id] = filename
            if debug >= 2:
                print("Look-up found a gpw: ", filename)

    for ref_db_id in msa_files:
        filename = msa_files[ref_db_id]
        # f = gzip.open(filename,'r')
        # msa = f.read()
        # f.close()
        # if msa == '':
        #    continue

        psic_name = util.get_msa_path(out_directory, prot_id, ref_db_id, psic=True)
        if not os.path.isfile(psic_name):
            if debug >= 1:
                print("Calc psic profiles from lookup", ref_db_id)

            f = gzip.open(filename, "r")
            msa = f.read()
            f.close()
            psic.psicFromFasta(msa, psic_name[:-3], config)

            if os.path.isfile(psic_name):
                os.remove(psic_name)
            try:
                os.system(f'gzip "{psic_name[:-3]}"')
            except:
                pass

    for ref_db_id in gpw_files:
        filename = gpw_files[ref_db_id]

        if update_mode:
            gpw = updateGPW(
                filename, ref_db_id, sequence, sequence_maps[ref_db_id], prot_id
            )

        """
        else:
            
        """

        psic_name = util.get_msa_path(
            out_directory, prot_id, ref_db_id, psic=True, gpw=True
        )
        if (not os.path.isfile(psic_name)) or update_mode:
            if debug >= 1:
                print("Calc psic profiles from lookup", prot_id, ref_db_id)
            try:
                f = gzip.open(filename, "r")
                gpw = f.read()
                f.close()
            except:
                print(f"\nERROR:\nCouldnt open: {filename}\n")
                continue
            if gpw == "":
                continue
            psic.psicFromGPW(gpw, psic_name[:-3], config, debug=debug)

            if os.path.isfile(psic_name):
                os.remove(psic_name)
                if debug >= 1:
                    print("removed old psic file: ", psic_name)
            try:
                os.system(f'gzip "{psic_name[:-3]}"')
            except:
                pass

    return msa_files, gpw_files


def parseFasta(path, lines=None):
    if lines == None:
        f = open(path, "rb")
        lines = f.read().split(b"\n")
        f.close()

    seq_map = {}
    for line in lines:
        if len(line) == 0:
            continue
        if line[0:1] == b">":
            entry_id = line[1:].split()[0].decode("ascii")
            seq_map[entry_id] = ""
        else:
            seq_map[entry_id] += line.decode("ascii")

    return seq_map


def write_fasta(seq_map, outfile=None):
    lines = []
    for prot_id in seq_map:
        lines.append(f">{prot_id}\n")
        lines.append(f"{seq_map[prot_id]}\n")
    page = "".join(lines)
    if outfile is None:
        return page

    f = open(outfile, "w")
    f.write(page)
    f.close()
    return page


def blast(config, seq, name, search_db, search_db_path, search_db_sequences={}):
    blast_path = config.blast_path

    blast_db_path = search_db_path

    if not search_db in search_db_sequences:
        search_db_sequences[search_db] = parseFasta(blast_db_path)

    sequences = search_db_sequences[search_db]

    FNULL = open(os.devnull, "w")
    cwd = os.getcwd()
    seq = seq.replace(" ", "")
    target_length = len(seq)

    blast_in = "%s/blast_in_%s_%s.tmp.fasta" % (cwd, name, search_db)
    blast_out = "%s/blast_out_%s_%s.tmp" % (cwd, name, search_db)

    f = open(blast_in, "wb")
    f.write(b">" + name.encode("ascii") + b"\n" + seq.encode("ascii"))
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
    p = subprocess.Popen(
        [path, arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10]
    )  # ,stdout=FNULL)
    p.wait()

    if not os.path.isfile(blast_out):
        print(name, search_db, "Blast error")
        return None

    f = open(blast_out, "r")
    lines = f.read()
    f.close()
    # print name,lines

    os.remove(blast_in)
    os.remove(blast_out)

    uniref_ids = []

    if len(lines) == 0:
        return ""

    root = ET.fromstring(lines)
    blast_iter_hits = root[8][0][4]

    for child in blast_iter_hits:
        uniref_id = child[2].text.split()[0]
        uniref_ids.append(uniref_id)

    # print uniref_ids
    fasta_lines = [
        ">%s" % name,
        seq.replace("U", "C").replace("O", "K").replace("J", "I"),
    ]
    for uniref_id in uniref_ids:
        fasta_lines.append(">%s" % uniref_id)
        fasta_lines.append(
            sequences[uniref_id].replace("U", "C").replace("O", "K").replace("J", "I")
        )

    page = "\n".join(fasta_lines)
    return page, search_db_sequences


def computeMSA(
    config,
    seq,
    u_ac,
    search_db="ref50",
    search_db_path="",
    debug=0,
    search_db_sequences={},
    sequence_map=None,
    sub_threads=1,
    target_file=None
):
    mafft_exe = config.mafft_path
    print("Compute MSA: ", u_ac, search_db)

    if sequence_map is None:
        # Blast against search database
        fasta_page, search_db_sequences = blast(
            seq,
            "%s_%s" % (u_ac, search_db),
            search_db,
            search_db_path,
            search_db_sequences=search_db_sequences,
        )

        if fasta_page is None:
            return None, None

        # Write the Blast results into a fasta file
        cwd = os.getcwd()

        temp_fasta = "%s/temp_fasta_%s_%s.fasta" % (cwd, u_ac, search_db)
        f = open(temp_fasta, "w")
        f.write(fasta_page)
        f.close()
    else:
        cwd = os.getcwd()

        temp_fasta = "%s/temp_fasta_%s_%s.fasta" % (
            cwd,
            u_ac.replace("(", "").replace(")", "").replace('/','_'),
            search_db,
        )
        sequence_map[u_ac] = seq
        fasta_page = write_fasta(sequence_map, outfile=temp_fasta)

    stderr = None
    # Run mafft
    try:

        cmds = ' '.join([
            'mafft',
            '--thread',
            str(sub_threads),
            f'"{temp_fasta}"'
            ])

        if target_file is not None:
            with open(target_file, 'w') as target:
                p = subprocess.Popen(cmds, shell=True, stdout=target)
                p.wait()
            f = open(target_file, 'r')
            stdout = f.read()
            f.close()

        else:
            p = subprocess.Popen(cmds, shell=True, stdout=subprocess.PIPE)
            p.wait()
            stdout, stderr = p.communicate()

        """
        stdout = None
        stderr = None
        mafft_cline = MafftCommandline(
            mafft_exe, input=temp_fasta, thread=sub_threads, amino=True
        )
        stdout, stderr = mafft_cline()
        """
        
    except:
        [e, f, g] = sys.exc_info()
        g = traceback.format_exc()
        print(f'MAFFT failed: {u_ac=}, {search_db=} {temp_fasta=}\n{e}\n{f}\n{g}\n{stdout=}\n{stderr=}')
        return None, None, None

    # delete temporary files
    try:
        os.remove(temp_fasta)
    except:
        pass

    print("Done computing MSA: ", u_ac, search_db)

    outlines = []
    inlines = stdout.split("\n")
    parse = False
    for line in inlines:
        if line == "":
            continue
        if line[0] == ">":
            entry_id = line[1:]
            if entry_id == u_ac:
                parse = True
            else:
                parse = False
        if parse:
            outlines.append(f"{line}\n")

    parse = False
    for line in inlines:
        if line == "":
            continue
        if line[0] == ">":
            entry_id = line[1:]
            if entry_id == u_ac:
                parse = False
            else:
                parse = True
        if parse:
            outlines.append(f"{line}\n")

    return "".join(outlines), fasta_page, search_db_sequences


def updateGPW(filename, search_db, seq, sequence_map, u_ac):
    print("Update GPW: ", filename)
    f = gzip.open(filename, "rb")
    gpw = f.read()
    f.close()
    gpw_ds, seed = parseGpwFasta(gpw, dict_out=True)

    # print gpw_ds

    out_fasta_lines = []

    new_fasta_lines = []

    n = 0
    m = 0
    target_seq = seq.replace("U", "C").replace("O", "K").replace("J", "I")
    for seq_id in sequence_map:
        # print seq_id
        if seq_id in gpw_ds:
            n += 1
            out_fasta_lines.append(">%s_%s" % (u_ac, seq_id))
            out_fasta_lines.append(gpw_ds[seq_id][0])
            out_fasta_lines.append(">%s" % seq_id)
            out_fasta_lines.append(gpw_ds[seq_id][1])
            continue
        m += 1
        template_seq = sequence_map[seq_id]

        try:
            (target_aligned_sequence, template_aligned_sequence, a, b, c) = (
                pairwise2.align.globalds(
                    target_seq,
                    template_seq,
                    matrix,
                    -10.0,
                    -0.5,
                    one_alignment_only=True,
                )[0]
            )
        except:
            print("GPW error: ", u_ac, seq_id)
            print(target_seq[:10], template_seq[:10])
            continue
        new_fasta_lines.append(">%s_%s" % (u_ac, seq_id))
        new_fasta_lines.append(target_aligned_sequence)
        new_fasta_lines.append(">%s" % seq_id)
        new_fasta_lines.append(template_aligned_sequence)

    print("Done updating GPW: ", u_ac, search_db)
    print("Used ", n, " old alignments and computed ", m, " new alignments")

    gpw = "%s\n%s" % ("\n".join(out_fasta_lines), "\n".join(new_fasta_lines))

    if m > 0:
        new_total_gpw = "%s\n%s" % (gpw, "\n".join(new_fasta_lines))
        f = gzip.open(filename, "w")
        f.write(new_total_gpw)
        f.close()

    return gpw


def computeGPW(
    config,
    seq,
    prot_id,
    sequence_store,
    aligner_class=None,
    search_db="ref50",
    search_db_path={},
    fasta_page=None,
    sequence_map=None,
    search_db_sequences={},
    sub_threads=1,
)->tuple[None | str, dict[str, str]]:
    print(
        f"Compute GPW: {prot_id}, {search_db} {sequence_map is None} {fasta_page is None}"
    )

    if sequence_map is None:
        if fasta_page is None:
            pass
            # not supported at the moment
            """
            #Blast against search database
            if debug >= 1:
                print('In computeGPW sequence_map is None and fasta_page is None, try BLAST:',prot_id,search_db)
            fasta_page,search_db_sequences = blast(seq,'%s_%s' % (prot_id,search_db),search_db,search_db_path,search_db_sequences=search_db_sequences)
            """
        if fasta_page is None:
            return None, search_db_sequences

        if config.verbosity >= 1:
            print("Blast result size: ", len(fasta_page))

        seq_map = parseFasta("", lines=fasta_page.split("\n"))

    else:
        seq_map = sequence_map

    if seq == 0 or seq == 1:
        if config.verbosity >= 1:
            print("Sequence error in computeGPW:", prot_id, search_db)
        return "", search_db_sequences

    out_fasta_lines = []
    target_seq = seq.replace("U", "C").replace("O", "K").replace("J", "I")
    if sub_threads == 1:
        for hit in seq_map:
            if isinstance(seq_map, dict):
                template_seq = seq_map[hit]
            else:
                template_seq = ray.get(sequence_store[hit])
            if config.verbosity >= 3:
                print("Aligning:", prot_id, hit)
            if config.verbosity >= 5:
                print(target_seq)
                print(template_seq)
            try:
                (target_aligned_sequence, template_aligned_sequence) = (
                    call_biopython_alignment(
                        target_seq, template_seq, aligner_class=aligner_class
                    )
                )
            except:
                if config.verbosity >= 1:
                    print("GPW error: ", prot_id, hit)
                    print(target_seq[:10], template_seq[:10])
                continue
            out_fasta_lines.append(f">{prot_id}_{hit}\n")
            out_fasta_lines.append(f"{target_aligned_sequence}\n")
            out_fasta_lines.append(f">{hit}\n")
            out_fasta_lines.append(f"{template_aligned_sequence}\n")
    else:
        store = ray.put((prot_id, target_seq))
        packages = []
        current_package = 0
        for hit in seq_map:
            if isinstance(seq_map, dict):
                hit_seq = seq_map[hit]
                if len(packages) == current_package:
                    packages.append([])
                packages[current_package].append((hit, hit_seq))
            else:
                if len(packages) == current_package:
                    packages.append([])
                packages[current_package].append(hit)
            current_package += 1
            if current_package >= sub_threads:
                current_package = 0

        para_alignment_ray_process_ids = []
        for package in packages:
            para_alignment_ray_process_ids.append(
                para_align_seqs.remote(store, sequence_store, package)
            )

        alignment_results = ray.get(para_alignment_ray_process_ids)
        for results_package in alignment_results:
            para_fasta_lines = results_package
            out_fasta_lines += para_fasta_lines

    if config.verbosity >= 1:
        print("Done computing GPW: ", prot_id, search_db)

    return "".join(out_fasta_lines), search_db_sequences


@ray.remote(max_calls=1)
def para_align_seqs(store, sequence_store, package):
    prot_id, target_seq = store
    aligner_class = init_bp_aligner_class()
    out_fasta_lines = []
    for content in package:
        if isinstance(content, tuple):
            hit, hit_seq = content
        else:
            hit = content
            hit_seq = ray.get(sequence_store[hit])
        try:
            (target_aligned_sequence, template_aligned_sequence) = (
                call_biopython_alignment(
                    target_seq, hit_seq, aligner_class=aligner_class
                )
            )
        except:
            continue
        out_fasta_lines.append(f">{prot_id}_{hit}\n")
        out_fasta_lines.append(f"{target_aligned_sequence}\n")
        out_fasta_lines.append(f">{hit}\n")
        out_fasta_lines.append(f"{template_aligned_sequence}\n")
    return out_fasta_lines


def saveMSA(config, msa, prot_id, ref_db_id, pdb_tuple):
    if msa is None:
        return

    out_directory = get_out_directory(prot_id, config, pdb_tuple=pdb_tuple)

    files = {}

    filename = util.get_msa_path(out_directory, prot_id, ref_db_id, unpacked=True)

    outlines = []
    inlines = msa.split("\n")
    parse = False
    for line in inlines:
        if line == "":
            continue
        if line[0] == ">":
            entry_id = line[1:]
            if entry_id == prot_id:
                parse = True
            else:
                parse = False
        if parse:
            outlines.append(f"{line}\n")

    parse = False
    for line in inlines:
        if line == "":
            continue
        if line[0] == ">":
            entry_id = line[1:]
            if entry_id == prot_id:
                parse = False
            else:
                parse = True
        if parse:
            outlines.append(f"{line}\n")

    f = open(filename, "w")
    f.write("".join(outlines))
    f.close()

    if os.path.isfile("%s.gz" % filename):
        os.remove("%s.gz" % filename)
    os.system(f'gzip "{filename}"')

    return f"{filename}.gz"


def saveGpw(config, gpw, prot_id, ref_db_id, pdb_tuple):
    if gpw == None:
        return

    out_directory = get_out_directory(prot_id, config, pdb_tuple=pdb_tuple)

    filename = util.get_msa_path(
        out_directory, prot_id, ref_db_id, gpw=True, unpacked=True
    )

    f = open(filename, "w")
    f.write(gpw)
    f.close()

    if os.path.isfile("%s.gz" % filename):
        os.remove("%s.gz" % filename)
    os.system(f'gzip "{filename}"')

    return f"{filename}.gz"


def parsePsicFile(infile):
    if not os.path.isfile(infile):
        print(f"Did not found psic-file: {infile}")
        return {}

    if infile[-3:] == '.gz':
        f = gzip.open(infile, "r")
    else:
        f = open(infile, 'rb')
    lines = f.read().decode("ascii").split("\n")
    f.close()

    aa_key = lines[1].split()[1:-1]

    psic_profiles = {}

    for pos, line in enumerate(lines[2:]):
        if line == "":
            continue
        data = line.split()[1:-1]
        psic_profiles[pos] = {}
        for aa_pos, aa in enumerate(aa_key):
            psic_profiles[pos][aa] = float(data[aa_pos])

    return psic_profiles


# called by sequence_feature_generation
def calcPsicProfiles(config, prot_id, aacs, seq, ref_db_id, gpw=False, psic_name = None):
    out_directory = get_out_directory(prot_id, config)

    if psic_name is None:
        psic_name = util.get_msa_path(out_directory, prot_id, ref_db_id, gpw=gpw, psic=True)

    if config.verbosity >= 2:
        config.logger.info(f'calcPsicProfiles: {prot_id=} {len(seq)=} {ref_db_id=} {gpw=} {psic_name=} {out_directory=}')

    psic_profiles = parsePsicFile(psic_name)

    psic_wt_map = {}
    psic_mut_map = {}
    dpsic_map = {}
    window_dpsic_map = {}
    positional_dpsic_map = {}

    if psic_profiles == {}:
        config.logger.info(f"Empty psic profiles: {prot_id=}, {ref_db_id=}")
        return psic_wt_map, psic_mut_map, dpsic_map, [], {}, None

    config.logger.info(f"{len(psic_profiles)=} {prot_id=} {ref_db_id=}")

    positional_median_dpsics = []
    for pos, wt in enumerate(seq):
        if pos not in psic_profiles:
            config.logger.info("pos not in psic_profiles:", prot_id, pos)
            continue
        if wt not in psic_profiles[pos]:
            config.logger.info(f"wt not in psic_profiles[pos]: {prot_id} {pos} {wt}\n{psic_profiles[pos]}")
            continue
        psic_wt = psic_profiles[pos][wt]
        positional_dpsics = []
        for mut_aa in psic_profiles[pos]:
            if mut_aa == wt:
                continue
            psic_mut = psic_profiles[pos][mut_aa]
            dpsic = psic_wt - psic_mut
            positional_dpsics.append(dpsic)
        positional_median_dpsic = median(positional_dpsics)
        positional_median_dpsics.append(positional_median_dpsic)

    protein_median_dpsic = median(positional_median_dpsics)

    for aac in aacs:
        aa_wt = aac[0]
        aa_mut = aac[-1]
        pos = int(aac[1:-1]) - 1

        window_a = pos - 10
        window_b = pos + 10

        if window_a < 0 and window_b >= len(seq):
            window_a = 0
            window_b = len(seq) - 1
        if window_a < 0:
            window_b -= window_a
            window_a = 0
        if window_b >= len(seq):
            window_a -= window_b - (len(seq) - 1)
            window_b = len(seq) - 1
            if window_a < 0:
                window_a = 0

        if pos not in psic_profiles:
            config.logger.info(f"psic error {prot_id=} {aac=} {ref_db_id=} {gpw=}")
            positional_dpsic_map[aac] = 0.0
            psic_wt_map[aac] = 0.0
            psic_mut_map[aac] = 0.0
            dpsic_map[aac] = 0.0
            continue

        if aa_wt not in psic_profiles[pos]:
            config.logger.info(f"psic error 2 {prot_id=} {aac=} {ref_db_id=} {gpw=}")
            positional_dpsic_map[aac] = 0.0
            psic_wt_map[aac] = 0.0
            psic_mut_map[aac] = 0.0
            dpsic_map[aac] = 0.0
            continue

        window_median_position_dpsics = positional_median_dpsics[
            window_a : (window_b + 1)
        ]
        window_median_dpsic = median(window_median_position_dpsics)

        window_dpsic_map[aac] = window_median_dpsic

        if pos >= len(positional_median_dpsics):
            config.logger.info(f"psic error 3 {prot_id=} {aac=} {ref_db_id=} {gpw=}")
            positional_dpsic_map[aac] = 0.0
            psic_wt_map[aac] = 0.0
            psic_mut_map[aac] = 0.0
            dpsic_map[aac] = 0.0
            continue

        positional_median_dpsic = positional_median_dpsics[pos]
        positional_dpsic_map[aac] = positional_median_dpsic

        psic_wt = psic_profiles[pos][aa_wt]
        if aa_mut in psic_profiles[pos]:
            psic_mut = psic_profiles[pos][aa_mut]
        else:
            psic_mut = -1.0

        dpsic = psic_wt - psic_mut

        psic_wt_map[aac] = psic_wt
        psic_mut_map[aac] = psic_mut
        dpsic_map[aac] = dpsic

    return (
        psic_wt_map,
        psic_mut_map,
        dpsic_map,
        positional_dpsic_map,
        window_dpsic_map,
        protein_median_dpsic,
    )


# called by structural_feature_generation
def getPosWiseGPW(gpw):
    pos_wise_map = {}

    first_hit = True
    for hit_id in gpw:
        query_seq, hit_seq = gpw[hit_id]

        pos_map = getPosMap(query_seq)

        if first_hit:
            for gl_pos, pos in enumerate(pos_map):
                pos_wise_map[gl_pos] = [query_seq[pos]]
                first_hit = False

        try:
            for gl_pos, pos in enumerate(pos_map):
                pos_wise_map[gl_pos].append(hit_seq[pos])
        except:
            print(f"{hit_id} {len(pos_map)} {len(query_seq)} {len(hit_seq)}")
            for gl_pos, pos in enumerate(pos_map):
                pos_wise_map[gl_pos].append(hit_seq[pos])

    return pos_wise_map


def getPosWiseMSA(msa, seed):
    pos_wise_map = {}
    pos_map = getPosMap(msa[seed])
    for gl_pos, pos in enumerate(pos_map):
        pos_wise_map[gl_pos] = [msa[seed][pos]]
    for prot_id in msa:
        if prot_id == seed:
            continue
        seq = msa[prot_id]
        for gl_pos, pos in enumerate(pos_map):
            pos_wise_map[gl_pos].append(seq[pos])
    return pos_wise_map


def getPosMap(seq):
    # print seq
    pos_map = []
    for pos, char in enumerate(seq):
        if char != "-":
            pos_map.append(pos)
    # print pos_map
    return pos_map


def get_out_directory(protein_id, config, pdb_tuple=None):
    if not config.fasta_mode:
        if pdb_tuple is None:
            folder_key = protein_id.split("-")[0][-2:]
        else:
            folder_key = pdb_tuple[1:3]
        out_directory = f"{config.msa_db}/{folder_key}"

    else:
        out_directory = f"{config.custom_msa_db}"  # gets set in parse_from_fasta in sequence_feature_generation

    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)

    return out_directory


def getMSA(
    config,
    prot_id,
    sequence_store,
    com_queue,
    sequence_maps=None,
    sequence=None,
    ref_db_ids=["ref50", "ref90"],
    gpw_ref_db_ids=["ref50", "ref90"],
    debug=0,
    update_mode=False,
    sub_threads=1,
):
    if len(prot_id) < 5:
        pdb_tuple = None
    elif prot_id[4] == ":" and len(prot_id) == 6:
        pdb_tuple = prot_id
    else:
        pdb_tuple = None

    search_dbs = {
        "ref50": config.mmseqs_search_db_ref50,
        "ref90": config.mmseqs_search_db_ref90,
        "ref100": config.mmseqs_search_db_ref100,
    }

    if debug >= 1:
        print("getMSA", prot_id, ref_db_ids, gpw_ref_db_ids, update_mode)

    # If the sequence is not given, get it
    if update_mode and sequence is None:
        print("Sequence error: ", prot_id)
        return {}, {}

    # Check if the protein is in the database
    msas, gpws = lookup(
        config,
        prot_id,
        ref_db_ids=ref_db_ids,
        gpw_ref_db_ids=gpw_ref_db_ids,
        debug=debug,
        update_mode=update_mode,
        sequence=sequence,
        sequence_maps=sequence_maps,
        pdb_tuple=pdb_tuple,
    )

    if debug >= 2:
        print("Lookup results: ", prot_id, list(msas.keys()), list(gpws.keys()))

    # If the msa is not in the database, compute it
    if len(msas) == len(ref_db_ids) and len(gpws) == len(gpw_ref_db_ids):
        if debug >= 1:
            print("found msas in the db")
        return msas, gpws

    fasta_results = {}

    out_directory = get_out_directory(prot_id, config, pdb_tuple=pdb_tuple)

    # compute the msa's
    search_db_sequences = {}
    for ref_db_id in ref_db_ids:
        if ref_db_id in msas:
            continue
        msa, fasta_page, search_db_sequences = computeMSA(
            config,
            sequence,
            prot_id,
            search_db=ref_db_id,
            search_db_path=search_dbs[ref_db_id],
            debug=debug,
            sequence_map=sequence_maps[ref_db_id],
            search_db_sequences=search_db_sequences,
            sub_threads=sub_threads,
        )

        psic_name = util.get_msa_path(out_directory, prot_id, ref_db_id, psic=True)
        if not os.path.isfile(psic_name):
            if debug >= 1:
                print("Calc psic profiles from getMSA", ref_db_id)
            psic.psicFromFasta(msa, psic_name[:-3], config)

            if os.path.isfile(psic_name):
                os.remove(psic_name)
            try:
                os.system(f'gzip "{psic_name[:-3]}"')
            except:
                pass

        # save them into the database
        msa_file_path = saveMSA(config, msa, prot_id, ref_db_id, pdb_tuple)
        msas[ref_db_id] = msa_file_path

    aligner_class = init_bp_aligner_class()

    for ref_db_id in gpw_ref_db_ids:
        if ref_db_id in gpws:
            continue
        if ref_db_id in fasta_results:
            fasta_page = fasta_results[ref_db_id]
        else:
            fasta_page = None
        gpw, search_db_sequences = computeGPW(
            config,
            sequence,
            prot_id,
            sequence_store,
            aligner_class=aligner_class,
            search_db=ref_db_id,
            search_db_path=search_dbs[ref_db_id],
            fasta_page=fasta_page,
            sequence_map=sequence_maps[ref_db_id],
            search_db_sequences=search_db_sequences,
            sub_threads=sub_threads,
        )
        if gpw == "" or gpw is None:
            if debug >= 1:
                print(f"computeGPW returned empty result: {prot_id=}, {ref_db_id=} {gpw=}")
            continue

        # save them into the database
        gpw_file_path = saveGpw(config, gpw, prot_id, ref_db_id, pdb_tuple)

        gpws[ref_db_id] = gpw_file_path

    com_queue.put(sub_threads -1)

    for ref_db_id in gpw_ref_db_ids:
        
        psic_name = util.get_msa_path(
            out_directory, prot_id, ref_db_id, psic=True, gpw=True
        )
        if not os.path.isfile(psic_name):
            if debug >= 1:
                print("Calc psic profiles from getMSA", ref_db_id)
            psic.psicFromGPW(gpw, psic_name[:-3], config, debug=debug)

            if os.path.isfile(psic_name):
                os.remove(psic_name)
            try:
                os.system(f'gzip "{psic_name[:-3]}"')
            except:
                pass

    return msas, gpws


def parseGpwFasta(page, dict_out=False):
    try:
        lines = page.split(b"\n")
    except:
        page = page.encode("ascii")
        lines = page.split(b"\n")

    if dict_out:
        seed = None
        seq_map = {}
        second_seq = False
        first_entry = True
        for line in lines:
            if len(line) == 0:
                continue
            if line[0:1] == b">":
                entry_id = line[1:].split()[0]
                if first_entry:
                    if entry_id.count(b"UniRef") > 0:
                        q_id = b"_".join(entry_id.split(b"_")[:-2])
                        t_id = (b"_".join(entry_id.split(b"_")[-2:])).decode("ascii")
                    else:
                        splits = entry_id.split(b"_")
                        h = len(splits) // 2
                        q_id = b"_".join(entry_id.split(b"_")[:-h])
                        t_id = (b"_".join(entry_id.split(b"_")[-h:])).decode("ascii")
                    seq_map[t_id] = ["", ""]
                    second_seq = False
                    first_entry = False
                    if seed is None:
                        seed = q_id.decode("ascii")
                else:
                    t_id = entry_id.decode("ascii")
                    second_seq = True
                    first_entry = True

            else:
                try:
                    if second_seq:
                        seq_map[t_id][1] += line.decode("ascii")
                    else:
                        seq_map[t_id][0] += line.decode("ascii")
                except:
                    print(f"{t_id} \n {list(seq_map.keys())}")
                    if second_seq:
                        seq_map[t_id][1] += line.decode("ascii")
                    else:
                        seq_map[t_id][0] += line.decode("ascii")
    else:
        seq_map = []
        seed = 0
        for line in lines:
            if len(line) == 0:
                continue
            if line[0:1] == b">":
                entry_id = line[1:].split()[0].decode("ascii")

                seq_map.append([entry_id, ""])
            else:
                seq_map[-1][1] += line.decode("ascii")
    return seq_map, seed


def parseMsaFasta(page):
    lines = page.split(b"\n")

    seq_map = {}
    seed = None
    for line in lines:
        if len(line) == 0:
            continue
        if line[0:1] == b">":
            entry_id = line[1:].split()[0].decode("ascii")
            seq_map[entry_id] = ""
            if seed is None:
                seed = entry_id
        else:
            seq_map[entry_id] += line.decode("ascii")
    return seq_map, seed
