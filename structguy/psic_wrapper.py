import subprocess
import os

def fastaToClustal(aln_fasta):
    seq_map = {}
    seed_id = None
    for line in aln_fasta.split(b'\n'):
        if len(line) == 0:
            continue
        if line[0:1] == b'>':
            entry_id = line[1:].split()[0].decode('ascii')
            seq_map[entry_id] = ''
            if seed_id == None:
                seed_id = entry_id
        else:
            seq_map[entry_id] += line.decode('ascii')

    if seed_id == None:
        return None

    seed_seq = seq_map[seed_id]
    pos_map = []
    for pos,aa in enumerate(seed_seq):
        if aa != '-':
            pos_map.append(pos)

    aln_clustal_lines = ['CLUSTAL\n']
    for seq_id in seq_map:
        seq = seq_map[seq_id]
        cutseq = []
        for pos in pos_map:
            cutseq.append(seq[pos])
        seq = ''.join(cutseq)

        spacer = ' '*(67 - len(seq_id))
        aln_clustal_lines.append('%s%s%s' %(seq_id,spacer,seq))

    aln_clustal = '\n'.join(aln_clustal_lines).encode('ascii')
    return aln_clustal

def getPosMap(seq):
    #print seq
    pos_map = []
    for pos,char in enumerate(seq):
        if char != '-':
            pos_map.append(pos)
    #print pos_map
    return pos_map

def gpwToClustal(page,debug=0):
    if not type(page) == type(b''):
        try:
            page = page.encode('ascii')
        except:
            lines = page.split('\n')
            for line in lines:
                try:
                    line = line.encode('ascii')
                except:
                    print(f'Error in encoding line:\n{line}')
    lines = page.split(b'\n')

    seq_map = []
    for line in lines:
        if len(line) == 0:
            continue
        if line[0:1] == b'>':
            entry_id = line[1:].split()[0].decode('ascii')
            seq_map.append([entry_id,''])
        else:
            seq_map[-1][1] += line.decode('ascii')

    aln_clustal_lines = ['CLUSTAL\n']
    target = True
    first_target = True
    for [seq_id,seq] in seq_map:
        if target:
            pos_map = getPosMap(seq)
            target = False
            if first_target:
                seq = seq.replace('-','')
                spacer = ' '*(67 - len(seq_id))
                aln_clustal_lines.append('%s%s%s' %(seq_id,spacer,seq))
            first_target = False
        else:
            target = True
            cutseq = []
            for pos in pos_map:
                cutseq.append(seq[pos])
            seq = ''.join(cutseq)

            spacer = ' '*(67 - len(seq_id))
            aln_clustal_lines.append('%s%s%s' %(seq_id,spacer,seq))

    aln_clustal = '\n'.join(aln_clustal_lines).encode('ascii')
    if debug >= 2:
        print('gpwToClustal done ',aln_clustal_lines[1].split()[0],' with ',len(seq_map),' sequences and total ',len(aln_clustal_lines),' lines')
    return aln_clustal

def callPsic(config,infile,outfile=None,debug=0):
    outf_opt = False
    if outfile == None:
        outf_opt = True
        outfile = '%s_psic_out' % infile

    if not os.path.isfile(infile) and debug >= 1:
        print("callPsic with unknown file: ",infile)

    exc_path = config.psic_source + '/psic'

    if debug >= 2:
        print('callPsic with ',infile,' and ',outfile,',path: ',exc_path)
    p = subprocess.Popen([exc_path,infile,config.blosum_path,outfile])
    p.wait()

    if debug >= 1:
        errors = p.communicate()[1]
        if errors != None:
            print('PSIC errors: ',errors)

    if outf_opt:
        f = open(outfile,'r')
        page = f.read()
        f.close()
        os.remove(outfile)
        return page
    else:
        return

def psicFromFasta(fasta_page,outfile,config):
    if fasta_page == None:
        print('Error in psicFromFasta: fasta_page is None')
        return

    cl_page = fastaToClustal(fasta_page)

    if cl_page == None:
        print('Error in psicFromFasta: could not convert from fasta to clustal')
        return

    clustal_file = '%s.clustal' % outfile
    
    f = open(clustal_file,'wb')
    f.write(cl_page)
    f.close()
    
    callPsic(config,clustal_file,outfile=outfile)

    if os.path.isfile(clustal_file):
        os.remove(clustal_file)
    else:
        print('after callPsic there was no clustal file')
    return

def psicFromGPW(gpw,outfile,config,debug=0):
    if gpw == None:
        if debug >= 1:
            print('psicFromGPW called with None')
        return
    #try:
    cl_page = gpwToClustal(gpw,debug=debug)
    #except:
    #    print(f'Error in gpwToClustal: {gpw}')
    #    sys.exit()

    clustal_file = '%s.clustal' % outfile
    f = open(clustal_file,'wb')
    f.write(cl_page)
    f.close()
    
    callPsic(config,clustal_file,outfile=outfile,debug=debug)

    if debug >= 1:
        if not os.path.isfile(outfile):
            print('callPSIC produced no output: ',clustal_file,' cl_page_size ',len(cl_page))
        elif os.path.isfile(clustal_file):
            os.remove(clustal_file)
    elif os.path.isfile(outfile) and os.path.isfile(clustal_file):
        os.remove(clustal_file)
    return


"""
#test box
t0 = time.time()
fasta_example = '/TL/sin/nobackup/MSAdb/00/O88700_ref50.fasta.gz'
f = gzip.open(fasta_example,'r')
page = f.read()
f.close()

cl_page = fastaToClustal(page)

clustal_example = 'cl_example.clustal'
f = open(clustal_example,'w')
f.write(cl_page)
f.close()

print callPsic(clustal_example)
t1 = time.time()
print t1-t0
"""
