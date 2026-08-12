import sys
from scipy import stats

class Results:
    def __init__(self):
        self.ids = []
        self.true_values = []
        self.pred_values = []

    def add_result(self, iden, true_value, pred_value):
        self.ids.append(iden)
        self.true_values.append(true_value)
        self.pred_values.append(pred_value)

    def calculate_pearsons(self):
        try:
            pearson, pearson_p = stats.pearsonr(self.true_values, self.pred_values)
        except:
            pearson = None
        self.pearson = pearson

def parse_results_table(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()

    protein_wise_results = {}

    for line in lines[1:]:
        words = line.split('\t')
        prot_id = words[0]
        aac = words[1]
        true_value = float(words[2])
        pred_value = float(words[3])

        if not prot_id in protein_wise_results:
            protein_wise_results[prot_id] = Results()

        protein_wise_results[prot_id].add_result(aac, true_value, pred_value)

    return protein_wise_results

def parse_feature_table(filepath):
    f = open(filepath, 'r')
    lines = f.readlines()
    f.close()

    protein_info = {}

    headers = lines[0].replace('\n','').split('\t')

    for header_pos, header in enumerate(headers):
        if header == 'Protein Size':
            size_pos = header_pos

    for line in lines[1:]:
        words = line.replace('\n','').split('\t')
        prot_id = words[0]
        aac = words[1]
        true_value = words[2]

        try:
            protein_size = int(words[size_pos])
        except:
            continue

        if true_value != 'None':
            if prot_id not in protein_info:
                protein_info[prot_id] = [protein_size, 0]
            protein_info[prot_id][1] += 1

    return protein_info


def write_protein_wise_performances(outfile, protein_wise_results: list[tuple[str, float]], protein_info: dict[str, tuple[int, int]], err_corrs: list[float]):

    lines = ["Protein Identifier\tPerformance\tProtein Length\t# of SAVs\tError-STD-Correlation\n"]

    for prot_nr, (prot_id, performance_value) in enumerate(protein_wise_results):
        #performance_value = protein_wise_results[prot_id]

        prot_len, num_of_savs = protein_info[prot_id]

        lines.append(f'{prot_id}\t{performance_value}\t{prot_len}\t{num_of_savs}\t{err_corrs[prot_nr]}\n')

    f = open(outfile, 'w')
    f.write(''.join(lines))
    f.close()

def main(infile, feature_file, outfile):
    protein_info = parse_feature_table(feature_file)
    protein_wise_results = parse_results_table(infile)
    write_protein_wise_pearsons(outfile, protein_wise_results, protein_info)

if __name__ == "__main__":
    args = sys.argv

    infile = args[1]
    feature_file = args [2]
    outfile = args[3]

    main(infile, feature_file, outfile)
