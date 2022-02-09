import MySQLdb
import sys
import os
sys.path.append("/TL/sin/work/agress/StructMAn/lib")
import database
msa_db = '/TL/sin/nobackup/MSAdb'
db_adress='bioinfodb'
db_user_name='agress'
db_password='3GccJS8c'
db_name = 'struct_man_db_6'
session = 4

db = MySQLdb.connect(db_adress,db_user_name,db_password,db_name)
cursor = db.cursor()



sql = "SELECT Mutation,New_AA,Tag FROM RS_Mutation_Session WHERE Session = %s" % str(session)
try:
    cursor.execute(sql)
    results = cursor.fetchall()
    db.commit()
except:
    raise NameError("Error: %s" % sql)

#print results

tag_map = {}
mut_ids = set()
for row in results:
    tag_map[(row[0],row[1])] = row[2]
    mut_ids.add(row[0])

gene_id_list = set()

mutation_dict = database.getMutationDict(mut_ids,db,cursor)
for m in mutation_dict:
    aac,gene_id = mutation_dict[m][:2]

    gene_id_list.add(gene_id)

gene_score_dict = database.getGeneScoreDict(gene_id_list,session,db,cursor)

db.close()



n = 0
for g_id in gene_score_dict:
    u_ac = gene_score_dict[g_id][0]
    folder_key = u_ac.split('-')[0][-2:]
    filename = '%s/%s/%s_ref50_gpw.fasta.gz' % (msa_db,folder_key,u_ac)
    if os.path.isfile(filename):
        n += 1
        continue
print 'Total: ',len(gene_score_dict)
print 'Done: ',n
print 'Remaining: ',len(gene_score_dict) -n
