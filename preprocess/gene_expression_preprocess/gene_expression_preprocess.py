import os
import json
import pandas as pd

# 设置工作目录
os.chdir("/home/yuhaowang/project/learning/BRCA-expression")

# 加载 JSON 文件
with open("metadata.repository.2024-11-26.json", 'r') as file:
    json_data = json.load(file)

# 提取需要的字段
file_sample0 = pd.DataFrame(json_data)[['file_name', 'associated_entities']]
file_sample0['sample_id'] = file_sample0['associated_entities'].apply(lambda x: x[0]['entity_submitter_id'])
file_sample = file_sample0.drop(columns=['associated_entities'])

# 处理 file_name 列
file_sample['file_name'] = file_sample['file_name'].str.split('e_counts.tsv').str[0]

# 读取表达矩阵文件名
count_files = [os.path.join(root, file) 
               for root, _, files in os.walk('./') 
               for file in files if file.endswith('rna_seq.augmented_star_gene_counts.tsv')]
count_file_names = [os.path.basename(file).split('/')[0] for file in count_files]

# 创建 COUNT_Ensembl_matrix
COUNT_Ensembl_matrix = pd.DataFrame()

for i, file_path in enumerate(count_files):
    print(file_path)
    data0 = pd.read_csv(file_path, sep="\t", header=1)

    data = data0.iloc[4:][['gene_id', 'tpm_unstranded']]  # 假设第4列是表达值
    column_name=file_sample[file_sample['file_name'] == 
                                   count_file_names[i].replace('star_gene_counts.tsv', 'star_gen')][['sample_id']].values[0][0]
    data=data.rename(columns={'tpm_unstranded': column_name})
    if COUNT_Ensembl_matrix.empty:
        COUNT_Ensembl_matrix = data
    else:
        COUNT_Ensembl_matrix = pd.merge(COUNT_Ensembl_matrix, data, on='gene_id', how='inner')
        COUNT_Ensembl_matrix = COUNT_Ensembl_matrix.loc[:, ~COUNT_Ensembl_matrix.columns.str.endswith('_y')]

        COUNT_Ensembl_matrix.columns = COUNT_Ensembl_matrix.columns.str.replace('_x', '', regex=True)

#COUNT_Ensembl_matrix.to_csv('COUNT_Ensembl_matrix.csv', index=False)

# ----------------------------- 1. Ensembl ID 矩阵转换为 Gene Symbol 矩阵 -------------------------------
Ensembl_Symbol = data0[['gene_id', 'gene_name']]
COUNT_Symbol_matrix = pd.merge(Ensembl_Symbol, COUNT_Ensembl_matrix, on='gene_id')
COUNT_Symbol_matrix = COUNT_Symbol_matrix.drop(columns=['gene_id'])

# 删除重复的基因，保留最大表达值
COUNT_Symbol_matrix = COUNT_Symbol_matrix.groupby('gene_name').max().reset_index()

# 将 gene_name 列设为索引
COUNT_Symbol_matrix.set_index('gene_name', inplace=True)

COUNT_Symbol_matrix.to_csv('COUNT_Symbol_matrix.csv')

# ----------------------------- 2. 分为 normal 和 tumor 矩阵 -------------------------------------
# samples = COUNT_Symbol_matrix.columns
# normal_samples = [sample for sample in samples if int(sample[13:15]) >= 10]
# tumor_samples = [sample for sample in samples if int(sample[13:15]) < 10]

# normal_matrix = COUNT_Symbol_matrix[normal_samples]
# tumor_matrix = COUNT_Symbol_matrix[tumor_samples]

# tumor_matrix.to_csv('tumor_matrix.csv')
# normal_matrix.to_csv('normal_matrix.csv')
