import pandas as pd
import random
from sklearn.metrics.pairwise import cosine_similarity

# 通过rdkit 绘制分子图像
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import DataStructs
from rdkit.DataStructs import TanimotoSimilarity

df_embedding = pd.read_csv("have_pretrain_embeddings.csv",header=None, index_col=False)
df_labels = pd.read_csv("have_pretrain_labels.csv",header=None, index_col=False)

print(df_labels.head(10))
label_index = 1
all_list = [0,1]
# all_list.remove(label_index) 
# 统计df_labels[label_index]的数量
print(df_labels[label_index].value_counts())
print(df_labels[label_index].notna())
df_embedding_filter = df_embedding[df_labels[label_index].notna()]
df_labels_filter = df_labels[df_labels[label_index].notna()][all_list]
print(df_labels_filter.head(10))
print(df_labels_filter.shape)

print(df_embedding_filter.head(10))
print(df_embedding_filter.shape)

print(df_embedding_filter.iloc[0])
def top_k_indices(numbers, k):
    # 创建一个包含 (索引, 值) 的列表
    indexed_numbers = list(enumerate(numbers))
    # 按值降序排序
    indexed_numbers.sort(key=lambda x: x[1], reverse=True)
    # 提取前 k 个索引
    top_k_indices = [index for index, value in indexed_numbers[:k]]
    return top_k_indices

def get_smiles(df_embedding, df_labels,rank, type="full"):
    if type == "graph":
        df_embedding = df_embedding.iloc[:, list(range(128, 256)) + list(range(256, 512))]
        
    elif type == "smiles":
        # pass
        df_embedding = df_embedding.iloc[:, list(range(0, 256)) + list(range(256+128, 512))]
    random_index = random.randint(0, df_embedding.shape[0])
    print("random_index",random_index)
    print(df_labels)
    smiles = df_labels.iloc[random_index][0]
    label = df_labels.iloc[random_index][1]
    embedding = df_embedding.iloc[random_index]
    # df_embedding = df_embedding.drop(index = random_index)
    df_embedding = df_embedding.iloc[list(set(range(len(df_embedding))) - {random_index})]
    # df_labels = df_labels.drop(index = random_index)
    df_labels = df_labels.iloc[list(set(range(len(df_labels))) - {random_index})]
    A_2d = embedding.tolist()
    # print("A_2d",A_2d)
    print("A_2d shape",len(A_2d))
    cosine_similarity_list = []
    for index, row in df_embedding.iterrows():
        B_2d = row.tolist()
        # print(cosine_similarity([A_2d], [B_2d])[0][0])
        # break
        cosine_similarity_list.append(cosine_similarity([A_2d], [B_2d])[0][0])
    rank_index = top_k_indices(cosine_similarity_list, rank)
    rank_smiles = df_labels.iloc[rank_index][0].values.tolist()
    return smiles, cosine_similarity_list, rank_smiles

def calcECFP(smiles, smiles_list):
    scores_list = []
    mol_anchor = Chem.MolFromSmiles(smiles) 
    # 计算Morgan指纹（ECFP）
    radius = 2  # 指纹半径
    nBits = 1024  # 指纹长度
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_anchor, radius, nBits)
    for item in smiles_list:
        mol2 = Chem.MolFromSmiles(item)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits)
        # 计算Tanimoto相似度
        # similarity = TanimotoSimilarity(fp1, fp2)
        # print("fp1",fp1.ToBitString())
        # print("fp2",fp2.ToBitString())
        similarity = DataStructs.FingerprintSimilarity(fp1, fp2)
        scores_list.append(similarity)
        similarity = TanimotoSimilarity(fp1, fp2)
        scores_list.append(similarity)
    return scores_list

def calcMoegen(smiles, smiles_list):
    scores_list = []


smiles, cosine_similarity_list, rank_smiles = get_smiles(df_embedding_filter, df_labels_filter, 5, type="full")

print("rank_smiles",rank_smiles)

query_molecule_path = 'path_to_query_molecule.png'
rank_1_path = 'path_to_rank_1.png'
rank_2_path = 'path_to_rank_2.png'
rank_3_path = 'path_to_rank_3.png'
rank_4_path = 'path_to_rank_4.png'
rank_5_path = 'path_to_rank_5.png'


def smiles2img(smiles, path):
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    img.save(path)

smiles2img(smiles, query_molecule_path)
smiles2img(rank_smiles[0], rank_1_path)
smiles2img(rank_smiles[1], rank_2_path)
smiles2img(rank_smiles[2], rank_3_path)
smiles2img(rank_smiles[3], rank_4_path)
smiles2img(rank_smiles[4], rank_5_path)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# 加载图片
query_molecule_img = mpimg.imread(query_molecule_path)
rank_1_img = mpimg.imread(rank_1_path)
rank_2_img = mpimg.imread(rank_2_path)
rank_3_img = mpimg.imread(rank_3_path)
rank_4_img = mpimg.imread(rank_4_path)
rank_5_img = mpimg.imread(rank_5_path)

ecfp_scores = calcECFP(smiles, rank_smiles)
print("ecfp_scores",ecfp_scores)