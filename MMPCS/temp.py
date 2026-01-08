# # from smiles_encoder import SmilesRoBERTaEncoder
# import torch
# import torch.nn as nn
# from transformers import RobertaModel, RobertaTokenizer

# device = "cuda:5"
# # smiles_encoder = SmilesRoBERTaEncoder(device)

# smiles = "CC(=O)OC1=CC=CC=C1C(=O)O"

# tokenizer = RobertaTokenizer.from_pretrained("roberta")
# roberta_model = RobertaModel.from_pretrained("roberta").to(device)

# for smiles in ["C","CC","CC("]:
#     inputs = tokenizer(
#             smiles,
#             add_special_tokens=True,
#             max_length=256,
#             padding='max_length',
#             return_tensors='pt',
#             truncation=True,
#         ).to(device)

#     inputs = {k: v for k, v in inputs.items()}
#     input_ids = inputs["input_ids"]

#     print(input_ids)

import pandas as pd
import numpy as np

dataset_name = "BBBP"
# dataset_name = "Estrogen"
df_embedding = pd.read_csv(f"/home/xcy/projects/bib_ddp/model_2023-11-23-18-48-58/finetune_epoch_1/df_embedding_{dataset_name}_heap.csv", index_col=False)
df_label = pd.read_csv(f"/home/xcy/projects/bib_ddp/model_2023-11-23-18-48-58/finetune_epoch_1/df_label_{dataset_name}_heap.csv", index_col=False)


print(len(df_embedding))
# print(len(df_label))
print(df_label.head(10))

# filtered_df = df_label.loc[
#     ( (df_label["1"] == 0))
# ]
# filtered_df = df_label.loc[
#     ((df_label["1"] == 1) & (df_label["2"] > 0.5)) |
#     ((df_label["1"] == 0) & (df_label["2"] < 0.2))
# ]
# print(len(filtered_df))
# print(filtered_df.head(10))
positive_index = []
negative_index = []
positive_error_index = []
negative_error_index = []
for index, row in df_label.iterrows():
    # print("index",index)
    if row["1"] == 1 and row["2"] > 0.95:
        positive_index.append((index,row["0"],row["2"], row["1"]))
    elif row["1"] == 0 and row["2"] < 0.15:
        negative_index.append((index,row["0"],row["2"], row["1"]))
    elif row["1"] == 1 and row["2"] < 0.5 and row["2"] > 0.4:
        positive_error_index.append((index,row["0"],row["2"],row["1"]))
    elif row["1"] == 0 and row["2"] > 0.5 and row["2"] < 0.6:
        negative_error_index.append((index,row["0"],row["2"],row["1"]))
# filtered_df_1 = filtered_df.loc[
#     (filtered_df["1"] == 1)
# ]
print(df_label.loc[0])
# print(positive_index)
print(negative_index)

import random
# random.seed(42)
# random_positve_list = random.sample(positive_index, 8)
num_molecules = 8
positive_index = sorted(positive_index, key=lambda x: x[2], reverse=True)
random_positve_list = [positive_index[min(i * (len(positive_index)//(num_molecules-1)),len(positive_index)-1)] for i in range(num_molecules)]
random_positve_list = sorted(random_positve_list, key=lambda x: x[2], reverse=True)
# error = random.sample(negative_error_index, 1)
# random_positve_list.append(error[0])
# random_negative_list = random.sample(negative_index, 8)
negative_index = sorted(negative_index, key=lambda x: x[2], reverse=True)
random_negative_list = [negative_index[min(i * (len(negative_index)//(num_molecules-1)),len(negative_index)-1)] for i in range(num_molecules)]
random_negative_list = sorted(random_negative_list, key=lambda x: x[2], reverse=True)
# error = random.sample(negative_error_index, 1)
# random_negative_list.append(error[0])
print(random_positve_list)
print(random_negative_list)

embedding_list = []
for index, smiles, _, _ in random_positve_list:
    embedding_list.append(df_embedding.loc[index].tolist())
for index, smiles, _, _ in random_negative_list:
    embedding_list.append(df_embedding.loc[index].tolist())

print(len(embedding_list))
print(len(embedding_list[0]))

# import numpy as np

def calc_c(embedding_list, cancat_type="full"):
    embedding_array = np.array(embedding_list)
    result = []
    result = [[] for _ in range(len(embedding_list))]
    for i in range(len(embedding_list)):
        for j in range(len(embedding_list)):
            if cancat_type == "full":
                A = embedding_array[i]
                B = embedding_array[j]
            elif cancat_type == "smiles":
                A = np.concatenate((embedding_array[i][:256],embedding_array[i][256+128:512]))
                B = np.concatenate((embedding_array[j][:256],embedding_array[j][256+128:512]))
            elif cancat_type == "graph":
                A = embedding_array[i][128:]
                B = embedding_array[j][128:]
            result[i].append(np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B)))

    return result

result = calc_c(embedding_list,"graph")
# print(result)

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,LinearSegmentedColormap,Normalize

plt.figure(figsize=(10, 10))
white = (1,1,1)
light_blue3 = (200/255, 212/255, 224/255)
light_blue2 = (186/255,197/255,214/255)
light_blue = (162/255,182/255,204/255)
deep_blue = (9/255, 48/255, 107/255)
cmap_name = 'custom_cmap'
colors = [white, light_blue2,light_blue,deep_blue]
cmap = LinearSegmentedColormap.from_list(cmap_name, colors)

norm = Normalize(vmin=-0.4, vmax=1)
# 使用 imshow 绘制矩阵
cax = plt.imshow(result, cmap=cmap, aspect='equal', norm=norm)

# 添加颜色条
plt.colorbar(cax, ticks=[-0.4, -0.2, 0, 0.2, 0.4, 0.6, 0.8, 1], shrink=0.8)

# 添加数字标签
for i in range(16):
    for j in range(16):
        text_color = 'white' if result[i][j] > 0.5 else 'black'
        text = f'{result[i][j]:.2f}'
        if text == "1.00":
            text = "1"
        plt.text(j, i, text, ha='center', va='center', color=text_color,fontweight=30)

# 设置标题和轴标签
labels = [f'M{i+1}' for i in range(16)]
plt.title('Molecules Cosine Similarity Heatmap')
plt.xticks(ticks=np.arange(16), labels=labels)
plt.yticks(ticks=np.arange(16), labels=labels)
plt.xlabel('Molecules')
plt.ylabel('Molecules')

# 显示图形
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # 调整子图参数
# plt.axis('off')
plt.show()
plt.savefig("temp-result------.png", bbox_inches='tight', pad_inches=0.3)

# print(len(filtered_df_1))
# print(filtered_df_1.head(10))

# filtered_df_0 = filtered_df.loc[
#     (filtered_df["1"] == 0)
# ]
# print(len(filtered_df_0))
# print(filtered_df_0.head(10))