from Dataset import MyDataset
from Dataset import FineTuneClassifierDataset, get_finetune_classification_dataset
from tqdm import tqdm
from Model import MyModel, AutoEncoder, FusionAE
from torch_geometric.loader import DataLoader
# 加载 globel mean pooling
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import os
from Dataset_info import DatasetInfo, DatasetLoadHelper
os.environ['CUDA_VISIBLE_DEVICES'] = '7'

import pandas as pd
import numpy as np

import argparse
argparse = argparse.ArgumentParser()
argparse.add_argument("--device", type=str, default="cuda:0")
args = argparse.parse_args()
device = args.device
dataset_name = "BBBP"
datasetinfo = DatasetInfo(dataset_name)
datasetloadhelper = DatasetLoadHelper(datasetinfo)
train_dataset, test_dataset, alpha_list = get_finetune_classification_dataset(datasetloadhelper=datasetloadhelper)
# print("alpha_list", alpha_list)
train_dataloader = DataLoader(train_dataset, batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)

# 创建模型
model = MyModel(device=device)
model = model.to(device)

all_embeddings = []
labels1 = []
# labels2 = []
smiles_list= []
with torch.no_grad():
    for data in train_dataloader:
        smiles, graph, label = data
        # smiles = smiles.to(device)
        graph = graph.to(device)
        # label = label.to(device)
        smiles_rep, graph_rep = model(smiles, graph)
        fusion_rep = torch.cat([graph_rep, smiles_rep], dim=1)
        # print("fusion_rep", fusion_rep.shape)
        # print("label", label)
        # print("shape", len(label))
        all_embeddings.append(fusion_rep)
        # all_labels.extend(label)
        labels1.extend(label[0])
        # labels2.extend(label[1])
        smiles_list.extend(smiles)
        # break
    for data in test_dataloader:
        smiles, graph, label = data
        # smiles = smiles.to(device)
        graph = graph.to(device)
        # label = label.to(device)
        smiles_rep, graph_rep = model(smiles, graph)
        fusion_rep = torch.cat([graph_rep, smiles_rep], dim=1)
        # print("fusion_rep", fusion_rep.shape)
        all_embeddings.append(fusion_rep)
        # all_labels.extend(label)
        labels1.extend(label[0])
        # labels2.extend(label[1])
        smiles_list.extend(smiles)
        # break
all_embeddings_tensor = torch.cat(all_embeddings, dim=0).cpu()
all_embeddings_np = all_embeddings_tensor.numpy()
print("all_embeddings_np", all_embeddings_np.shape)

# 创建 DataFrame
df = pd.DataFrame(all_embeddings_np)
# print(df.shape())

# 保存为 CSV 文件
df.to_csv('no_pretrain_embeddings.csv', index=False,header=False)
# 保存标签
# print(all_lables)


# print("labels1", labels1)
# print("labels2", labels2)
# 对应两个列
print(type(labels1))
print(len(labels1))
print(type(labels1[0]))
# 把 list 里面每一个tensor 转化为 numpy
labels1 = [i.numpy() for i in labels1]
# labels2 = [i.numpy() for i in labels2]
df = pd.DataFrame({'smiles':smiles_list,'labels1': labels1})

df.to_csv('no_pretrain_labels.csv', index=False,header=False)

model.load_state_dict(torch.load("/home/xcy/projects/bib_ddp/model_2023-11-23-18-48-58/model_pretrain_1.pth"))
all_embeddings = []
labels1 = []
labels2 = []
smiles_list= []
with torch.no_grad():
    for data in train_dataloader:
        smiles, graph, label = data
        # smiles = smiles.to(device)
        graph = graph.to(device)
        # label = label.to(device)
        smiles_rep, graph_rep = model(smiles, graph)
        fusion_rep = torch.cat([graph_rep, smiles_rep], dim=1)
        # print("fusion_rep", fusion_rep.shape)
        # print("label", label)
        # print("shape", len(label))
        all_embeddings.append(fusion_rep)
        # all_labels.extend(label)
        labels1.extend(label[0])
        # labels2.extend(label[1])
        smiles_list.extend(smiles)
        # break
    for data in test_dataloader:
        smiles, graph, label = data
        # smiles = smiles.to(device)
        graph = graph.to(device)
        # label = label.to(device)
        smiles_rep, graph_rep = model(smiles, graph)
        fusion_rep = torch.cat([graph_rep, smiles_rep], dim=1)
        # print("fusion_rep", fusion_rep.shape)
        all_embeddings.append(fusion_rep)
        # all_labels.extend(label)
        labels1.extend(label[0])
        # labels2.extend(label[1])
        smiles_list.extend(smiles)
        # break
all_embeddings_tensor = torch.cat(all_embeddings, dim=0).cpu()
all_embeddings_np = all_embeddings_tensor.numpy()
print("all_embeddings_np", all_embeddings_np.shape)

# 创建 DataFrame
df = pd.DataFrame(all_embeddings_np)
# print(df.shape())

# 保存为 CSV 文件
df.to_csv('have_pretrain_embeddings.csv', index=False,header=False)
# 保存标签
# print(all_lables)


# print("labels1", labels1)
# print("labels2", labels2)
# 对应两个列
print(type(labels1))
print(len(labels1))
print(type(labels1[0]))
# 把 list 里面每一个tensor 转化为 numpy
labels1 = [i.numpy() for i in labels1]
# labels2 = [i.numpy() for i in labels2]
df = pd.DataFrame({'smiles':smiles_list,'labels1': labels1})

df.to_csv('have_pretrain_labels.csv', index=False,header=False)
