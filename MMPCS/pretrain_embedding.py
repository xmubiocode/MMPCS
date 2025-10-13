from Model import MyModel
from Dataset import MyDataset
from torch_geometric.loader import DataLoader

from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
import pandas as pd
import numpy as np

import torch

# 加载预训练模型
device = "cuda:6"
model = MyModel(device=device)
model = model.to(device)

model.load_state_dict(torch.load("/home/xcy/projects/bib_ddp/model_2023-11-23-18-48-58/model_pretrain_1.pth"))

dataset = MyDataset(filename="zinc10M.csv", jump = True)
dataloader = DataLoader(dataset, batch_size=1000, shuffle=False, num_workers=0, pin_memory=True)


labels_list = []
df_embedding_list = []

with torch.no_grad():
    for smiles, graph in tqdm(dataloader):
        # smiles = smiles.to(device)
        graph = graph.to(device)
        # label = label.to(device)
        smiles_rep, graph_rep = model(smiles, graph)
        print(smiles_rep.shape, graph_rep.shape)
        smiles_mol_weight = []
        for smiles_item in smiles:
            mol = Chem.MolFromSmiles(smiles_item)
            mol_weight = Descriptors.MolWt(mol)
            smiles_mol_weight.append(mol_weight)
        # print(smiles_mol_weight)
        df_embedding_list.append(torch.cat([smiles_rep, graph_rep], dim=1).cpu().numpy())
        labels_list.extend(smiles_mol_weight)
        
df_embedding = pd.DataFrame(np.concatenate(df_embedding_list, axis=0))
df_lable = pd.DataFrame(labels_list)

df_embedding.to_csv("df_embedding.csv", index=False)
df_lable.to_csv("df_lable.csv", index=False)