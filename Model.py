
from smiles_encoder import SmilesRoBERTaEncoder
from graph_encoder import GNN
import torch.nn as nn
import torch
from torch_geometric.nn import global_mean_pool

# 定义一个深度学习模型
class MyModel(nn.Module):
    def __init__(self, device="cpu"):
        super(MyModel, self).__init__()
        self.device = device
        self.smiles_encoder = SmilesRoBERTaEncoder(device)
        self.grap_encoder = GNN(3, 256, JK="last", drop_ratio=0.1, gnn_type="gin")
        self.encoder = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
        )
    
    def forward(self, smiles, graph):
        smiles_rep = self.smiles_encoder(smiles)
        graph_rep = self.grap_encoder(graph)
        graph_rep = global_mean_pool(graph_rep, graph.batch)
        graph_rep = self.encoder(graph_rep)
        return smiles_rep, graph_rep
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
        )
    def forward(self, input, map_output):
        latent = self.encoder(input)
        recon_map_output = self.decoder(latent)
        # 计算重构损失
        loss = torch.mean((recon_map_output - map_output) ** 2)
        return loss
    
class FusionAE(nn.Module):
    def __init__(self):
        super(FusionAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(128*3, 128*2),
            nn.ReLU(),
            nn.Linear(128*2, 128),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 128*2),
            nn.ReLU(),
            nn.Linear(128*2, 128*2),
        )
    def forward(self, fusion_rep, origin_rep):
        latent = self.encoder(fusion_rep)
        recon_map_output = self.decoder(latent)
        # 计算重构损失
        loss = torch.mean((recon_map_output - origin_rep) ** 2)
        return loss