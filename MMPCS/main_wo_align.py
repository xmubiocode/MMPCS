from Dataset import MyDataset
from tqdm import tqdm
from Model import MyModel, AutoEncoder, FusionAE
from torch_geometric.loader import DataLoader
# 加载 globel mean pooling
from torch_geometric.nn import global_mean_pool
import torch
import torch.nn as nn
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
import argparse
import numpy as np

from loss import compute_info_nce_loss, compute_cov_loss

parser = argparse.ArgumentParser()
parser.add_argument("--plot_name", type=str, default="plot-5e-5.png")
parser.add_argument("--lr", type=float, default=5e-5)
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()


print("args:", args)
epoch = 1
device = args.device
# 创建数据集
dataset = MyDataset(filename="zinc10M.csv")

# 创建数据加载器
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

# 创建模型
model = MyModel(device=device)
model = model.to(device)
# ae_smiles_to_graph = AutoEncoder()
# ae_smiles_to_graph = ae_smiles_to_graph.to(device)
# ae_graph_to_smiles = AutoEncoder()
# ae_graph_to_smiles = ae_graph_to_smiles.to(device)

# fusion_ae_to_smiles = FusionAE()
# fusion_ae_to_smiles = fusion_ae_to_smiles.to(device)

# fusion_ae_to_graph = FusionAE()
# fusion_ae_to_graph = fusion_ae_to_graph.to(device)


# 定义优化器
# optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
optimizer = torch.optim.Adam(list(model.parameters()), 
                            lr=args.lr)

# 定义损失函数，InfoNCE Loss
import torch
import torch.nn.functional as F



# 模型训练

all_loss = []
x = [] 
# 查看有几个可用cuda
print(torch.cuda.device_count())

# 根据时间创建一个文件夹
import time
import os
# 创建文件夹
dir_name = f"model_wo_align_{time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time()))}"
os.mkdir(dir_name)

# 保存参数文件
with open(os.path.join(dir_name, "args.txt"), "w") as f:
    f.write("epoch: "+str(epoch)+"\n")
    f.write("lr: "+str(args.lr)+"\n")


for i in range(epoch):
    print(f"第{i}轮训练")
    train_bar = tqdm(dataloader, desc="训练")
    model.train()
    # ae_smiles_to_graph.train()
    # ae_graph_to_smiles.train()
    
    for smiles, graph in train_bar:
        optimizer.zero_grad()
        graph = graph.to(device)
        # 得到 SMILES 的表达, Graph 的表达
        smiles_rep, graph_rep = model(smiles, graph)
        # pooling 
        # graph_rep = global_mean_pool(graph_rep, graph.batch)
        # print()
        shared_smiles_rep = smiles_rep[:,:128]
        unique_smiles_rep = smiles_rep[:,128:]
        shared_graph_rep = graph_rep[:,:128]
        unique_graph_rep = graph_rep[:,128:]
        loss_cov_smiles = compute_cov_loss(shared_smiles_rep, unique_smiles_rep)
        loss_cov_graph = compute_cov_loss(shared_graph_rep, unique_graph_rep)

        fusion_rep_1 =  torch.cat((smiles_rep[:,:128],smiles_rep[:,128:], graph_rep[:,128:]), 1)
        fusion_rep_2 =  torch.cat((smiles_rep[:,128:], graph_rep[:,:128], graph_rep[:,128:]), 1)
        # loss_fu_smiles = fusion_ae_to_smiles(fusion_rep_1, smiles_rep)
        # loss_fu_graph = fusion_ae_to_graph(fusion_rep_2, graph_rep)



        # 重构损失
        # recon_loss_smiles_to_graph = ae_smiles_to_graph(shared_smiles_rep, shared_graph_rep)
        # recon_loss_graph_to_smiles = ae_graph_to_smiles(shared_graph_rep, shared_smiles_rep)
       
        
        # loss_info_nce = compute_info_nce_loss(shared_smiles_rep, shared_smiles_rep)
        loss_info_nce = 0
        print(f" loss_cov_smiles:{loss_cov_smiles.item():.4f}, loss_cov_graph:{loss_cov_graph.item():.4f}")
        loss = loss_info_nce + 1*(loss_cov_smiles + loss_cov_graph)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        all_loss.append(loss.item())
        train_bar.set_description(f"loss:{loss.item():.4f}")
        train_bar.update(1)
    # all loss 对应的x轴
    x= [*x, *np.linspace(i, i+1, len(all_loss)-len(x))]
    model.eval()
    # 保存model
    model_path = os.path.join(dir_name, f"model_pretrain_{i+1}.pth")
    torch.save(model.state_dict(), model_path)
    # ae_smiles_to_graph_path = os.path.join(dir_name, f"ae_smiles_to_graph_{i+1}.pth")
    # torch.save(ae_smiles_to_graph.state_dict(), ae_smiles_to_graph_path)
    # ae_graph_to_smiles_path = os.path.join(dir_name, f"ae_graph_to_smiles_{i+1}.pth")
    # torch.save(ae_graph_to_smiles.state_dict(), ae_graph_to_smiles_path)
    # fusion_ae_to_smiles_path = os.path.join(dir_name, f"fusion_ae_to_smiles_{i+1}.pth")
    # torch.save(fusion_ae_to_smiles.state_dict(), fusion_ae_to_smiles_path)
    # fusion_ae_to_graph_path = os.path.join(dir_name, f"fusion_ae_to_graph_{i+1}.pth")
    # torch.save(fusion_ae_to_graph.state_dict(), fusion_ae_to_graph_path)

    # 加载模型
train_bar.close()

# 画loss图像保存
# import matplotlib.pyplot as plt
# # x 轴为x ，y 轴为 all_loss
# plt.plot(x, all_loss)
# plot_path = os.path.join(dir_name, args.plot_name)
# plt.savefig(plot_path)
# # plt.show()
