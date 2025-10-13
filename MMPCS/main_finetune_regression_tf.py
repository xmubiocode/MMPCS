
from tqdm import tqdm

from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch_geometric
import numpy as np
import torch
import argparse

# 自己写的
from finetune_model import FineTuneModelRegression
from Dataset import FineTuneClassifierDataset, get_finetune_classification_dataset, get_finetune_regression_dataset
from Dataset_info import DatasetInfo, DatasetLoadHelper
from loss import compute_info_nce_loss, compute_cov_loss, compute_regression_loss
from metrics import compute_metrics_regression
import Config

argparse = argparse.ArgumentParser()
argparse.add_argument("--dataset", type=str, default="ESOL")
argparse.add_argument("--gpu", type=str, default="cuda:0")
argparse.add_argument("--loss_type", type=int, default=0)
argparse.add_argument("--epoch", type=int, default=100)
argparse.add_argument("--seed", type=int, default=1)

classification_loss_list = ["MSE", "RMSE"]

args = argparse.parse_args()

device = args.gpu
dataset_name = args.dataset
loss_type = classification_loss_list[args.loss_type]
epoch = args.epoch
seed = args.seed






# 设置随机数种子
# np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# 创建数据集
is_pretrain = True

datasetinfo = DatasetInfo(dataset_name)
datasetloadhelper = DatasetLoadHelper(datasetinfo)

# 创建训练集和测试集合
train_dataset, test_dataset = get_finetune_regression_dataset(datasetloadhelper=datasetloadhelper)

# 创建数据加载器
# batch_size 设为
batch_size = min(int(len(train_dataset)/100), 64)
# if batch_size < 16:
#     batch_size = 16
batch_size = 32
print("batch_size", batch_size)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=0, pin_memory=True)


# 创建模型
model = FineTuneModelRegression(device=device)
model_dir = "model_2023-12-26-20-10-47"
model_index = 1
if is_pretrain == True:
    model.load_model(model_dir, model_index)
model = model.to(device)

# 定义优化器
lr = 5e-5
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

all_loss = []
x = [] 
# 查看有几个可用cuda
print(torch.cuda.device_count())

# 根据时间创建一个文件夹
import time
import os
# 创建文件夹
dir_name = os.path.join(model_dir, f"finetune_epoch_{model_index}_tf")
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

log_file_path = os.path.join(dir_name, f"log_regre_{dataset_name}.txt")

# 记录基本信息
with open(log_file_path, "a") as f:
    f.write(f"Dataset: {dataset_name}\n")
    f.write(f"是否预训练: {is_pretrain}\n")
    if is_pretrain == True:
        f.write(f"预训练模型: {model_dir}\n")
        f.write(f"预训练模型序号: {model_index}\n")
        f.write(f"batch size: {batch_size}\n")
        f.write(f"使用的设备: {device}\n")
        f.write(f"种子: {seed}\n")
        f.write(f"学习率: {Config.lr}\n")
        f.write(f"损失函数: {loss_type}\n")
    # 开始时间
    f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}\n")



min_eval_metrics = 100000
min_epoch = 0

for i in tqdm(range(epoch)):
    print(f"第{i+1}轮训练")
    train_bar = tqdm(train_dataloader, desc="训练")
    model.train()

    for smiles, graph, label in train_bar:
        optimizer.zero_grad()
        graph = graph.to(device)
        label = label.to(device)
        # 得到 SMILES 的表达, Graph 的表达
        try:
            output = model.forward_tf(smiles, graph)
        except:
            print("smiles", smiles)
            print("graph", graph)
            print("label", label)
            # raise Exception("error")
            continue
        # print(labels)
        # print(outputs)
        # pooling 
        
        # print()
        # shared_smiles_rep = smiles_rep[:,:128]
        # unique_smiles_rep = smiles_rep[:,128:]
        # shared_graph_rep = graph_rep[:,:128]
        # unique_graph_rep = graph_rep[:,128:]
        # loss_cov_smiles = compute_cov_loss(shared_smiles_rep, unique_smiles_rep)
        # loss_cov_graph = compute_cov_loss(shared_graph_rep, unique_graph_rep)

        loss_regression = compute_regression_loss(output, label, loss_type=loss_type)
        
        # loss_info_nce = compute_info_nce_loss(shared_smiles_rep, shared_smiles_rep)
        loss_info_nce = 0
        print(f" loss_regression:{loss_regression.item():.4f}")
        
        loss = loss_regression
        loss.backward()
        optimizer.step()
        # print(loss.item())
        all_loss.append(loss.item())
        train_bar.set_description(f"loss:{loss.item():.4f}")
        train_bar.update(1)

    # 测试模型    
    model.eval()
    with torch.no_grad():
        preds = []
        labels_all = []
        for smiles, graph, labels in tqdm(test_dataloader):
            graph = graph.to(device)
            # label = label.to(device)
            output = model.forward_tf(smiles, graph)
            output = output.squeeze(1)
            # print("output", output.shape)
            # print("labels", labels.shape)
            preds = [*preds, *output.tolist()]
            labels_all = [*labels_all, *labels.tolist()]
            

        print("===============epoch: ", i+1, "====================")
        # print("preds", preds)
        # print("labels_all", labels_all)
        eval_metrics = compute_metrics_regression(preds, labels_all, datasetinfo.eval_metrics)
        print(f"{datasetinfo.eval_metrics}: {eval_metrics}")
        if eval_metrics < min_eval_metrics:
            min_eval_metrics = eval_metrics
            min_epoch = i+1
        with open(log_file_path, "a") as f:
            f.write(f"===============epoch: {i+1}====================\n")
            timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
            f.write(f"epoch: {i+1}, {datasetinfo.eval_metrics}: {eval_metrics}, timestamp: {timestamp}\n")
        if i+1 == 100:
            with open(os.path.join(dir_name, f"log_summary.txt"), "a") as f:
                f.write(f"\n\n")
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                f.write(f"dataset:{dataset_name}, RMSE: {eval_metrics}, 最好RMSE: {min_eval_metrics}, 在epoch: {min_epoch}, timestamp: {timestamp}\n")
    # all loss 对应的x轴
    x= [*x, *np.linspace(i, i+1, len(all_loss)-len(x))]
    model.eval()
    # 保存model
    # 加载模型

with open(log_file_path, "a") as f:
    print(f"{dataset_name} 最好 {datasetinfo.eval_metrics} 为: {min_eval_metrics}, epoch: {min_epoch}\n\n")
    f.write(f"{dataset_name} 最好 {datasetinfo.eval_metrics} 为: {min_eval_metrics}, epoch: {min_epoch}\n\n\n\n")
train_bar.close()

# 画loss图像保存
# import matplotlib.pyplot as plt
# # x 轴为x ，y 轴为 all_loss
# plt.plot(x, all_loss)
# plot_path = os.path.join(dir_name, "plot-5e-5.png")
# plt.savefig(plot_path)
# plt.show()
