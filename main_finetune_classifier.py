from Dataset import FineTuneClassifierDataset, get_finetune_classification_dataset
from tqdm import tqdm
from finetune_model import FineTuneModelClassifier
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import torch_geometric
import numpy as np
import torch
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import torch.nn.functional as F

# 自己写的
from Dataset_info import DatasetInfo, DatasetLoadHelper
from loss import compute_info_nce_loss, compute_cov_loss, compute_classification_loss
from metrics import compute_metrics_classification
import Config

argparse = argparse.ArgumentParser()
argparse.add_argument("--dataset", type=str, default="BACE")
argparse.add_argument("--gpu_idx", type=str, default="cuda:0")
argparse.add_argument("--loss_type", type=int, default=2)
argparse.add_argument("--is_multi_gpu", type=bool, default=False)
argparse.add_argument("--local-rank", type=int, default=-1)
argparse.add_argument("--decay", type=float, default=0.0)
argparse.add_argument("--seed", type=int, default=42)
argparse.add_argument("--batch_size", type=int, default=64)

classification_loss_list = ["CELoss", "FocalLoss", "BCELoss"]
args = argparse.parse_args()

device = args.gpu_idx
# device = args.local_rank

dataset_name = args.dataset
loss_type = classification_loss_list[args.loss_type]
is_multi_gpu = args.is_multi_gpu

epoch = 100

if is_multi_gpu == True:
    device = args.local_rank
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl')


def set_seed():
    # 设置随机数种子
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # random.seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # torch_geometric.seed_all(args.seed)

set_seed()



# 创建数据集
# dataset = FineTuneClassifierDataset(filename="HIV.csv", smiles_col="smiles", label_col="HIV_active")


is_pretrain = True

datasetinfo = DatasetInfo(dataset_name)
datasetloadhelper = DatasetLoadHelper(datasetinfo)

# 创建训练集和测试集合
train_dataset, test_dataset, alpha_list = get_finetune_classification_dataset(datasetloadhelper=datasetloadhelper)
print("alpha_list", alpha_list)
print("len(train_dataset)", len(train_dataset))
print("len(test_dataset)", len(test_dataset))
# test_dataset  = FineTuneClassifierDataset(filename=datasetinfo.filename, smiles_col=datasetinfo.smiles_col, label_col=datasetinfo.label_col, type="test")

# print(len(dataset))
# 创建数据加载器
# batch_size 设为
batch_size = min(int(len(train_dataset)/100), 128)

batch_size = args.batch_size
print("batch_size", batch_size)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
if is_multi_gpu:
    # batch_size = args.batch_size
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True, sampler=train_sampler)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0, pin_memory=True)


# 创建模型
model = FineTuneModelClassifier(datasetloadhelper=datasetloadhelper, device=device, is_multi_gpu=is_multi_gpu)
model_dir = "model_2023-11-23-18-48-58"
# model_dir = "model_2023-12-26-20-10-47"

model_index = 1
if is_pretrain == True:
    model.load_model(model_dir, model_index)
print("device:", device)
model = model.to(device)
if is_multi_gpu == True:
    torch.cuda.set_device(args.local_rank)
    model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)

# 定义优化器
optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr, weight_decay=args.decay)

all_loss = []
x = [] 
# 查看有几个可用cuda
print(torch.cuda.device_count())

# 根据时间创建一个文件夹
import time
import os
# 创建文件夹
dir_name = os.path.join(model_dir, f"finetune_epoch_{model_index}")
if not os.path.exists(dir_name):
    os.mkdir(dir_name)

log_file_path = os.path.join(dir_name, f"log_{dataset_name}.txt")
log_summary_path = os.path.join(dir_name, f"log_summary.txt")

# 记录基本信息
if (is_multi_gpu == True and dist.get_rank() == 1) or is_multi_gpu == False:
    with open(log_file_path, "a") as f:
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"是否预训练: {is_pretrain}\n")
        if is_pretrain == True:
            f.write(f"预训练模型: {model_dir}\n")
            f.write(f"预训练模型序号: {model_index}\n")
        f.write(f"batch size: {batch_size}\n")
        f.write(f"使用的设备: {device}\n")
        f.write(f"种子: {args.seed}\n")
        f.write(f"学习率: {Config.lr}\n")
        f.write(f"损失函数: {loss_type}\n")
        f.write(f"decay: {args.decay}\n")
        # 开始时间
        start_time = time.time()
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))}\n")



max_eval_metrics = 0
max_other_eval_metrics = 0
max_epoch = 0

for i in range(epoch):
    if is_multi_gpu:
        train_dataloader.sampler.set_epoch(i)
    print(f"第{i+1}轮训练")
    train_bar = tqdm(train_dataloader, desc="训练")
    model.train()

    for smiles, graph, labels in train_bar:
        optimizer.zero_grad()
        graph = graph.to(device)
        # label = label.to(device)
        # 得到 SMILES 的表达, Graph 的表达
        smiles_rep, graph_rep, recon_loss, outputs = model(smiles, graph)
        # print(labels)
        # print(outputs)
        # pooling 
        
        # print()
        shared_smiles_rep = smiles_rep[:,:128]
        unique_smiles_rep = smiles_rep[:,128:]
        shared_graph_rep = graph_rep[:,:128]
        unique_graph_rep = graph_rep[:,128:]
        loss_cov_smiles = compute_cov_loss(shared_smiles_rep, unique_smiles_rep)
        loss_cov_graph = compute_cov_loss(shared_graph_rep, unique_graph_rep)

        loss_classification_list = []   
        valid_count = []     
        for index, (output, label) in enumerate(zip(outputs, labels)):
            mask = label >= 0

            # 统计 mask 中 True 的数量
            # 如果 True 的数量等于 0 就跳过
            if sum(mask) == 0:
                # print("mask 跳过")
                continue
            valid_count.append(sum(mask))
            # print("计算loss")
            label = label[mask].long()
            output = output[mask]
            # print("label", label)
            # print("output", output)
            # print("label len", len(label), "output len", len(output))
            
            label = label.to(device)
            output = output.to(device)
            # 这里要改的 加权平均的loss
            loss_classification = compute_classification_loss(output, label, loss_type, device=device, alpha = alpha_list[index])
            loss_classification_list.append(loss_classification)
        
        # print("valid_count", valid_count)
        loss_classification_sum = sum([x*y for x,y in zip(loss_classification_list, valid_count)])/sum(valid_count)
        # loss_info_nce = compute_info_nce_loss(shared_smiles_rep, shared_smiles_rep)
        loss_info_nce = 0
        print(f" loss_cov_smiles:{loss_cov_smiles.item():.4f}, loss_cov_graph:{loss_cov_graph.item():.4f}, recon_loss:{recon_loss.item():.4f}, loss_classification:{loss_classification_sum.item():.4f}")
        
        loss = loss_info_nce + 1 * (loss_cov_smiles + loss_cov_graph) + 1 * (recon_loss) + 10*(loss_classification_sum)
        loss.backward()
        optimizer.step()
        # print(loss.item())
        all_loss.append(loss.item())
        train_bar.set_description(f"loss:{loss.item():.4f}")
        train_bar.update(1)

    # 测试模型    
    model.eval()
    preds = []
    preds_label = [] 
    labels_all = []
    other_pred = [[] for i in range(len(alpha_list))]
    other_labels_all = [[] for i in range(len(alpha_list))]
    # print("dist.get_rank()", dist.get_rank())
    with torch.no_grad():
        for smiles, graph, labels in tqdm(test_dataloader):
            graph = graph.to(device)
            # label = label.to(device)
            _, _, _, outputs = model(smiles, graph)
            for task_id, (output, label) in enumerate(zip(outputs, labels)):
                # 如果 label 是 nan 就跳过
                mask = label >= 0
                if sum(mask) == 0:
                # print("mask 跳过")
                    continue
                label = label[mask].to(device).long()
                if datasetinfo.eval_metrics == "ROC_AUC":
                    # output = torch.argmax(output, dim=1)
                    output = F.softmax(output, dim=-1)
                elif datasetinfo.eval_metrics == "PRC_AUC":
                    output = torch.softmax(output, dim=1)
                    # output = output[:,1]
                output = output[mask]
                # preds.append(output.item())
                preds.extend(output.tolist())
                preds_label.extend(output[:,1].tolist())
                # labels_all.append(label.item())
                labels_all.extend(label.tolist())
                # other_pred[task_id].append(output.item())
                # other_labels_all[task_id].append(label.item())
    # print("preads", preds)
    print("===============epoch: ", i+1, "====================")
    # print("preds", preds)
    # print("labels_all", labels_all)
    if is_multi_gpu == True and  dist.get_rank() != 1:
        continue
    # predictions = preds[:,1]
    eval_metrics, acc, _ = compute_metrics_classification(preds_label, labels_all, datasetinfo.eval_metrics)
    other_eval_metrics = 0
    # for task_id in range(len(alpha_list)):
    #     if len(other_pred[task_id]) == 0:
    #         other_eval_metrics += 1
    #         continue
        # other_eval_metrics += compute_metrics_classification(other_pred[task_id], other_labels_all[task_id], datasetinfo.eval_metrics)[0]
    # other_eval_metrics /= len(alpha_list)
    if eval_metrics > max_eval_metrics:
        max_eval_metrics = eval_metrics
        max_epoch = i+1
    # if other_eval_metrics > max_other_eval_metrics:
    #     max_other_eval_metrics = other_eval_metrics
    #     max_other_epoch = i+1
    
    with open(log_file_path, "a") as f:
        f.write(f"===============epoch: {i+1}====================\n")
        timestamp = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
        str1 = f"epoch: {i+1}, {datasetinfo.eval_metrics}: {eval_metrics}, other:{other_eval_metrics} , acc: {acc}. timestamp: {timestamp}\n"
        print(str1)
        f.write(str1)

    # all loss 对应的x轴
    x= [*x, *np.linspace(i, i+1, len(all_loss)-len(x))]
    # 保存model
    # 加载模型



if (is_multi_gpu == True and dist.get_rank() == 1) or is_multi_gpu == False:
    with open(log_file_path, "a") as f:
        str2 = f"{dataset_name} 最好 {datasetinfo.eval_metrics} 为: {max_eval_metrics}, epoch: {max_epoch}\n\n\n\n"
        print(str2)
        f.write(str2)

    with open(log_summary_path, "a") as f:
        end_time = time.time()
        f.write(f"Dataset: {dataset_name}\n")
        f.write(f"是否预训练: {is_pretrain}\n")
        if is_pretrain == True:
            f.write(f"预训练模型: {model_dir}\n")
            f.write(f"预训练模型序号: {model_index}\n")
        f.write(f"batch size: {batch_size}\n")
        f.write(f"使用的设备: {device}\n")
        f.write(f"种子: {args.seed}\n")
        f.write(f"学习率: {Config.lr}\n")
        f.write(f"损失函数: {loss_type}\n")
        f.write(f"decay: {args.decay}\n")
        durition = end_time - start_time
        # 按时分秒的格式打印出来
        m, s = divmod(durition, 60)
        h, m = divmod(m, 60)
        f.write(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(start_time))}\n")
        f.write(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(end_time))}\n")
        f.write(f"总共用时: {h:.0f}小时{m:.0f}分钟{s:.0f}秒\n")
        # f.write(f"总共用时: {}\n")
        str2 = f"{dataset_name} 最好 {datasetinfo.eval_metrics} 为: {max_eval_metrics}, epoch: {max_epoch}, timestamp: {end_time}\n\n\n\n"
        print(str2)
        f.write(f"最终结果为: {eval_metrics}")
        f.write(str2)

    # 创建一个csv文件, 保存所有的rep
    import pandas as pd
    import numpy as np
    # # 保存rep
    df1 = pd.DataFrame()
    # # 保存labels
    df_labels_list = []
    df_embedding_list = []
    with torch.no_grad():
        model.eval()
        loaders = [train_dataloader, test_dataloader]
        for loader in loaders:  
            print("loader", len(loader))
            for smiles, graph, labels in tqdm(loader):
                graph = graph.to(device)
                smiles_rep, graph_rep, recon_loss, outputs = model(smiles, graph)
                # print("len(smiles)", len(smiles))
                # print("outputs", outputs)
                # print("len(outputs)",len(outputs))
                # outputs = torch.softmax(outputs, dim=1)
                # outputs = F.softmax(outputs[0], dim=-1)
                # print(outputs)
                outputs = [F.softmax(item, dim=-1) for item in outputs]
                # print(outputs)
                # outputs = outputs[:,:,1]
                
                print(len(outputs))
                print(len(outputs[0]))
                print(len(outputs[0][0]))
                # print(outputs.shape)
                pred_outputs = []
                for output in outputs:
                    pred_outputs.append(output[:,1].cpu().numpy())
                outputs = pred_outputs
                # print("outputs", outputs)
                # outputs = [max(item[0],item[1]).cpu().numpy() for item in outputs]
                
                # print("outputs", outputs)
                # print("smiles_rep", smiles_rep.shape)
                # print("graph_rep", graph_rep.shape)
                # print("concat", torch.cat([smiles_rep, graph_rep], dim=1).shape)
                df_embedding_list.append(torch.cat([smiles_rep, graph_rep], dim=1).cpu().numpy())

                if len(df_labels_list) == 0:
                    df_labels_list = [[] for i in range(2*len(labels)+1)]
                df_labels_list[0].extend(smiles)
                for index, label in enumerate(labels):
                    df_labels_list[2*index+1].extend(label.tolist())
                # df_labels_list[-1].extend(outputs)
                for index, output in enumerate(outputs):
                    df_labels_list[2*index+2].extend(output)
    # print("df_embedding_list", df_embedding_list.)
    # print("df_labels_list", df_labels_list)
    df_embedding = pd.DataFrame(np.concatenate(df_embedding_list, axis=0))
    # 对df_labels_list 进行转置
    # print(df_labels_list)
    print("df_label len", len(df_labels_list))
    for item in df_labels_list:
        print("item len", len(item))
    # df_labels_list = np.array(df_labels_list).T.tolist()
    df_lable = pd.DataFrame(df_labels_list).T
    # print("df_embedding", df_embedding)
    # print("df_lable", df_lable)
    
    df_embedding.to_csv(os.path.join(dir_name, f"df_embedding_{dataset_name}_heap.csv"), index=False)
    df_lable.to_csv(os.path.join(dir_name, f"df_label_{dataset_name}_heap.csv"), index=False)



    

if is_multi_gpu == True:
    dist.destroy_process_group()