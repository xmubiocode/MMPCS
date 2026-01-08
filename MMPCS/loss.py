import torch

def compute_info_nce_loss(tensor1, tensor2):
    # tensor1和tensor2分别表示两组表示向量，每个Tensor的每一行是一个100维的表示向量
    
    num_samples = tensor1.shape[0]
    loss = 0.0

    for i in range(num_samples):
        temperature = 1.0
        # 计算正样本对的相似度分数 内积
        similarity_positive = torch.dot(tensor1[i], tensor2[i])

        similarity_positive /= temperature
        similarity_positive = torch.exp(similarity_positive)
        # print(similarity_positive)

        # 构造负样本并计算相似度分数
        similarity_negative = 0.0
        for j in range(num_samples):
            if j != i:
                temp = torch.dot(tensor1[i], tensor2[j])
                similarity_negative += torch.exp(temp / temperature)

        # 计算当前正样本对的InfoNCE损失
        current_loss = -torch.log(similarity_positive / (similarity_positive + similarity_negative))
        
        # 将当前正样本对的损失累加到总损失中
        loss += current_loss

    # 对所有正样本对的损失取平均
    loss /= num_samples

    return loss

def compute_cov_loss(tensor1,tensor2):
    # print(tensor1.shape)
    # print("tensor1.shape[1]", tensor1.shape[1])
    covariances = torch.zeros(tensor1.shape[0])
    for i in range(tensor1.shape[0]):
        covariances[i] = torch.dot(tensor1[i]-tensor1[i].mean(), tensor2[i]-tensor2[i].mean())

    return torch.sum(covariances**2)/covariances.shape[0]

def CELoss(output, label):
    loss = torch.nn.CrossEntropyLoss()
    return loss(output, label)

# 分类损失
def compute_classification_loss(output, label, loss_type="CELoss",alpha=0.8, device="cpu"):
    if loss_type == "CELoss":
        loss = CELoss(output, label)
    elif loss_type == "FocalLoss":
        loss = focal_loss(output, label, alpha=alpha, device=device)
    # loss = torch.nn.CrossEntropyLoss()
    elif loss_type == "BCELoss":
        output = F.softmax(output, dim=1)
        output = output[:,1]
        label = label.float()
        loss = torch.nn.BCELoss()(output, label)
    else:
        raise ValueError("不存在此Loss")
    return loss

def compute_regression_loss(output, label, loss_type="MSE"):
    if loss_type == "MSE":
        loss = torch.nn.MSELoss()
        # print("output", output.\\)
        output = output.squeeze(1)
        # print("label", label)
        return loss(output, label)
    if loss_type == "RMSE":
        loss = torch.nn.MSELoss()
        return torch.sqrt(loss(output, label))
    else:
        raise ValueError("不存在此Loss")
    


import torch
import torch.nn as nn
import torch.nn.functional as F



def focal_loss(inputs, targets, alpha=0.75, gamma=2, device = "cpu", reduction='mean'):
    # 计算交叉熵损失
    ce_loss = F.cross_entropy(inputs, targets, reduction='none')
    # print("alpha", alpha)

    # 计算类别权重
    class_weights = torch.tensor([1 - alpha, alpha], dtype=torch.float32)
    class_weights = class_weights.to(device)
    weights = class_weights[targets]
    # weights = weights.to(device)

    # 计算 Focal Loss
    focal_loss = weights * (1 - torch.exp(-ce_loss))**gamma * ce_loss

    # 根据 reduction 参数进行降维
    if reduction == 'mean':
        return focal_loss.mean()
    elif reduction == 'sum':
        return focal_loss.sum()
    else:
        return focal_loss

