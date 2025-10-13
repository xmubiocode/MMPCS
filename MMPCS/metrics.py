from sklearn.metrics import roc_auc_score, \
                            accuracy_score, \
                            recall_score, \
                            average_precision_score, \
                            mean_squared_error

import numpy as np
import torch

# 计算 ROC AUC
def get_ROC_AUC(pred, label):
    roc_auc = roc_auc_score(label, pred)
    return roc_auc

# 计算准确率
def get_accuracy(pred, label):
    
    accuracy = accuracy_score(label, pred)
    return accuracy

# 计算召回率
def get_recall(pred, label):
    
    recall = recall_score(label, pred)
    return recall

def get_PRC_AUC(pred, label):
    print("pred", pred)
    # prob = torch.softmax(torch.tensor(pred), dim=1)
    # positive = pred.tolist()
    print("positive", pred)
    print("label", label)
    prc_auc = average_precision_score(label, pred)
    return prc_auc

# 计算RMSE
def get_RMSE(pred, label):
    pred = np.array(pred)
    label = np.array(label)
    RMSE = np.sqrt(np.mean((pred-label)**2))
    return RMSE

# 给分类任务计算指标
def compute_metrics_classification(pred, label, eval_metrics="ROC_AUC"):

    label = label
    pred = pred
    # 计算 ROC AUC
    roc_auc = None
    prc_auc = None
    # accuracy = get_accuracy(pred, label)
    
    if eval_metrics == "ROC_AUC":
        try:
            roc_auc = get_ROC_AUC(pred, label)
        except OSError as e:
            print("pred", pred)
            print("label", label)
            print(e)
        # pred_label = torch.argmax(pred, dim=1)
        # accuracy = get_accuracy(pred, label)
        accuracy = 0
        # recall = get_recall(pred, label)
        recall = 0
        print(f"roc_auc:{roc_auc:.4f}, accuracy:{accuracy:.4f}")
        return roc_auc, accuracy, recall
    elif eval_metrics == "PRC_AUC":
        prc_auc = get_PRC_AUC(pred, label)
        # pred_label = torch.argmax(pred, dim=1)
        # pred_label 如果大于 0.5 就为 1， 否则为 0
        pred_label = []
        for p in pred:
            if p >= 0.5:
                pred_label.append(1)
            else:
                pred_label.append(0)
        accuracy = get_accuracy(pred_label, label)
        recall = get_recall(pred_label, label)
        print(f"prc_auc:{prc_auc:.4f}, accuracy:{accuracy:.4f}, recall:{recall:.4f}")
        return prc_auc, accuracy, recall


# 给回归任务计算指标
def compute_metrics_regression(pred, label, eval_metrics="RMSE"):
    # 计算 RMSE
    rmse1 = mean_squared_error(label, pred, squared=False)
    return rmse1

# 计算回归任务的指标
def rmse(y, f):
    return sqrt(((y - f) ** 2).mean(axis=0))