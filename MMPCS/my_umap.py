import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE


def plot_embedding(ax, data, label, dataset_name, embedding_name="full"):
    # embedding = reducer.fit_transform(data)
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    embedding = tsne.fit_transform(data)

    # for i in range(len(labels[0])):
        # label = [label[i] for label in labels]
        # label = labels[:, i]
    # label = np.array(label)
    mask = label >= 0
    # print("mask",i , len(mask))
    # print("embedding",i , len(embedding))
    # print("label",i , len(label))
    
    embedding_mask = embedding[mask,:]
    label = label[mask]
    print("embedding_mask",i , len(embedding_mask))
    print(f"umap_{dataset_name}")
    # print("label",i , len(label))
    colors = {0: '#a2d2e2', 1: '#12507b'}
    label_colors  = [colors[l] for l in label]
    # label = [int(item) for item in label]
    # print("label",i , label)
    unique_label = np.unique(label)
    unique_label = [int(item) for item in unique_label]
    
    ax.scatter(embedding_mask[:, 0], embedding_mask[:, 1], c=label_colors,  s=5, label=label)
    dbi = davies_bouldin_score(embedding_mask, label)
    ax.set_title(f"{embedding_name} - task: {i} - dbi: {dbi:.2f}")
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', label=unique_label[i], markerfacecolor=colors[i], markersize=5) for i in range(len(unique_label))]
    ax.legend(handles=legend_elements)
    # plt.title('UMAP Projection to 2D')
    # plt.show()
    # 保存图片
    # plt.savefig(f"umap_{dataset_name}_{embedding_name}_{i}.png")


dataset_name = "Estrogen"
# dataset_name = "ClinTox"

data_path = f"model_2023-11-23-18-48-58/finetune_epoch_1/df_embedding_{dataset_name}.csv"
label_path = f"model_2023-11-23-18-48-58/finetune_epoch_1/df_label_{dataset_name}.csv"

data = pd.read_csv(data_path).to_numpy().tolist()
# 对数据进行归一化
data = np.array(data)
print("100,100", data[100,100])
print("300,300", data[300,300])
data[:,:256] = (data[:,:256] - data[:,:256].min()) / (data[:,:256].max() - data[:,:256].min())
data[:,256:] = (data[:,256:] - data[:,256:].min()) / (data[:,256:].max() - data[:,256:].min())
print("100,100", data[100,100])
print("300,300", data[300,300])



labels = pd.read_csv(label_path).to_numpy().tolist()
label_0 = [label[0] for label in labels]
# print("label_0", label_0)
print("label_0", len(label_0))
# label_1 = [label[1] for label in labels]
# print("data", data)
# print("label", label)
# print(len(label))
# print(len(label[0]))
reducer = umap.UMAP()
data = np.array(data)
labels = np.array(labels)

fig, axes = plt.subplots(3, 2, figsize=(8, 12))

import time

for i in range(0,2):
    label = [label[i] for label in labels]
    print("label",i , len(label))
    label = np.array(label)
    # label = label[:, i]
    start = time.time()
    plot_embedding(axes[0,i], np.hstack((data[:,:256],data[:,256+128:])), label, dataset_name, "fusion_smiles")

    print("fusion_smiles time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    start = time.time()

    plot_embedding(axes[1,i], data[:,128:], label, dataset_name, "fusion_graph")

    print("fusion_graph time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    start = time.time()

    plot_embedding(axes[2,i],data, label, dataset_name, "concat")

    print("concat time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    start = time.time()

    # plot_embedding(axes[3,i],data[:,:256], label, dataset_name, "smiles")

    # print("smiles time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    # start = time.time()

    # plot_embedding(axes[4,i],data[:,256:], label, dataset_name, "graph")

    # print("graph time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    start = time.time()

    # plot_embedding(axes[5,i],data[:,:128], label, dataset_name, "ss")

    # print("ss time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    # start = time.time()

    # plot_embedding(axes[6,i],data[:,128:256], label, dataset_name, "su")

    # print("su time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    # start = time.time()

    

    # plot_embedding(axes[7,i],data[:,256:256+128], label, dataset_name, "gs")

    # print("gs time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    # start = time.time()

    # plot_embedding(axes[8,i],data[:,256+128:], label, dataset_name, "gu")

    # print("gu time", time.strftime("%H:%M:%S", time.gmtime(time.time()-start)))
    # start = time.time()

plt.tight_layout()
plt.savefig(f"umap_{dataset_name}.png", dpi=300)