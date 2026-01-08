import numpy as np
import pandas as pd
import umap
import matplotlib.pyplot as plt

def plot_embedding(data, labels, dataset_name, embedding_name="full"):
    embedding = reducer.fit_transform(data)
    print("embedding_name", embedding_name)
    for i in range(len(labels[0])):
        label = labels[:, i]
        # 将label画一个分布图
        # plt.figure()
        # plt.hist(label, bins=100)
        # plt.title('label distribution')
        # plt.show()
        sc = plt.scatter(embedding[:, 0], embedding[:, 1], c=label, cmap='viridis', s=0.05, marker='.')
        plt.title('UMAP Projection to 2D')
        plt.colorbar(sc)
        plt.show()
        # 保存图片
        plt.savefig(f"umap_{dataset_name}_{embedding_name}_{i}.png", dpi=300)



dataset_name = "ori"
# dataset_name = "ClinTox"

data_path = f"/home/xcy/projects/bib_ddp/df_embedding.csv"
label_path = f"/home/xcy/projects/bib_ddp/df_lable.csv"

data = pd.read_csv(data_path).to_numpy().tolist()[:20000]
labels = pd.read_csv(label_path).to_numpy().tolist()[:20000]
# 根据 labels 的分布选取均匀分布的数据

# label_0 = [label[0] for label in labels]
# print("label_0", label_0)
# print("label_0", len(label_0))
# label_1 = [label[1] for label in labels]
# print("data", data)
# print("label", label)
# print(len(label))
# print(len(label[0]))
reducer = umap.UMAP()
data = np.array(data)
labels = np.array(labels)

import time

start = time.time()

# plot_embedding(data[:,256:256+128], labels, dataset_name, "gs")

# print("gs time", time.strftime("%Y-%m-%d %H:%M:%S", time.time()-start))

# plot_embedding(data[:,:128], labels, dataset_name, "ss")

# print("ss time", time.strftime("%Y-%m-%d %H:%M:%S", time.time()-start))


plot_embedding(np.hstack((data[:,:256],data[:,256+128:])), labels, dataset_name, "fusion_2")

# plot_embedding(data[:,128:], labels, dataset_name, "fusion_2_other")


# # end = time.time()
# print("fusion time", time.strftime("%Y-%m-%d %H:%M:%S", time.time()-start))

# plot_embedding(data, labels, dataset_name, "full_2")

# print("full time", time.strftime("%Y-%m-%d %H:%M:%S", time.time()-start))

# plot_embedding(data[:,:256], labels, dataset_name, "smiles_2")

# print("smiles time", time.strftime("%Y-%m-%d %H:%M:%S", time.time()-start))

# plot_embedding(data[:,256:], labels, dataset_name, "graph_2")

# print("graph time", time.strftime("%Y-%m-%d %H:%M:%S", time.time()-start))