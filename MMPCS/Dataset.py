from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data
import torch
import pandas as pd
from rdkit import Chem
import numpy as np
import pickle
from tqdm import  tqdm
import os
from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed

from rdkit.Chem.Scaffolds import MurckoScaffold


# 自己的文件
import Config






allowable_features = {
    'possible_atomic_num_list':       list(range(1, 119)),
    'possible_formal_charge_list':    [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list':        [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list':    [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list':             [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list':           [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds':                 [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs':             [  # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ]
}

def mol_to_graph_data_obj_simple(mol):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    
    atom_features_list = []
    for atom in mol.GetAtoms():
        atom_feature = [allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + \
                       [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    if len(mol.GetBonds()) <= 0:  # mol has no bonds
        num_bond_features = 2  # bond type & direction
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)
    else:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = [allowable_features['possible_bonds'].index(bond.GetBondType())] + \
                           [allowable_features['possible_bond_dirs'].index(bond.GetBondDir())]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    return data

def init_process_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            data = mol_to_graph_data_obj_simple(mol)
            smiles = Chem.MolToSmiles(mol)
            return smiles, data
        else:
            return None
    except Exception as e:
        print("Error", e, smiles)
        return None

def generate_scaffold(smiles, include_chirality=False):
    """ Obtain Bemis-Murcko scaffold from smiles
    :return: smiles of scaffold """
    # print(smiles)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(
        smiles=smiles, includeChirality=include_chirality)
    return scaffold

# 创建数据集
# 包含smiles和图数据

def get_scaffold_split_ind(smiles_list):
    scaffolds = {} 

    for ind, smiles in enumerate(smiles_list):
        scaffold = generate_scaffold(smiles=smiles,include_chirality= True)
        if scaffold is not None:
            if scaffold not in scaffolds:
                scaffolds[scaffold] = [ind]
            else:
                scaffolds[scaffold].append(ind)

    scaffolds = {key: sorted(value) for key, value in scaffolds.items()}
    scaffold_sets = [
                scaffold_set
                for (scaffold,
                    scaffold_set) in sorted(scaffolds.items(),
                                            key=lambda x: (len(x[1]), x[1][0]),
                                            reverse=True)
            ]
    train_inds = []
    valid_inds = []
    test_inds = []
    train_cutoff = 0.8 * len(smiles_list)
    valid_cutoff = 0.9 * len(smiles_list)

    for scaffold_set in scaffold_sets:
        # set_len = len(scaffold_set)
        # if set_len >= 10:
        #     train_inds += scaffold_set[:int(0.8)*set_len]
        #     valid_inds += scaffold_set[int(0.8)*set_len:int(0.9)*set_len]
        #     test_inds  += scaffold_set[int(0.9)*set_len:]
        # else:
            if len(train_inds) + len(scaffold_set) > train_cutoff:
                if len(train_inds) + len(valid_inds) + len(
                        scaffold_set) > valid_cutoff:
                    test_inds += scaffold_set
                else:
                    valid_inds += scaffold_set
            else:
                train_inds += scaffold_set
    return train_inds, valid_inds, test_inds


class MyDataset(Dataset):
    def __init__(self,filename="zinc10M.csv", jump=False):
        smiles_list = pd.read_csv(filename, index_col=None)["smiles"].tolist()
        smiles_list = smiles_list[:500000]
        self.smiles_list = []
        # 数据过滤
        # for smiles in tqdm(smiles_list, desc="数据过滤"):
        #     try:
        #         mol = Chem.MolFromSmiles(smiles)
        #         if mol is not None:
        #             mol = Chem.MolFromSmiles(smiles)
        #             data = mol_to_graph_data_obj_simple(mol)
        #             self.smiles_list.append(smiles)
        #         else:
        #             print(smiles)
        #     except Exception as e:
        #         print("Error", e, smiles)
        num_processes = -1

        # 使用joblib进行并行处理，并显示进度条
        if jump:
            self.smiles_list = smiles_list
            return
        results = Parallel(n_jobs=num_processes)(
            delayed(init_process_smiles)(smiles) for smiles in tqdm(smiles_list, desc="数据过滤", position=0, leave=True)
        )

        # 处理结果，将有效的 SMILES 和相应的分子数据添加到 self.smiles_list 中
        self.smiles_list.extend([smiles for smiles, data in results if data is not None])
        print("数据过滤后的数据量：", len(self.smiles_list))


    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        graph = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smiles))
        return smiles, graph
    
def get_finetune_regression_dataset(datasetloadhelper):
    smiles_list = datasetloadhelper.smiles_list
    label_list = datasetloadhelper.tasks_labels_list
    np.random.seed(3)

    # 准备 shuflle
    indices = list(range(len(smiles_list)))
    # indices = np.array(indices)
    #shuffle(indices)
    np.random.shuffle(indices)

    smiles_list = [smiles_list[i] for i in indices]
    label_list = [label_list[i] for i in indices]

    # 获取有效数据集
    valid_label_list = []
    valid_smiles_list = []
    # 数据过滤
    for index,smiles in tqdm(enumerate(smiles_list), desc="数据过滤"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.MolFromSmiles(smiles)
                data = mol_to_graph_data_obj_simple(mol)
                smiles = Chem.MolToSmiles(mol)
                valid_smiles_list.append(smiles)
                valid_label_list.append(label_list[index])
            else:
                print(smiles)
        except Exception as e:
            print("Error", e, smiles)
    print("数据过滤后的数据量：", len(valid_smiles_list))
    
    # 切割成训练集和测试集
    train_smiles_list = valid_smiles_list[:int(len(valid_smiles_list)*0.8)]
    train_label_list = valid_label_list[:int(len(valid_smiles_list)*0.8)]
    test_smiles_list = valid_smiles_list[int(len(valid_smiles_list)*0.9):]
    test_label_list = valid_label_list[int(len(valid_smiles_list)*0.9):]
    train_dataset = FineTuneRegressionDataset( train_smiles_list, train_label_list)
    test_dataset = FineTuneRegressionDataset(test_smiles_list, test_label_list)
    return train_dataset, test_dataset


def get_finetune_classification_dataset(datasetloadhelper):
    smiles_list = datasetloadhelper.smiles_list
    tasks_labels_list = datasetloadhelper.tasks_labels_list
    np.random.seed(Config.SEED)

    #

    # 获取有效数据集
    valid_tasks_labels_list = [[] for i in range(len(tasks_labels_list))]
    valid_smiles_list = []
    # 数据过滤
    for index,smiles in tqdm(enumerate(smiles_list), desc="数据过滤"):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mol = Chem.MolFromSmiles(smiles)
                data = mol_to_graph_data_obj_simple(mol)
                valid_smiles_list.append(smiles)
                for i, task_labels_list in enumerate(tasks_labels_list):
                    valid_tasks_labels_list[i].append(task_labels_list[index])
                # self.label_list.append(label_list[index])
            else:
                print(smiles)
        except Exception as e:
            print("Error", e, smiles)
    print("数据过滤后的数据量：", len(valid_smiles_list))
    # 切割成训练集和测试集
    
    train_smiles_list = []
    train_tasks_labels_list = []
    
    test_smiles_list = [] 
    test_tasks_labels_list = []

    if datasetloadhelper.datasetinfo.split_type == "Rondom":
        # 准备 shuflle
        indices = list(range(len(valid_smiles_list)))

        # 使用相同的索引列表对主列表和子列表进行 shuffle
        np.random.shuffle(indices)
        valid_smiles_list = [valid_smiles_list[i] for i in indices]
        valid_tasks_labels_list = [[valid_task_labels_list[i] for i in indices] for valid_task_labels_list in valid_tasks_labels_list]


        train_smiles_list = valid_smiles_list[:int(len(valid_smiles_list)*0.8)]
        train_tasks_labels_list = [valid_tasks_labels_list[i][:int(len(valid_smiles_list)*0.8)] for i in range(len(valid_tasks_labels_list))]

        test_smiles_list = valid_smiles_list[int(len(valid_smiles_list)*0.9):]
        test_tasks_labels_list = [valid_tasks_labels_list[i][int(len(valid_smiles_list)*0.9):] for i in range(len(valid_tasks_labels_list))]
    elif datasetloadhelper.datasetinfo.split_type == "Scaffold":
        train_inds, _, test_inds = get_scaffold_split_ind(valid_smiles_list)
        train_smiles_list = [valid_smiles_list[i] for i in train_inds]
        train_tasks_labels_list = [[ valid_tasks_labels_list[i][j] for j in train_inds] for i in range(len(valid_tasks_labels_list))]
        test_smiles_list = [valid_smiles_list[i] for i in test_inds]
        test_tasks_labels_list = [[ valid_tasks_labels_list[i][j] for j in test_inds] for i in range(len(valid_tasks_labels_list))]
    else:
        raise ValueError(f"分割方式 {datasetloadhelper.datasetinfo.split_type} 不存在")
    print("train len: ", len(train_smiles_list))
    print("test len: ", len(test_tasks_labels_list))
    alpha_list = []
    for i in range(len(train_tasks_labels_list)):
        # 统计正样本的数量
        positive_num = 0
        negative_num = 0
        for label in train_tasks_labels_list[i]:
            if label == 1:
                positive_num += 1
            if label == 0:
                negative_num += 1
        print("positive_num", positive_num)

        # 计算正负样本的比例
        alpha = negative_num / (positive_num + negative_num)
        alpha_list.append(alpha)
    train_dataset = FineTuneClassifierDataset(train_smiles_list, train_tasks_labels_list)
    test_dataset = FineTuneClassifierDataset(test_smiles_list, test_tasks_labels_list)
    # alpha 的每一个元素进行开根号
    # alpha_list = [np.sqrt(alpha) for alpha in alpha_list]
    return train_dataset, test_dataset, alpha_list


class FineTuneClassifierDataset(Dataset):
    def __init__(self, smiles_list, tasks_labels_list):

        self.smiles_list = smiles_list
        self.tasks_labels_list = tasks_labels_list

        # if type == "train":
        #     smiles_list = smiles_list[:int(len(smiles_list)*0.8)]
            
        #     label_list = label_list[:int(len(label_list)*0.8)]
        #     if is_balance == True:
        #         print("数据平衡前的数据量：", len(smiles_list))
        #         index_1 = [i for i, x in enumerate(label_list) if x == 1]
        #         index_0 = [i for i, x in enumerate(label_list) if x == 0]
        #         if len(index_1) > len(index_0):
        #             ratio = round(len(index_1) / len(index_0)) - 1
        #             selected_smiles_list = [smiles_list[i] for i in index_0]
        #             smiles_list = smiles_list + selected_smiles_list*ratio
        #             label_list = label_list + [0]*len(selected_smiles_list)*ratio
        #         else:
        #             ratio = round(len(index_0) / len(index_1)) - 1
        #             selected_smiles_list = [smiles_list[i] for i in index_1]
        #             smiles_list = smiles_list + selected_smiles_list*ratio
        #             label_list = label_list + [1]*len(selected_smiles_list)*ratio
        #         print("数据平衡后的数据量：", len(smiles_list))

        #     combined_list = list(zip(smiles_list, label_list))
        #     np.random.shuffle(combined_list)
        #     smiles_list, label_list = zip(*combined_list)
        # elif type == "test":
        #     smiles_list = smiles_list[int(len(smiles_list)*0.9):]
        #     label_list = label_list[int(len(label_list)*0.9):]
        
        

        # smiles_list = smiles_list[:10000]
        
        #切割成训练集和测试集



    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        graph = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smiles))
        # label = self.label_list[idx]
        # 这里怪怪的，不确定这边这样写对不对
        labels = [task_labels_list[idx] for task_labels_list in self.tasks_labels_list]
        return smiles, graph, labels
    
class FineTuneRegressionDataset(Dataset):
    def __init__(self, smiles_list, label_list):
        self.smiles_list = smiles_list
        self.label_list = label_list
    
    def __len__(self):
        return len(self.smiles_list)
    
    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]
        graph = mol_to_graph_data_obj_simple(Chem.MolFromSmiles(smiles))
        label = self.label_list[idx]
        return smiles, graph, label