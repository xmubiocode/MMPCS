from rdkit import Chem
from rdkit.Chem import QED
import pandas as pd
from tqdm import tqdm

file_path = "aaa.txt"
smiles_list = []
# smiles_list = pd.read_csv(file_path)["smiles"].values.tolist()
with open(file_path, "r") as f:
    for line in f.readlines():
        smiles_list.append(line.strip())

print("分子的数量为: ", len(smiles_list))

# smiles_list  = smiles_list[:100]
degree_list = []

# 读取 smiles 的文件

def get_atom_and_bond_num(smiles):
    """
    获取分子的原子数和键数
    :param smiles: smiles 格式的分子
    :return: 原子数和键数和摩尔质量
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise Exception("分子格式错误")
        atom_num = 0
        bond_num = 0
        weight = 0
        # atom_num = mol.GetNumAtoms()
        # bond_num = mol.GetNumBonds()
        # weight = QED.properties(mol).MW
        
        graph = Chem.rdmolops.GetAdjacencyMatrix(mol)
        atom_degree = [sum(row) for row in graph]
        # print(atom_degree)
        degree_list.extend(atom_degree)
        # print("degree_list: ", degree_list)
        return atom_num, bond_num, weight
    except Exception as e:
        raise Exception("分子格式错误")

def get_all_mocule_parameters(smiles_list):
    """
    获取分子的所有参数, 包括原子数和键数, 并取平均
    :param smiles_list: smiles 格式的分子列表
    :return: 分子的所有参数
    """
    atom_num_list = []
    bond_num_list = []
    weight_list = []
    for smiles in tqdm(smiles_list):
        try:
            atom_num, bond_num, weight = get_atom_and_bond_num(smiles)
            atom_num_list.append(atom_num)
            bond_num_list.append(bond_num)
            weight_list.append(weight)
        except Exception as e:
            # print(e)
            pass
    atom_num_mean = sum(atom_num_list) / len(atom_num_list)
    bond_num_mean = sum(bond_num_list) / len(bond_num_list)
    weight_mean = sum(weight_list) / len(weight_list)
    print("len(atom_num_list): ", len(atom_num_list))
    return atom_num_mean, bond_num_mean, weight_mean

atom_num_mean, bond_num_mean, weight_mean = get_all_mocule_parameters(smiles_list)
print("原子数的平均值为: ", atom_num_mean)
print("键数的平均值为: ", bond_num_mean)
print("摩尔质量的平均值为: ", weight_mean)
print("原子度的平均值为: ", sum(degree_list) / len(degree_list))