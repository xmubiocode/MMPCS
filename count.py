import pandas as pd
import rdkit
from rdkit import Chem
from tqdm import tqdm

df = pd.read_csv("tox21.csv", index_col=None)
smiles_list = df["smiles"].tolist()
atom_counts = {}
atom_count_wtih_h = {}
atoms_set = set('H')
error_smiles = []
none_smiles = []
max = 0
max_with_h = 0

# print(smiles_list[:10])
for smiles in tqdm(smiles_list, desc="Processing SMILES"):
    try:
        mol = Chem.MolFromSmiles(smiles)
        temp = 0
        temp_with_h = 0
        if mol is not None:
            
            # 统计
            for atom in mol.GetAtoms():
                # print(atom)
                symbol = atom.GetSymbol()  # 获取原子符号
                atoms_set.add(symbol)

            temp = mol.GetNumAtoms()
            if temp > max:
                max = temp
            mol = Chem.AddHs(mol)
            temp_with_h = mol.GetNumAtoms()
            if temp_with_h > max_with_h:
                max_with_h = temp_with_h
            if temp not in atom_counts:
                atom_counts[temp] = 1
            else: 
                atom_counts[temp] += 1
            if temp_with_h not in atom_count_wtih_h:
                atom_count_wtih_h[temp_with_h] = 1
            else:
                atom_count_wtih_h[temp_with_h] += 1
        else:
            none_smiles.append(smiles)
    except:
        error_smiles.append(smiles)

print("统计字典", atom_counts)
print("统计字典含氢", atom_count_wtih_h)
print("None smiles", none_smiles)
print("错误smiles", error_smiles)
print("原子种类", atoms_set)
print("共 {} 种原子".format(len(atoms_set)))
print("最多原子数",max)
print("加氢最多原子数", max_with_h)
