import rdkit
from rdkit import Chem
from rdkit.Chem import AllChem, rdDepictor
import torch
import numpy as np
from torch_geometric.data import Data

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

def smiles_to_graph_data_obj_simple(smiles):
    """ used in MoleculeDataset() class
    Converts rdkit mol objects to graph data object in pytorch geometric
    NB: Uses simplified atom and bond features, and represent as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr """

    # atoms
    # num_atom_features = 2  # atom type, chirality tag
    mol = Chem.MolFromSmiles(smiles)
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
        print("edges_list", edges_list)
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

smiles = "CCOC(=O)CC1(O)C(CO[Si](C)(C)C(C)(C)C)OC(n2ccc(=O)[nH]c2=O)C1O[Si](C)(C)C(C)(C)C"
# smiles = "c1ccccc1"

print("smiles:", smiles)

mol = Chem.MolFromSmiles(smiles)
# 指定Morgan指纹的半径和长度
radius = 2  # 指纹半径
n_bits = 1024  # 指纹位数

# 生成Morgan分子指纹
fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
# 打印分子指纹的二进制表示
print("FP:", fingerprint.ToBitString())

# 获取拓扑信息
for atom in mol.GetAtoms():
    atom_index = atom.GetIdx()  # 原子的索引
    atom_symbol = atom.GetSymbol()  # 原子符号 (如 'C' 表示碳)
    atom_degree = atom.GetDegree()  # 原子的度 (与其他原子相连的键的数量)
    atom_valence = atom.GetImplicitValence()  # 原子的隐式价
    atom_hydrogens = atom.GetTotalNumHs()  # 原子的氢原子数
    atom_charge = atom.GetFormalCharge()  # 原子的形式电荷
    atom_num = atom.GetAtomicNum()
    print(f"Atom {atom_index}: {atom_symbol}, Degree: {atom_degree}, Valence: {atom_valence}, "
          f"Hydrogens: {atom_hydrogens}, Charge: {atom_charge}, atom_num: {atom_num}, aalo: {[allowable_features['possible_atomic_num_list'].index(atom.GetAtomicNum())] + [allowable_features['possible_chirality_list'].index(atom.GetChiralTag())]}")
    
for bond in mol.GetBonds():
    bond_index = bond.GetIdx()  # 键的索引
    bond_type = bond.GetBondType()  # 键的类型 (如 'SINGLE', 'DOUBLE'，等等)
    bond_begin_atom = bond.GetBeginAtom().GetIdx()  # 键的起始原子索引
    bond_end_atom = bond.GetEndAtom().GetIdx()  # 键的结束原子索引
    print(f"Bond {bond_index}: Type: {bond_type}, "
          f"Between atoms {bond_begin_atom} and {bond_end_atom}")
    
# bond_types= dir(Chem.BondType)
# # 过滤出键类型（通常以大写字母表示）
# bond_types = [bond_type for bond_type in bond_types if bond_type.isupper()]

# # 打印所有键类型
# for bond_type in bond_types:
#     print(bond_type)

# 生成3D构象
mol_h = Chem.AddHs(mol)  # 添加氢原子
# 构象生成方法
# AllChem.ETKDGv2
# AllChem.UFF
# AllChem.MMFF94s
AllChem.EmbedMolecule(mol_h, AllChem.ETKDG())  # 使用ETKDG构象生成方法生成3D构象

# 获取3D坐标信息
coords = mol_h.GetConformer().GetPositions()

# 打印3D坐标信息
# for atom_idx, coord in enumerate(coords):
#     atom_symbol = mol_h.GetAtomWithIdx(atom_idx).GetSymbol()
#     x, y, z = coord
#     print(f"Atom {atom_idx} ({atom_symbol}): X={x:.3f}, Y={y:.3f}, Z={z:.3f}")

data = smiles_to_graph_data_obj_simple("CCO")
print(data)
print(data.x)
print(data.edge_index)
print(data.edge_attr)
print(smiles_to_graph_data_obj_simple("CO"))