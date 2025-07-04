import re
import torch
import pubchempy as pcp
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from collections import defaultdict

elem_list = [
    'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr', 
    'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 
    'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr', 
    'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og']
ELEM_LIST = ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Cr', 'Mn', 'Fe', 'Co', 'Al', 'I', 'B', 'K', 'Se', 'Zn', 'H', 'Cu', 'Mn', 'unknown'] # 26
TM_LIST = ['Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Cn']
NM_LIST = ['H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Se', 'Br', 'I']
VE_DICT = {'H': 1, 'He': 2, 'Li': 1, 'Be': 2, 'B': 3, 'C': 4, 'N': 5, 'O': 6, 'F': 7, 'Ne': 8, 'Na': 1, 'Mg': 2, 'Al': 3, 'Si': 4, 'P': 5, 'S': 6, 'Cl': 7, 'Ar': 8}
VE_DICT.update({'K': 1, 'Ca': 2, 'Sc': 3, 'Ti': 4, 'V': 5, 'Cr': 6, 'Mn': 7, 'Fe': 8, 'Co': 9, 'Ni': 10, 'Cu': 11, 'Zn': 2, 'Ga': 3, 'Ge': 4, 'As': 5, 'Se': 6, 'Br': 7, 'Kr': 8})
VE_DICT.update({'Rb': 1, 'Sr': 2, 'Y': 3, 'Zr': 4, 'Nb': 5, 'Mo': 6, 'Tc': 7, 'Ru': 8, 'Rh': 9, 'Pd': 10, 'Ag': 11, 'Cd': 2, 'In': 3, 'Sn': 4, 'Sb': 5, 'Te': 6, 'I': 7, 'Xe': 8})
VE_DICT.update({'Cs': 1, 'Ba': 2, 'La': 3, 'Ce': 4, 'Pr': 5, 'Nd': 6, 'Pm': 7, 'Sm': 8, 'Eu': 9, 'Gd': 10, 'Tb': 11, 'Dy': 12, 'Ho': 13, 'Er': 14, 'Tm': 15, 'Yb': 16, 'Lu': 17})
VE_DICT.update({'Hf': 4, 'Ta': 5, 'W': 6, 'Re': 7, 'Os': 8, 'Ir': 9, 'Pt': 10, 'Au': 11, 'Hg': 2, 'Tl': 3, 'Pb': 4, 'Bi': 5, 'Po': 6, 'At': 7, 'Rn': 8})
VE_DICT.update({'Fr': 1, 'Ra': 2, 'Ac': 3, 'Th': 4, 'Pa': 5, 'U': 6, 'Np': 7, 'Pu': 8, 'Am': 9, 'Cm': 10, 'Bk': 11, 'Cf': 12, 'Es': 13, 'Fm': 14, 'Md': 15, 'No': 16, 'Lr': 17})
VE_DICT.update({'Rf': 4, 'Db': 5, 'Sg': 6, 'Bh': 21, 'Hs': 22, 'Mt': 23, 'Ds': 24, 'Rg': 25, 'Cn': 26, 'Nh': 27, 'Fl': 28, 'Mc': 29, 'Lv': 30, 'Ts': 31, 'Og': 32})

def get_metal_oxidation_state(metal):
    oxidation_states = ''.join(filter(str.isdigit, metal))

    if len(oxidation_states) == 0:
        if "+" in metal:
            return 1 
        elif "-" in metal:
            return -1
        else:
            return 0
    else:
        return int(oxidation_states)
    
def onek_encoding_unk(value, allowable_set):
    if value in allowable_set:
        return [1 if v == value else 0 for v in allowable_set]
    else:
        return [0] * len(allowable_set)

def atom_features(atom, oxidation_state=None, features=153):
    atom_symbol_encoding = onek_encoding_unk(atom.GetSymbol(), elem_list)  # 118
    atom_degree_encoding = onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 6
    formal_charge = atom.GetFormalCharge() if oxidation_state is None else oxidation_state
    formal_charge_encoding = onek_encoding_unk(formal_charge, [-4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8])  # 13
    chiral_tag_encoding = onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])  # 4
    num_h_encoding = onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 5
    hybridization_encoding = onek_encoding_unk(
        int(atom.GetHybridization()), 
        [0, 1, 2, 3, 4, 5]  # Example encoding: [S, SP, SP2, SP3, SP3D, SP3D2]
    )  # 6
    is_aromatic = [atom.GetIsAromatic()]  # 1

    return torch.Tensor(
        atom_symbol_encoding +
        atom_degree_encoding +
        formal_charge_encoding+
        chiral_tag_encoding +
        num_h_encoding +
        hybridization_encoding +
        is_aromatic
        ) # 118+13 

def metal_features(metal, features=153):
    oxidation_state = get_metal_oxidation_state(metal)
    metal_symbol = metal.split(str(oxidation_state))[0]
    mol = Chem.MolFromSmiles("[{}]".format(metal_symbol))
    atom = mol.GetAtomWithIdx(0) 
    edge_index = torch.tensor([[0],[0]], dtype=torch.long).cuda()
    batch1 = torch.tensor([0], dtype=torch.long).cuda()
    edge_attr = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0]], dtype=torch.float).cuda()

    return (atom_features(atom, oxidation_state, features)), (edge_index, batch1), edge_attr

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

def tensorize_with_subgraphs(smiles_batch, metal, features=153):
    for smi in smiles_batch:
        minds = []  # List to store all metal indices
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() in TM_LIST:
                minds.append(i)  # Collect all metal indices

        #organic compound
        fatoms = []
        if not minds: # No metal atoms found
            mol.UpdatePropertyCache(strict=False)
            for i, atom in enumerate(mol.GetAtoms()):
                fatoms.append(atom_features(atom, features=features))
            fatoms = torch.stack(fatoms, 0)
            bond_feature_dict = {}
            ligand_edge_idx = []
            for bond in mol.GetBonds():
                bond_feat = bond_features(bond)
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                bond_feature_dict[(start, end)] = bond_feat
                bond_feature_dict[(end, start)] = bond_feat
                ligand_edge_idx.append([start, end])
                ligand_edge_idx.append([end, start])
            ligand_edge_idx = torch.Tensor(ligand_edge_idx).long().permute(1, 0)
            ligand_bond_features = []
            for start, end in ligand_edge_idx.T.tolist():
                if (start, end) in bond_feature_dict:
                    ligand_bond_features.append(bond_feature_dict[(start, end)])
                else:
                    ligand_bond_features.append(torch.zeros((1, 11)))
            ligand_bond_features = [t.flatten() for t in ligand_bond_features]
            ligand_bond_features = torch.stack(ligand_bond_features) 
            ligand_batch_idx = np.zeros((mol.GetNumAtoms()))
            ligand_batch_idx = torch.Tensor(ligand_batch_idx).long()

            return ((fatoms, smiles_batch),(ligand_edge_idx, ligand_batch_idx),ligand_bond_features, minds)
        
        # complex
        else:
            oxidation_states = {}
            metals = metal.split('.')  # Split multiple metals
            for m in metals:
                metal_symbol = m.split(str(get_metal_oxidation_state(m)))[0]
                oxidation_states[metal_symbol] = get_metal_oxidation_state(m)
            
            # Collect all neighbors of all metal atoms
            metal_neighbor_indices_set = set()
            for midx in minds:
                atom = mol.GetAtomWithIdx(midx)
                for nei in atom.GetNeighbors():
                    metal_neighbor_indices_set.add(nei.GetIdx())
            ninds_to_rmove = list(metal_neighbor_indices_set)

            editable_mol = Chem.EditableMol(mol)
            # Remove all metal-ligand and metal-metal bonds
            inds_bond_removed_metal = [] # edge index for metal-binding atoms
            for midx in minds:
                metal_atom = mol.GetAtomWithIdx(midx)
                for neighbor in metal_atom.GetNeighbors():
                    inds_to_remove = [midx, neighbor.GetIdx()]
                    inds_bond_removed_metal.append(inds_to_remove)
                    editable_mol.RemoveBond(*inds_to_remove)

            mol_modified = editable_mol.GetMol()
            mol_modified.UpdatePropertyCache(strict=False)
            fatoms = []
            for i, atom in enumerate(mol_modified.GetAtoms()):
                if atom.GetSymbol() in TM_LIST:
                    # Get the corresponding oxidation state for this metal
                    metal_oxidation_state = oxidation_states.get(atom.GetSymbol(), 0)
                    fatoms.append(atom_features(atom, metal_oxidation_state, features=features))
                else:
                    fatoms.append(atom_features(atom, features=features))
            fatoms = torch.stack(fatoms, 0)
            frag_indss = Chem.GetMolFrags(mol_modified, sanitizeFrags=False) #Finds the disconnected fragments from a molecule
            frag_idx_dict = dict(zip(range(len(frag_indss)), frag_indss))
            atoms = mol_modified.GetAtoms()
            ligand_edge_idx = []
            for i, frag_inds in enumerate(frag_indss):
                for frag_ind in frag_inds:
                    neis = atoms[frag_ind].GetNeighbors()
                    if len(neis) == 0:
                        ligand_edge_idx.append(torch.Tensor([frag_ind, frag_ind]).long()) # metal and neighbors and ligands neighbors bonds broken
                    for nei in neis:
                        nei_idx = nei.GetIdx()
                        # all bonds in ligands backbones / if not have bonds, tensor is the same index
                        ligand_edge_idx.append(torch.Tensor([frag_ind, nei_idx]).long())
            ligand_edge_idx = torch.stack(ligand_edge_idx, 0).T if ligand_edge_idx else torch.empty((2, 0)).long()
            # bond features
            bond_feature_dict = {}
            for bond in mol_modified.GetBonds():
                bond_feat = bond_features(bond)
                start = bond.GetBeginAtomIdx()
                end = bond.GetEndAtomIdx()
                bond_feature_dict[(start, end)] = bond_feat
                bond_feature_dict[(end, start)] = bond_feat
            ligand_bond_features = []
            for start, end in ligand_edge_idx.T.tolist():
                if (start, end) in bond_feature_dict:
                    ligand_bond_features.append(bond_feature_dict[(start, end)])
                else:
                    ligand_bond_features.append(torch.zeros((1, 11)))
            ligand_bond_features = [t.flatten() for t in ligand_bond_features]
            ligand_bond_features = torch.stack(ligand_bond_features) if ligand_bond_features else torch.empty((0, 11))

            ligand_batch_idx = np.zeros((mol_modified.GetNumAtoms()))
            G = nx.Graph()
            G.add_edges_from(ligand_edge_idx.t().tolist())
            for fragment_id, component in frag_idx_dict.items():
                for atom in component:
                    ligand_batch_idx[atom] = fragment_id
            ligand_batch_idx = torch.Tensor(ligand_batch_idx).long()

            ### GNN2: binding atoms with ligands ###
            removed_bonds_set = set()
            inds_bond_removed_non_metal = []
            for nind_to_rmove in ninds_to_rmove:
                # print(nind_to_rmove)
                for neighbor in mol.GetAtomWithIdx(nind_to_rmove).GetNeighbors():
                    if neighbor.GetIdx() not in minds:
                        bond_tuple = tuple(sorted([nind_to_rmove, neighbor.GetIdx()]))
                        if bond_tuple not in removed_bonds_set:
                            # print(list(bond_tuple))
                            inds_bond_removed_non_metal.append(list(bond_tuple))
                            editable_mol.RemoveBond(*bond_tuple)
                            removed_bonds_set.add(bond_tuple)

            mol_modified_2 = editable_mol.GetMol()
            mol_modified_2.UpdatePropertyCache(strict=False)
            frag_indss = Chem.GetMolFrags(mol_modified_2, sanitizeFrags=False) #Finds the disconnected fragments from a molecule
            frag_idx_dict = dict(zip(range(len(frag_indss)), frag_indss))
            atoms = mol_modified_2.GetAtoms()
            intrafrag_edge_idx = []
            for i, frag_inds in enumerate(frag_indss):
                for frag_ind in frag_inds:
                    neis = atoms[frag_ind].GetNeighbors()
                    if len(neis) == 0:
                        intrafrag_edge_idx.append(torch.Tensor([frag_ind, frag_ind]).long()) # metal and neighbors and ligands neighbors bonds broken
                    for nei in neis:
                        nei_idx = nei.GetIdx()
                        # all bonds in ligands backbones / if not have bonds, tensor is the same index
                        intrafrag_edge_idx.append(torch.Tensor([frag_ind, nei_idx]).long())
            intrafrag_edge_idx = torch.stack(intrafrag_edge_idx, 0).T if intrafrag_edge_idx else torch.empty((2, 0)).long()

            intrafrag_batch_idx = np.zeros((mol_modified.GetNumAtoms()))
            G = nx.Graph()
            G.add_edges_from(intrafrag_edge_idx.t().tolist())
            for fragment_id, component in frag_idx_dict.items():
                for atom in component:
                    intrafrag_batch_idx[atom] = fragment_id
            intrafrag_batch_idx = torch.Tensor(intrafrag_batch_idx).long()
            frag_ind_list = []
            for frag_inds in frag_indss:
                frag_ind_list += frag_inds
            intrafrag_batch_idx_dict = {atom_idx: intrafrag_batch_idx[atom_idx] for atom_idx in frag_ind_list}
            interfrag_batch_idx = np.zeros(len(set(intrafrag_batch_idx.tolist())))
            interfrag_edge_idx = []
            for inds in inds_bond_removed_non_metal:
                ind1, ind2 = inds # bond removed between "metal neighbors" and its "neighbors in ligands"
                frag_idx1 = intrafrag_batch_idx_dict[ind1]
                frag_idx2 = intrafrag_batch_idx_dict[ind2]
                interfrag_edge_idx.append([frag_idx1, frag_idx2])
                interfrag_edge_idx.append([frag_idx2, frag_idx1])
            
            # Add self-loops for all metal atoms
            for midx in minds:
                if midx in intrafrag_batch_idx_dict:
                    frag_midx = intrafrag_batch_idx_dict[midx]
                    interfrag_edge_idx.append([frag_midx, frag_midx])

            inds_bond_removed_non_metal_flattened = [ind for inds in inds_bond_removed_non_metal for ind in inds]
            for nidx in ninds_to_rmove:
                if nidx not in inds_bond_removed_non_metal_flattened:
                    frag_nidx = intrafrag_batch_idx_dict[nidx]
                    interfrag_edge_idx.append([frag_nidx, frag_nidx])
            interfrag_edge_idx = torch.Tensor(interfrag_edge_idx).long().T if interfrag_edge_idx else torch.empty((2, 0)).long()
            
            G = nx.Graph()
            G.add_edges_from(interfrag_edge_idx.t().tolist())
            connected_components = list(nx.connected_components(G))
            for fragment_id, component in enumerate(connected_components):
                for atom in component:
                    interfrag_batch_idx[atom] = fragment_id
            interfrag_batch_idx = torch.Tensor(interfrag_batch_idx).long()

            # filtered_masks
            intrafrag_ninds_to_rmove = []  # binding atoms in interfrag_batch_idx
            for nind in ninds_to_rmove:
                for frag_id, atom_indices in frag_idx_dict.items():
                    if nind in atom_indices:
                        intrafrag_ninds_to_rmove.append(frag_id) 
                        break
            
            # create binding atom mask and group by interfrag_batch_idx
            binding_atom_mask = torch.zeros_like(interfrag_batch_idx)
            for idx in range(len(interfrag_batch_idx)):
                if idx in intrafrag_ninds_to_rmove:
                    binding_atom_mask[idx] = 1

            # group to mask
            group_to_mask = {}
            for idx in range(len(interfrag_batch_idx)):
                group = interfrag_batch_idx[idx].item()
                if group not in group_to_mask:
                    group_to_mask[group] = []
                group_to_mask[group].append(binding_atom_mask[idx].item())

            # separate each group's mask into a separate dictionary
            separated_masks = []
            for group, masks in group_to_mask.items():
                one_positions = [i for i, x in enumerate(masks) if x == 1]
                for pos in one_positions:
                    new_mask = [0] * len(masks)
                    new_mask[pos] = 1
                    separated_masks.append({group: new_mask})

            # calculate the number of times each key appears
            key_counts = {}
            for mask_dict in separated_masks:
                for key in mask_dict:
                    key_counts[key] = key_counts.get(key, 0) + 1

            # only keep keys that appear more than once
            filtered_masks = [mask for mask in separated_masks if list(mask.keys())[0] in [k for k, v in key_counts.items() if v > 1]]

            # get intrafrag bonds
            excluded_bonds = set()
            for bond in mol_modified_2.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                excluded_bonds.add((min(begin_idx, end_idx), max(begin_idx, end_idx)))
            interfrag_bonds, interfrag_bond_idx = [], [] 
            # get all bonds in complex
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                # print(begin_idx , end_idx)
                bond_tuple = (min(begin_idx, end_idx), max(begin_idx, end_idx))
                # get metal-lig bonds and metal-metal bonds
                if bond_tuple not in excluded_bonds and not any(
                    (begin_idx == midx and end_idx in ninds_to_rmove) or
                    (end_idx == midx and begin_idx in ninds_to_rmove)
                    for midx in minds
                ):
                    interfrag_bonds.append(bond)
                    interfrag_bond_idx.append((begin_idx, end_idx))
            bond_features_list = [bond_features(bond) for bond in interfrag_bonds]
            def atom_idx_to_tensor(atom_idx, intrafrag_dict):
                return intrafrag_dict.get(atom_idx, None)
            bond_tensor_map = []
            for bond in interfrag_bonds:
                # print(begin_idx , end_idx)
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                begin_tensor = atom_idx_to_tensor(begin_idx, intrafrag_batch_idx_dict)
                end_tensor = atom_idx_to_tensor(end_idx, intrafrag_batch_idx_dict)
                bond_tensor_map.append((begin_tensor, end_tensor))
                # print(begin_idx , end_idx)
            bond_feature_dict = {(min(a, b), max(a, b)): feature for (a, b), feature in zip(interfrag_bond_idx, bond_features_list)}

            def map_to_tensor_indices(atom_idx1, atom_idx2, mapping_dict):
                return (mapping_dict.get(atom_idx1).item(), mapping_dict.get(atom_idx2).item())

            bond_feature_tensor_list = []
            for (a, b), feature in bond_feature_dict.items():
                tensor_pair = map_to_tensor_indices(a, b, intrafrag_batch_idx_dict)
                bond_feature_tensor_list.append((tensor_pair, feature))

            zero_bond_feature = torch.zeros(11)
            def get_bond_feature(bond_pair):
                pair_1 = (min(bond_pair), max(bond_pair))
                pair_2 = (max(bond_pair), min(bond_pair))

                matching_features = [
                    (pair, feature) for pair, feature in bond_feature_tensor_list if pair == pair_1 or pair == pair_2
                ]

                if matching_features:
                    pair_to_remove, feature = matching_features[0]
                    bond_feature_tensor_list.remove((pair_to_remove, feature))  
                    return torch.stack((feature, feature)) 
                else:
                    return torch.zeros(11)

            ordered_bond_features = []

            i = 0
            while i < len(interfrag_edge_idx[0]):
                bond_pair = (interfrag_edge_idx[0][i].item(), interfrag_edge_idx[1][i].item())
                if bond_pair[0] == bond_pair[1]:
                    ordered_bond_features.append(zero_bond_feature)
                    i += 1
                else:
                    feature = get_bond_feature(bond_pair)
                    ordered_bond_features.append(feature)
                    i += 2   
            flattened_features = []
            for feature in ordered_bond_features:
                if feature.dim() == 2:  # If the tensor has shape (2, 11)
                    flattened_features.extend(feature)  # Add both rows separately
                else:  # If the tensor has shape (11,)
                    flattened_features.append(feature)
            interfrag_bond_features = torch.stack(flattened_features) if flattened_features else torch.empty((0, 11))

            complex_edge_idx = []
            for midx, nidx in inds_bond_removed_metal:
                # print(midx, nidx)
                complex_idx1 = interfrag_batch_idx[intrafrag_batch_idx_dict[midx].item()]
                complex_idx2 = interfrag_batch_idx[intrafrag_batch_idx_dict[nidx].item()]
                complex_edge_idx.append([complex_idx1, complex_idx2])
                complex_edge_idx.append([complex_idx2, complex_idx1])
            complex_edge_idx = torch.Tensor(complex_edge_idx).long().T if complex_edge_idx else torch.empty((2, 0)).long()
            complex_batch_idx = torch.Tensor([0] * len(set(complex_edge_idx.flatten().tolist()))).long()
            
            # Handle metal-ligand and metal-metal bonds
            complex_bond_features = []
            for bond in mol.GetBonds():
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                # Check if either end is a metal atom
                if begin_idx in minds or end_idx in minds:
                    # Check if it's a metal-ligand bond or metal-metal bond
                    if (begin_idx in minds and end_idx in ninds_to_rmove) or \
                       (end_idx in minds and begin_idx in ninds_to_rmove) or \
                       (begin_idx in minds and end_idx in minds):
                        bond_feat = bond_features(bond)  
                        complex_bond_features.append(bond_feat)
                        complex_bond_features.append(bond_feat)
            complex_bond_features = torch.stack(complex_bond_features) if complex_bond_features else torch.empty((0, 11))

            return ((fatoms, smiles_batch),
                    ((ligand_edge_idx, ligand_batch_idx), (intrafrag_batch_idx), (interfrag_edge_idx, interfrag_batch_idx, filtered_masks), (complex_edge_idx, complex_batch_idx)),
                    (ligand_bond_features, interfrag_bond_features, complex_bond_features), minds, ninds_to_rmove)


def redox_each_num(smiles_batch, metal, redox_sites):
    for smi in smiles_batch:
        midx = None
        mol = Chem.MolFromSmiles(smi, sanitize=False)
        
        for i, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() in TM_LIST:
                midx = i  # Center metal index
                break
        
    ligand_edge_idx = []
    minds, ninds_to_rmove, inds_bond_removed_metal = [], [], []
    
    for nei in atom.GetNeighbors():
        ninds_to_rmove.append(nei.GetIdx())
    minds.append(midx)
    
    editable_mol = Chem.EditableMol(mol)
    for mind in minds:
        for neighbor in mol.GetAtomWithIdx(mind).GetNeighbors():
            inds_to_remove = [mind, neighbor.GetIdx()]
            inds_bond_removed_metal.append(inds_to_remove)
            editable_mol.RemoveBond(*inds_to_remove)
    
    mol_modified = editable_mol.GetMol()
    mol_modified.UpdatePropertyCache(strict=False)
    mol_smiles = Chem.MolToSmiles(mol_modified).split('.')
    frag_indss = Chem.GetMolFrags(mol_modified, sanitizeFrags=False)  # Finds the disconnected fragments
    frag_idx_dict = dict(zip(range(len(frag_indss)), frag_indss))
    
    atoms = mol_modified.GetAtoms()
    for i, frag_inds in enumerate(frag_indss):
        for frag_ind in frag_inds:
            neis = atoms[frag_ind].GetNeighbors()
            if len(neis) == 0:
                ligand_edge_idx.append(torch.Tensor([frag_ind, frag_ind]).long())  # Bonds broken
            for nei in neis:
                nei_idx = nei.GetIdx()
                ligand_edge_idx.append(torch.Tensor([frag_ind, nei_idx]).long())
    
    ligand_edge_idx = torch.stack(ligand_edge_idx, 0).T
    ligand_batch_idx = np.zeros((mol_modified.GetNumAtoms()))
    G = nx.Graph()
    G.add_edges_from(ligand_edge_idx.t().tolist())
    
    for fragment_id, component in frag_idx_dict.items():
        for atom in component:
            ligand_batch_idx[atom] = fragment_id
    
    ligand_batch_idx = torch.Tensor(ligand_batch_idx).long()
    atom_symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    batch_atom_mapping = [atom_symbols[i] for i in range(ligand_batch_idx.shape[0])]
    
    grouped_atoms = defaultdict(list)
    for batch_idx, atom_symbol in zip(ligand_batch_idx, batch_atom_mapping):
        grouped_atoms[int(batch_idx)].append(atom_symbol)
    grouped_atoms = dict(grouped_atoms)
    
    frag_to_group = defaultdict(list)
    for frag_smile in mol_smiles:
        frag_mol = Chem.MolFromSmiles(frag_smile)
        atom_symbols = [atom.GetSymbol() for atom in frag_mol.GetAtoms()]
        for group, symbols in grouped_atoms.items():
            if sorted(atom_symbols) == sorted(symbols) and frag_smile not in frag_to_group[group]:
                frag_to_group[group].append(frag_smile)
    
    if pd.isnull(redox_sites):
        redox_num_dict = { key: [frag_list[0], 0] for key, frag_list in frag_to_group.items()}
    else:
        redox_sites_list = redox_sites.split('/')
        redox_num_dict = {}
        for key, frag_list in frag_to_group.items():
            redox_num_dict[key] = [frag_list[0], 0]

        for redox_site in redox_sites_list:
            for key, frag_list in frag_to_group.items():
                for frag in frag_list:
                    if frag in redox_site or redox_site in frag:
                        redox_num_dict[key][1] += 1

        redox_num_dict = dict(redox_num_dict)

    return redox_num_dict

