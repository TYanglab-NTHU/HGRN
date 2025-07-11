import re
import torch
import numpy    as np
import networkx as nx     
import pymatgen as mg
from rdkit import Chem

from chic import Structure as ChIC_Structure #process mof to metal node and linkers

from pymatgen.core import Structure            # for loading crystal structures :contentReference[oaicite:1]{index=1}
from pymatgen.analysis.local_env import CrystalNN, NearNeighbors  # or any NearNeighbor class :contentReference[oaicite:2]{index=2}
from pymatgen.analysis.graphs import StructureGraph  # for creating periodic graphs :contentReference[oaicite:3]{index=3}
from pymatgen.transformations.standard_transformations import SupercellTransformation
from pymatgen.core import Molecule
from pymatgen.io.xyz import XYZ

from openbabel import openbabel, pybel

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
    match = re.search(r'([+-]?)(\d*)', metal)
    if match:
        sign, number = match.groups()
        if number:
            value = int(number)
            return value if sign != '-' else -value
        elif sign == '+':
            return 1
        elif sign == '-':
            return -1
    return 0

def onek_encoding_unk(value, allowable_set):
    if value in allowable_set:
        return [1 if v == value else 0 for v in allowable_set]
    else:
        return [0] * len(allowable_set)

def atom_features(atom, oxidation_state=None):
    atom_symbol_encoding = onek_encoding_unk(atom.GetSymbol(), elem_list)  # 118
    atom_sym = atom.GetSymbol()
    atom_degree_encoding = onek_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])  # 6
    formal_charge = atom.GetFormalCharge() if oxidation_state is None else oxidation_state
    formal_charge_encoding = onek_encoding_unk(formal_charge, [-4, -3, -2, -1, 1, 2, 3, 4, 5, 6, 7, 8, 0])  # 13
    chiral_tag_encoding = onek_encoding_unk(int(atom.GetChiralTag()), [0, 1, 2, 3])  # 4
    num_h_encoding = onek_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])  # 5
    hybridization_encoding = onek_encoding_unk(
        int(atom.GetHybridization()), 
        [0, 1, 2, 3, 4, 5]  # Example encoding: [S, SP, SP2, SP3, SP3D, SP3D2]
    )  # 6
    is_aromatic = [atom.GetIsAromatic()]  # 1
    atomic_mass = [0.01 * atom.GetMass()]  # 1s
    return torch.Tensor(
        atom_symbol_encoding +
        atom_degree_encoding +
        formal_charge_encoding+
        chiral_tag_encoding +
        num_h_encoding +
        hybridization_encoding +
        is_aromatic
        ) # 118+13 

def metal_features(metal):
    oxidation_state = get_metal_oxidation_state(metal)
    #remove number from metal
    metal_symbol = metal.split(str(oxidation_state))[0]
    mol = Chem.MolFromSmiles("[{}]".format(metal_symbol))
    atom = mol.GetAtomWithIdx(0) 
    edge_index = torch.tensor([[0],[0]], dtype=torch.long).cuda()
    batch1 = torch.tensor([0], dtype=torch.long).cuda()
    edge_attr = torch.tensor([[0,0,0,0,0,0,0,0,0,0,0]], dtype=torch.float).cuda()

    return (atom_features(atom, oxidation_state)), (edge_index, batch1), edge_attr

def bond_features(bond):
    bt = bond.GetBondType()
    stereo = int(bond.GetStereo())
    fbond = [bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE, bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC, bond.IsInRing()]
    fstereo = onek_encoding_unk(stereo, [0,1,2,3,4,5])
    return torch.Tensor(fbond + fstereo)

# def MO_tensorize_with_subgraphs(cif_file, metal):
#     # cif_file = '/work/u7069586/E-hGNN_f/zero-shot/metal_oxide/test_MO/MnO2.cif'

#     structure = Structure.from_file(cif_file)

#     sc_mat = [[2, 0, 0],
#             [0, 2, 0],
#             [0, 0, 2]]
#     transform = SupercellTransformation(sc_mat)
#     supercell = transform.apply_transformation(structure)

#     nn = CrystalNN()
#     sg = StructureGraph.with_local_env_strategy(supercell, nn, weights=False)

#     supercell = transform.apply_transformation(structure)
#     sg = StructureGraph.with_local_env_strategy(supercell, CrystalNN(), weights=False)

#     metal_symbol = metal_symbol = ''.join(c for c in metal if c.isalpha())
#     metal_idxs = [i for i, site in enumerate(supercell) if site.specie.symbol == metal_symbol]

#     center_frac = np.array([0.5, 0.5, 0.5])

#     metal_fracs = np.vstack([supercell.frac_coords[i] for i in metal_idxs])
#     central_idx = metal_idxs[np.argmin(np.linalg.norm(metal_fracs - center_frac, axis=1))]

#     first_shell_O = []
#     for nbr in sg.get_connected_sites(central_idx):
#         if nbr.site.specie.symbol == "O":
#             target = nbr.site.coords
#             dists = np.linalg.norm(supercell.cart_coords - target, axis=1)
#             o_idx = int(np.argmin(dists))
#             if dists[o_idx] < 1e-3:
#                 first_shell_O.append(o_idx)

#     second_shell_M = set()
#     for o_idx in first_shell_O:
#         for nbr in sg.get_connected_sites(o_idx):
#             if nbr.site.specie.symbol == metal_symbol:
#                 target = nbr.site.coords
#                 dists = np.linalg.norm(supercell.cart_coords - target, axis=1)
#                 m_idx = int(np.argmin(dists))
#                 if dists[m_idx] < 1e-3:
#                     second_shell_M.add(m_idx)
#     second_shell_M = list(second_shell_M)

#     first_shell_edges = [(central_idx, o_idx) for o_idx in first_shell_O]

#     second_shell_edges = []
#     for o_idx in first_shell_O:
#         for nbr in sg.get_connected_sites(o_idx):
#             if nbr.site.specie.symbol == metal_symbol:
#                 target = nbr.site.coords
#                 dists = np.linalg.norm(supercell.cart_coords - target, axis=1)
#                 m_idx = int(np.argmin(dists))
#                 if dists[m_idx] < 1e-3:
#                     second_shell_edges.append((o_idx, m_idx))


#     mapping = {}
#     mapping[central_idx] = 0

#     fitst_shell_idx = []
#     for new_idx, o in enumerate(first_shell_O, start=1):
#         mapping[o] = new_idx
#         fitst_shell_idx.append(new_idx)

#     ninds_to_rmove = fitst_shell_idx 

#     second_shell_idx = []
#     second_metals = [m for m in second_shell_M if m != central_idx]
#     for new_idx, m in enumerate(second_metals, start=1+len(first_shell_O)):
#         mapping[m] = new_idx
#         second_shell_idx.append(new_idx)

#     all_idx = [0] + fitst_shell_idx + second_shell_idx

#     all_edges = first_shell_edges + second_shell_edges
#     u_list = [mapping[u] for u, v in all_edges]
#     v_list = [mapping[v] for u, v in all_edges]

#     sub_edges = [u_list, v_list]

#     minds = list(second_shell_M)

#     metal_idx = [mapping[i] for i in minds]

#     ox_state = get_metal_oxidation_state(metal)

#     midx = metal_idx[0]

#     fatoms = []
#     for i in all_idx:
#         if i in metal_idx:
#             mol_m = Chem.MolFromSmiles(f'[{metal_symbol}{ox_state:+d}]')
#             atom_m = mol_m.GetAtomWithIdx(0)
#             fatoms.append(atom_features(atom_m, ox_state))
#         else:
#             mol_o = Chem.MolFromSmiles('[O-2]')
#             atom_o = mol_o.GetAtomWithIdx(0)
#             fatoms.append(atom_features(atom_o))
#     fatoms = torch.stack(fatoms, dim=0)

#     idx = torch.tensor(all_idx, dtype=torch.long)
#     ligand_edge_idx = torch.stack([idx, idx], dim=0)   # shape [2, n_atoms]
#     ligand_batch_idx = torch.tensor(all_idx, dtype=torch.long)

#     ligand_bond_features = []
#     for start, end in ligand_edge_idx.T.tolist():
#         ligand_bond_features.append(torch.zeros((1, 11)))

#     intrafrag_batch_idx  = ligand_batch_idx.clone()

#     intrafrag_edge_idx = []
#     intrafrag_edge_idx = torch.stack([idx, idx], dim=0) 
#     interfrag_batch_idx = torch.tensor(all_idx, dtype=torch.long)

#     filtered_masks = []
#     intrafrag_edge_idx = ligand_edge_idx.clone()
#     interfrag_edge_idx = intrafrag_edge_idx.clone()
#     interfrag_bond_features = []
#     for start, end in interfrag_edge_idx.T.tolist():
#         interfrag_bond_features.append(torch.zeros((1, 11)))

#     complex_edge_idx = torch.tensor(sub_edges)
#     complex_batch_idx = torch.zeros(len(all_idx), dtype=torch.long)

#     complex_bond_features = []
#     print(complex_edge_idx.T.tolist())
#     for start, end in complex_edge_idx.T.tolist():
#         f = torch.tensor([1,0,0,0,0,1,0,0,0,0,0], dtype=torch.float).unsqueeze(0)  # [1,11]
#         complex_bond_features.append(f)

#     ligand_bond_features = torch.cat(ligand_bond_features, dim=0)     
#     interfrag_bond_features = torch.cat(interfrag_bond_features, dim=0) 
#     complex_bond_features = torch.cat(complex_bond_features, dim=0)   
#     print(complex_bond_features.shape)

#     return ((fatoms),
#         ((ligand_edge_idx, ligand_batch_idx), (intrafrag_batch_idx), (interfrag_edge_idx, interfrag_batch_idx, filtered_masks), (complex_edge_idx, complex_batch_idx)),
#         (ligand_bond_features, interfrag_bond_features, complex_bond_features), midx, ninds_to_rmove)

def MO_tensorize_with_subgraphs(cif_file, metal):
    structure = Structure.from_file(cif_file) #pymatgen
    nn_strategy = CrystalNN() 
    sg = StructureGraph.with_local_env_strategy(structure, nn_strategy, weights=False)


    metal_symbol = ''.join(c for c in metal if c.isalpha())
    structure_metal_idx = [i for i, site in enumerate(structure) if site.specie.symbol == metal_symbol]
    metal_idx = structure_metal_idx[0]
    
    neighbor_atom_idx = list(set([i.index for i in sg.get_connected_sites(metal_idx)]))


    structure_to_all_atoms_idx = {}
    all_atoms = []
    count = 0
    for nei in neighbor_atom_idx:
        elem = structure[nei].specie.name
        coord = structure[nei].coords
        all_atoms.append((elem, coord[0], coord[1], coord[2]))
        if elem != 'H':
            structure_to_all_atoms_idx[nei] = count
            count += 1

    metal_structure_idx = metal_idx
    all_atoms.append((''.join(c for c in metal if c.isalpha()), structure.cart_coords[metal_structure_idx][0], structure.cart_coords[metal_structure_idx][1], structure.cart_coords[metal_structure_idx][2]))
    structure_to_all_atoms_idx[metal_idx] = count
    neighbor_atom_idx = [structure_to_all_atoms_idx[i] for i in neighbor_atom_idx]
    xyz_string = f"{len(all_atoms)}\n"
    xyz_string += "generated by script\n"  # 第二行註解
    for elem, x, y, z in all_atoms:
        xyz_string += f"{elem} {x:.6f} {y:.6f} {z:.6f}\n"

    # pybel mol
    mol = pybel.readstring("xyz", xyz_string)
    smiles = mol.write("smi").strip().split("\t")[0]
    # rdkit mol
    molblock = mol.write("mol")
    mol = Chem.MolFromMolBlock(molblock)

    oxidation_state = get_metal_oxidation_state(metal)
    minds = []
    for i, atom in enumerate(mol.GetAtoms()):
        if atom.GetSymbol() in TM_LIST:
            minds.append(i)  # Collect all metal indices
    midx = minds[0]
    # Collect all neighbors of all metal atoms
    ninds_to_rmove = neighbor_atom_idx
    editable_mol = Chem.EditableMol(mol)

    for midx in minds:
        for neighbor in ninds_to_rmove:
            inds_to_remove = [midx, neighbor]
            editable_mol.RemoveBond(*inds_to_remove)

    mol_modified = editable_mol.GetMol()
    mol_modified.UpdatePropertyCache(strict=False)
    fatoms = []
    for i, atom in enumerate(mol_modified.GetAtoms()):
        if atom.GetSymbol() in TM_LIST:
            fatoms.append(atom_features(atom, oxidation_state))
        else:
            fatoms.append(atom_features(atom))
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
    ligand_bond_features = []
    bond_feature_dict = {}
    for bond in mol_modified.GetBonds():
        bond_feat = bond_features(bond)
        start = bond.GetBeginAtomIdx()
        end = bond.GetEndAtomIdx()
        bond_feature_dict[(start, end)] = bond_feat
        bond_feature_dict[(end, start)] = bond_feat
    for start, end in ligand_edge_idx.T.tolist():
        if (start, end) in bond_feature_dict:
            ligand_bond_features.append(bond_feature_dict[(start, end)])
        else:
            ligand_bond_features.append(torch.zeros((1, 11)))
    ligand_bond_features = [t.flatten() for t in ligand_bond_features]
    ligand_bond_features = torch.stack(ligand_bond_features) if ligand_bond_features else torch.empty((0, 11))

    ligand_batch_idx = []
    ligand_batch_idx = np.zeros((mol_modified.GetNumAtoms()))
    G = nx.Graph()
    G.add_edges_from(ligand_edge_idx.t().tolist())
    for fragment_id, component in frag_idx_dict.items():
        for atom in component:
            ligand_batch_idx[atom] = fragment_id
    ligand_batch_idx = torch.Tensor(ligand_batch_idx).long()

    ### GNN2: binding atoms with ligands ###
    # remove bonds atoms bonded to metal and its ligands atoms
    inds_bond_removed_non_metal = []
    for nind_to_rmove in ninds_to_rmove:
        for neighbor in mol.GetAtomWithIdx(nind_to_rmove).GetNeighbors():
            if neighbor.GetIdx() not in minds:
                inds_to_remove = [nind_to_rmove, neighbor.GetIdx()]
                inds_bond_removed_non_metal.append(inds_to_remove)
                editable_mol.RemoveBond(*inds_to_remove)

    intrafrag_edge_idx = []
    mol_modified_2 = editable_mol.GetMol()
    mol_modified_2.UpdatePropertyCache(strict=False)
    frag_indss = Chem.GetMolFrags(mol_modified_2, sanitizeFrags=False) #Finds the disconnected fragments from a molecule
    frag_idx_dict = dict(zip(range(len(frag_indss)), frag_indss))
    atoms = mol_modified_2.GetAtoms()
    for i, frag_inds in enumerate(frag_indss):
        for frag_ind in frag_inds:
            # print(frag_ind)
            neis = atoms[frag_ind].GetNeighbors()
            if len(neis) == 0:
                intrafrag_edge_idx.append(torch.Tensor([frag_ind, frag_ind]).long()) # metal and neighbors and ligands neighbors bonds broken
            for nei in neis:
                nei_idx = nei.GetIdx()
                # all bonds in ligands backbones / if not have bonds, tensor is the same index
                intrafrag_edge_idx.append(torch.Tensor([frag_ind, nei_idx]).long())
    intrafrag_edge_idx = torch.stack(intrafrag_edge_idx, 0).T

    intrafrag_batch_idx = np.zeros((mol_modified.GetNumAtoms()))
    G = nx.Graph()
    G.add_edges_from(intrafrag_edge_idx.t().tolist())
    for fragment_id, component in frag_idx_dict.items():
        # print(fragment_id, component)
        for atom in component:
            intrafrag_batch_idx[atom] = fragment_id
    intrafrag_batch_idx = torch.Tensor(intrafrag_batch_idx).long()
    frag_ind_list = []
    for frag_inds in frag_indss:
        # print(frag_inds)
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
    frag_midx = intrafrag_batch_idx_dict[midx]
    interfrag_edge_idx.append([frag_midx, frag_midx])
    inds_bond_removed_non_metal_flattened = [ind for inds in inds_bond_removed_non_metal for ind in inds]
    for nidx in ninds_to_rmove:
        if nidx not in inds_bond_removed_non_metal_flattened:
            frag_nidx = intrafrag_batch_idx_dict[nidx]
            interfrag_edge_idx.append([frag_nidx, frag_nidx])
    interfrag_edge_idx = torch.Tensor(interfrag_edge_idx).long().T
    G = nx.Graph()
    G.add_edges_from(interfrag_edge_idx.t().tolist())
    connected_components = list(nx.connected_components(G))
    for fragment_id, component in enumerate(connected_components):
        for atom in component:
            interfrag_batch_idx[atom] = fragment_id
    interfrag_batch_idx = torch.Tensor(interfrag_batch_idx).long()
    intrafrag_ninds_to_rmove = []
    for nind in ninds_to_rmove:
        for frag_id, atom_indices in frag_idx_dict.items():
            if nind in atom_indices:
                intrafrag_ninds_to_rmove.append(frag_id)
                break

    
    # 創建binding atom mask並根據interfrag_batch_idx分組
    binding_atom_mask = torch.zeros_like(interfrag_batch_idx)
    for idx in range(len(interfrag_batch_idx)):
        if idx in intrafrag_ninds_to_rmove:
            binding_atom_mask[idx] = 1

    # 創建group到binding atom mask的映射
    group_to_mask = {}
    for idx in range(len(interfrag_batch_idx)):
        group = interfrag_batch_idx[idx].item()
        if group not in group_to_mask:
            group_to_mask[group] = []
        group_to_mask[group].append(binding_atom_mask[idx].item())

    # 將每個group的mask分離成單獨的字典
    separated_masks = []
    for group, masks in group_to_mask.items():
        one_positions = [i for i, x in enumerate(masks) if x == 1]
        for pos in one_positions:
            new_mask = [0] * len(masks)
            new_mask[pos] = 1
            separated_masks.append({group: new_mask})

    key_counts = {}
    for mask_dict in separated_masks:
        for key in mask_dict:
            key_counts[key] = key_counts.get(key, 0) + 1

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
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        begin_tensor = atom_idx_to_tensor(begin_idx, intrafrag_batch_idx_dict)
        end_tensor = atom_idx_to_tensor(end_idx, intrafrag_batch_idx_dict)
        bond_tensor_map.append((begin_tensor, end_tensor))
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

    # complex_bond_features = torch.stack(complex_bond_features) if complex_bond_features else torch.empty((0, 11))
    metal = interfrag_batch_idx[intrafrag_batch_idx[minds][0]].item()
    complex_edge_idx = []
    complex_bond_features = []
    for i, nidx in enumerate(neighbor_atom_idx):
        # 為每個 metal-linker 連接建立 edge，使用實際的 metal index
        neighbor_frag = interfrag_batch_idx[intrafrag_batch_idx[nidx]].item()
        complex_edge_idx.append([metal, neighbor_frag])  # 使用實際的 metal index
        complex_edge_idx.append([neighbor_frag, metal])
        # 加入對應的 bond features (metal-linker bond)
        complex_bond_features.extend([torch.tensor([1,0,0,0,0,1,0,0,0,0,0])] * 2)  # 雙向連接
    
    complex_edge_idx = torch.tensor(complex_edge_idx).T if complex_edge_idx else torch.empty((2, 0)).long()
    complex_batch_idx = torch.zeros(len(set(complex_edge_idx.flatten().tolist()))).long()
    complex_bond_features = torch.stack(complex_bond_features) if complex_bond_features else torch.empty((0, 11))
    
    return ((fatoms),
            ((ligand_edge_idx, ligand_batch_idx), (intrafrag_batch_idx), (interfrag_edge_idx, interfrag_batch_idx, filtered_masks), (complex_edge_idx, complex_batch_idx)),
            (ligand_bond_features, interfrag_bond_features, complex_bond_features), midx, ninds_to_rmove)