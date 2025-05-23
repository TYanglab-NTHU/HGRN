import os,sys
import json
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader 
from optparse   import OptionParser
import numpy as np
from models.pretrain_models import *
from models.model import *

from utils.trainutils_v2 import *
from utils.chemutils import *
from utils.datautils import *
import torch 
import torch.nn as nn
import warnings
from torch_geometric.data import Data
from torch_geometric.nn   import global_mean_pool, global_max_pool
from torch_geometric.nn   import MessagePassing, GCNConv,  Linear, BatchNorm, GlobalAttention, GATConv


# print("Loading data...")
file_path = 'data/organo_rp_site_raw1.csv'
tensorize_fn = tensorize_with_subgraphs
batchsize = 1
test_size = 0.2

train_data, test_loader = data_loader(file_path=file_path,tensorize_fn=tensorize_fn,batch_size=batchsize,test_size=test_size)
train_loader  = DataLoader(train_data, batch_size=batchsize, shuffle=False)

batch = train_loader.dataset[1]
print(batch)


class BondMessagePassing(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=3, dropout=0.3):
        super(BondMessagePassing, self).__init__()
        self.W_i  = nn.Linear(node_features + bond_features, hidden_size)
        self.W_h  = nn.Linear(hidden_size, hidden_size)
        self.W_o  = nn.Linear(node_features + hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.depth = depth 

    def update(self, M_t, H_0):
        H_t = self.W_h(M_t)
        H_t = self.relu(H_0 + H_t)
        H_t = self.dropout(H_t)

        return H_t

    def message(self, H, batch):
        index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])
        M_all = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)[batch.edge_index[0]]
        M_rev = H[batch.rev_edge_index]

        # degree = torch.bincount(batch.edge_index[1], minlength=len(batch.edge_index[1])).unsqueeze(1).to(H.device)
        # degree = torch.where(degree == 0, torch.ones_like(degree), degree)
        # M_all = M_all / degree
        return M_all - M_rev

    def forward(self, batch):
        H_0 = self.W_i(torch.cat([batch.x[batch.edge_index[0]], batch.edge_attr], dim=1))
        H   = self.relu(H_0)
        for _ in range(1, self.depth):
            M = self.message(H, batch)
            H = self.update(M, H_0)
        index_torch = batch.edge_index[1].unsqueeze(1).repeat(1, H.shape[1]) # Noramal GNN tran
        M = torch.zeros(len(batch.x), H.shape[1], dtype=H.dtype, device=H.device).scatter_reduce_(0, index_torch, H, reduce="sum", include_self=False)
        M = torch.where(M.sum(dim=1, keepdim=True) == 0, batch.x, M)            
        H = self.W_o(torch.cat([batch.x, M], dim=1))
        H = self.relu(H)    
        H = self.dropout(H)
        return H             


node_dim = 153
bond_dim = 11
hidden_dim = 153
depth1 = 3
depth2 = 2
depth3 = 2
dropout = 0.3
output_dim = 1

GCN1 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth1, dropout=0.3)
GCN2 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth2, dropout=0.3)
GCN3 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth3, dropout=0.3)

pool = global_mean_pool
num_peaks_red = nn.Sequential(
    nn.Linear(hidden_dim, 512),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, 5))
num_peaks_ox = nn.Sequential(
    nn.Linear(hidden_dim, 512),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, 5))
E12_reg_red = nn.Sequential(
    nn.Linear(hidden_dim , 512),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, output_dim))
E12_reg_ox = nn.Sequential(
    nn.Linear(hidden_dim , 512),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(512, 128),
    nn.ReLU(),
    nn.Dropout(dropout),
    nn.Linear(128, output_dim))
gate_GCN1 = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh())
gate_GCN3 = nn.Sequential(
    # nn.Linear(hidden_dim, hidden_dim),
    # nn.ReLU(),
    nn.Linear(hidden_dim, hidden_dim),
    nn.Tanh())

def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index
    
def forward_subgraph(x, edge_index, batch, edge_attr, gcn, pre_proc=None, transform_edge_attr=None):
    if pre_proc is not None:
        x = pre_proc(x)

    rev_edge_index = _rev_edge_index(edge_index)

    if transform_edge_attr is not None:
        edge_attr = transform_edge_attr(edge_attr)

    data = Data(x=x, edge_index=edge_index, rev_edge_index=rev_edge_index, edge_attr=edge_attr)

    if isinstance(gcn, GATConv):
        result = gcn(x, edge_index, edge_attr) 
    else:
        result = gcn(data) 

    result_pooled = pool(result, batch)
    return result, result_pooled

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



def forward(batch):
    batch = train_loader.dataset[1]
    # for graph in batch.to_data_list():
    graph = batch
    x, edge_index, edge_attr, midx, real_E12, reaction, redox = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction, graph.redox
    
    subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
    subgraph1_edge_index, batch1 = subgraph1
    subgraph2_edge_index, batch2, filtered_mask = subgraph2
    subgraph3_edge_index, batch3 = subgraph3

    #"results after GCN and result_ after global pooling"
    subgraph1_result, subgraph1_pooled = forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=GCN1)
    subgraph2_result, subgraph2_pooled = forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
    subgraph3_result, subgraph3_pooled = forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=GCN3)

    total_loss = 0
    # convert batch1 index to batch3 index
    m_batch1  = batch1[midx[0]]
    new_batch = batch2[batch1_2.long()]

    mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
    ordered_indices = [mapping_dict[k] for k in sorted(mapping_dict)]
    # real_num_peaks = torch.tensor([graph.redox[i][1] for i in range(len(graph.redox))])
    print(f"graph.redox : {graph.redox}")
    real_num_peaks = torch.tensor([graph.redox[i][1] for i in range(len(graph.redox))]).to(device)
    real_num_peaks = torch.tensor(
        [v[1] for v in graph.redox.values()], device=device
    )
    redox_sites = [i for i, value in enumerate(real_num_peaks) for _ in range(int(value))]

    E12s = []
    count = 0
    while redox_sites:
        count += 1
        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        # if real_E12.numel() == 0:
        #     break
        if   reaction == 'reduction':
            each_num_redox = num_peaks_red(batch1_subgraph3_result)
        elif reaction == 'oxidation':
            each_num_redox = num_peaks_ox(batch1_subgraph3_result)
        
        unique_redox_sites = list(set(redox_sites))
        if   reaction == 'reduction':
            lig_potentials = E12_reg_red(batch1_subgraph3_result[unique_redox_sites])
            E12, idx = lig_potentials.max(dim=0)
        elif reaction == 'oxidation':
            lig_potentials = E12_reg_ox(batch1_subgraph3_result[unique_redox_sites])
            E12, idx = lig_potentials.min(dim=0)
        E12s.append(E12)
        # target
        
        print("logits :", each_num_redox)
        print("target :", real_num_peaks)
        loss_cla    = nn.CrossEntropyLoss()(each_num_redox.to(device), real_num_peaks.to(device))
        # loss_cla    = nn.CrossEntropyLoss()(each_num_redox, real_num_peaks)
        loss_reg    = nn.MSELoss()(E12, real_E12[0].unsqueeze(0))
        total_loss += loss_reg + loss_cla
        real_E12       = real_E12[1:]
        redox_site_idx = unique_redox_sites[idx]

        # gat x with GCN1
        redox_x_idx = [i for i, idx in enumerate(batch1) if idx == redox_site_idx]
        redox_x_    = x[redox_x_idx]
        redox_subgraph1_result_  = subgraph1_result[redox_x_idx]
        if redox_site_idx == m_batch1:
            if reaction == 'reduction':
                new_tensor =  torch.roll(redox_x_[124:137], shifts=-1, dims=1)
            if reaction == 'oxidation':
                new_tensor =  torch.roll(redox_x_[124:137], shifts=1, dims=1)
            redox_x_change = redox_x_.clone()
            redox_x_change[124:137] = new_tensor
        else:
            redox_x_change = redox_subgraph1_result_ * gate_GCN1(redox_subgraph1_result_) + redox_x_
            # redox_x_change =  redox_x_ * gate_GCN1( redox_x_) + redox_x_

        x_              = x.clone()
        x_[redox_x_idx] = redox_x_change
        subgraph1_result, subgraph1_pooled = forward_subgraph(x=x_, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=GCN1)
        subgraph2_result, subgraph2_pooled = forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

        # gat GCN2 with GCN3
        batch2_redox_idx = mapping_dict.get(redox_site_idx)
        all_indices      = np.arange(subgraph2_pooled.shape[0])
        nonredox_subgraph2_pooled = subgraph2_pooled[all_indices != batch2_redox_idx]
        updated_subgraph2_pooled  = nonredox_subgraph2_pooled.clone()
        redox_subgraph2_pooled    = subgraph2_pooled[batch2_redox_idx]
        redox_subgraph3_result_   = subgraph3_result[batch2_redox_idx]
        redox_site_change = redox_subgraph3_result_ * gate_GCN3(redox_subgraph3_result_) + redox_subgraph2_pooled
        # redox_site_change = redox_subgraph2_pooled  * self.gate_GCN3(redox_subgraph2_pooled) + redox_subgraph2_pooled
        subgraph2_result_ = torch.cat([updated_subgraph2_pooled[:batch2_redox_idx], redox_site_change.unsqueeze(0), updated_subgraph2_pooled[batch2_redox_idx:]], dim=0)

        subgraph3_result, subgraph3_pooled = forward_subgraph(x=subgraph2_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=GCN3)

        # redox_pos = [redox[redox_site_idx][0]]
        # sites = [i for i in range(len(redox)) if redox[i][0] in redox_pos]
        # for site in sites:
        #     if site in redox_sites:
        #         redox_sites.remove(site)
        redox_sites.remove(redox_site_idx)

        real_num_peaks = real_num_peaks.clone()  # ensure a separate copy
        real_num_peaks[redox_site_idx] = real_num_peaks[redox_site_idx] - 1

    if redox_sites != []:
        if E12s == []:
            subgraph3_pooled = pool(subgraph3_result, batch3)
            if   reaction == 'reduction':
                E12 = E12_reg_red(subgraph3_pooled)
            elif reaction == 'oxidation':
                E12 = E12_reg_ox(subgraph3_pooled)
            E12s.append(E12)
            if torch.isnan(E12) or torch.isnan(real_E12[0]):
                warnings.warn("NaN detected in E12 or real_E12[0]")
            else:
                loss_reg    = nn.MSELoss()(E12, real_E12[0])
                total_loss += loss_reg
            # loss_reg.backward(retain_graph=True)
    return total_loss







def forward(self, batch, device):
    for graph in batch.to_data_list():
        x, edge_index, edge_attr, midx, real_E12, reaction, redox = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction, graph.redox
    
    subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
    subgraph1_edge_index, batch1 = subgraph1
    subgraph2_edge_index, batch2, filtered_mask = subgraph2
    subgraph3_edge_index, batch3 = subgraph3

    #"results after GCN and result_ after global pooling"
    subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
    subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
    subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

    total_loss = 0
    # convert batch1 index to batch3 index
    m_batch1  = batch1[midx[0]]
    new_batch = batch2[batch1_2.long()]

    mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
    ordered_indices = [mapping_dict[k] for k in sorted(mapping_dict)]
    # real_num_peaks = torch.tensor([graph.redox[i][1] for i in range(len(graph.redox))])
    real_num_peaks = torch.tensor([graph.redox[i][1] for i in range(len(graph.redox))]).to(device)
    redox_sites = [i for i, value in enumerate(real_num_peaks) for _ in range(int(value))]

    E12s = []
    count = 0
    while redox_sites:
        count += 1
        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        # if real_E12.numel() == 0:
        #     break
        if   reaction == 'reduction':
            each_num_redox = self.num_peaks_red(batch1_subgraph3_result)
        elif reaction == 'oxidation':
            each_num_redox = self.num_peaks_ox(batch1_subgraph3_result)
        
        unique_redox_sites = list(set(redox_sites))
        if   reaction == 'reduction':
            lig_potentials = self.E12_reg_red(batch1_subgraph3_result[unique_redox_sites])
            E12, idx = lig_potentials.max(dim=0)
        elif reaction == 'oxidation':
            lig_potentials = self.E12_reg_ox(batch1_subgraph3_result[unique_redox_sites])
            E12, idx = lig_potentials.min(dim=0)
        E12s.append(E12)
        # target
        
        print("logits :", each_num_redox)
        print("target :", real_num_peaks)
        loss_cla    = nn.CrossEntropyLoss()(each_num_redox.to(device), real_num_peaks.to(device))
        # loss_cla    = nn.CrossEntropyLoss()(each_num_redox, real_num_peaks)
        loss_reg    = nn.MSELoss()(E12, real_E12[0].unsqueeze(0))
        total_loss += loss_reg + loss_cla
        real_E12       = real_E12[1:]
        redox_site_idx = unique_redox_sites[idx]

        # gat x with GCN1
        redox_x_idx = [i for i, idx in enumerate(batch1) if idx == redox_site_idx]
        redox_x_    = x[redox_x_idx]
        redox_subgraph1_result_  = subgraph1_result[redox_x_idx]
        if redox_site_idx == m_batch1:
            if reaction == 'reduction':
                new_tensor =  torch.roll(redox_x_[124:137], shifts=-1, dims=1)
            if reaction == 'oxidation':
                new_tensor =  torch.roll(redox_x_[124:137], shifts=1, dims=1)
            redox_x_change = redox_x_.clone()
            redox_x_change[124:137] = new_tensor
        else:
            redox_x_change = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_
            # redox_x_change =  redox_x_ * self.gate_GCN1( redox_x_) + redox_x_

        x_              = x.clone()
        x_[redox_x_idx] = redox_x_change
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

        # gat GCN2 with GCN3
        batch2_redox_idx = mapping_dict.get(redox_site_idx)
        all_indices      = torch.arange(subgraph2_pooled.shape[0], device=device)
        nonredox_subgraph2_pooled = subgraph2_pooled[all_indices != batch2_redox_idx]
        updated_subgraph2_pooled  = nonredox_subgraph2_pooled.clone()
        redox_subgraph2_pooled    = subgraph2_pooled[batch2_redox_idx]
        redox_subgraph3_result_   = subgraph3_result[batch2_redox_idx]
        redox_site_change = redox_subgraph3_result_ * self.gate_GCN3(redox_subgraph3_result_) + redox_subgraph2_pooled
        # redox_site_change = redox_subgraph2_pooled  * self.gate_GCN3(redox_subgraph2_pooled) + redox_subgraph2_pooled
        subgraph2_result_ = torch.cat([updated_subgraph2_pooled[:batch2_redox_idx], redox_site_change.unsqueeze(0), updated_subgraph2_pooled[batch2_redox_idx:]], dim=0)

        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

        # redox_pos = [redox[redox_site_idx][0]]
        # sites = [i for i in range(len(redox)) if redox[i][0] in redox_pos]
        # for site in sites:
        #     if site in redox_sites:
        #         redox_sites.remove(site)
        redox_sites.remove(redox_site_idx)

        real_num_peaks = real_num_peaks.clone()  # ensure a separate copy
        real_num_peaks[redox_site_idx] = real_num_peaks[redox_site_idx] - 1

    if redox_sites != []:
        if E12s == []:
            subgraph3_pooled = self.pool(subgraph3_result, batch3)
            if   reaction == 'reduction':
                E12 = self.E12_reg_red(subgraph3_pooled)
            elif reaction == 'oxidation':
                E12 = self.E12_reg_ox(subgraph3_pooled)
            E12s.append(E12)
            if torch.isnan(E12) or torch.isnan(real_E12[0]):
                warnings.warn("NaN detected in E12 or real_E12[0]")
            else:
                loss_reg    = nn.MSELoss()(E12, real_E12[0])
                total_loss += loss_reg
            # loss_reg.backward(retain_graph=True)
    return total_loss


smi = "C1CCC2=C(C1)SC(S2)=C(C=C1SC2=C(SCCS2)S1)c12c3c4c5c1[Fe+]16782345c2c1c6c7(c82)C(C=C1SC2=C(SCCS2)S1)=C1SC2=C(CCCC2)S1"
smi_list = [smi]
metal = "Fe"
redox_sites = "oxidation"


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
        if frag_mol is None:
            raise ValueError(f"[frag_mol] 無法解析SMILES字符串: {smi}")
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

res = redox_each_num(smi_list, metal, redox_sites)