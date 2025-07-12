import torch 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn   import global_mean_pool, global_max_pool
from torch_geometric.nn   import MessagePassing, GCNConv,  Linear, BatchNorm, GlobalAttention, GATConv

import pandas as pd 
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from torch.nn import functional as F

import warnings
warnings.filterwarnings("ignore")

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

class OMGNN_RNN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, output_dim, depth1=3, depth2=3, depth3=4,dropout=0.3):
        super(OMGNN_RNN, self).__init__()
        self.GCN1 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth1, dropout=0.3)
        self.GCN2 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth2, dropout=0.3)
        self.GCN3 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth3, dropout=0.3)

        self.pool = global_mean_pool
        self.num_peaks_red = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5))
        self.num_peaks_ox = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5))
        self.E12_reg_red = nn.Sequential(
            nn.Linear(hidden_dim , 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.E12_reg_ox = nn.Sequential(
            nn.Linear(hidden_dim , 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.gate_GCN1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())
        self.gate_GCN3 = nn.Sequential(
            # nn.Linear(hidden_dim, hidden_dim),
            # nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())

    @staticmethod
    def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index

    def forward_subgraph(self, x, edge_index, batch, edge_attr, gcn, pre_proc=None, transform_edge_attr=None):
        if pre_proc is not None:
            x = pre_proc(x)

        rev_edge_index = self._rev_edge_index(edge_index)

        if transform_edge_attr is not None:
            edge_attr = transform_edge_attr(edge_attr)

        data = Data(x=x, edge_index=edge_index, rev_edge_index=rev_edge_index, edge_attr=edge_attr)

        if isinstance(gcn, GATConv):
            result = gcn(x, edge_index, edge_attr) 
        else:
            result = gcn(data) 

        result_pooled = self.pool(result, batch)
        return result, result_pooled

    def forward(self, batch, device):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx, real_E12, reaction = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction
        
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
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device=device)

        real_num_peaks  = torch.tensor([graph.redox[i][1] for i in range(len(graph.redox))]).to(device)

        redox_sites = [i for i, value in enumerate(real_num_peaks) for _ in range(int(value))]

        delocalized_potential_indices_t_1 = list(set(redox_sites))
        E12s = []
        count = 0
        while redox_sites:
            count += 1
            batch1_subgraph3_result = subgraph3_result[ordered_indices]
            if real_E12.numel() == 0:
                break
            if   reaction == 'reduction':
                each_num_redox = self.num_peaks_red(batch1_subgraph3_result)
            elif reaction == 'oxidation':
                each_num_redox = self.num_peaks_ox(batch1_subgraph3_result)
  
            unique_redox_sites = list(set(redox_sites))
            if   reaction == 'reduction':
                all_potentials = self.E12_reg_red(batch1_subgraph3_result)
                lig_potentials = all_potentials[unique_redox_sites]
                E12, idx = lig_potentials.max(dim=0)
            elif reaction == 'oxidation':
                all_potentials = self.E12_reg_ox(batch1_subgraph3_result)
                lig_potentials = all_potentials[unique_redox_sites]
                E12, idx = lig_potentials.min(dim=0)
            E12s.append(E12)

            redox_site_idx = unique_redox_sites[idx]

            if reaction == 'reduction':
                redox_potentials = all_potentials[delocalized_potential_indices_t_1]
                potential_p = torch.softmax(redox_potentials, dim=0)
                potential_P = potential_p * len(all_potentials)
                delocalized_potential_indices = torch.where(potential_P > 0.96)[0]
                if delocalized_potential_indices.numel() == 1:
                    delocalized_potential_indices = torch.tensor([redox_site_idx], device=device)
            elif reaction == 'oxidation':
                redox_potentials = all_potentials[delocalized_potential_indices_t_1]
                potential_p = torch.softmax(-redox_potentials, dim=0)
                potential_P = potential_p * len(redox_potentials)
                delocalized_potential_indices = torch.where(potential_P > 0.96)[0]
                if delocalized_potential_indices.numel() == 1:
                    delocalized_potential_indices = torch.tensor([redox_site_idx], device=device)


            all_redox_x_idx = []
            same_idx_tensor = torch.tensor(delocalized_potential_indices, device=real_num_peaks.device)
            if same_idx_tensor.numel() != 1:
                mapped_indices = [delocalized_potential_indices_t_1[i] for i in delocalized_potential_indices.tolist()]
                delocalized_potential_indices = list(set(mapped_indices + delocalized_potential_indices_t_1))
                for site_idx in delocalized_potential_indices:
                    site_indices = [i for i, idx in enumerate(batch1) if idx == site_idx]
                    all_redox_x_idx.extend(site_indices)
                        
                real_num_peaks_ = real_num_peaks.clone()
            else:
                site_indices = [i for i, idx in enumerate(batch1) if idx == redox_site_idx]
                all_redox_x_idx.extend(site_indices)
                real_num_peaks_ = real_num_peaks.clone()
            


            loss_mask = torch.ones_like(real_num_peaks_, dtype=torch.bool)
            if len(delocalized_potential_indices_t_1) > 1:
                delocalized_peaks = real_num_peaks_[delocalized_potential_indices_t_1]

                if delocalized_peaks.max() != delocalized_peaks.min():
                    min_peaks = delocalized_peaks.min()
                    min_indices = [delocalized_potential_indices_t_1[i] for i, peaks in enumerate(delocalized_peaks) if peaks == min_peaks]
                    
                    for idx in min_indices:
                        loss_mask[idx] = False
            
            loss_per_sample = nn.CrossEntropyLoss(reduction='none')(each_num_redox, real_num_peaks_)
            loss_cla = (loss_per_sample * loss_mask.float()).sum() / (loss_mask.sum().float() + 1e-8)
            # loss_cla    = nn.CrossEntropyLoss()(each_num_redox, real_num_peaks_)
            loss_reg    = nn.MSELoss()(E12, real_E12[0])
            loss        = loss_cla + loss_reg
            total_loss += loss

            real_E12       = real_E12[1:]

            if count == 1:
                x_t_1 = x.clone()

            # gat x with GCN1
            if midx[0] in all_redox_x_idx:
                redox_x_metal  = x_t_1[midx]
                redox_x_    = x_t_1[all_redox_x_idx]
                redox_subgraph1_result_ = subgraph1_result[all_redox_x_idx]
                if reaction == 'reduction':
                    new_tensor =  torch.roll(redox_x_metal[:,124:137], shifts=-1, dims=1)
                if reaction == 'oxidation':
                    new_tensor =  torch.roll(redox_x_metal[:,124:137], shifts=1, dims=1)
                redox_x_metal_change = redox_x_metal.clone()
                redox_x_metal_change[:,124:137] = new_tensor

                redox_x_change = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_

                x_t_1[all_redox_x_idx] = redox_x_change
                x_t_1[midx[0]] = redox_x_metal_change
            else:
                redox_x_    = x_t_1[all_redox_x_idx]
                redox_subgraph1_result_ = subgraph1_result[all_redox_x_idx]
                redox_x_change = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_

                x_t_1[all_redox_x_idx] = redox_x_change

            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_t_1, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))


            all_redox_site_idx = unique_redox_sites + [idx for idx in delocalized_potential_indices if idx not in unique_redox_sites]
            batch2_redox_idx = [mapping_dict.get(site) for site in all_redox_site_idx]
            all_indices      = torch.arange(subgraph2_pooled.shape[0], device=device)
            updated_subgraph2_pooled  = subgraph2_pooled.clone()

            all_redox_site_idx_tensor = torch.tensor(all_redox_site_idx, device=device)
            same_potentials = all_potentials[all_redox_site_idx_tensor]

            potentials_mapping = {}
            for site_idx, potential in enumerate(same_potentials):
                if isinstance(batch2_redox_idx[site_idx], list):
                    for sub_idx in batch2_redox_idx[site_idx]:
                        potentials_mapping[sub_idx] = potential
                else:
                    potentials_mapping[batch2_redox_idx[site_idx]] = potential

            redox_sites_ = []
            for idx in batch2_redox_idx:
                if isinstance(idx, list):
                    redox_sites_.extend(idx)
                else:
                    redox_sites_.append(idx)
            
            redox_subgraph2_pooled  = subgraph2_pooled[redox_sites_]
            redox_subgraph3_result_ = subgraph3_result[redox_sites_]

            # 根據映射獲取對應的電位
            site_potentials = torch.stack([potentials_mapping[site] for site in redox_sites_])
            
            def boltzmann_distribution(potentials):
                if reaction == 'reduction':
                    potential_p = torch.softmax(potentials, dim=0)
                    potential_P = potential_p * len(potentials)
                    potential_p = torch.where(potential_P > 0.96, potential_p, torch.tensor(0.0, device=potential_p.device))
                elif reaction == 'oxidation':
                    potential_p = torch.softmax(-potentials, dim=0)
                    potential_P = potential_p * len(potentials)
                    potential_p = torch.where(potential_P > 0.96, potential_p, torch.tensor(0.0, device=potential_p.device))
                potential_p = F.normalize(potential_p, p=1, dim=0)
                return potential_p
            
            gate_weights = boltzmann_distribution(site_potentials)

            redox_site_change = redox_subgraph3_result_ * (1 - gate_weights) + redox_subgraph2_pooled * gate_weights 
            updated_subgraph2_pooled[redox_sites_] = redox_site_change
            subgraph2_result_ = updated_subgraph2_pooled.clone()

            delocalized_potential_indices_t_1 = delocalized_potential_indices

            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

            real_num_peaks = real_num_peaks.clone() 
            
            real_num_peaks[redox_site_idx] = real_num_peaks[redox_site_idx] - 1
            redox_sites.remove(redox_site_idx)
            
        if redox_sites != []:
            if E12s == []:
                subgraph3_pooled = self.pool(subgraph3_result, batch3)
                if   reaction == 'reduction':
                    E12 = self.E12_reg_red(subgraph3_pooled)
                elif reaction == 'oxidation':
                    E12 = self.E12_reg_ox(subgraph3_pooled)
                E12s.append(E12)
                loss_reg    = nn.MSELoss()(E12, real_E12[0])
                total_loss += loss_reg
        if redox_sites == []:
            if reaction == 'reduction':
                each_num_redox = self.num_peaks_red(subgraph3_result[ordered_indices])
                loss_cla    = nn.CrossEntropyLoss()(each_num_redox, real_num_peaks)
                total_loss += loss_cla * 0.1
            elif reaction == 'oxidation':
                each_num_redox = self.num_peaks_ox(subgraph3_result[ordered_indices])
                loss_cla    = nn.CrossEntropyLoss()(each_num_redox, real_num_peaks)
                total_loss += loss_cla * 0.1

        return total_loss

    def sample(self, batch, device, warmup=False):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx, reaction = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.reaction
        
        subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2, filtered_mask = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        #"results after GCN and result_ after global pooling"
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

        # convert batch1 index to batch3 index
        m_batch1  = batch1[midx[0]]
        new_batch = batch2[batch1_2.long()]

        mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device=device)

        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        if   reaction == 'reduction':
            num_redox_all = self.num_peaks_red(batch1_subgraph3_result)
        elif reaction == 'oxidation':
            num_redox_all = self.num_peaks_ox(batch1_subgraph3_result)
        
        num_redox_ = torch.argmax(num_redox_all, dim=1)
        pred_num_redox_ = num_redox_.clone()
        pred_E12s  = torch.tensor([], device=device)

        redox_mask    = num_redox_ > 0
        redox_indices = torch.nonzero(redox_mask, as_tuple=False).flatten()
        delocalized_potential_indices_t_1 = list(set([i.item() for i in redox_indices]))

        count = 0
        while num_redox_.sum() != 0:
            count += 1
            batch1_subgraph3_result = subgraph3_result[ordered_indices]
            if   reaction == 'reduction':
                E12s          = self.E12_reg_red(batch1_subgraph3_result)
            elif reaction == 'oxidation':
                E12s          = self.E12_reg_ox(batch1_subgraph3_result)
            E12s          = E12s.squeeze()
            redox_mask    = num_redox_ > 0
            redox_indices = torch.nonzero(redox_mask, as_tuple=False).flatten()
            E12s_redox    = E12s[redox_mask]
            if reaction == "reduction":
                E12, filtered_idx = torch.max(E12s_redox, dim=0)
            elif reaction == "oxidation":
                E12, filtered_idx = torch.min(E12s_redox, dim=0)

            redox_site_idx = redox_indices[filtered_idx].item()

            if reaction == 'reduction':
                redox_potentials = E12s[delocalized_potential_indices_t_1]
                potential_p = torch.softmax(redox_potentials, dim=0)
                potential_P = potential_p * len(redox_potentials)
                delocalized_potential_indices = torch.where(potential_P > 0.96)[0]
                if delocalized_potential_indices.numel() == 1:
                    delocalized_potential_indices = torch.tensor([], device=device)
            elif reaction == 'oxidation':
                redox_potentials = E12s[delocalized_potential_indices_t_1]
                potential_p = torch.softmax(-redox_potentials, dim=0)
                potential_P = potential_p * len(redox_potentials)
                delocalized_potential_indices = torch.where(potential_P > 0.96)[0]
                if delocalized_potential_indices.numel() == 1:
                    delocalized_potential_indices = torch.tensor([], device=device)

            # # 找到所有 delocalized_potential_indices 在 batch1 中對應的索引
            all_redox_x_idx = []
            same_idx_tensor = torch.tensor(delocalized_potential_indices, device=device)
            if same_idx_tensor.numel() != 0:
                mapped_indices = [delocalized_potential_indices_t_1[i] for i in delocalized_potential_indices.tolist()]
                delocalized_potential_indices = list(set(mapped_indices + delocalized_potential_indices_t_1))
                for site_idx in delocalized_potential_indices:
                    site_indices = [i for i, idx in enumerate(batch1) if idx == site_idx]
                    all_redox_x_idx.extend(site_indices)

            else:
                site_indices = [i for i, idx in enumerate(batch1) if idx == redox_site_idx]
                all_redox_x_idx.extend(site_indices)
            

            if count == 1:
                x_t_1 = x.clone()

            # gat x with GCN1
            if midx[0] in all_redox_x_idx:
                redox_x_metal    = x_t_1[midx]
                redox_x_    = x_t_1[all_redox_x_idx]
                redox_subgraph1_result_ = subgraph1_result[all_redox_x_idx]
                if reaction == 'reduction':
                    new_tensor =  torch.roll(redox_x_metal[:,124:137], shifts=-1, dims=1)
                if reaction == 'oxidation':
                    new_tensor =  torch.roll(redox_x_metal[:,124:137], shifts=1, dims=1)
                redox_x_metal_change = redox_x_metal.clone()
                redox_x_metal_change[:,124:137] = new_tensor

                redox_x_change = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_

                x_t_1[all_redox_x_idx] = redox_x_change
                x_t_1[midx] = redox_x_metal_change
            else:
                redox_x_    = x_t_1[all_redox_x_idx]
                redox_subgraph1_result_ = subgraph1_result[all_redox_x_idx]
                redox_x_change = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_

                x_t_1[all_redox_x_idx] = redox_x_change

            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_t_1, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

            all_redox_site_idx = redox_indices.tolist() + [idx for idx in delocalized_potential_indices if idx not in redox_indices.tolist()]
            batch2_redox_idx = [mapping_dict.get(site) for site in all_redox_site_idx]
            all_indices      = torch.arange(subgraph2_pooled.shape[0], device=device)
            updated_subgraph2_pooled  = subgraph2_pooled.clone()

            all_redox_site_idx_tensor = torch.tensor(all_redox_site_idx, device=device)
            same_potentials = E12s[all_redox_site_idx_tensor]

            potentials_mapping = {}
            for site_idx, potential in enumerate(same_potentials):
                if isinstance(batch2_redox_idx[site_idx], list):
                    for sub_idx in batch2_redox_idx[site_idx]:
                        potentials_mapping[sub_idx] = potential
                else:
                    potentials_mapping[batch2_redox_idx[site_idx]] = potential

            redox_sites_ = []
            for idx in batch2_redox_idx:
                if isinstance(idx, list):
                    redox_sites_.extend(idx)
                else:
                    redox_sites_.append(idx)
            
            redox_subgraph2_pooled  = subgraph2_pooled[redox_sites_]
            redox_subgraph3_result_ = subgraph3_result[redox_sites_]

            site_potentials = torch.stack([potentials_mapping[site] for site in redox_sites_])
            
            def boltzmann_distribution(potentials):
                if reaction == 'reduction':
                    potential_p = torch.softmax(potentials, dim=0)
                    potential_P = potential_p * len(potentials)
                    potential_p = torch.where(potential_P > 0.96, potential_p, torch.tensor(0.0, device=potential_p.device))
                elif reaction == 'oxidation':
                    potential_p = torch.softmax(-potentials, dim=0)
                    potential_P = potential_p * len(potentials)
                    potential_p = torch.where(potential_P > 0.96, potential_p, torch.tensor(0.0, device=potential_p.device))
                potential_p = F.normalize(potential_p, p=1, dim=0)
                return potential_p

            gate_weights = boltzmann_distribution(site_potentials)

            gate_weights = gate_weights.unsqueeze(-1)  
            redox_site_change = redox_subgraph3_result_ * (1 - gate_weights) + redox_subgraph2_pooled * gate_weights
            updated_subgraph2_pooled[redox_sites_] = redox_site_change
            subgraph2_result_ = updated_subgraph2_pooled.clone()

            delocalized_potential_indices_t_1 = list(delocalized_potential_indices)

            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

            pred_E12s = torch.cat((pred_E12s, E12.unsqueeze(0)), 0)

            # redox_pos = [redox[redox_site_idx][0]]
            # sites = [i for i in range(len(redox)) if redox[i][0] in redox_pos]

            num_redox_[redox_site_idx] = num_redox_[redox_site_idx] - 1
            
        return num_redox_all, pred_num_redox_, pred_E12s
    
    def sample_no_reaction(self, batch, device):
        # 創建兩個不同的 batch 用於氧化和還原反應
        reduction_batch = batch.clone()
        oxidation_batch = batch.clone()
        
        # 設置 reaction 屬性
        data_list_red = reduction_batch.to_data_list()
        data_list_ox = oxidation_batch.to_data_list()
        
        for graph in data_list_red:
            graph.reaction = 'reduction'
            
        for graph in data_list_ox:
            graph.reaction = 'oxidation'
            
        from torch_geometric.data import Batch
        reduction_batch = Batch.from_data_list(data_list_red)
        oxidation_batch = Batch.from_data_list(data_list_ox)
            
        # 分別調用 sample 函數進行預測
        num_redox_all_red, pred_num_redox_red, pred_E12s_red = self.sample(reduction_batch, device, warmup=True)
        num_redox_all_ox, pred_num_redox_ox, pred_E12s_ox = self.sample(oxidation_batch, device, warmup=True)
        
        return (num_redox_all_red, pred_num_redox_red, pred_E12s_red,
                num_redox_all_ox, pred_num_redox_ox, pred_E12s_ox)
  


class complex_HGRN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, cla_output_dim, depth1=3, depth2=3, depth3=4,dropout=0.3):
        super(complex_HGRN, self).__init__()
        self.GCN1 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth1, dropout=0.3)
        self.GCN2 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth2, dropout=0.3)
        self.GCN3 = BondMessagePassing(node_dim, bond_dim, hidden_dim, depth=depth3, dropout=0.3)

        self.pool = global_mean_pool
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, cla_output_dim))
        
        self.gate_GCN1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())
        self.gate_GCN3 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())


    @staticmethod
    def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index

    def forward_subgraph(self, x, edge_index, batch, edge_attr, gcn, pre_proc=None, transform_edge_attr=None):
        if pre_proc is not None:
            x = pre_proc(x)

        rev_edge_index = self._rev_edge_index(edge_index)

        if transform_edge_attr is not None:
            edge_attr = transform_edge_attr(edge_attr)

        data = Data(x=x, edge_index=edge_index, rev_edge_index=rev_edge_index, edge_attr=edge_attr)

        if isinstance(gcn, GATConv):
            result = gcn(x, edge_index, edge_attr) 
        else:
            result = gcn(data) 

        result_pooled = self.pool(result, batch)
        return result, result_pooled

    def forward(self, batch, device, global_graph=False):
        if global_graph:
            # no site in dataset
            for graph in batch.to_data_list():
                x, edge_index, edge_attr, midx, true_vals = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys

            subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
            subgraph1_edge_index, batch1 = subgraph1
            subgraph2_edge_index, batch2, filtered_mask = subgraph2
            subgraph3_edge_index, batch3 = subgraph3

            #"results after GCN and result_ after global pooling"
            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

            m_batch1  = batch1[midx[0]]
            new_batch = batch2[batch1_2.long()]

            mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
            ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device=device)
            real_num_peaks  = torch.tensor([len(true_vals)]).cuda()

            count = 0
            total_loss = 0 

            while real_num_peaks[0].item() > 0:
                count +=1 

                nums = self.classifier(subgraph3_pooled)
                pred = self.regressor(subgraph3_pooled)

                loss_cla = nn.CrossEntropyLoss()(nums, real_num_peaks)
                loss_reg = nn.MSELoss()(pred, true_vals[0].unsqueeze(0).to(device))
                total_loss += loss_cla + loss_reg

                true_vals = true_vals[1:]

                reaction_x_    = x[:]
                reaction_subgraph1_result_ = subgraph1_result[:]
                reaction_x_change = reaction_subgraph1_result_ * self.gate_GCN1(reaction_subgraph1_result_) + reaction_x_

                x_   = x.clone()
                x_[:] = reaction_x_change

                subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
                subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
                subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

                real_num_peaks = real_num_peaks.clone()  # ensure a separate copy
                real_num_peaks[0] = real_num_peaks[0] - 1

        else:
            for graph in batch.to_data_list():
                x, edge_index, edge_attr, midx, true_vals, sites = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.redox
            
            subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
            subgraph1_edge_index, batch1 = subgraph1
            subgraph2_edge_index, batch2, filtered_mask = subgraph2
            subgraph3_edge_index, batch3 = subgraph3

            #"results after GCN and result_ after global pooling"
            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

            new_batch = batch2[batch1_2.long()]

            mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
            ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device=device)
            real_num_peaks  = torch.tensor([sites[i][1] for i in range(len(sites))]).to(device)

            reaction_sites = [i for i, value in enumerate(real_num_peaks) for _ in range(int(value))]
    
            delocalized_potential_indices_t_1 = list(set(reaction_sites))

            count = 0
            total_loss = 0 
            while reaction_sites:
                count +=1 
                batch1_subgraph3_result = subgraph3_result[ordered_indices]

                each_nums = self.classifier(batch1_subgraph3_result)
                
                unique_reaction_sites = list(set(reaction_sites))
                all_preds = self.regressor(batch1_subgraph3_result)
                preds     = all_preds[unique_reaction_sites]
                pred, idx = preds.min(dim=0)

                reaction_sites_idx = unique_reaction_sites[idx]
                energy = all_preds[delocalized_potential_indices_t_1]
                energy_p = torch.softmax(-energy, dim=0)
                energy_P = energy_p * len(energy)
                delocalized_potential_indices = torch.where(energy_P > 0.96)[0]
                if delocalized_potential_indices.numel() == 1:
                    delocalized_potential_indices = torch.tensor([reaction_sites_idx], device=device)

                all_reaction_x_idx = []
                same_idx_tensor = torch.tensor(delocalized_potential_indices, device=device)
                if same_idx_tensor.numel() != 1:
                    mapped_indices = [delocalized_potential_indices_t_1[i] for i in delocalized_potential_indices.tolist()]
                    delocalized_potential_indices = list(set(mapped_indices + delocalized_potential_indices_t_1))
                    for site_idx in delocalized_potential_indices:
                        site_indices = [i for i, idx in enumerate(batch1) if idx == site_idx]
                        all_reaction_x_idx.extend(site_indices)
                    
                    real_num_peaks_ = real_num_peaks.clone()
                else:
                    site_indices = [i for i, idx in enumerate(batch1) if idx == reaction_sites_idx]
                    all_reaction_x_idx.extend(site_indices)
                    real_num_peaks_ = real_num_peaks.clone()

                loss_mask = torch.ones_like(real_num_peaks_, dtype=torch.bool)
                if len(delocalized_potential_indices_t_1) > 1:
                    delocalized_peaks = real_num_peaks_[delocalized_potential_indices_t_1]

                    if delocalized_peaks.max() != delocalized_peaks.min():
                        min_peaks = delocalized_peaks.min()
                        min_indices = [delocalized_potential_indices_t_1[i] for i, peaks in enumerate(delocalized_peaks) if peaks == min_peaks]

                        for idx in min_indices:
                            loss_mask[idx] = False
                        
                loss_per_sample = nn.CrossEntropyLoss(reduction='none')(each_nums, real_num_peaks_)
                loss_cla = (loss_per_sample * loss_mask.float()).sum() / (loss_mask.sum().float() + 1e-8)
                loss_reg = nn.MSELoss()(pred, true_vals[0].unsqueeze(0).to(device))
                total_loss += loss_cla + loss_reg

                true_vals = true_vals[1:]

                if count == 1:
                    x_t_1 = x.clone()

                redox_x_    = x_t_1[all_reaction_x_idx]
                redox_subgraph1_result_ = subgraph1_result[all_reaction_x_idx]
                redox_x_change = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_

                x_t_1[all_reaction_x_idx] = redox_x_change

                subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_t_1, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
                subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

                all_reaction_site_idx = unique_reaction_sites + [idx for idx in delocalized_potential_indices if idx not in unique_reaction_sites]
                batch2_reaction_idx = [mapping_dict.get(site) for site in all_reaction_site_idx]
                updated_subgraph2_pooled  = subgraph2_pooled.clone()

                all_reaction_site_idx_tensor = torch.tensor(all_reaction_site_idx, device=device)
                same_energy = all_preds[all_reaction_site_idx_tensor]

                potentials_mapping = {}
                for site_idx, potential in enumerate(same_energy):
                    if isinstance(batch2_reaction_idx[site_idx], list):
                        for sub_idx in batch2_reaction_idx[site_idx]:
                            potentials_mapping[sub_idx] = potential
                    else:
                        potentials_mapping[batch2_reaction_idx[site_idx]] = potential

                reaction_sites_ = []
                for idx in batch2_reaction_idx:
                    if isinstance(idx, list):
                        reaction_sites_.extend(idx)
                    else:
                        reaction_sites_.append(idx)
                
                reaction_subgraph2_pooled  = subgraph2_pooled[reaction_sites_]
                reaction_subgraph3_result_ = subgraph3_result[reaction_sites_]

                def boltzmann_distribution(energies):
                    energy_p = torch.softmax(-energies, dim=0)
                    energy_P = energy_p * len(energies)
                    energy_p = torch.where(energy_P > 0.96, energy_p, torch.tensor(0.0, device=energy_p.device))
                    energy_p = F.normalize(energy_p, p=1, dim=0)
                    return energy_p

                site_potentials   = torch.stack([potentials_mapping[site] for site in reaction_sites_])
                gate_weights      = boltzmann_distribution(site_potentials)
                reaction_site_change = reaction_subgraph3_result_ * (1 - gate_weights) + reaction_subgraph2_pooled  * gate_weights 

                updated_subgraph2_pooled[reaction_sites_] = reaction_site_change
                subgraph2_result_ = updated_subgraph2_pooled.clone()

                subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

                delocalized_potential_indices_t_1 = delocalized_potential_indices

                reaction_sites.remove(reaction_sites_idx)

                real_num_peaks = real_num_peaks.clone()  # ensure a separate copy
                real_num_peaks[reaction_sites_idx] = real_num_peaks[reaction_sites_idx] - 1

        return total_loss

    def sample(self, batch, device, global_graph=False):
        if global_graph:
            for graph in batch.to_data_list():
                x, edge_index, edge_attr, midx, true_vals = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys
        
            subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
            subgraph1_edge_index, batch1 = subgraph1
            subgraph2_edge_index, batch2, filtered_mask = subgraph2
            subgraph3_edge_index, batch3 = subgraph3

            #"results after GCN and result_ after global pooling"
            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

            m_batch1  = batch1[midx[0]]
            new_batch = batch2[batch1_2.long()]

            num_peaks = self.classifier(subgraph3_pooled)

            num_peaks_pred = torch.argmax(num_peaks, dim=1)

            preds_output = torch.tensor([], device=device)

            count = 0
            while num_peaks_pred.sum() != 0:
                count +=1 
                pred = self.regressor(subgraph3_result) 

                reaction_x_    = x[:]
                reaction_subgraph1_result_ = subgraph1_result[:]
                reaction_x_change = reaction_subgraph1_result_ * self.gate_GCN1(reaction_subgraph1_result_) + reaction_x_

                x_ = x.clone()
                x_[:] = reaction_x_change

                subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
                subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
                subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

                preds_output = torch.cat((preds_output, pred.unsqueeze(0)), 0)

                num_peaks_pred[0] = num_peaks_pred[0] - 1

        else:
            for graph in batch.to_data_list():
                x, edge_index, edge_attr, midx, true_vals, sites = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.redox
            
            subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
            subgraph1_edge_index, batch1 = subgraph1
            subgraph2_edge_index, batch2, filtered_mask = subgraph2
            subgraph3_edge_index, batch3 = subgraph3

            #"results after GCN and result_ after global pooling"
            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

            m_batch1  = batch1[midx[0]]
            new_batch = batch2[batch1_2.long()]

            mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
            ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device=device)


            batch1_subgraph3_result = subgraph3_result[ordered_indices]
            num_peaks = self.classifier(batch1_subgraph3_result)
            num_peaks_pred = torch.argmax(num_peaks, dim=1)

            preds_output = torch.tensor([], device=device)

            num_mask = num_peaks_pred > 0
            reaction_indices = torch.nonzero(num_mask, as_tuple=False).flatten()
            delocalized_potential_indices_t_1 = list(set([i.item() for i in reaction_indices]))


            count = 0
            while num_peaks_pred.sum() != 0:
                count +=1 
                batch1_subgraph3_result = subgraph3_result[ordered_indices]
                all_preds = self.regressor(batch1_subgraph3_result) 
                all_preds = all_preds.squeeze()
                num_mask = num_peaks_pred > 0
                reaction_indices = torch.nonzero(num_mask, as_tuple=False).flatten()
                preds = all_preds[num_mask]
                pred, idx = preds.min(dim=0)

                reaction_site_idx = reaction_indices[idx].item()
                
                energy = all_preds[delocalized_potential_indices_t_1]
                energy_p = torch.softmax(-energy, dim=0)
                energy_P = energy_p * len(energy)
                delocalized_potential_indices = torch.where(energy_P > 0.96)[0]
                if delocalized_potential_indices.numel() == 1:
                    delocalized_potential_indices = torch.tensor([], device=device)

                all_reaction_x_idx = []
                same_idx_tensor = torch.tensor(delocalized_potential_indices, device=device)
                if same_idx_tensor.numel() != 0:
                    mapped_indices = [delocalized_potential_indices_t_1[i] for i in delocalized_potential_indices.tolist()]
                    delocalized_potential_indices = list(set(mapped_indices + delocalized_potential_indices_t_1))
                    for site_idx in delocalized_potential_indices:
                        site_indices = [i for i, idx in enumerate(batch1) if idx == site_idx]
                        all_reaction_x_idx.extend(site_indices)

                else:
                    site_indices = [i for i, idx in enumerate(batch1) if idx == reaction_site_idx]
                    all_reaction_x_idx.extend(site_indices)

                
                if count == 1:
                    x_t_1 = x.clone()

                redox_x_    = x_t_1[all_reaction_x_idx]
                redox_subgraph1_result_ = subgraph1_result[all_reaction_x_idx]
                redox_x_change = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_

                x_t_1[all_reaction_x_idx] = redox_x_change

                subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_t_1, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
                subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))


                all_reaction_site_idx = reaction_indices.tolist() + [idx for idx in delocalized_potential_indices if idx not in reaction_indices.tolist()]
                batch2_reaction_idx = [mapping_dict.get(site) for site in all_reaction_site_idx]
                updated_subgraph2_pooled  = subgraph2_pooled.clone()

                all_reaction_site_idx_tensor = torch.tensor(all_reaction_site_idx, device=device)
                same_energy = all_preds[all_reaction_site_idx_tensor]

                potentials_mapping = {}
                for site_idx, potential in enumerate(same_energy):
                    if isinstance(batch2_reaction_idx[site_idx], list):
                        for sub_idx in batch2_reaction_idx[site_idx]:
                            potentials_mapping[sub_idx] = potential
                    else:
                        potentials_mapping[batch2_reaction_idx[site_idx]] = potential

                reaction_sites_ = []
                for idx in batch2_reaction_idx:
                    if isinstance(idx, list):
                        reaction_sites_.extend(idx)
                    else:
                        reaction_sites_.append(idx)
                
                reaction_subgraph2_pooled  = subgraph2_pooled[reaction_sites_]
                reaction_subgraph3_result_ = subgraph3_result[reaction_sites_]

                def boltzmann_distribution(energies):
                    energy_p = torch.softmax(-energies, dim=0)
                    energy_P = energy_p * len(energies)
                    energy_p = torch.where(energy_P > 0.96, energy_p, torch.tensor(0.0, device=energy_p.device))
                    energy_p = F.normalize(energy_p, p=1, dim=0)
                    return energy_p

                site_potentials   = torch.stack([potentials_mapping[site] for site in reaction_sites_])
                gate_weights      = boltzmann_distribution(site_potentials)
                gate_weights = gate_weights.unsqueeze(-1) 
                reaction_site_change = reaction_subgraph3_result_ * (1 - gate_weights) + reaction_subgraph2_pooled  * gate_weights 

                updated_subgraph2_pooled[reaction_sites_] = reaction_site_change
                subgraph2_result_ = updated_subgraph2_pooled.clone()
                subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

                delocalized_potential_indices_t_1 = list(delocalized_potential_indices)

                preds_output = torch.cat((preds_output, pred.unsqueeze(0)), 0)

                num_peaks_pred[reaction_site_idx] = num_peaks_pred[reaction_site_idx] - 1

        return num_peaks, num_peaks_pred, preds_output
