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
        # --- Initial Graph Processing (Common for both Red/Ox) ---
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx = graph.x, graph.edge_index, graph.edge_attr, graph.midx
            # Assuming redox site info might be needed later, grab it here if available
            # redox = graph.redox # Example, adjust if attribute name is different

        subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2, filtered_mask = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        # --- Initial Forward Pass (Common for both Red/Ox) ---
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        # Note: The original code had a potential issue applying transform_edge_attr only here. Assuming it's correct.
        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

        # --- Batch Mapping (Common) ---
        m_batch1  = batch1[midx[0]]
        new_batch = batch2[batch1_2.long()]
        mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device=device) # Use specified device

        # --- Initial Peak Predictions ---
        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        num_redox_all_red = self.num_peaks_red(batch1_subgraph3_result)
        num_redox_all_ox  = self.num_peaks_ox(batch1_subgraph3_result)

        num_redox_red_ = torch.argmax(num_redox_all_red, dim=1)
        num_redox_ox_  = torch.argmax(num_redox_all_ox, dim=1)

        pred_num_redox_red = num_redox_red_.clone()
        pred_num_redox_ox  = num_redox_ox_.clone()

        pred_E12s_red = torch.tensor([], device=device)
        pred_E12s_ox  = torch.tensor([], device=device)

        # --- Reduction Prediction Loop ---
        x_red = x.clone()
        subgraph1_result_red = subgraph1_result.clone()
        subgraph2_pooled_red = subgraph2_pooled.clone()
        subgraph3_result_red = subgraph3_result.clone()
        num_redox_red_loop = num_redox_red_.clone() # Use a separate counter for the loop

        while num_redox_red_loop.sum() > 0: # Check against zero
            current_batch1_subgraph3_result = subgraph3_result_red[ordered_indices]
            E12s_red = self.E12_reg_red(current_batch1_subgraph3_result).squeeze()

            redox_mask_red    = num_redox_red_loop > 0
            redox_indices_red = torch.nonzero(redox_mask_red, as_tuple=False).flatten()

            if redox_indices_red.numel() == 0: break # Exit if no sites left

            E12s_redox_red = E12s_red[redox_mask_red]

            # Handle case where E12s_redox_red might be empty if mask is all False
            if E12s_redox_red.numel() == 0: break

            E12_red, filtered_idx_red = torch.max(E12s_redox_red, dim=0)
            redox_site_idx_red = redox_indices_red[filtered_idx_red].item()

            # --- Update State for Reduction ---
            redox_x_idx_red = (batch1 == redox_site_idx_red).nonzero(as_tuple=False).flatten() # More robust way to get indices
            redox_x_red = x_red[redox_x_idx_red]

            # Apply reduction change to atom features (assuming indices 124:137 are correct)
            # Ensure redox_x_red is not empty before indexing
            if redox_x_red.numel() > 0 and redox_x_red.shape[1] > 136: # Check dimensions
                 # Check if this indexing is still relevant and correct for your features
                if 124 < redox_x_red.shape[1] and 137 <= redox_x_red.shape[1]:
                    new_tensor_red = torch.roll(redox_x_red[:, 124:137], shifts=-1, dims=-1) # Use last dim for features
                    x_red[redox_x_idx_red, 124:137] = new_tensor_red
                else:
                    print(f"Warning: Feature dimension mismatch for reduction update at site {redox_site_idx_red}. Skipping feature update.")
                    # Handle the case where features are not as expected - maybe skip update or log error

            else:
                 # Handle case where no atoms found for the site or feature dim too small
                 print(f"Warning: Could not find atoms or features for reduction site {redox_site_idx_red}. Skipping feature update.")


            # --- Recalculate Forward Pass for Reduction ---
            subgraph1_result_red, subgraph1_pooled_red = self.forward_subgraph(x=x_red, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result_red, subgraph2_pooled_red = self.forward_subgraph(x=subgraph1_result_red, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

            # Apply GCN3 Gate Update (carefully replicating original logic for reduction)
            batch2_redox_idx_red = mapping_dict.get(redox_site_idx_red)
            if batch2_redox_idx_red is not None: # Check if site exists in batch2
                redox_subgraph2_pooled_red = subgraph2_pooled_red[batch2_redox_idx_red]
                # Original code used subgraph3_result - assume that's the intended input for gating
                redox_subgraph3_result_for_gate = subgraph3_result_red[batch2_redox_idx_red] # Use the state BEFORE GCN3 recalculation for gating? Check logic.
                # Or maybe use the new subgraph3_result_red? Let's assume previous state based on original code.
                # **This gating logic might need careful review based on your self's intention**
                redox_site_change_red = redox_subgraph3_result_for_gate * self.gate_GCN3(redox_subgraph3_result_for_gate) + redox_subgraph2_pooled_red # Gating applied

                # Create updated pooled result for GCN3 input
                subgraph2_pooled_updated_red = subgraph2_pooled_red.clone()
                subgraph2_pooled_updated_red[batch2_redox_idx_red] = redox_site_change_red # Update the specific site
            else:
                # Site not found in batch2, use original pooled result
                subgraph2_pooled_updated_red = subgraph2_pooled_red

            subgraph3_result_red, subgraph3_pooled_red = self.forward_subgraph(x=subgraph2_pooled_updated_red, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

            # --- Store Result and Decrement Count ---
            pred_E12s_red = torch.cat((pred_E12s_red, E12_red.unsqueeze(0)), 0)

            # Decrement count for the site (and potentially related sites if needed)
            # Assuming only the selected site's count decreases by 1
            # If 'redox' structure is needed:
            # redox_pos = [r[0] for r_idx, r in enumerate(redox) if r_idx == redox_site_idx_red] # Get position info if needed
            # sites_to_decrement = [i for i, r in enumerate(redox) if r[0] in redox_pos]
            # num_redox_red_loop[sites_to_decrement] -= 1
            # Simplified: Decrement only the selected site index
            if redox_site_idx_red < len(num_redox_red_loop):
                 num_redox_red_loop[redox_site_idx_red] -= 1
            else:
                 print(f"Warning: redox_site_idx_red {redox_site_idx_red} out of bounds for num_redox_red_loop.")


        # --- Oxidation Prediction Loop ---
        x_ox = x.clone()
        subgraph1_result_ox = subgraph1_result.clone()
        subgraph2_pooled_ox = subgraph2_pooled.clone()
        subgraph3_result_ox = subgraph3_result.clone()
        num_redox_ox_loop = num_redox_ox_.clone() # Use a separate counter

        while num_redox_ox_loop.sum() > 0: # Check against zero
            current_batch1_subgraph3_result = subgraph3_result_ox[ordered_indices]
            E12s_ox = self.E12_reg_ox(current_batch1_subgraph3_result).squeeze()

            redox_mask_ox    = num_redox_ox_loop > 0
            redox_indices_ox = torch.nonzero(redox_mask_ox, as_tuple=False).flatten()

            if redox_indices_ox.numel() == 0: break # Exit if no sites left

            E12s_redox_ox = E12s_ox[redox_mask_ox]

            if E12s_redox_ox.numel() == 0: break

            E12_ox, filtered_idx_ox = torch.min(E12s_redox_ox, dim=0) # Use min for oxidation
            redox_site_idx_ox = redox_indices_ox[filtered_idx_ox].item()

            # --- Update State for Oxidation ---
            redox_x_idx_ox = (batch1 == redox_site_idx_ox).nonzero(as_tuple=False).flatten()
            redox_x_ox = x_ox[redox_x_idx_ox]

            # Apply oxidation change to atom features
            if redox_x_ox.numel() > 0 and redox_x_ox.shape[1] > 136:
                if 124 < redox_x_ox.shape[1] and 137 <= redox_x_ox.shape[1]:
                    new_tensor_ox = torch.roll(redox_x_ox[:, 124:137], shifts=1, dims=-1) # Use shift=1 for oxidation
                    x_ox[redox_x_idx_ox, 124:137] = new_tensor_ox
                else:
                    print(f"Warning: Feature dimension mismatch for oxidation update at site {redox_site_idx_ox}. Skipping feature update.")
            else:
                print(f"Warning: Could not find atoms or features for oxidation site {redox_site_idx_ox}. Skipping feature update.")


            # --- Recalculate Forward Pass for Oxidation ---
            subgraph1_result_ox, subgraph1_pooled_ox = self.forward_subgraph(x=x_ox, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result_ox, subgraph2_pooled_ox = self.forward_subgraph(x=subgraph1_result_ox, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

            # Apply GCN3 Gate Update for Oxidation
            batch2_redox_idx_ox = mapping_dict.get(redox_site_idx_ox)
            if batch2_redox_idx_ox is not None:
                redox_subgraph2_pooled_ox = subgraph2_pooled_ox[batch2_redox_idx_ox]
                redox_subgraph3_result_for_gate_ox = subgraph3_result_ox[batch2_redox_idx_ox] # Check if this state is correct for gating
                redox_site_change_ox = redox_subgraph3_result_for_gate_ox * self.gate_GCN3(redox_subgraph3_result_for_gate_ox) + redox_subgraph2_pooled_ox
                subgraph2_pooled_updated_ox = subgraph2_pooled_ox.clone()
                subgraph2_pooled_updated_ox[batch2_redox_idx_ox] = redox_site_change_ox
            else:
                subgraph2_pooled_updated_ox = subgraph2_pooled_ox

            subgraph3_result_ox, subgraph3_pooled_ox = self.forward_subgraph(x=subgraph2_pooled_updated_ox, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

            # --- Store Result and Decrement Count ---
            pred_E12s_ox = torch.cat((pred_E12s_ox, E12_ox.unsqueeze(0)), 0)

            # Decrement count for the site
            if redox_site_idx_ox < len(num_redox_ox_loop):
                num_redox_ox_loop[redox_site_idx_ox] -= 1
            else:
                 print(f"Warning: redox_site_idx_ox {redox_site_idx_ox} out of bounds for num_redox_ox_loop.")


        # --- Return Results ---
        return (num_redox_all_red, pred_num_redox_red, pred_E12s_red,
                num_redox_all_ox, pred_num_redox_ox, pred_E12s_ox)
