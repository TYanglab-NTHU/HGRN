import torch 
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn   import global_mean_pool, global_max_pool
from torch_geometric.nn   import MessagePassing, GCNConv,  Linear, BatchNorm, GlobalAttention, GATConv

import pandas as pd 

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
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device=device)

        real_num_peaks = torch.tensor([graph.redox[i][1] for i in range(len(graph.redox))]).cuda()

        redox_sites = [i for i, value in enumerate(real_num_peaks) for _ in range(int(value))]
        # if redox_sites == []:
        #     loss_cla    = nn.CrossEntropyLoss()(each_num_redox, real_num_peaks)
        #     # loss_cla.backward(retain_graph=True)
        #     total_loss += loss_cla

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
                lig_potentials = self.E12_reg_red(batch1_subgraph3_result[unique_redox_sites])
                E12, idx = lig_potentials.max(dim=0)
            elif reaction == 'oxidation':
                lig_potentials = self.E12_reg_ox(batch1_subgraph3_result[unique_redox_sites])
                E12, idx = lig_potentials.min(dim=0)
            E12s.append(E12)
        
            loss_cla    = nn.CrossEntropyLoss()(each_num_redox, real_num_peaks)
            loss_reg    = nn.MSELoss()(E12, real_E12[0])
            loss        = loss_cla + loss_reg
            total_loss += loss
            # loss.backward(retain_graph=True)

            real_E12       = real_E12[1:]
            redox_site_idx = unique_redox_sites[idx]

            # gat x with GCN1
            redox_x_idx = [i for i, idx in enumerate(batch1) if idx == redox_site_idx]
            redox_x_    = x[redox_x_idx]
            redox_subgraph1_result_  = subgraph1_result[redox_x_idx]
            if redox_site_idx == m_batch1:
                if reaction == 'reduction':
                    new_tensor =  torch.roll(redox_x_[:,124:137], shifts=-1, dims=1)
                if reaction == 'oxidation':
                    new_tensor =  torch.roll(redox_x_[:,124:137], shifts=1, dims=1)
                redox_x_change = redox_x_.clone()
                redox_x_change[:,124:137] = new_tensor
            else:
                redox_x_change = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_
                # redox_x_change =  redox_x_ * self.gate_GCN1( redox_x_) + redox_x_

            x_              = x.clone()
            x_[redox_x_idx] = redox_x_change
            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

            # gat GCN2 with GCN3
            batch2_redox_idx = [mapping_dict.get(site) for site in unique_redox_sites]
            all_indices      = torch.arange(subgraph2_pooled.shape[0], device=device)
            updated_subgraph2_pooled  = subgraph2_pooled.clone()

            potentials_mapping = {}
            # 創建電位映射
            for site_idx, potential in enumerate(lig_potentials):
                if isinstance(batch2_redox_idx[site_idx], list):
                    for sub_idx in batch2_redox_idx[site_idx]:
                        potentials_mapping[sub_idx] = potential
                else:
                    potentials_mapping[batch2_redox_idx[site_idx]] = potential

            redox_sites_ = []
            # 收集所有 redox sites
            for idx in batch2_redox_idx:
                if isinstance(idx, list):
                    redox_sites_.extend(idx)
                else:
                    redox_sites_.append(idx)
            
            redox_subgraph2_pooled  = subgraph2_pooled[redox_sites_]
            redox_subgraph3_result_ = subgraph3_result[redox_sites_]

            # 根據映射獲取對應的電位
            site_potentials = torch.stack([potentials_mapping[site] for site in redox_sites_])
            
            def boltzmann_distribution(potentials, temperature=1.0):
                weights = torch.exp(-potentials / temperature)
                return weights / weights.sum()

            gate_weights = boltzmann_distribution(site_potentials)
            redox_site_change = redox_subgraph3_result_ * gate_weights + redox_subgraph2_pooled 

            updated_subgraph2_pooled[redox_sites_] = redox_site_change
            subgraph2_result_ = updated_subgraph2_pooled.clone()

            # redox_site_change = redox_subgraph3_result_ * self.gate_GCN3(redox_subgraph3_result_) + redox_subgraph2_pooled
            # redox_site_change = redox_subgraph2_pooled  * self.gate_GCN3(redox_subgraph2_pooled) + redox_subgraph2_pooled
            # subgraph2_result_ = torch.cat([updated_subgraph2_pooled[:batch2_redox_idx], redox_site_change.unsqueeze(0), updated_subgraph2_pooled[batch2_redox_idx:]], dim=0)

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
                loss_reg    = nn.MSELoss()(E12, real_E12[0])
                total_loss += loss_reg
                # loss_reg.backward(retain_graph=True)

        return total_loss

    def sample(self, batch, device):
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
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)]).to(device)

        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        if   reaction == 'reduction':
            num_redox_all = self.num_peaks_red(batch1_subgraph3_result)
        elif reaction == 'oxidation':
            num_redox_all = self.num_peaks_ox(batch1_subgraph3_result)
        
        num_redox_ = torch.argmax(num_redox_all, dim=1)
        pred_num_redox_ = num_redox_.clone()
        pred_E12s  = torch.tensor([], device=device)

        while num_redox_.sum() != 0:
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
            # gat x with GCN1
            redox_x_idx = [i for i, idx in enumerate(batch1) if idx == redox_site_idx]
            redox_x_    = x[redox_x_idx]
            redox_subgraph1_result_ = subgraph1_result[redox_x_idx]
            if redox_site_idx == m_batch1:
                if reaction == 'reduction':
                    new_tensor =  torch.roll(redox_x_[:,124:137], shifts=-1, dims=1)
                if reaction == 'oxidation':
                    new_tensor =  torch.roll(redox_x_[:,124:137], shifts=1, dims=1)
                redox_x_change = redox_x_.clone()
                redox_x_change[:,124:137] = new_tensor
            else:
                redox_x_change          = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_

            x_              = x.clone()
            x_[redox_x_idx] = redox_x_change
            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x_, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

            # gat GCN2 with GCN3
            batch2_redox_idx = [mapping_dict.get(site.item()) for site in redox_indices]
            # batch2_redox_idx = mapping_dict.get(redox_site_idx)
            all_indices      = torch.arange(subgraph2_pooled.shape[0], device=device)
            updated_subgraph2_pooled  = subgraph2_pooled.clone()
            potentials_mapping = {}
            # 創建電位映射
            for site_idx, potential in enumerate(E12s_redox):
                if isinstance(batch2_redox_idx[site_idx], list):
                    for sub_idx in batch2_redox_idx[site_idx]:
                        potentials_mapping[sub_idx] = potential
                else:
                    potentials_mapping[batch2_redox_idx[site_idx]] = potential

            redox_sites_ = []
            # 收集所有 redox sites
            for idx in batch2_redox_idx:
                if isinstance(idx, list):
                    redox_sites_.extend(idx)
                else:
                    redox_sites_.append(idx)
            
            redox_subgraph2_pooled  = subgraph2_pooled[redox_sites_]
            redox_subgraph3_result_ = subgraph3_result[redox_sites_]

            # 根據映射獲取對應的電位
            site_potentials = torch.stack([potentials_mapping[site] for site in redox_sites_])
            
            def boltzmann_distribution(potentials, temperature=1.0):
                weights = torch.exp(-potentials / temperature)
                return weights / weights.sum()

            gate_weights = boltzmann_distribution(site_potentials)
            redox_site_change = redox_subgraph3_result_ * gate_weights.unsqueeze(-1) + redox_subgraph2_pooled
            
            updated_subgraph2_pooled[redox_sites_] = redox_site_change
            subgraph2_result_ = updated_subgraph2_pooled.clone()
            # redox_site_change = redox_subgraph3_result_ * self.gate_GCN3(redox_subgraph3_result_) + redox_subgraph2_pooled
            # redox_site_change = redox_subgraph2_pooled  * self.gate_GCN3(redox_subgraph2_pooled) + redox_subgraph2_pooled
            # subgraph2_result_ = torch.cat([updated_subgraph2_pooled[:batch2_redox_idx], redox_site_change.unsqueeze(0), updated_subgraph2_pooled[batch2_redox_idx:]], dim=0)

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
        num_redox_all_red, pred_num_redox_red, pred_E12s_red = self.sample(reduction_batch, device)
        num_redox_all_ox, pred_num_redox_ox, pred_E12s_ox = self.sample(oxidation_batch, device)
        
        return (num_redox_all_red, pred_num_redox_red, pred_E12s_red,
                num_redox_all_ox, pred_num_redox_ox, pred_E12s_ox)


# with torch.set_grad_enabled(True):      
    # subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
    # subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=model.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
    # subgraph3_result, subgraph3_pooled = model.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=model.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))
