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
        subgraph2_edge_index, batch2, filtered_masks = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        #"results after GCN and result_ after global pooling"
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

        total_loss = 0
        # convert batch1 index to batch3 index
        m_batch1  = batch1[midx]
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
                loss_reg    = nn.MSELoss()(E12, real_E12[0])
                total_loss += loss_reg
                # loss_reg.backward(retain_graph=True)

        return total_loss

    def get_subgraph2_representation(self, batch, device):
        """
        計算並返回 subgraph2_pooled 特徵表示。
        """
        for graph in batch.to_data_list():
            # 只需要 x, edge_index, edge_attr
            x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr

        subgraph1, batch1_2, subgraph2, _ = edge_index # 不需要 subgraph3
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2 = subgraph2
        # subgraph3_edge_index, batch3 = subgraph3 # 不需要

        # 計算 subgraph1
        subgraph1_result, _ = self.forward_subgraph( # 不需要 subgraph1_pooled
            x=x, edge_index=subgraph1_edge_index, batch=batch1,
            edge_attr=edge_attr[0], gcn=self.GCN1
        )
        # 計算 subgraph2
        _, subgraph2_pooled = self.forward_subgraph( # 只需要 subgraph2_pooled
            x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2,
            edge_attr=edge_attr[1], gcn=self.GCN2,
            pre_proc=lambda x: global_mean_pool(x, batch1_2)
        )

        return subgraph2_pooled

    def sample(self, batch, device):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx, reaction, redox = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.reaction, graph.redox
        
        subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2, filtered_masks = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        #"results after GCN and result_ after global pooling"
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

        # convert batch1 index to batch3 index
        m_batch1  = batch1[midx]
        new_batch = batch2[batch1_2.long()]

        mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device='cuda')

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
                    new_tensor =  torch.roll(redox_x_[124:137], shifts=-1, dims=1)
                if reaction == 'oxidation':
                    new_tensor =  torch.roll(redox_x_[124:137], shifts=1, dims=1)
                redox_x_change = redox_x_.clone()
                redox_x_change[124:137] = new_tensor
            else:
                redox_x_change          = redox_subgraph1_result_ * self.gate_GCN1(redox_subgraph1_result_) + redox_x_

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
        subgraph2_edge_index, batch2, filtered_masks = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        # --- Initial Forward Pass (Common for both Red/Ox) ---
        subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
        subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        # Note: The original code had a potential issue applying transform_edge_attr only here. Assuming it's correct.
        subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)

        # --- Batch Mapping (Common) ---
        m_batch1  = batch1[midx]
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
                # **This gating logic might need careful review based on your model's intention**
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


    def compute_grad_cam(self, batch, device):
        """計算所有節點的 Grad-CAM，不進行任何遮罩
        
        Args:
            batch: 輸入批次數據
            device: 計算設備
            
        Returns:
            atom_grad_cam: 原子層級的重要性分數
            ligand_grad_cam: 配體層級的重要性分數
            complex_grad_cam: 複合物層級的重要性分數
        """
        # 前向傳播
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx, real_E12, reaction, redox = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction, graph.redox
        
        subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2 = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        # 保存中間結果並啟用梯度計算
        with torch.set_grad_enabled(True):
            # 確保所有中間結果都保留梯度
            subgraph1_result, subgraph1_pooled = self.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=self.GCN1)
            subgraph1_result.retain_grad()
            
            subgraph2_result, subgraph2_pooled = self.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=self.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
            subgraph2_result.retain_grad()
            
            subgraph3_result, subgraph3_pooled = self.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=self.GCN3)
            subgraph3_result.retain_grad()
            
            # 計算 E12 預測
            if reaction == 'reduction':
                E12 = self.E12_reg_red(subgraph3_result)
                E12_scalar = torch.max(E12)
            else:
                E12 = self.E12_reg_ox(subgraph3_result)
                E12_scalar = torch.min(E12)

            # 清除之前的梯度
            if subgraph3_result.grad is not None:
                subgraph3_result.grad.zero_()
            if subgraph2_result.grad is not None:
                subgraph2_result.grad.zero_()
            if subgraph1_result.grad is not None:
                subgraph1_result.grad.zero_()
            
            # 計算梯度
            E12_scalar.backward(retain_graph=True)
            
            # 計算所有節點的 Grad-CAM
            # 1. 複合物層級
            gradients3 = subgraph3_result.grad
            features3 = subgraph3_result
            weights3 = torch.mean(gradients3, dim=1, keepdim=True)
            complex_grad_cam = torch.sum(weights3 * features3, dim=1)
            complex_grad_cam = torch.relu(complex_grad_cam)
            complex_grad_cam = complex_grad_cam / (torch.max(complex_grad_cam) + 1e-8)
            
            # 2. 配體層級
            gradients2 = subgraph2_result.grad
            features2 = subgraph2_result
            weights2 = torch.mean(gradients2, dim=1, keepdim=True)
            ligand_grad_cam = torch.sum(weights2 * features2, dim=1)
            ligand_grad_cam = torch.relu(ligand_grad_cam)
            ligand_grad_cam = ligand_grad_cam / (torch.max(ligand_grad_cam) + 1e-8)
            
            # 3. 原子層級
            gradients1 = subgraph1_result.grad
            features1 = subgraph1_result
            weights1 = torch.mean(gradients1, dim=1, keepdim=True)
            atom_grad_cam = torch.sum(weights1 * features1, dim=1)
            atom_grad_cam = torch.relu(atom_grad_cam)
            atom_grad_cam = atom_grad_cam / (torch.max(atom_grad_cam) + 1e-8)
            
            return atom_grad_cam, ligand_grad_cam, complex_grad_cam



# with torch.set_grad_enabled(True):      
    # subgraph1_result, subgraph1_pooled = model.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=model.GCN1)
    # subgraph2_result, subgraph2_pooled = model.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=model.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
    # subgraph3_result, subgraph3_pooled = model.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=model.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))