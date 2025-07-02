import pandas as pd 

import torch 
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import Data
from torch_geometric.nn   import global_mean_pool
from torch_geometric.nn   import MessagePassing, GCNConv,  Linear, BatchNorm, GlobalAttention, GATConv

#dielectric constant, refractive index
solvent_dict = {'ACN': [36.6*0.01, 1.3441*0.1],   
                'DMF': [36.7*0.01, 1.4305*0.1],
                'H2O': [78.5*0.01, 1.3330*0.1]}   #3.92*0.1 , / 3.46*0.1 , 

class DMPNN(nn.Module):
    def __init__(self, node_features, bond_features, hidden_size, depth=3, dropout=0.3):
        super(DMPNN, self).__init__()
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

class GAT(nn.Module):
    """
    使用 GATConv 取代手寫 message-passing：
    - 支援邊特徵：edge_dim=bond_features
    - 多頭注意力，可自行調整 heads
    - 仍保留 residual、dropout 與輸出線性層，方便銜接後續模型
    """
    def __init__(self,node_features: int,bond_features: int,hidden_size: int,depth: int = 5,heads: int = 4,dropout: float = 0.3):
        super(GAT, self).__init__()

        self.depth   = depth
        self.dropout = dropout

        # (1) 第一層把 node+bond 映射到 hidden_size
        #     之後的層直接吃 hidden_size*heads
        gat_layers = []
        in_dim = node_features
        for i in range(depth):
            gat_layers.append(
                GATConv(
                    in_channels=in_dim,
                    out_channels=hidden_size,
                    heads=heads,
                    edge_dim=bond_features,
                    dropout=dropout,
                    add_self_loops=True,   # 保留自迴圈
                )
            )
            in_dim = hidden_size * heads  # 下一層輸入維度
        self.gats = nn.ModuleList(gat_layers)

        # (2) 最終線性層：concat(x, h_final) → hidden_size
        self.fc_out = nn.Linear(node_features + hidden_size * heads,
                                hidden_size)

        # (3) 初始化
        self.reset_parameters()

    def reset_parameters(self):
        for gat in self.gats:
            gat.reset_parameters()
        nn.init.kaiming_normal_(self.fc_out.weight,
                                mode='fan_out',
                                nonlinearity='relu')
        nn.init.zeros_(self.fc_out.bias)

    @staticmethod
    def _sanitize(t: torch.Tensor) -> torch.Tensor:
        """把 NaN / Inf 轉成有限值以避免梯度爆炸"""
        if torch.isfinite(t).all():
            return t
        return torch.nan_to_num(t, nan=0.0, posinf=1e3, neginf=-1e3)

    def forward(self,batch) -> torch.Tensor:
        """
        x            : [N, node_features]
        edge_index   : [2, E]
        edge_attr    : [E, bond_features]
        rev_edge_index : 保留參數位置以相容既有呼叫；GAT 不會用到
        """
        x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        device = x.device
        edge_index = edge_index.to(device)
        edge_attr  = edge_attr.to(device)

        h = self._sanitize(x)

        # ----- (1) 多層 GAT -----
        for gat in self.gats:
            h = gat(h, edge_index, edge_attr)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self._sanitize(h)

        # ----- (2) Residual  + 輸出層 -----
        h_cat = torch.cat([x, h], dim=1)          # [N, node+hidden*heads]
        out   = F.relu(self.fc_out(h_cat))        # [N, hidden_size]
        out   = F.dropout(out, p=self.dropout, training=self.training)
        out   = self._sanitize(out)
        return out         
    

class OGNN_RNN_allmask(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, output_dim, depth=3 ,dropout=0.3, model_type='DMPNN'):
        super(OGNN_RNN_allmask, self).__init__()
        """
        Train all potentials of organic compounds in gas and solvent phase,
        and GCN1 and gate_GCN1 transfer learning to OGNN_RNN_solvent and OMGNN_RNN"""
        if model_type == 'DMPNN':
            self.GCN1 = DMPNN(node_dim, bond_dim, hidden_dim, depth=depth, dropout=0.3)
        elif model_type == 'GAT':
            self.GCN1 = GAT(node_dim, bond_dim, hidden_dim, depth=depth, dropout=0.3)
        self.num_peaks_IP = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5))
        self.num_peaks_EA = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5))
        self.num_peaks_red = nn.Sequential(
            nn.Linear(hidden_dim + 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5))
        self.num_peaks_ox = nn.Sequential(
            nn.Linear(hidden_dim+ 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 5))  
        self.reg_IP = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.reg_EA = nn.Sequential(
            nn.Linear(hidden_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.reg_red = nn.Sequential(
            nn.Linear(hidden_dim + 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.reg_ox = nn.Sequential(
            nn.Linear(hidden_dim + 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim))
        self.pool = global_mean_pool
        self.gate_GCN1 = nn.Sequential(
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

    def forward_subgraph1(self, x, subgraph1_edge_index, batch1, edge_attr):
        subgraph1_rev_edge_index = self._rev_edge_index(subgraph1_edge_index)
        subgraph1_batch   = Data(x=x, edge_index=subgraph1_edge_index, rev_edge_index=subgraph1_rev_edge_index, edge_attr=edge_attr)
        subgraph1_result  = self.GCN1(subgraph1_batch)
        subgraph1_result_ = self.pool(subgraph1_result, batch1)
        return subgraph1_result, subgraph1_result_

    def forward(self, batch, device):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, true_potentials, reaction= graph.x, graph.edge_index, graph.edge_attr, graph.ys, graph.reaction
            if graph.solvent is not None:
                solvent = graph.solvent
            else:
                solvent = 'None'
        subgraph1_edge_index, batch1 = edge_index
        
        keys = true_potentials.keys()
        total_loss = 0
        
        if 'IE' in keys and true_potentials['IE'] is not None:
            ie_potential = true_potentials['IE']
            subgraph1_result, subgraph1_pooled = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr) 
            real_num_peaks = torch.tensor(len(ie_potential), device=device)
            for i, true_potential in enumerate(ie_potential):
                # 確保 true_potential 的維度正確
                true_potential = true_potential.view(-1, 1) if true_potential.dim() == 0 else true_potential.view(-1, 1)

                if x.shape[0] == 1:
                    potential = self.reg_IP(subgraph1_pooled)
                    loss_reg = nn.MSELoss()(potential, true_potential)
                    loss = loss_reg
                    total_loss += loss

                    x_ = x.clone()
                    x_[:,124:137]  = torch.roll(x_[:,124:137], shifts=1, dims=1)
                    update_nodes = x_

                else:
                    potential = self.reg_IP(subgraph1_pooled)
                    num_peaks = self.num_peaks_IP(subgraph1_pooled)
                    loss_reg = nn.MSELoss()(potential, true_potential)
                    loss_cla = nn.CrossEntropyLoss()(num_peaks, real_num_peaks.unsqueeze(0))
                    loss = loss_reg + loss_cla
                    total_loss += loss

                # update ligand node features (after redox)
                    x_ = x.clone()
                    subgraph1_result_ = subgraph1_result.clone()
                    update_nodes      = subgraph1_result_ * self.gate_GCN1(subgraph1_result_) + x_
                    # update_nodes      = x_ * self.gate_GCN1(x_) + x_

                subgraph1_result, subgraph1_pooled = self.forward_subgraph1(update_nodes, subgraph1_edge_index, batch1, edge_attr)

                real_num_peaks = real_num_peaks - 1

        if 'EA' in keys and true_potentials['EA'] is not None:
            ea_potential = true_potentials['EA']
            subgraph1_result, subgraph1_pooled = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr) 
            real_num_peaks = torch.tensor(len(ea_potential), device=device)
            for i, true_potential in enumerate(ea_potential):
                # 確保 true_potential 的維度正確
                true_potential = true_potential.view(-1, 1) if true_potential.dim() == 0 else true_potential.view(-1, 1)

                if x.shape[0] == 1:
                    potential = self.reg_EA(subgraph1_pooled)
                    loss_reg = nn.MSELoss()(potential, true_potential)
                    loss = loss_reg
                    total_loss += loss

                    x_ = x.clone()
                    x_[:,124:137]  = torch.roll(x_[:,124:137], shifts=-1, dims=1)
                    update_nodes = x_
                else:
                    potential = self.reg_EA(subgraph1_pooled)
                    num_peaks = self.num_peaks_EA(subgraph1_pooled)
                    loss_reg = nn.MSELoss()(potential, true_potential)
                    loss_cla = nn.CrossEntropyLoss()(num_peaks, real_num_peaks.unsqueeze(0))
                    loss = loss_reg + loss_cla
                    total_loss += loss

                # update ligand node features (after redox)
                    x_ = x.clone()
                    subgraph1_result_ = subgraph1_result.clone()
                    update_nodes      = subgraph1_result_ * self.gate_GCN1(subgraph1_result_) + x_
                    # update_nodes      = x_ * self.gate_GCN1(x_) + x_

                subgraph1_result, subgraph1_pooled = self.forward_subgraph1(update_nodes, subgraph1_edge_index, batch1, edge_attr)

                real_num_peaks = real_num_peaks - 1

        if 'E12' in keys and true_potentials['E12'] is not None:
            e12_potential = true_potentials['E12']
            subgraph1_result, subgraph1_pooled = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr) 
            reaction = reaction['E12']
            real_num_peaks = torch.tensor(len(e12_potential), device=device)
            for i, true_potential in enumerate(e12_potential):
                # 確保 true_potential 的維度正確
                true_potential = true_potential.view(-1, 1) if true_potential.dim() == 0 else true_potential.view(-1, 1)

                if solvent != 'None':
                    solvent_features  = solvent_dict[solvent]
                    solvent_features  = torch.Tensor(solvent_features).cuda().unsqueeze(0)
                    subgraph1_pooled_ = torch.cat([subgraph1_pooled, solvent_features], dim=1)
                else:
                    solvent_features  = solvent_dict['ACN']
                    solvent_features  = torch.Tensor(solvent_features).cuda().unsqueeze(0)
                    subgraph1_pooled_ = torch.cat([subgraph1_pooled, solvent_features], dim=1)
                if reaction  == 'reduction':
                    if x.shape[0] == 1:
                        potential = self.reg_red(subgraph1_pooled_)
                        loss_reg = nn.MSELoss()(potential, true_potential)
                        loss = loss_reg 
                        total_loss += loss
                        # update ligand node features (after redox)
                        x_ = x.clone()
                        x_[:,124:137]  = torch.roll(x_[:,124:137], shifts=1, dims=1)
                        update_nodes = x_
                    else:
                        potential = self.reg_red(subgraph1_pooled_)
                        num_peaks = self.num_peaks_red(subgraph1_pooled_)
                        loss_reg = nn.MSELoss()(potential, true_potential)
                        loss_cla = nn.CrossEntropyLoss()(num_peaks, real_num_peaks.unsqueeze(0))
                        loss = (loss_reg + loss_cla ) * (i+0.5)
                        total_loss += loss

                        # update ligand node features (after redox)
                        x_ = x.clone()
                        subgraph1_result_ = subgraph1_result.clone()
                        update_nodes      = subgraph1_result_ * self.gate_GCN1(subgraph1_result_) + x_
                        # update_nodes      = x_ * self.gate_GCN1(x_) + x_
                elif reaction == 'oxidation':
                    if x.shape[0] == 1:
                        potential = self.reg_ox(subgraph1_pooled_)
                        loss_reg = nn.MSELoss()(potential, true_potential)
                        loss = loss_reg 
                        total_loss += loss

                        # update ligand node features (after redox)
                        x_ = x.clone()
                        x_[:,124:137]  = torch.roll(x_[:,124:137], shifts=-1, dims=1)
                        update_nodes = x_                        
                    else:
                        potential = self.reg_ox(subgraph1_pooled_)
                        num_peaks = self.num_peaks_ox(subgraph1_pooled_)
                        loss_reg = nn.MSELoss()(potential, true_potential)
                        loss_cla = nn.CrossEntropyLoss()(num_peaks, real_num_peaks.unsqueeze(0))
                        loss = (loss_reg  + loss_cla ) * (i+0.5)
                        total_loss += loss

                        # update ligand node features (after redox)
                        x_ = x.clone()
                        subgraph1_result_ = subgraph1_result.clone()
                        update_nodes      = subgraph1_result_ * self.gate_GCN1(subgraph1_result_) + x_
                        # update_nodes      = x_ * self.gate_GCN1(x_) + x_

                subgraph1_result, subgraph1_pooled = self.forward_subgraph1(update_nodes, subgraph1_edge_index, batch1, edge_attr)

                real_num_peaks = real_num_peaks - 1
        
        return total_loss
    
    def sample(self, batch, device):
        potentials = {'IE': [], 'EA': [], 'E12': []}
        clas       = {'IE': [], 'EA': [], 'E12': []}
        
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, true_potentials, reaction= graph.x, graph.edge_index, graph.edge_attr, graph.ys, graph.reaction
            if graph.solvent is not None:
                solvent = graph.solvent
            else:
                solvent = 'None'

            subgraph1_edge_index, batch1 = edge_index

            subgraph1_result, subgraph1_pooled = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr)
            x_orig = x.clone()
            subgraph1_result_orig = subgraph1_result.clone()
            subgraph1_pooled_orig = subgraph1_pooled.clone()

            if reaction.get('IE') is not None:
                peaks_ie = self.num_peaks_IP(subgraph1_pooled_orig)
                num_peaks_ie = int(torch.argmax(peaks_ie, dim=1).item())
                clas['IE'].append(peaks_ie)
                
                subgraph1_result_ie = subgraph1_result_orig.clone()
                subgraph1_pooled_ie = subgraph1_pooled_orig.clone()
                for i in range(num_peaks_ie):
                    potential = self.reg_IP(subgraph1_pooled_ie)
                    potentials['IE'].append(potential)
                    
                    x_update = x_orig.clone()
                    sub_result_update = subgraph1_result_ie.clone()
                    update_nodes = sub_result_update * self.gate_GCN1(sub_result_update) + x_update
                    subgraph1_result_ie, subgraph1_pooled_ie = self.forward_subgraph1(update_nodes, subgraph1_edge_index, batch1, edge_attr)
            
            if reaction.get('EA') is not None:
                peaks_ea = self.num_peaks_EA(subgraph1_pooled_orig)
                clas['EA'].append(peaks_ea)
                num_peaks_ea = int(torch.argmax(peaks_ea, dim=1).item())
                
                subgraph1_result_ea = subgraph1_result_orig.clone()
                subgraph1_pooled_ea = subgraph1_pooled_orig.clone()
                for i in range(num_peaks_ea):
                    potential = self.reg_EA(subgraph1_pooled_ea)
                    potentials['EA'].append(potential)
                    
                    x_update = x_orig.clone()
                    sub_result_update = subgraph1_result_ea.clone()
                    update_nodes = sub_result_update * self.gate_GCN1(sub_result_update) + x_update
                    subgraph1_result_ea, subgraph1_pooled_ea = self.forward_subgraph1(update_nodes, subgraph1_edge_index, batch1, edge_attr)
                    
            if reaction.get('E12') is not None:
                if solvent != 'None':
                    solvent_features  = solvent_dict[solvent]
                    solvent_features  = torch.Tensor(solvent_features).cuda().unsqueeze(0)
                else:
                    solvent_features  = solvent_dict['ACN']
                    solvent_features  = torch.Tensor(solvent_features).cuda().unsqueeze(0)
                subgraph1_result_e12  = subgraph1_result_orig.clone()
                subgraph1_pooled_e12  = torch.cat([subgraph1_pooled_orig.clone(), solvent_features], dim=1)
                
                if pd.notnull(reaction['E12']):
                    if reaction['E12'] == 'reduction':
                        peaks_e12 = self.num_peaks_red(subgraph1_pooled_e12)
                    elif reaction['E12'] == 'oxidation':
                        peaks_e12 = self.num_peaks_ox(subgraph1_pooled_e12)
                else:
                    peaks_e12 = self.num_peaks_red(subgraph1_pooled_e12)
                clas['E12'].append(peaks_e12)
                num_peaks_e12 = int(torch.argmax(peaks_e12, dim=1).item())
                
                subgraph1_result_e12 = subgraph1_result_orig.clone()
                subgraph1_pooled     = subgraph1_pooled_orig.clone()
                for i in range(num_peaks_e12):
                    subgraph1_pooled_e12 = torch.cat([subgraph1_pooled.clone(), solvent_features], dim=1)
                    if reaction['E12'] == 'reduction':
                        potential = self.reg_red(subgraph1_pooled_e12)
                    else:
                        potential = self.reg_ox(subgraph1_pooled_e12)
                    potentials['E12'].append(potential) 
                    
                    x_update = x_orig.clone()
                    sub_result_update = subgraph1_result_e12.clone()
                    update_nodes      = sub_result_update * self.gate_GCN1(sub_result_update) + x_update
                    subgraph1_result_e12, subgraph1_pooled_e12 = self.forward_subgraph1(update_nodes, subgraph1_edge_index, batch1, edge_attr)
        
        return clas, potentials
    
class Organic_GRN(nn.Module):
    def __init__(self, node_dim, bond_dim, hidden_dim, cla_dim, depth=3, dropout=0.3):
        super(Organic_GRN, self).__init__()
        self.GCN1 = DMPNN(node_dim, bond_dim, hidden_dim, depth=depth, dropout=0.3)
        self.pool = global_mean_pool
        self.gate_GCN1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh())
        
        self.predictor = nn.Sequential(
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
            nn.Linear(128, cla_dim))
        
    @staticmethod
    def _rev_edge_index(edge_index):
        edge_to_index = {(edge_index[0, i].item(), edge_index[1, i].item()): i for i in range(edge_index.shape[1])}
        rev_edge_index = torch.full((edge_index.shape[1],), -1, dtype=torch.long)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            if (v, u) in edge_to_index:
                rev_edge_index[i] = edge_to_index[(v, u)]
        return rev_edge_index
    def forward_subgraph1(self, x, subgraph1_edge_index, batch1, edge_attr):
        subgraph1_rev_edge_index = self._rev_edge_index(subgraph1_edge_index)
        subgraph1_batch   = Data(x=x, edge_index=subgraph1_edge_index, rev_edge_index=subgraph1_rev_edge_index, edge_attr=edge_attr)
        subgraph1_result  = self.GCN1(subgraph1_batch)
        subgraph1_result_ = self.pool(subgraph1_result, batch1)
        return subgraph1_result, subgraph1_result_
    def forward(self, batch, device):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, true_vals = graph.x, graph.edge_index, graph.edge_attr, graph.ys 
        subgraph1_edge_index, batch1 = edge_index
   
        subgraph1_result, subgraph1_pooled = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr)
        numbers = len(true_vals)
          
        total_loss = 0
        for i, true in enumerate(true_vals):
            pred_reg = self.predictor(subgraph1_pooled) 
            pred_cla = self.classifier(subgraph1_pooled)
            
            # 確保張量尺寸匹配
            true_tensor = true.view(-1, 1) if true.dim() == 0 else true.view(-1, 1)
            loss_reg = nn.MSELoss()(pred_reg, true_tensor)
            
            # 轉換 numbers 為張量
            numbers_tensor = torch.tensor([numbers], dtype=torch.long, device=pred_cla.device)
            loss_cla = nn.CrossEntropyLoss()(pred_cla, numbers_tensor)
            
            loss = loss_reg + loss_cla
            total_loss += loss
            # update ligand node features (after redox)
            x_ = x.clone()
            subgraph1_result_ = subgraph1_result.clone()
            update_nodes      = subgraph1_result_ * self.gate_GCN1(subgraph1_result_) + x_
            subgraph1_result, subgraph1_pooled = self.forward_subgraph1(update_nodes, subgraph1_edge_index, batch1, edge_attr)
            numbers -= 1
        return total_loss
    def sample(self,batch, device):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr = graph.x, graph.edge_index, graph.edge_attr
        subgraph1_edge_index, batch1 = edge_index

        subgraph1_result, subgraph1_pooled = self.forward_subgraph1(x, subgraph1_edge_index, batch1, edge_attr)
        pred_cla = self.classifier(subgraph1_pooled)
        times = torch.argmax(pred_cla, dim=1).item()
        preds = []
        for i in range(times):
            pred_reg = self.predictor(subgraph1_pooled)
            preds.append(pred_reg)
            # update ligand node features (after redox)
            x_ = x.clone()
            subgraph1_result_ = subgraph1_result.clone()
            update_nodes      = subgraph1_result_ * self.gate_GCN1(subgraph1_result_) + x_
            subgraph1_result, subgraph1_pooled = self.forward_subgraph1(update_nodes, subgraph1_edge_index, batch1, edge_attr)
        return preds, pred_cla