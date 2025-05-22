

import pandas as pd
import numpy as np
import torch
from torch_geometric.data     import Data, DataLoader
from torch_geometric.loader   import DataLoader
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from utils.chemutils import metal_features, tensorize_with_subgraphs, redox_each_num
from utils.datautils import dataloader
label_columns=None
default_reactions=None
test_size=0.2
is_metal=False
features=153
k_fold=False
unlabeled=False

label_columns=['IE', 'EA', 'E12']
label_keys = label_columns
file_path = "data/organic_ip_ea_rp.csv"
# df = pd.read_csv(file_path)
# row = df[df['smiles'] == 'CC(=O)C(C)(C)C']



def load_data(cls, 
              file_path, 
              test_size=0.2, 
              is_metal=False, 
              features=153, 
              k_fold=False, 
              unlabeled=False, 
              label_columns=None, 
              default_reactions=None
    ):
    """統一的數據加載方法
    
    Args:
        file_path: 數據文件路徑
        test_size: 測試集比例
        is_metal: 是否為金屬數據
        features: 特徵數量
        k_fold: 是否使用交叉驗證
        unlabeled: 是否為無標籤數據
        label_columns: 要處理的標籤列表，例如 ['E12']。如果為 None，則使用默認值。
        default_reactions: 字典，定義每個標籤的默認反應類型
    """
    df = pd.read_csv(file_path)
    if label_columns is None:
        label_columns = ['E12']
    for col in label_columns:
        if col in df.columns:
            df[col] = df[col].apply(dataloader.process_numeric_lists)
    row = df[df['Name'] == '2-Butanone, 3,3-dimethyl-']
    def create_data_object(row, is_metal=False):
        # smiles = row.get("smiles") or row.get("pubchem_smiles")
        try:
            if is_metal:
                fatoms, graphs, edge_features = metal_features(row["Metal"], features)
                fatoms = torch.unsqueeze(fatoms, dim=0)
                name = [row["Metal"]]
            else:
                smiles = row.get("smiles").item() or row.get("pubchem_smiles").item()
                [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([smiles], 'None', features)
                name = fatoms[1]
            labels, reaction_info = dataloader.process_labels(row, label_columns, default_reactions)
            # labels, reaction_info = process_labels(row, label_columns, default_reactions)
            solvent = row.get('Solvent', "None")
            
            data_item = Data(
                x=fatoms if is_metal else fatoms[0],
                edge_index=graphs,
                edge_attr=edge_features,
                ys=labels,
                solvent=solvent,
                name=name,
                reaction=reaction_info
            )
            
            if not is_metal and 'redox_site_smiles' in row:
                redox_idxs = redox_each_num([smiles], row["Metal"], row["redox_site_smiles"])
                data_item.redox = redox_idxs
                data_item.oreder_site = row["redox_site_smiles"]
                data_item.midx = midx

            return data_item
        except Exception as e:
            print(f"Error processing row: {e} for {smiles}")
            return None

    def create_dataset(data):
        return [item for item in (create_data_object(row, is_metal) for _, row in data.iterrows()) if item is not None]

    if k_fold:
        kf = KFold(n_splits=5, shuffle=True, random_state=12)
        return [(DataLoader(create_dataset(df.iloc[train_idx]), batch_size=1, shuffle=True),
                DataLoader(create_dataset(df.iloc[test_idx]), batch_size=1, shuffle=False))
                for train_idx, test_idx in kf.split(df)]
    
    if is_metal:
        dataset = create_dataset(df)
        return DataLoader(dataset, batch_size=1, shuffle=True), None
    
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    train_dataset = create_dataset(train_data)
    test_dataset = create_dataset(test_data)
    
    return (DataLoader(train_dataset, batch_size=1, shuffle=True),
            DataLoader(test_dataset, batch_size=1, shuffle=False))
    
train_dataset_organic, test_dataset_organic = dataloader.load_data(
    file_path, 
    test_size, 
    is_metal=False, 
    features=153,
    label_columns=label_columns
)

smiles = 'C(C)(C)C(=O)O'
ret = tensorize_with_subgraphs([smiles], 'None', features)
print(len(ret))          # organic → 5，organometallic → 6
for i, v in enumerate(ret):
    print(i, type(v))