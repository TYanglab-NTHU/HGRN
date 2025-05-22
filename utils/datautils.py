import os, ast
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch_geometric.data     import Data, DataLoader

from sklearn.metrics         import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from utils.chemutils import *
from optparse  import OptionParser

import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
from matplotlib           import ticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec  import GridSpec, GridSpecFromSubplotSpec


class dataloader:
    @staticmethod
    def process_numeric_lists(x):
        """處理數值列表的通用方法"""
        if isinstance(x, (list, float, int)):
            return x if isinstance(x, list) else [x]
        if isinstance(x, str) and ',' in x:
            return list(map(float, x.split(',')))
        return [float(x)] if isinstance(x, str) else None

    @staticmethod
    def process_labels(row, label_keys, default_reactions=None):
        """處理標籤數據的通用方法
        
        Args:
            row: 數據行
            label_keys: 標籤鍵值列表，可以包含任意標籤名稱
            default_reactions: 字典，定義每個標籤的默認反應類型，例如 {'E12': 'reduction', 'custom_label': 'custom_reaction'}
            
        Returns:
            labels: 包含所有標籤值的字典
            reaction_info: 包含所有反應信息的字典
        """
        labels = {}
        reaction_info = {}
        
        # 設置默認反應類型
        if default_reactions is None:
            default_reactions = {
                'IE': 'oxidation',
                'EA': 'reduction',
                'E12': 'redox'  # 設置 E12 的默認反應類型
            }
        
        for key in label_keys:
            # 檢查是否有特定的單位後綴
            unit_suffix = ''
            if key in ['IE', 'EA']:
                unit_suffix = ' / eV'
            
            col = f"{key}{unit_suffix}"
            
            # 獲取標籤值
            if col in row and pd.notna(row[col][0] if isinstance(row[col], list) else row[col]):
                value = row[col]
                # 轉換值為浮點數列表
                if isinstance(value, list):
                    try:
                        value = [float(v) for v in value]
                    except (ValueError, TypeError):
                        print(f"Warning: Could not convert value {value} to float for {key}")
                        labels[key] = None
                        reaction_info[key] = None
                        continue
                else:
                    try:
                        value = [float(value)]
                    except (ValueError, TypeError):
                        labels[key] = None
                        reaction_info[key] = None
                        continue
                labels[key] = torch.tensor(value, dtype=torch.float32)
                
                # 獲取反應信息
                reaction_type = None
                if 'Reaction_' + key in row:
                    reaction_type = row['Reaction_' + key]
                elif key == 'IE':
                    reaction_type = 'oxidation'
                elif key == 'EA':
                    reaction_type = 'reduction'
                elif key == 'E12':
                    # 如果數據中有 Reaction，使用它；否則使用默認值
                    reaction_type = row.get('Reaction') or default_reactions.get('E12')
                else:
                    # 對於其他標籤，使用默認反應類型（如果有的話）
                    reaction_type = default_reactions.get(key)
                
                reaction_info[key] = reaction_type
            else:
                labels[key] = None
                reaction_info[key] = None
        
        return labels, reaction_info

    @classmethod
    def load_data(cls, file_path, test_size=0.2, is_metal=False, features=153, k_fold=False, unlabeled=False, label_columns=None, default_reactions=None):
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
        
        # 如果沒有指定標籤列，使用默認值
        if label_columns is None:
            label_columns = ['E12']
        
        # 處理數值列表
        for col in label_columns:
            if col in df.columns:
                df[col] = df[col].apply(cls.process_numeric_lists)

        def create_data_object(row, is_metal=False):
            try:
                if is_metal:
                    metal = row.get("Metal")
                    fatoms, graphs, edge_features = metal_features(row["Metal"], features)
                    fatoms = torch.unsqueeze(fatoms, dim=0)
                    name = metal
                else:
                    smiles = row.get("smiles") or row.get("pubchem_smiles")
                    if pd.isna(smiles):
                        return None
                    [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([smiles], 'None', features)
                    name = fatoms[1]

                # 處理所有可能的標籤，並傳入默認反應類型
                labels, reaction_info = cls.process_labels(row, label_columns, default_reactions)
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
                print(f"Error processing row: {e}")
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


def data_loader(file_path, tensorize_fn, batch_size, test_size=0.2):
    df = pd.read_csv(file_path)

    df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))) if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=12)

    def tensorize_dataset(data):
        dataset = []
        for _, row in data.iterrows():
            try:
                [fatoms, graphs, edge_features, midx, binding_atoms] = tensorize_fn([row["smiles"]], row["Metal"])
                label = torch.Tensor(row['E12'])
                name = fatoms[1]
                redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                data_item  = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, redox=redox_idxs, ys=label, name=name, reaction=row["Reaction"], oreder_site=row["redox_site_smiles"], binding_atoms=binding_atoms)
                data_item.midx = midx
                dataset.append(data_item)
            except Exception as e:
                print(f"Error processing row: {e}")
                continue
        return dataset

    train_dataset = tensorize_dataset(train_data)
    test_dataset = tensorize_dataset(test_data)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_loader
