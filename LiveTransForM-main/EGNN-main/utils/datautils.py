import os, ast
import numpy as np
import pandas as pd

# 添加 RDKit 警告抑制
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import torch
import torch.optim as optim
from torch_geometric.data     import Data
from torch_geometric.loader   import DataLoader

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

from ast import literal_eval
class dataloader_v2:
    @staticmethod
    def process_numeric_lists(x):
        """處理數值列表的通用方法"""
        if isinstance(x, (list, float, int)):
            return x if isinstance(x, list) else [x]
        if isinstance(x, str):
            # 移除方括號和引號
            x = x.strip().strip('"\'')
            if x.startswith('[') and x.endswith(']'):
                x = x[1:-1]  # 移除方括號
            
            # 處理逗號分隔的值
            if ',' in x:
                return [float(val.strip()) for val in x.split(',')]
            else:
                try:
                    return [float(x)]
                except ValueError:
                    return None
        return None
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
        
        # 處理數值列表
        for col in label_columns:
            if col in df.columns:
                df[col] = df[col].apply(cls.process_numeric_lists)

        def create_data_object(row, is_metal=False):
            try:
                smiles = row.get("smiles")
                if pd.isna(smiles):
                    return None
                
                # 檢查 E12 值是否有效
                e12_value = row.get("E12")
                if pd.isna(e12_value) or e12_value == "" or e12_value is None:
                    return None
                
                # 處理 E12 值
                if isinstance(e12_value, str):
                    if e12_value.strip() == "":
                        return None
                    try:
                        e12_parsed = literal_eval(e12_value)
                    except (ValueError, SyntaxError):
                        return None
                else:
                    e12_parsed = e12_value
                
                # 確保 e12_parsed 是列表
                if not isinstance(e12_parsed, list):
                    e12_parsed = [e12_parsed]
                
                # 轉換為張量
                e12_tensor = torch.tensor(e12_parsed, dtype=torch.float32)
                
                [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([smiles], 'None', features)

                data_item = Data(
                    x=fatoms if is_metal else fatoms[0],
                    edge_index=graphs,
                    edge_attr=edge_features,
                    ys=e12_tensor
                )
                return data_item
            except Exception as e:
                print(f"Error processing row: {e}")
                return None

        def create_dataset(data):
            return [item for item in (create_data_object(row, is_metal) for _, row in data.iterrows()) if item is not None]

        train_data, test_data = train_test_split(df, test_size=test_size, random_state=8)
        train_dataset = create_dataset(train_data)
        test_dataset = create_dataset(test_data)
        
        return (DataLoader(train_dataset, batch_size=1, shuffle=True),
                DataLoader(test_dataset, batch_size=1, shuffle=False))
    
    
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
                # print(f"Error processing row: {e}")
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
        
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=8)
        train_dataset = create_dataset(train_data)
        test_dataset = create_dataset(test_data)
        
        return (DataLoader(train_dataset, batch_size=1, shuffle=True),
                DataLoader(test_dataset, batch_size=1, shuffle=False))

    @classmethod
    def all_load_data(cls, file_path, is_metal=False, features=153, label_columns=None, default_reactions=None):
        """統一的數據加載方法，不分割訓練和測試集
        
        Args:
            file_path: 數據文件路徑
            is_metal: 是否為金屬數據
            features: 特徵數量
            label_columns: 要處理的標籤列表，例如 ['E12']。如果為 None，則使用默認值。
            default_reactions: 字典，定義每個標籤的默認反應類型
            
        Returns:
            DataLoader: 包含所有數據的 DataLoader
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
                return None

        def create_dataset(data):
            return [item for item in (create_data_object(row, is_metal) for _, row in data.iterrows()) if item is not None]
        
        dataset = create_dataset(df)
        return DataLoader(dataset, batch_size=1, shuffle=False)


def data_loader_v2(file_path, tensorize_fn, batch_size, test_size=0.2, random_state=12):
    df = pd.read_csv(file_path)

    # 確保 E12 列表的正確處理
    df['E12'] = df['E12'].apply(lambda x: literal_eval(x))
    
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

    def tensorize_dataset(data):
        dataset = []
        for _, row in data.iterrows():
            try:
                [fatoms, graphs, edge_features, midx, _] = tensorize_fn([row["smiles"]], row["Metal"])
                # 確保所有 E12 值都被轉換為張量
                e12_values = row['E12']
                if not isinstance(e12_values, list):
                    e12_values = [e12_values]
                label = torch.tensor(e12_values, dtype=torch.float32)
                
                name = fatoms[1]
                redox_idxs = redox_each_num_v2([row["smiles"]], _, row["redox_site_smiles"])
                
                # 創建包含所有 E12 值的數據項
                data_item = Data(
                    x=fatoms[0],
                    edge_index=graphs,
                    edge_attr=edge_features,
                    redox=redox_idxs,
                    ys=label,  # 現在包含所有 E12 值
                    name=name,
                )
                data_item.midx = midx
                dataset.append(data_item)
            except Exception as e:
                continue
        return dataset

    train_dataset = tensorize_dataset(train_data)
    test_dataset = tensorize_dataset(test_data)
    
    # if train_dataset is None or test_dataset is None:
    #     raise ValueError("無法處理數據集")
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_loader

def data_loader(file_path, tensorize_fn, batch_size, test_size=0.2, random_state=12):
    df = pd.read_csv(file_path)

    # 確保 E12 列表的正確處理
    df['E12'] = df['E12'].apply(lambda x: 
        list(map(float, x.split(','))) if isinstance(x, str) and ',' in x 
        else ([float(x)] if isinstance(x, str) 
        else ([x] if not isinstance(x, list) else x)))
    
    train_data, test_data = train_test_split(df, test_size=test_size, random_state=random_state)

    def tensorize_dataset(data):
        dataset = []
        for _, row in data.iterrows():
            try:
                [fatoms, graphs, edge_features, midx, binding_atoms] = tensorize_fn([row["smiles"]], row["Metal"])
                # 確保所有 E12 值都被轉換為張量
                e12_values = row['E12']
                if not isinstance(e12_values, list):
                    e12_values = [e12_values]
                label = torch.tensor(e12_values, dtype=torch.float32)
                
                name = fatoms[1]
                redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                
                # 創建包含所有 E12 值的數據項
                data_item = Data(
                    x=fatoms[0],
                    edge_index=graphs,
                    edge_attr=edge_features,
                    redox=redox_idxs,
                    ys=label,  # 現在包含所有 E12 值
                    name=name,
                    reaction=row["Reaction"],
                    oreder_site=row["redox_site_smiles"],
                    binding_atoms=binding_atoms
                )
                data_item.midx = midx
                dataset.append(data_item)
            except Exception as e:
                # print(f"Error processing row: {e}")
                continue
        return dataset

    train_dataset = tensorize_dataset(train_data)
    test_dataset = tensorize_dataset(test_data)
    
    # if train_dataset is None or test_dataset is None:
    #     raise ValueError("無法處理數據集")
        
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_loader


def alldata_loader(file_path, tensorize_fn):
    df = pd.read_csv(file_path)

    # 確保 E12 列表的正確處理
    df['E12'] = df['E12'].apply(lambda x: 
        list(map(float, x.split(','))) if isinstance(x, str) and ',' in x 
        else ([float(x)] if isinstance(x, str) 
        else ([x] if not isinstance(x, list) else x)))
    
    def tensorize_dataset(data):
        dataset = []
        for _, row in data.iterrows():
            try:
                [fatoms, graphs, edge_features, midx, binding_atoms] = tensorize_fn([row["smiles"]], row["Metal"])
                # 確保所有 E12 值都被轉換為張量
                e12_values = row['E12']
                if not isinstance(e12_values, list):
                    e12_values = [e12_values]
                label = torch.tensor(e12_values, dtype=torch.float32)
                
                name = fatoms[1]
                redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                
                # 創建包含所有 E12 值的數據項
                data_item = Data(
                    x=fatoms[0],
                    edge_index=graphs,
                    edge_attr=edge_features,
                    redox=redox_idxs,
                    ys=label,  # 現在包含所有 E12 值
                    name=name,
                    reaction=row["Reaction"],
                    oreder_site=row["redox_site_smiles"],
                    binding_atoms=binding_atoms
                )
                data_item.midx = midx
                dataset.append(data_item)
            except Exception as e:
                # print(f"Error processing row: {e}")
                continue
        return dataset

    all_data = tensorize_dataset(df)
    return all_data
