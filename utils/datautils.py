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


def process_e12_value(x):
    if isinstance(x, str):
        if x.startswith('[') and x.endswith(']'):
            content = x[1:-1] 
            if ',' in content:
                return [float(val.strip()) for val in content.split(',')]
            else:
                return [float(content.strip())]
        elif ',' in x:
            return [float(val.strip()) for val in x.split(',')]
        else:
            return [float(x.strip())]
    elif isinstance(x, list):
        return [float(val) for val in x]
    else:
        return [float(x)]

class dataloader_v2:
    @staticmethod
    def process_numeric_lists(x):
        """處理數值列表的通用方法"""
        if isinstance(x, (list, float, int)):
            return x if isinstance(x, list) else [x]
        if isinstance(x, str):
            parsed = literal_eval(x)
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            else:
                return [parsed]  # Convert single value to list
        return x
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
                
                # 動態獲取標籤值（使用 label_columns 中的第一個標籤）
                label_col = label_columns[0] if label_columns else "E12"
                label_value = row.get(label_col)
                
                # 檢查標籤值是否有效
                if label_value is None or label_value == "":
                    return None
                
                # 處理數組的 NaN 檢查
                if isinstance(label_value, (list, np.ndarray)):
                    if len(label_value) == 0 or all(pd.isna(val) for val in label_value):
                        return None
                elif pd.isna(label_value):
                    return None
                
                # 處理標籤值
                if isinstance(label_value, str):
                    if label_value.strip() == "":
                        return None
                    try:
                        label_parsed = literal_eval(label_value)
                    except (ValueError, SyntaxError):
                        return None
                else:
                    label_parsed = label_value
                
                # 確保 label_parsed 是列表
                if not isinstance(label_parsed, list):
                    label_parsed = [label_parsed]
                
                # 轉換為張量
                label_tensor = torch.tensor(label_parsed, dtype=torch.float32)
                
                [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([smiles], 'None', features)

                data_item = Data(
                    x=fatoms if is_metal else fatoms[0],
                    edge_index=graphs,
                    edge_attr=edge_features,
                    ys=label_tensor
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
    
    @classmethod
    def load_TMCs_data(cls, file_path, test_size=0.2, is_metal=False, features=153, k_fold=False, unlabeled=False, label_columns=None, default_reactions=None):
        df = pd.read_csv(file_path)
        
        # 處理數值列表
        for col in label_columns:
            if col in df.columns:
                df[col] = df[col].apply(cls.process_numeric_lists)

        def create_data_object(row, is_metal=False):
            try:
                smiles = row.get("smiles")
                metal = row.get("Metal")
                if pd.isna(smiles):
                    return None
                
                # 動態獲取標籤值（使用 label_columns 中的第一個標籤）
                label_col = label_columns[0] if label_columns else "E12"
                label_value = row.get(label_col)
                
                # 檢查標籤值是否有效
                if label_value is None or label_value == "":
                    return None
                
                # 處理數組的 NaN 檢查
                if isinstance(label_value, (list, np.ndarray)):
                    if len(label_value) == 0 or all(pd.isna(val) for val in label_value):
                        return None
                elif pd.isna(label_value):
                    return None
                
                # 處理標籤值
                if isinstance(label_value, str):
                    if label_value.strip() == "":
                        return None
                    try:
                        label_parsed = literal_eval(label_value)
                    except (ValueError, SyntaxError):
                        return None
                else:
                    label_parsed = label_value
                
                # 確保 label_parsed 是列表
                if not isinstance(label_parsed, list):
                    label_parsed = [label_parsed]
                
                # 轉換為張量
                label_tensor = torch.tensor(label_parsed, dtype=torch.float32)
                
                [fatoms, graphs, edge_features, midx, ninds_to_rmove] = tensorize_with_subgraphs([smiles], metal, features)

                data_item = Data(
                    x=fatoms if is_metal else fatoms[0],
                    edge_index=graphs,
                    edge_attr=edge_features,
                    midx=midx,
                    ys=label_tensor
                )
                
                # 新增 redox_idxs
                if 'site' in row:
                    redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["site"])
                    data_item.redox = redox_idxs

                return data_item
            except Exception as e:
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


def load_traintest_dataset(file_path, tensorize_fn, batch_size=1):
    df = pd.read_csv(file_path)
    
    print("使用原始 split 列進行分割...")
    train_data = df[df['split'] == 'train']
    test_data = df[df['split'] == 'test']
    print(f"原始分割結果: 訓練 {len(train_data)}, 測試 {len(test_data)}")

    def tensorize_dataset(data):
        dataset = []
        for _, row in data.iterrows():
            try:
                [fatoms, graphs, edge_features, midx, binding_atoms] = tensorize_fn([row["smiles"]], row["Metal"])
                # 確保所有 E12 值都被轉換為張量
                e12_values = row['E12']
                if not isinstance(e12_values, list):
                    e12_values = [e12_values]
                # 強制轉成 float
                e12_values = process_e12_value(row['E12'])
                label = torch.tensor(e12_values, dtype=torch.float32)
                
                name = fatoms[1]
                redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                
                data_item = Data(
                    x=fatoms[0],
                    edge_index=graphs,
                    edge_attr=edge_features,
                    redox=redox_idxs,
                    ys=label,  
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
    
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataset, test_loader
