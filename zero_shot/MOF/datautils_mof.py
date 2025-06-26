import os, ast
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data     import Data, DataLoader

from sklearn.metrics         import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from periodic_chemutils import *

def data_loader(file_path, batch_size, label='E12'):
    df = pd.read_csv(file_path)

    dataset = []
    for index, row in df.iterrows():
        fatoms, graphs, edge_features, midx, binding_atoms = MOF_tensorize_with_subgraphs(row['cif_path'], row['metal'])
        data = Data(x=fatoms, edge_index=graphs, edge_attr=edge_features, y=row[label], binding_atoms=binding_atoms, midx=midx)
        dataset.append(data)
    return dataset
        
