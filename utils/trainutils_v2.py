import os, ast
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch    import nn
from torch.nn import functional as F
from torch.optim.lr_scheduler import ExponentialLR, StepLR
from torch_geometric.data     import Data, DataLoader
from torch_geometric.nn       import global_mean_pool

from sklearn.metrics         import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics         import roc_auc_score


from utils.chemutils import *
from optparse  import OptionParser

import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
from matplotlib           import ticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec  import GridSpec, GridSpecFromSubplotSpec


class OrganicMetal_potential():
    def __init__(self):
        pass

    # def data_loader(file_path, test_size=0.2, is_metal=False, features=153):
    #     df = pd.read_csv(file_path)

    #     df['IE / eV']  = df['IE / eV'].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))   
    #     df['EA / eV']  = df["EA / eV"].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))   
    #     df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))            

    #     if is_metal:
    #         train_data, test_data = df, None
    #     else:
    #         train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    #     def tensorize_dataset(data, is_metal=False):
    #         dataset = []
    #         for _, row in data.iterrows():
    #             try:
    #                 if is_metal:
    #                     fatoms, graphs, edge_features = metal_features(row["Metal"], features)
    #                     fatoms = torch.unsqueeze(fatoms, dim=0)
    #                     label_keys = ['IE', 'EA', 'E12']
    #                     columns = ['IE / eV', 'EA / eV', 'E12']
    #                     labels, reaction_info = {}, {}
    #                     for key, col in zip(label_keys, columns):
    #                         if pd.notna(row[col][0]):
    #                             labels[key] = torch.Tensor(row[col])
    #                             if key == 'IE':
    #                                 reaction_info[key] = 'oxidation'
    #                             elif key == 'EA':
    #                                 reaction_info[key] = 'reduction'
    #                             elif key == 'E12':
    #                                 reaction_info[key] = row['Reaction']
    #                         else:
    #                             labels[key] = None
    #                             reaction_info[key] = None
    #                     name   = [row["Metal"]]
    #                     solvent = row['Solvent'] if 'Solvent' in row else "None"
    #                     data_item = Data(x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=labels, solvent=solvent, name=name, reaction=reaction_info)
    #                 else:
    #                     if pd.notna(row["smiles"]):
    #                         [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([row["smiles"]], 'None', features)
    #                     else:
    #                         [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([row["pubchem_smiles"]], 'None', features)
    #                     label_keys = ['IE', 'EA', 'E12']
    #                     columns = ['IE / eV', 'EA / eV', 'E12']
    #                     labels, reaction_info = {},  {}
    #                     for key, col in zip(label_keys, columns):
    #                         if pd.notna(row[col][0]):
    #                             labels[key] = torch.Tensor(row[col])
    #                             if key == 'IE':
    #                                 reaction_info[key] = 'oxidation'
    #                             elif key == 'EA':
    #                                 reaction_info[key] = 'reduction'
    #                             elif key == 'E12':
    #                                 reaction_info[key] = row['Reaction']
    #                         else:
    #                             labels[key] = None
    #                             reaction_info[key] = None
    #                     name    = fatoms[1]
    #                     solvent = row['Solvent'] if 'Solvent' in row else "None"
    #                     data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=labels,  solvent=solvent, name=name, reaction=reaction_info)                    
    #                 dataset.append(data_item)
    #             except Exception as e:
    #                 print(f"Error processing row: {e}")
    #                 continue
    #         return dataset

    #     train_dataset = tensorize_dataset(train_data, is_metal=is_metal)
    #     if is_metal:
    #         test_dataset = None
    #     else:
    #         test_dataset = tensorize_dataset(test_data)

    #     return train_dataset, test_dataset
    
    # def sample_loader(file_path, is_metal=False, features=153, unlabeled=False):
    #     df = pd.read_csv(file_path)

    #     df['IE / eV']  = df['IE / eV'].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))   
    #     df['EA / eV']  = df["EA / eV"].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))   
    #     df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))            

    #     data = df

    #     def tensorize_dataset(data, is_metal=False, features=153):
    #         dataset = []
    #         for _, row in data.iterrows():
    #             try:
    #                 if is_metal:
    #                     fatoms, graphs, edge_features = metal_features(row["Metal"], features)
    #                     fatoms = torch.unsqueeze(fatoms, dim=0)
    #                     label_keys = ['IE', 'EA', 'E12']
    #                     columns = ['IE / eV', 'EA / eV', 'E12']
    #                     labels, reaction_info = {}, {}
    #                     for key, col in zip(label_keys, columns):
    #                         if pd.notna(row[col][0]):
    #                             labels[key] = torch.Tensor(row[col])
    #                             if key == 'IE':
    #                                 reaction_info[key] = 'oxidation'
    #                             elif key == 'EA':
    #                                 reaction_info[key] = 'reduction'
    #                             elif key == 'E12':
    #                                 reaction_info[key] = row['Reaction']
    #                     name   = [row["Metal"]]
    #                     solvent = row['Solvent']
    #                     data_item = Data(x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=labels, solvent=solvent, name=name, reaction=reaction_info)
    #                 else:
    #                     if pd.notna(row["smiles"]):
    #                         [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([row["smiles"]], 'None', features)
    #                     else:
    #                         [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([row["pubchem_smiles"]], 'None', features)
    #                     if unlabeled:
    #                         name    = fatoms[1]
    #                         solvent = row['Solvent']
    #                         data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=labels,  solvent=solvent, name=name, reaction=reaction_info)                    
    #                     else:
    #                         label_keys = ['IE', 'EA', 'E12']
    #                         columns = ['IE / eV', 'EA / eV', 'E12']
    #                         labels, reaction_info = {},  {}
    #                         for key, col in zip(label_keys, columns):
    #                             if pd.notna(row[col][0]):
    #                                 labels[key] = torch.Tensor(row[col])
    #                                 if key == 'IE':
    #                                     reaction_info[key] = 'oxidation'
    #                                 elif key == 'EA':
    #                                     reaction_info[key] = 'reduction'
    #                                 elif key == 'E12':
    #                                     reaction_info[key] = row['Reaction']
    #                             else:
    #                                 labels[key] = None
    #                                 reaction_info[key] = None
    #                         name    = fatoms[1]
    #                         solvent = row['Solvent']
    #                         data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=labels,  solvent=solvent, name=name, reaction=reaction_info)                    
    #                     dataset.append(data_item)
    #             except Exception as e:
    #                 print(f"Error processing row: {e}")
    #                 continue
    #         return dataset

    #     dataset = tensorize_dataset(data, is_metal=is_metal, features=features)
    #     loader = DataLoader(dataset, batch_size=1, shuffle=True)

    #     return dataset, loader

    def evaluate_model(model, loader, device, output_file=""):
        model.eval()

        reg_data_list = []
        cla_data_list = []

        count = 0
        total_reg_loss, total_cla_loss = 0.0, 0.0
        correct_batches,total_batches  = 0.0, 0.0

        with torch.no_grad():
            for data in loader:
                try:
                    data = data.to(device)  
                    potential_clas, potential_regs = model.sample(data, device)
                    true_labels = data.ys
                    sample_name = data.name
    
                    sample_reg_dict = {"SMILES": sample_name}
                    sample_cla_dict = {"SMILES": sample_name}
                    for key in ['IE', 'EA', 'E12', 'E12_inv']:
                        if isinstance(true_labels.get(key), torch.Tensor):
                            gt_tensor = true_labels[key]
                            sample_reg_dict[f"{key}_actual"] = gt_tensor.squeeze().cpu().numpy()
                        else:
                            sample_reg_dict[f"{key}_actual"] = np.nan

                        if potential_regs.get(key) and len(potential_regs[key]) > 0:
                            pred_tensor = potential_regs[key][0]
                            sample_reg_dict[f"{key}_pred"] = pred_tensor.squeeze().cpu().numpy()
                            if true_labels.get(key) is not None:
                                loss_reg = F.mse_loss(pred_tensor.squeeze(), true_labels[key].squeeze())
                                total_reg_loss += loss_reg.item()
                        else:
                            sample_reg_dict[f"{key}_pred"] = np.nan
                            
                        if isinstance(true_labels.get(key), torch.Tensor):
                            target_class = true_labels[key].numel()  # integer target
                            sample_cla_dict[f"{key}_actual"] = target_class
                        else:
                            sample_cla_dict[f"{key}_actual"] = np.nan
                            
                        if potential_clas.get(key) and len(potential_clas[key]) > 0:
                            logits = potential_clas[key][0]
                            pred_class = int(torch.argmax(logits, dim=1).item())
                            sample_cla_dict[f"{key}_pred"] = pred_class
                            if true_labels.get(key) is not None:
                                loss_cla = F.cross_entropy(logits, torch.tensor([target_class]).to(device))
                                total_cla_loss += loss_cla.item()
                                total_batches += 1
                                # For accuracy, compare predicted class to target class.
                                if pred_class == target_class:
                                    correct_batches += 1           
                        else:
                            sample_cla_dict[f"{key}_pred"] = np.nan
                    reg_data_list.append(sample_reg_dict)
                    cla_data_list.append(sample_cla_dict)
                    count += 1
                    # total_batches += 1
                except Exception as e:
                    print(f"Error evaluating model: {e}")
                    continue
                    
        total_loss   = (total_reg_loss + total_cla_loss) / count if count > 0 else 0.0
        avg_reg_loss = total_reg_loss / count if count > 0 else 0.0
        avg_cla_loss = total_cla_loss / count if count > 0 else 0.0
        accuracy = correct_batches / total_batches if total_batches > 0 else 0.0
        
        # Create DataFrames for regression and classification.
        df_reg = pd.DataFrame(reg_data_list)
        df_cla = pd.DataFrame(cla_data_list)
        
        if output_file:
            reg_outfile = os.path.join(os.getcwd(), f"reg_{output_file}")
            cla_outfile = os.path.join(os.getcwd(), f"cla_{output_file}")
            df_reg.to_csv(reg_outfile, index=False)
            df_cla.to_csv(cla_outfile, index=False)

        return total_loss ,avg_reg_loss, avg_cla_loss, accuracy
    
    def parity_plot(train_file, valid_file):
        def convert_str_to_float(val):
            if isinstance(val, (int, float)):
                return float(val)
            
            if isinstance(val, str):
                val = val.strip()
                if val.startswith('[') and val.endswith(']'):
                    try:
                        parsed = ast.literal_eval(val)
                        if isinstance(parsed, (list, tuple, np.ndarray)):
                            arr = np.array(parsed, dtype=float)
                            return float(np.mean(arr))
                        else:
                            return float(parsed)
                    except Exception as e:
                        print(f"Error parsing value {val}: {e}")
                        return np.nan
                else:
                    try:
                        return float(val)
                    except Exception as e:
                        print(f"Error converting value {val}: {e}")
                        return np.nan
            return np.nan

        train_data = pd.read_csv(train_file)
        valid_data = pd.read_csv(valid_file)
        
        # We are plotting only IE and EA (E12 data is not included here).
        keys = ['IE', 'EA', 'E12']
        markers = {'IE': 'o', 'EA': 's', 'E12': '^'}
        
        fig, ax = plt.subplots(figsize=(6, 6))
        
        # Lists to collect all points for determining overall axis limits.
        train_true_all = []
        train_pred_all = []
        valid_true_all = []
        valid_pred_all = []
        
        # Process training data.
        for key in keys:
            act_col = f"{key}_actual"
            pred_col = f"{key}_pred"
            if act_col in train_data.columns and pred_col in train_data.columns:
                # Drop rows where values are missing.
                df_temp = train_data[[act_col, pred_col]].dropna()
                if not df_temp.empty:
                    # Convert each element using the helper function.
                    t_actual = np.array([convert_str_to_float(x) for x in df_temp[act_col].values])
                    t_pred   = np.array([convert_str_to_float(x) for x in df_temp[pred_col].values])
                    ax.scatter(t_actual, t_pred,
                            c='royalblue',
                            edgecolors='black',
                            marker=markers[key],
                            s=50,
                            label=f"Train {key}")
                    train_true_all.extend(t_actual)
                    train_pred_all.extend(t_pred)
        
        # Process validation data.
        for key in keys:
            act_col = f"{key}_actual"
            pred_col = f"{key}_pred"
            if act_col in valid_data.columns and pred_col in valid_data.columns:
                df_temp = valid_data[[act_col, pred_col]].dropna()
                if not df_temp.empty:
                    v_actual = np.array([convert_str_to_float(x) for x in df_temp[act_col].values])
                    v_pred   = np.array([convert_str_to_float(x) for x in df_temp[pred_col].values])
                    ax.scatter(v_actual, v_pred,
                            c='red',
                            edgecolors='black',
                            marker=markers[key],
                            s=50,
                            label=f"Valid {key}")
                    valid_true_all.extend(v_actual)
                    valid_pred_all.extend(v_pred)
        
        # Compute overall error metrics for training.
        if len(train_true_all) > 0:
            train_true_all = np.array(train_true_all)
            train_pred_all = np.array(train_pred_all)
            mask = ~np.isnan(train_true_all) & ~np.isnan(train_pred_all)
            filtered_true = train_true_all[mask]
            filtered_pred = train_pred_all[mask]
            train_rmse = np.sqrt(np.mean((filtered_true - filtered_pred) ** 2))
            train_mae  = np.mean(np.abs(filtered_true - filtered_pred))
            # train_rmse = np.sqrt(np.mean((train_true_all - train_pred_all) ** 2))
            # train_mae  = np.mean(np.abs(train_true_all - train_pred_all))
        else:
            train_rmse, train_mae = 0, 0
            
        # Compute overall error metrics for validation.
        if len(valid_true_all) > 0:
            valid_true_all = np.array(valid_true_all)
            valid_pred_all = np.array(valid_pred_all)
            mask = ~np.isnan(valid_true_all) & ~np.isnan(valid_pred_all)
            filtered_true = valid_true_all[mask]
            filtered_pred = valid_pred_all[mask]
            valid_rmse = np.sqrt(np.mean((filtered_true - filtered_pred) ** 2))
            valid_mae  = np.mean(np.abs(filtered_true - filtered_pred))
            # valid_rmse = np.sqrt(np.mean((valid_true_all - valid_pred_all) ** 2))
            # valid_mae  = np.mean(np.abs(valid_true_all - valid_pred_all))
        else:
            valid_rmse, valid_mae = 0, 0
        
        all_true = np.concatenate([train_true_all, valid_true_all])
        all_pred = np.concatenate([train_pred_all, valid_pred_all])
        overall_min, overall_max = -4, 20
            
        ax.set_xlim(overall_min, overall_max)
        ax.set_ylim(overall_min, overall_max)
        
        # Set up tick spacing.
        xmajor, xminor = 6, 3
        ymajor, yminor = 6, 3
        x_major_ticks = np.arange(overall_min, overall_max + 0.1, xmajor)
        y_major_ticks = np.arange(overall_min, overall_max + 0.1, ymajor)
        ax.set_xticks(x_major_ticks)
        ax.set_yticks(y_major_ticks)
        x_minor_ticks = np.arange(overall_min, overall_max + 0.1, xminor)
        y_minor_ticks = np.arange(overall_min, overall_max + 0.1, yminor)
        ax.set_xticks(x_minor_ticks, minor=True)
        ax.set_yticks(y_minor_ticks, minor=True)
        ax.xaxis.set_tick_params(labelsize=16)
        ax.yaxis.set_tick_params(labelsize=16)
        ax.set_aspect("equal", adjustable="box")
        
        ax.set_xlabel("E1/2$_{true}$", fontsize=12)
        ax.set_ylabel("E1/2$_{pred}$", fontsize=12)
        
        # Create a text box with error metrics.
        info_text = (
            f"Train RMSE: {train_rmse:.3f}\n"
            f"Train MAE:  {train_mae:.3f}\n"
            f"Valid RMSE: {valid_rmse:.3f}\n"
            f"Valid MAE:  {valid_mae:.3f}"
        )
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.05, 0.95, info_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        line_x = np.linspace(overall_min, overall_max, 100)
        ax.plot(line_x, line_x, '--', lw=2, c='black')
        
        # ax.legend(bbox_to_anchor=(0.98, 0.2))
        
        plt.savefig("parity_plot.png", dpi=300)
        plt.show()
class OM():
    def __init__(self):
        pass

    # def data_loader(file_path, test_size=0.2, is_metal=False, features=153, k_fold=False):
    #     df = pd.read_csv(file_path)
    #     df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))) if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))

    #     def tensorize_dataset(data, is_metal=False):
    #         dataset = []
    #         for _, row in data.iterrows():
    #             try:
    #                 [fatoms, graphs, edge_features, midx] = tensorize_with_subgraphs([row["smiles"]], row["Metal"])
    #                 label = torch.Tensor(row['E12'])
    #                 name = fatoms[1]
    #                 redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
    #                 data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, redox=redox_idxs, ys=label, name=name, reaction=row["Reaction"], oreder_site=row["redox_site_smiles"])
    #                 data_item.midx = midx
    #                 dataset.append(data_item)
    #             except Exception as e:
    #                 # print(f"Error processing row: {e}")
    #                 continue
    #         return dataset

    #     if k_fold:
    #         loaders = []
    #         Kfold = KFold(n_splits=5, shuffle=True, random_state=12)
    #         for train_index, test_index in Kfold.split(df):
    #             train_data = df.iloc[train_index]
    #             test_data  = df.iloc[test_index]

    #             train_dataset = tensorize_dataset(train_data)
    #             test_dataset  = tensorize_dataset(test_data)
    #             train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
    #             test_loader   = DataLoader(test_dataset,  batch_size=1, shuffle=False)
                
    #             loaders.append((train_loader, test_loader))

    #         return loaders
    #     else:
    #         train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
    #         train_dataset = tensorize_dataset(train_data)
    #         test_dataset  = tensorize_dataset(test_data)

    #         train_loader  = DataLoader(train_dataset, batch_size=1, shuffle=False)
    #         test_loader   = DataLoader(test_dataset,  batch_size=1, shuffle=False)

    #         return train_loader, test_loader

    def evaluate_model(model, loader, device, output_file=""):
        model.eval()
        names = []
        eval_actuals_reg, eval_predictions_reg = [], []
        eval_actuals_cla, eval_predictions_cla = [], []
        total_loss, total_cla_loss, total_reg_loss, count, total_batches, correct_batches = 0, 0, 0, 0, 0, 0
        with torch.no_grad():
            for data in loader:
                try:
                    actuals, predictions = "" , ""
                    loss_cla, loss_reg = 0, 0
                    data = data.to(device)
                    num_logit, num_peak, E12_regs = model.sample(data, device)

                    for i, real in enumerate(data.ys):
                        if pd.isna(real.item()):
                            break
                        actuals     += str(real.cpu().numpy()) + ","
                        if i < len(E12_regs):
                            predictions += str(E12_regs[i].squeeze().cpu().detach().numpy()) + ","
                            loss_reg    += F.mse_loss(E12_regs[i].squeeze(), real).item() 
                        else:
                            break

                    # real_num_redox = OM.real_num_redox_poly(model, data)
                    # actuals_cla = ",".join(str(x.item()) for x in real_num_redox)
                    real_num_redox = [data.redox[i][0][1] for i in range(len(data.redox))]
                    actuals_cla     = ",".join(map(str, real_num_redox))
                    predictions_cla = ",".join(map(str, num_peak.cpu().tolist()))
                    loss_cla       = F.cross_entropy(num_logit, torch.tensor(real_num_redox).to(device))

                    for j, num in enumerate(num_peak):
                        total_batches += 1
                        if num == real_num_redox[j]:
                            correct_batches += 1    
                        
                    eval_actuals_reg.append(actuals.strip(','))
                    eval_predictions_reg.append(predictions.strip(','))

                    eval_actuals_cla.append(actuals_cla)
                    eval_predictions_cla.append(predictions_cla)

                    names.append(data.name)
                    all_loss = loss_cla + loss_reg
                    total_loss += all_loss.item()
                    total_cla_loss += loss_cla.item()
                    total_reg_loss += loss_reg
                    count += 1

                except Exception as e:
                    print(f"Error evaluating model: {e}")

        df_reg = pd.DataFrame({
        "Actuals"    : eval_actuals_reg,
        "Predictions": eval_predictions_reg,
        "SMILES"     : names,
        "Reaction"   : [data.reaction for data in loader.dataset]
        })
        df_cla = pd.DataFrame({
            "Actuals"    : eval_actuals_cla,
            "Predictions": eval_predictions_cla,
            "SMILES"     : names,
            "Reaction"   : [data.reaction for data in loader.dataset]
        })
        if output_file == "":
            pass
        else:
            df_reg.to_csv(os.path.join(os.getcwd(), f"reg-{output_file}"), index=False)
            df_cla.to_csv(os.path.join(os.getcwd(), f"cla-{output_file}"), index=False)
        return total_loss / count, total_reg_loss / count, total_cla_loss / count, correct_batches / total_batches if total_batches > 0 else 0.0



def analysis(model, test_loader, device):
    model.eval()
    eval_actuals_reg, eval_predictions_reg = [], []
    o_y_proba, o_y_true, r_y_proba, r_y_true = [], [], [], []
    eval_actuals_cla, eval_predictions_cla,  = [], []
    with torch.no_grad():
        for data in test_loader:
            try:
                actuals, predictions = "" , ""
                data = data.to(device)
                num_logit, num_peak, E12_regs = model.sample(data, device)

                for i, real in enumerate(data.ys):
                    if pd.isna(real.item()):
                        break
                    actuals     += str(real.cpu().numpy()) + ","
                    if i < len(E12_regs):
                        predictions += str(E12_regs[i].squeeze().cpu().detach().numpy()) + ","
                    else:
                        break

                y_proba = torch.softmax(num_logit, dim=1)
                y_true = torch.tensor([data.redox[i][0][1] for i in range(len(data.redox))])
                if data.reaction == ['reduction']:
                    r_y_proba.append(y_proba)
                    r_y_true.append(y_true)
                else:
                    o_y_proba.append(y_proba)
                    o_y_true.append(y_true)

                real_num_redox  = [data.redox[i][0][1] for i in range(len(data.redox))]
                actuals_cla     = ",".join(map(str, real_num_redox))
                predictions_cla = ",".join(map(str, num_peak.cpu().tolist()))
                    
                eval_actuals_reg.append(actuals.strip(','))
                eval_predictions_reg.append(predictions.strip(','))

                eval_actuals_cla.append(actuals_cla)
                eval_predictions_cla.append(predictions_cla)
            except Exception as e:
                return None
                # print(f"Error evaluating model: {e}")

        reg_df = pd.DataFrame({
        "Actuals"    : eval_actuals_reg,
        "Predictions": eval_predictions_reg,
        "Reaction"   : [data.reaction for data in test_loader.dataset]
        })

    ox_mask = reg_df['Reaction'] == 'oxidation'
    red_mask = reg_df['Reaction'] == 'reduction'
    results = {
        'oxidation': {'rmse_by_position': []},
        'reduction': {'rmse_by_position': []}
    }
    ox_data = reg_df[ox_mask]
    for idx, row in ox_data.iterrows():
        actuals = [float(x) for x in str(row['Actuals']).split(',') if str(x).strip()]
        predictions = [float(x) for x in str(row['Predictions']).split(',') if str(x).strip()]
        if not actuals or not predictions:
            continue
        for pos, (actual, pred) in enumerate(zip(actuals, predictions)):
            if np.isnan(actual) or np.isnan(pred):
                continue
            if len(results['oxidation']['rmse_by_position']) <= pos:
                results['oxidation']['rmse_by_position'].append([])
            results['oxidation']['rmse_by_position'][pos].append((actual - pred) ** 2)
    red_data = reg_df[red_mask]
    for idx, row in red_data.iterrows():
        actuals = [float(x) for x in str(row['Actuals']).split(',') if str(x).strip()]
        predictions = [float(x) for x in str(row['Predictions']).split(',') if str(x).strip()]
        if not actuals or not predictions:
            continue
        for pos, (actual, pred) in enumerate(zip(actuals, predictions)):
            if np.isnan(actual) or np.isnan(pred):
                continue
            if len(results['reduction']['rmse_by_position']) <= pos:
                results['reduction']['rmse_by_position'].append([])
            results['reduction']['rmse_by_position'][pos].append((actual - pred) ** 2)
    for reaction_type in ['oxidation', 'reduction']:
        for pos in range(len(results[reaction_type]['rmse_by_position'])):
            if results[reaction_type]['rmse_by_position'][pos]:
                mse = np.mean(results[reaction_type]['rmse_by_position'][pos])
                rmse = np.sqrt(mse)
                results[reaction_type]['rmse_by_position'][pos] = rmse
            else:
                results[reaction_type]['rmse_by_position'][pos] = np.nan      

    for reaction_type in results:
        print(f"  {reaction_type.upper()}:")
        for pos, rmse in enumerate(results[reaction_type]['rmse_by_position']):
            print(f"    Position {pos+1} RMSE: {rmse:.4f}")
            if pos == 1:
                break

    def safe_calculate_roc_auc(y_true, y_proba, classes):
        """安全地計算ROC和AUC，處理各種邊界情況"""
        try:
            # print(f"ROC計算輸入檢查: y_true={len(y_true)}, y_proba={y_proba.shape}, classes={classes}")
            
            # 基本輸入驗證
            if len(y_true) != y_proba.shape[0]:
                # print(f"錯誤：樣本數量不匹配 {len(y_true)} vs {y_proba.shape[0]}")
                return np.nan, np.nan
            
            unique_classes = np.unique(y_true)
            # print(f"測試集中的類別: {unique_classes}")
            
            # 如果只有一個類別，無法計算AUC
            if len(unique_classes) <= 1:
                # print("警告：測試集中只有一個類別，無法計算AUC")
                return np.nan, np.nan
            
            # 創建類別到索引的映射
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            # 檢查測試集中的類別是否都在訓練類別中
            valid_classes = [cls for cls in unique_classes if cls in class_to_idx]
            # if len(valid_classes) != len(unique_classes):
                # print(f"警告：測試集中有未知類別，有效類別: {valid_classes}")
            
            if len(valid_classes) <= 1:
                # print("警告：有效類別少於2個，無法計算AUC")
                return np.nan, np.nan
            
            # 使用手動方法計算每個類別的AUC
            # print("使用手動方法計算AUC...")
            aucs = []
            class_counts = []
            
            for cls in valid_classes:
                if cls in class_to_idx:
                    cls_idx = class_to_idx[cls]
                    # 創建二值標籤 (當前類別 vs 其他)
                    y_binary = (y_true == cls).astype(int)
                    # 使用對應類別的概率
                    y_prob_cls = y_proba[:, cls_idx]
                    
                    # 檢查是否有足夠的正例和負例
                    if len(np.unique(y_binary)) > 1:
                        try:
                            auc_cls = roc_auc_score(y_binary, y_prob_cls)
                            aucs.append(auc_cls)
                            class_counts.append(np.sum(y_true == cls))
                            print(f"類別 {cls} AUC: {auc_cls:.3f}")
                        except Exception as e:
                            print(f"類別 {cls} AUC計算失敗: {e}")
            
            if aucs:
                macro_auc = np.mean(aucs)
                if len(class_counts) == len(aucs):
                    weighted_auc = np.average(aucs, weights=class_counts)
                else:
                    weighted_auc = macro_auc
                
                return macro_auc, weighted_auc
            else:
                return np.nan, np.nan
                        
        except Exception as e:
            return np.nan, np.nan
    #cla 
    o_y_proba = torch.cat(o_y_proba, dim=0)
    o_y_true = torch.cat(o_y_true, dim=0)
    o_y_true = o_y_true.detach().cpu().numpy()
    o_y_proba = o_y_proba.detach().cpu().numpy()
    r_y_proba = torch.cat(r_y_proba, dim=0)
    r_y_true = torch.cat(r_y_true, dim=0)
    r_y_true = r_y_true.detach().cpu().numpy()
    r_y_proba = r_y_proba.detach().cpu().numpy()

    print("oxidation:")
    o_macro_auc, o_weighted_auc = safe_calculate_roc_auc(o_y_true, o_y_proba, np.unique(o_y_true))
    print("reduction:")
    r_macro_auc, r_weighted_auc = safe_calculate_roc_auc(r_y_true, r_y_proba, np.unique(r_y_true))

    
    # print(o_macro_auc, o_weighted_auc)
    # print(r_macro_auc, r_weighted_auc)