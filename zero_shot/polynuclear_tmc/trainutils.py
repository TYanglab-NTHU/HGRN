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
from sklearn.metrics         import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

from chemutils import *
from optparse  import OptionParser

import matplotlib.pyplot  as plt
import matplotlib.patches as mpatches
from matplotlib           import ticker
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.gridspec  import GridSpec, GridSpecFromSubplotSpec


class OrganicMetal_potential():
    def __init__(self):
        pass


    def data_loader(file_path, tensorize_fn, batch_size, reaction_type='reduction', test_size=0.2, is_metal=False):
        df = pd.read_csv(file_path)

        if not is_metal:
            df['IE / eV']  = df['IE / eV'].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))   
            df['EA / eV']  = df["EA / eV"].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))   
            df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))            
            train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

        def tensorize_dataset(data, is_metal=False):
            dataset = []
            for _, row in data.iterrows():
                try:
                    if is_metal:
                        fatoms, graphs, edge_features = tensorize_fn(row["Metal"])
                        fatoms = torch.unsqueeze(fatoms, dim=0)
                        label_keys = ['IE', 'EA', 'E12']
                        columns = ['IE / eV', 'EA / eV', 'E12']
                        labels, reaction_info, is_metal_info = {}, {}, {}
                        for key, col in zip(label_keys, columns):
                            if pd.notna(row[col][0]):
                                labels[key] = torch.Tensor(row[col])
                                if key == 'IE':
                                    reaction_info[key] = 'oxidation'
                                elif key == 'EA':
                                    reaction_info[key] = 'reduction'
                                elif key == 'E12':
                                    reaction_info[key] = row['Reaction']
                                is_metal_info[key] = True if row["Metal"] in metal_list else False
                            else:
                                labels[key] = None
                                reaction_info[key] = None
                                is_metal_info[key] = None
                        name   = [row["Metal"]]
                        data_item = Data(x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=labels, name=name, reaction=reaction_info, is_metal=is_metal_info)
                    else:
                        [fatoms, graphs, edge_features, midx] = tensorize_fn([row["smiles"]], 'None')
                        label_keys = ['IE', 'EA', 'E12']
                        columns = ['IE / eV', 'EA / eV', 'E12']
                        labels, reaction_info = {},  {}
                        for key, col in zip(label_keys, columns):
                            if pd.notna(row[col][0]):
                                labels[key] = torch.Tensor(row[col])
                                if key == 'IE':
                                    reaction_info[key] = 'oxidation'
                                elif key == 'EA':
                                    reaction_info[key] = 'reduction'
                                elif key == 'E12':
                                    reaction_info[key] = row['Reaction']
                            else:
                                labels[key] = None
                                reaction_info[key] = None
                        name    = fatoms[1]
                        solvent = row['Solvent']
                        data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=labels,  solvent=solvent, name=name, reaction=reaction_info)                    
                        dataset.append(data_item)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return dataset

        train_dataset = tensorize_dataset(train_data, is_metal=is_metal)
        if is_metal:
            return train_dataset, None
        test_dataset = tensorize_dataset(test_data)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_dataset, test_loader

    def evaluate_model(model, loader, device, output_file=""):
        model.eval()
        # Lists to accumulate each sample's regression and classification results.
        reg_data_list = []
        cla_data_list = []
        
        total_reg_loss = 0.0
        total_cla_loss = 0.0
        count = 0
        correct_batches = 0
        total_batches = 0
        
        with torch.no_grad():
            for data in loader:
                try:
                    data = data.to(device)
                    potential_clas, potential_regs = model.sample(data, device)
                    true_labels = data.ys  
                    sample_name = data.name  
                    
                    sample_reg_dict = {"SMILES": sample_name}
                    sample_cla_dict = {"SMILES": sample_name}
                    
                    for key in ['IE', 'EA', 'E12']:
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
                                # For accuracy, compare predicted class to target class.
                                if pred_class == target_class:
                                    correct_batches += 1
                        else:
                            sample_cla_dict[f"{key}_pred"] = np.nan
                            
                    reg_data_list.append(sample_reg_dict)
                    cla_data_list.append(sample_cla_dict)
                    count += 1
                    total_batches += 1
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

    # def data_loader(file_path, tensorize_fn, batch_size, reaction_type='reduction', test_size=0.2, is_metal=False):
    #     df = pd.read_csv(file_path)

    #     if not is_metal:
    #         df['IE / eV']  = df['IE / eV'].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))   
    #         df['EA / eV']  = df["EA / eV"].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))   
    #         df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(',')))if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))            
    #         train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)

    #     if is_metal:
    #         train_data, _ = df, None

    #     def tensorize_dataset(data, is_metal=False):
    #         dataset = []
    #         for _, row in data.iterrows():
    #             try:
    #                 if is_metal:
    #                     fatoms, graphs, edge_features = tensorize_fn(row["Metal"])
    #                     fatoms = torch.unsqueeze(fatoms, dim=0)
    #                     if pd.notna(row['IE / eV'][0]):
    #                         label    = torch.Tensor(row['IE / eV'])
    #                         reaction = 'oxidation'
    #                     if pd.notna(row['EA / eV'][0]):
    #                         label    = torch.Tensor(row['EA / eV'])
    #                         reaction = 'reduction'
    #                     if pd.notna(row['E12'][0]):
    #                         label    = torch.Tensor(row['E12'])
    #                         reaction = 'reduction'
    #                     name   = [row["Metal"]]
    #                     data_item = Data(x=fatoms, edge_index=graphs, edge_attr=edge_features, ys=label, name=name, reaction=row["Reaction"])
    #                 else:
    #                     [fatoms, graphs, edge_features, midx] = tensorize_fn([row["smiles"]], 'None')
    #                     if pd.notna(row['IE / eV'][0]):
    #                         label    = torch.Tensor(row['IE / eV'])
    #                         reaction = 'oxidation'
    #                     if pd.notna(row['EA / eV'][0]):
    #                         label    = torch.Tensor(row['EA / eV'])
    #                         reaction = 'reduction'
    #                     if pd.notna(row['E12'][0]):
    #                         label    = torch.Tensor(row['E12'])
    #                         reaction = row['Reaction']
    #                     name       = fatoms[1]
    #                     solvent    = row['Solvent']
    #                     data_item  = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, ys=label, solvent=solvent, name=name, reaction=reaction)
    #                 dataset.append(data_item)
    #             except Exception as e:
    #                 print(f"Error processing row: {e}")
    #                 continue
    #         return dataset

    #     train_dataset = tensorize_dataset(train_data, is_metal=is_metal)
    #     if is_metal:
    #         return train_dataset, None
    #     test_dataset = tensorize_dataset(test_data)
    #     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    #     return train_dataset, test_loader

    # def evaluate_model(model, loader, device, output_file=""):
    #     model.eval()
    #     names = []
    #     eval_actuals_reg, eval_predictions_reg = [], []
    #     eval_actuals_del, eval_predictions_del = [], []
    #     eval_actuals_cla, eval_predictions_cla = [], []
    #     total_loss, total_cla_loss, total_reg_loss, total_delta_loss, count, total_batches, correct_batches = 0, 0, 0, 0, 0, 0, 0
    #     with torch.no_grad():
    #         for data in loader:
    #             try:
    #                 all_loss, loss_reg, loss_cla = 0, 0, 0
    #                 # if isinstance(data.solvent[0], torch.Tensor):
    #                     # if data.reaction[0] == 'reduction':
    #                 actuals, predictions, delta_actuals, delta_preds = "" , "", "", ""
    #                 data = data.to(device)
    #                 potential_clas, potential_regs = model.sample(data, device)

    #                 for i, real in enumerate(data.ys):
    #                     actuals     += str(real.squeeze().cpu().detach().numpy()) + ","
    #                     loss_reg    += F.mse_loss(potential_regs[i].squeeze(), real).item()
    #                 for i, potential_reg in enumerate(potential_regs):
    #                     predictions += str(potential_reg.squeeze().cpu().detach().numpy()) + ","
    #                 all_loss    += loss_reg

    #                 real_num_redox   = len(data.ys)
    #                 redox_num_tensor = potential_clas[0]
    #                 num_peaks        = torch.argmax(redox_num_tensor, dim=0).item()
    #                 loss_cla        += F.cross_entropy(redox_num_tensor, torch.tensor(real_num_redox).to(device))
    #                 all_loss        += loss_cla
    #                 actuals_cla      = "".join(str(len(data.ys)))
    #                 predictions_cla  = "".join(str(num_peaks))

    #                 if num_peaks == len(data.ys):
    #                     correct_batches += 1
    #                 total_batches += 1

    #                 eval_actuals_reg.append(actuals.strip(','))
    #                 eval_predictions_reg.append(predictions.strip(','))

    #                 eval_actuals_cla.append(actuals_cla)
    #                 eval_predictions_cla.append(predictions_cla)

    #                 names.append(data.name)
    #                 total_loss += all_loss.item()
    #                 total_cla_loss += loss_cla.item()
    #                 total_reg_loss += loss_reg
    #                 count += 1

    #             except Exception as e:
    #                 print(f"Error evaluating model: {e}")

    #     df_reg = pd.DataFrame({
    #     "Actuals"     : eval_actuals_reg,
    #     "Predictions" : eval_predictions_reg,
    #     "SMILES"      : names
    #     })
    #     df_cla = pd.DataFrame({
    #         "Actuals"    : eval_actuals_cla,
    #         "Predictions": eval_predictions_cla,
    #         "SMILES"     : names
    #     })
    #     if pd.isnull(output_file):
    #         pass
    #     else:
    #         df_reg.to_csv(os.path.join(os.getcwd(), f"reg-{output_file}"), index=False)
    #         df_cla.to_csv(os.path.join(os.getcwd(), f"cla-{output_file}"), index=False)
    #     return total_loss / count, total_reg_loss / count, total_cla_loss / count, correct_batches / total_batches if total_batches > 0 else 0.0    

    # def parity_plot(train_file,valid_file):
    #     train_data = pd.read_csv(train_file)
    #     valid_data = pd.read_csv(valid_file)
    #     train_true, train_pred = train_data['Actuals'], train_data['Predictions']
    #     valid_true, valid_pred = valid_data['Actuals'], valid_data['Predictions']

    #     lfs_rmse1 = np.sqrt(np.mean((np.array(train_true)-np.array(train_pred))**2))
    #     lfs_mae_1 = np.mean(abs(np.array(train_true)-np.array(train_pred)))
    #     lfs_rmse2 = np.sqrt(np.mean((np.array(valid_true)-np.array(valid_pred))**2))
    #     lfs_mae_2 = np.mean(abs(np.array(valid_true)-np.array(valid_pred)))
    #     fig, ax = plt.subplots(figsize=(6,6))
    #     xmin, xmax = -4,20
    #     ymin, ymax = -4,20
    #     xmajor, xminor = 6, 3
    #     ymajor, yminor = 6, 3
    #     ax.set_xlim(xmin, xmax)
    #     ax.set_ylim(ymin, ymax)
    #     x_major_ticks = np.arange(xmin, xmax + 0.1, xmajor)
    #     y_major_ticks = np.arange(ymin, ymax + 0.1, ymajor)
    #     ax.set_xticks(x_major_ticks)
    #     ax.set_yticks(y_major_ticks)
    #     x_minor_ticks = np.arange(xmin, xmax + 0.1, xminor)
    #     y_minor_ticks = np.arange(ymin, ymax + 0.1, yminor)
    #     ax.set_xticks(x_minor_ticks, minor=True)
    #     ax.set_yticks(y_minor_ticks, minor=True)
    #     ax.xaxis.set_tick_params(labelsize=16)
    #     ax.yaxis.set_tick_params(labelsize=16)
    #     ax.set_aspect("equal", adjustable="box")
    #     ax.set_xlabel("E1/2$_{true}$", fontsize=12)
    #     ax.set_ylabel("E1/2$_{pred}$", fontsize=12)
    #     dist = [[0,5],[0,5]]
    #     string = 'train RMSE %.3f \ntrain MAE   %.3f \ntest  RMSE %.3f \ntest  MAE   %.3f'%(lfs_rmse1,lfs_mae_1,lfs_rmse2,lfs_mae_2)#%(rmse1,rmse2,mae_1,mae_2)
    #     props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    #     ax.text(0.05, 0.95, string, transform=ax.transAxes, fontsize=10,
    #             verticalalignment='top',bbox=props)
    #     ax.scatter(np.array(train_true).flatten(), np.array(train_pred).flatten(),c='royalblue',edgecolors='black',label='train')
    #     ax.scatter(np.array(valid_true).flatten(),np.array(valid_pred).flatten(),c='red',edgecolors='black',label='test')
    #     ax.plot([0, 1], [0, 1], "--",lw=2,c = 'black',transform=ax.transAxes)
    #     ax.legend(bbox_to_anchor=(0.98,0.15))
    #     plt.savefig('parity_plot.png')
    #     plt.show()


class OM():
    def __init__(self):
        pass

    def data_loader_binding(file_path, tensorize_fn, batch_size, test_size=0.2):
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
                    data_item  = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, redox=redox_idxs, ys=label, name=name, reaction=row["Reaction"], oreder_site=row["redox_site_smiles"])
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


    def kfold_loader(file_path, tensorize_fn, batch_size, test_size=0.2):
        df = pd.read_csv(file_path)

        df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))) if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))

        def tensorize_dataset(data):
            dataset = []
            for _, row in data.iterrows():
                try:
                    [fatoms, graphs, edge_features, midx] = tensorize_fn([row["smiles"]], row["Metal"])
                    label = torch.Tensor(row['E12'])
                    name = fatoms[1]
                    redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                    data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, redox=redox_idxs, ys=label, name=name, reaction=row["Reaction"], oreder_site=row["redox_site_smiles"])
                    data_item.midx = midx
                    dataset.append(data_item)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return dataset

        loaders = []
        Kfold   = KFold(n_splits=5, shuffle=True, random_state=12)
        for train_index, test_index in Kfold.split(df):
            train_data = df.iloc[train_index]
            test_data  = df.iloc[test_index]
            
            train_dataset = tensorize_dataset(train_data)
            test_dataset  = tensorize_dataset(test_data)
            train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
            test_loader   = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
            
            loaders.append((train_loader, test_loader))
        return loaders

    def real_num_redox_poly(model, data):
        edge = data.edge_index
        subgraph1, batch1_2, subgraph2, subgraph3 = edge[0]
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2, filtered_masks = subgraph2
        subgraph3_edge_index, batch3 = subgraph3
 
        new_batch = batch2[batch1_2.long()] #每個原子對應在原來subgraph2_pooled的index

        orig_sub2_poly_sub2 = model.build_index_mapping(batch2, filtered_masks) 

        mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()} #在subgraph1_pooled的index對應在subgraph2_pooled的index
        # 把subgraph3_result的index對應到subgraph1_pooled的index 因為real_num_peaks對應的是subgraph1_pooled的index
        mapping_sub3tosub1 = {}
        for orig_group, mid in mapping_dict.items():
            ids = orig_sub2_poly_sub2[mid]
            # 如果只有一個 pooled node，就取單值；否則保留列表
            mapping_sub3tosub1[orig_group] = ids[0] if len(ids) == 1 else ids

        # 3. 根據 mapping_sub3tosub1  重新排列 poly subgraph3_result 到subgraph1_pooled的index
        order = []
        for g in sorted(mapping_sub3tosub1 .keys()):
            v = mapping_sub3tosub1 [g]
            if isinstance(v, list):
                order.extend(v)
            else:
                order.append(v)

        # 新向量长度 = 最大 pooled 索引 + 1
        length = max(idx for idxs in orig_sub2_poly_sub2.values() for idx in idxs) + 1
        real_num_peaks = torch.tensor([data.redox[i][0][1] for i in range(len(data.redox))]).cuda()
        # 初始化全 0，然后依组填值
        expanded = torch.zeros(length, dtype=real_num_peaks.dtype, device=real_num_peaks.device)
        for grp, new_idx in mapping_sub3tosub1.items():
            expanded[new_idx] = real_num_peaks[grp]
        real_num_peaks = expanded[order]

        return real_num_peaks

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
        "SMILES"     : names
        })
        df_cla = pd.DataFrame({
            "Actuals"    : eval_actuals_cla,
            "Predictions": eval_predictions_cla,
            "SMILES"     : names
        })
        if pd.isnull(output_file):
            pass
        else:
            df_reg.to_csv(os.path.join(os.getcwd(), f"reg-{output_file}"), index=False)
            df_cla.to_csv(os.path.join(os.getcwd(), f"cla-{output_file}"), index=False)
        return total_loss / count, total_reg_loss / count, total_cla_loss / count, correct_batches / total_batches if total_batches > 0 else 0.0


    def redox_site(model, batch, device):
        for graph in batch.to_data_list():
            x, edge_index, edge_attr, midx, real_E12, reaction, redox = graph.x, graph.edge_index, graph.edge_attr, graph.midx, graph.ys, graph.reaction, graph.redox
        
        subgraph1, batch1_2, subgraph2, subgraph3 = edge_index
        subgraph1_edge_index, batch1 = subgraph1
        subgraph2_edge_index, batch2 = subgraph2
        subgraph3_edge_index, batch3 = subgraph3

        #"results after GCN and result_ after global pooling"
        subgraph1_result, subgraph1_pooled = model.forward_subgraph(x=x, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=model.GCN1)
        subgraph2_result, subgraph2_pooled = model.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=model.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))
        subgraph3_result, subgraph3_pooled = model.forward_subgraph(x=subgraph2_pooled, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=model.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

        # convert batch1 index to batch3 index
        m_batch1  = batch1[midx]
        new_batch = batch2[batch1_2.long()]

        mapping_dict    = {val.item(): new_batch[batch1 == val].unique().item() for val in batch1.unique()}
        ordered_indices = torch.tensor([mapping_dict[k] for k in sorted(mapping_dict)], device='cuda')

        batch1_subgraph3_result = subgraph3_result[ordered_indices]
        if   reaction == 'reduction':
            num_redox_all = model.num_peaks_red(batch1_subgraph3_result)
        elif reaction == 'oxidation':
            num_redox_all = model.num_peaks_ox(batch1_subgraph3_result)
        
        num_redox_ = torch.argmax(num_redox_all, dim=1)
        pred_num_redox_ = num_redox_.clone()
        pred_E12s  = torch.tensor([], device=device)

        redox_idxs = []
        while num_redox_.sum() != 0:
            batch1_subgraph3_result = subgraph3_result[ordered_indices]
            if   reaction == 'reduction':
                E12s          = model.E12_reg_red(batch1_subgraph3_result)
            elif reaction == 'oxidation':
                E12s          = model.E12_reg_ox(batch1_subgraph3_result)
            E12s          = E12s.squeeze()
            redox_mask    = num_redox_ > 0
            redox_indices = torch.nonzero(redox_mask, as_tuple=False).flatten()
            E12s_redox    = E12s[redox_mask]
            if reaction == "reduction":
                E12, filtered_idx = torch.max(E12s_redox, dim=0)
            elif reaction == "oxidation":
                E12, filtered_idx = torch.min(E12s_redox, dim=0)

            redox_site_idx = redox_indices[filtered_idx].item()
            redox_idxs.append(redox_site_idx)
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
                redox_x_change             = redox_subgraph1_result_ * model.gate_GCN1(redox_subgraph1_result_) + redox_x_

            x_              = x.clone()
            x_[redox_x_idx] = redox_x_change
            subgraph1_result, subgraph1_pooled = model.forward_subgraph(x=x_, edge_index=subgraph1_edge_index, batch=batch1, edge_attr=edge_attr[0], gcn=model.GCN1)
            subgraph2_result, subgraph2_pooled = model.forward_subgraph(x=subgraph1_result, edge_index=subgraph2_edge_index, batch=batch2, edge_attr=edge_attr[1], gcn=model.GCN2,pre_proc=lambda x: global_mean_pool(x, batch1_2))

            # gat GCN2 with GCN3
            batch2_redox_idx = mapping_dict.get(redox_site_idx)
            all_indices      = torch.arange(subgraph2_pooled.shape[0], device=device)
            nonredox_subgraph2_pooled = subgraph2_pooled[all_indices != batch2_redox_idx]
            updated_subgraph2_pooled  = nonredox_subgraph2_pooled.clone()
            redox_subgraph2_pooled    = subgraph2_pooled[batch2_redox_idx]
            redox_subgraph3_result_   = subgraph3_result[batch2_redox_idx]
            redox_site_change = redox_subgraph3_result_ * model.gate_GCN3(redox_subgraph3_result_) + redox_subgraph2_pooled
            subgraph2_result_ = torch.cat([updated_subgraph2_pooled[:batch2_redox_idx], redox_site_change.unsqueeze(0), updated_subgraph2_pooled[batch2_redox_idx:]], dim=0)

            subgraph3_result, subgraph3_pooled = model.forward_subgraph(x=subgraph2_result_, edge_index=subgraph3_edge_index, batch=batch3, edge_attr=edge_attr[2], gcn=model.GCN3,transform_edge_attr=lambda attr: torch.stack(attr, dim=0))

            pred_E12s = torch.cat((pred_E12s, E12.unsqueeze(0)), 0)

            redox_pos = [redox[redox_site_idx][0]]
            sites = [i for i in range(len(redox)) if redox[i][0] in redox_pos]

            num_redox_[sites] = num_redox_[sites] - 1
            
        return pred_E12s, pred_num_redox_, redox_idxs

    def E12_site(model, loader, device):
        model.eval()
        with torch.no_grad():
            data_list = []
            for data in loader:
                smiles        = data.name[0][0]
                reaction      = data.reaction[0]
                E12_site_pred = {}
                E12_site_real = {}
                data = data.to(device)

                E12s, redox_num, sites = OM.redox_site(model,data, device)
                redox_site_            = [data.redox[sites[i]][0][0] for i in range(len(sites))]
                E12_site_pred['E12']   = [i.cpu().detach().numpy().item() for i in E12s]
                E12_site_pred['site']  = redox_site_

                E12_site_real['E12']   = data.ys.tolist()
                E12_site_real['site']  = data.oreder_site[0].split('/')

                data_list.append([smiles, reaction, E12_site_pred, E12_site_real])

            return data_list

    def sample_loader(file_path, tensorize_fn, batch_size):
        df = pd.read_csv(file_path)
        df['E12'] = df['E12'].apply(lambda x: list(map(float, x.split(','))) if isinstance(x, str) and ',' in x else ([float(x)] if isinstance(x, str) else ([x] if not isinstance(x, list) else x)))
        # df['E12'] = df['E12'].apply(lambda x: [(i - mean) / std for i in x])

        def tensorize_dataset(data, is_metal=False):
            dataset = []
            for _, row in data.iterrows():
                try:
                    [fatoms, graphs, edge_features, midx, binding_atoms] = tensorize_fn([row["smiles"]], row["Metal"])
                    label = torch.Tensor(row['E12'])
                    name = fatoms[1]
                    redox_idxs = redox_each_num([row["smiles"]], row["Metal"], row["redox_site_smiles"])
                    data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, redox=redox_idxs, ys=label, name=name, reaction=row["Reaction"], oreder_site=row["redox_site_smiles"], binding_atoms=binding_atoms)
                    data_item.midx = midx
                    dataset.append(data_item)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return dataset

        sample_dataset = tensorize_dataset(df)
        loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False)
        return loader

    def unlabeled_sample_loader(file_path, batch_size):
        df = pd.read_csv(file_path)

        def tensorize_dataset(data, is_metal=False):
            dataset = []
            for _, row in data.iterrows():
                try:
                    [fatoms, graphs, edge_features, midx, binding_atoms] = tensorize_with_subgraphs([row["smiles"]], row["Metal"])
                    name = fatoms[1]
                    data_item = Data(x=fatoms[0], edge_index=graphs, edge_attr=edge_features, name=name, binding_atoms=binding_atoms)
                    data_item.midx = midx
                    dataset.append(data_item)
                except Exception as e:
                    print(f"Error processing row: {e}")
                    continue
            return dataset

        sample_dataset = tensorize_dataset(df)
        loader = DataLoader(sample_dataset, batch_size=batch_size, shuffle=False)
        return sample_dataset, loader

def plot_loss(loss_csv,num_epochs):
    df = pd.read_csv(loss_csv)
    df['train_loss'], df['train_reg_loss'], df['train_cla_loss']
    df['test_loss'],  df['test_reg_loss'],  df['test_cla_loss'], df['test_cla_accuray']
    df['epoch'] = df['Unnamed: 0']

    fig, ax = plt.subplots(figsize=(6,6))
    xmin, xmax = 0, num_epochs
    ymin, ymax = 0, 1.0
    xmajor, xminor = num_epochs/4, num_epochs/8
    ymajor, yminor = ymax/4, ymax/8
    ax.xaxis.set_major_locator(ticker.MultipleLocator(xmajor))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(xminor))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(ymajor))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(yminor))
    ax.xaxis.set_tick_params(labelsize=12)
    ax.yaxis.set_tick_params(labelsize=12)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.plot(df['epoch'],df['train_reg_loss'],label='train_reg')
    ax.plot(df['epoch'],df['test_reg_loss'],label='test_reg')
    ax.plot(df['epoch'],df['train_cla_loss'],label='train_cla')

    ax.set_xlabel('epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.legend()
    plt.savefig("loss.png", dpi=300, bbox_inches='tight')
    plt.show()


def parity_plot(train_file,valid_file):
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)
    def convert_cell(cell):
        if pd.isna(cell) or cell == '':
            return None
        try:
            return list(map(float, cell.split(',')))
        except Exception as e:
            return None

    if train_data['Actuals'].dtype != 'float64':
        train_data['Actuals'] = train_data['Actuals'].apply(convert_cell)
    if train_data['Predictions'].dtype != 'float64':
        train_data['Predictions'] = train_data['Predictions'].apply(convert_cell)

    if valid_data['Actuals'].dtype != 'float64':
        valid_data['Actuals'] = valid_data['Actuals'].apply(convert_cell)
    if valid_data['Predictions'].dtype != 'float64':
        valid_data['Predictions'] = valid_data['Predictions'].apply(convert_cell)

    train_data = train_data[train_data['Actuals'].notna() & train_data['Predictions'].notna()]
    valid_data = valid_data[valid_data['Actuals'].notna() & valid_data['Predictions'].notna()]

    def extract_values(row):
        actuals = row['Actuals']
        preds = row['Predictions']

        # 1. 檢查 None
        if actuals is None or preds is None:
            return None, None, []

        # 2. 檢查是否為單一的 NaN 浮點數
        #    (pd.isna 在標量上工作正常)
        is_actuals_scalar_nan = isinstance(actuals, float) and pd.isna(actuals)
        is_preds_scalar_nan = isinstance(preds, float) and pd.isna(preds)
        if is_actuals_scalar_nan or is_preds_scalar_nan:
            return None, None, []

        # --- 後續的類型檢查和處理邏輯保持不變 ---
        main_actual, main_pred = None, None
        multiples = []

        # --- 處理 Actuals ---
        if isinstance(actuals, (list, tuple, np.ndarray)):
            if len(actuals) > 0:
                main_actual = actuals[0]
                actuals_rest = actuals[1:]
            else:
                return None, None, []
        elif isinstance(actuals, (int, float)):
            main_actual = float(actuals)
            actuals_rest = []
        else:
            print(f"警告：處理 Actuals 時遇到未預期類型: {type(actuals)}, 值: {actuals}")
            return None, None, []

        # --- 處理 Preds ---
        if isinstance(preds, (list, tuple, np.ndarray)):
            if len(preds) > 0:
                main_pred = preds[0]
                preds_rest = preds[1:]
            else:
                return None, None, []
        elif isinstance(preds, (int, float)):
            main_pred = float(preds)
            preds_rest = []
        else:
            print(f"警告：處理 Preds 時遇到未預期類型: {type(preds)}, 值: {preds}")
            return None, None, []

        # --- 組合額外的值 (Multiples) ---
        for a, p in zip(actuals_rest, preds_rest):
            multiples.append((a, p))

        return main_actual, main_pred, multiples

    train_true ,train_pred = [], []
    train_true_add, train_pred_add = [], []

    for idx, row in train_data.iterrows():
        main_actual, main_prediction, additional = extract_values(row)
        if main_actual is not None:
            train_true.append(main_actual)
            train_pred.append(main_prediction)
            for a, p in additional:
                train_true_add.append(a)
                train_pred_add.append(p)

    valid_true ,valid_pred = [], []
    valid_true_add, valid_pred_add = [], []

    for idx, row in valid_data.iterrows():
        main_actual, main_prediction, additional = extract_values(row)
        if main_actual is not None:
            valid_true.append(main_actual)
            valid_pred.append(main_prediction)
            for a, p in additional:
                valid_true_add.append(a)
                valid_pred_add.append(p)

    lfs_rmse1 = np.sqrt(np.mean((np.array(train_true) - np.array(train_pred)) ** 2))
    lfs_mae_1 = np.mean(np.abs(np.array(train_true) - np.array(train_pred)))
    lfs_rmse2 = np.sqrt(np.mean((np.array(valid_true) - np.array(valid_pred)) ** 2))
    lfs_mae_2 = np.mean(np.abs(np.array(valid_true) - np.array(valid_pred)))

    fig, ax = plt.subplots(figsize=(6, 6))
    xmin, xmax = -4, 4
    ymin, ymax = -4, 4
    xmajor, xminor = 2, 1.0
    ymajor, yminor = 2, 1.0
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    x_major_ticks = np.arange(xmin, xmax + 0.1, xmajor)
    y_major_ticks = np.arange(ymin, ymax + 0.1, ymajor)
    ax.set_xticks(x_major_ticks)
    ax.set_yticks(y_major_ticks)
    x_minor_ticks = np.arange(xmin, xmax + 0.1, xminor)
    y_minor_ticks = np.arange(ymin, ymax + 0.1, yminor)
    ax.set_xticks(x_minor_ticks, minor=True)
    ax.set_yticks(y_minor_ticks, minor=True)
    ax.xaxis.set_tick_params(labelsize=16)
    ax.yaxis.set_tick_params(labelsize=16)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("E1/2$_{true}$", fontsize=12)
    ax.set_ylabel("E1/2$_{pred}$", fontsize=12)

    stats_text = 'train RMSE %.3f \ntrain MAE   %.3f \ntest  RMSE %.3f \ntest  MAE   %.3f' % (
        lfs_rmse1, lfs_mae_1, lfs_rmse2, lfs_mae_2)
    props = dict(boxstyle='round', facecolor='white', alpha=0.8)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)

    ax.scatter(np.array(train_true), np.array(train_pred), c='royalblue',
            edgecolors='black', label='train first')
    ax.scatter(np.array(valid_true), np.array(valid_pred), c='red',
            edgecolors='black', label='test first')

    if train_true_add:
        ax.scatter(np.array(train_true_add), np.array(train_pred_add), c='#669bbc',
                edgecolors='black', label='train multiple')
    if valid_true_add:
        ax.scatter(np.array(valid_true_add), np.array(valid_pred_add), c='#780000',
                edgecolors='black', label='test multiple')

    # Plot the unity line.
    ax.plot([xmin, xmax], [xmin, xmax], "--", lw=2, c='black')

    ax.legend(bbox_to_anchor=(0.98, 0.25))
    plt.savefig('parity_plot.png', dpi=300, bbox_inches='tight')
    plt.show()


def get_peaks(peaks): 
    num_list = [num for num in peaks] 
    num_list_ = [num.split(',') for num in num_list]
    num = [int(item) for sublist in num_list_ for item in sublist]
    return num

def precision_recall(num_classes, cm):
        precision = []
        recall = []
        for i in range(num_classes):
            TP = cm[i, i]
            actual = np.sum(cm[i, :])
            predicted = np.sum(cm[:, i])
            
            prec = TP / predicted if predicted > 0 else np.nan
            rec = TP / actual if actual > 0 else np.nan
            
            precision.append(prec)
            recall.append(rec)
            
        return precision, recall

def cm(train_file,valid_file, output=""):
    train_data = pd.read_csv(train_file)
    valid_data = pd.read_csv(valid_file)

    train_actual_peaks = train_data['Actuals'].values.tolist() #[eval(act) for act in train_data['Actuals']]
    valid_actual_peaks = valid_data['Actuals'].values.tolist() #[eval(act) for act in valid_data['Actuals']]
    train_pred_peaks = train_data['Predictions'].values.tolist()
    valid_pred_peaks = valid_data['Predictions'].values.tolist()
    
    train_actuals_peaks = get_peaks(train_actual_peaks)
    train_preds_peaks   = get_peaks(train_pred_peaks)
    valid_actuals_peaks = get_peaks(valid_actual_peaks)
    valid_preds_peaks   = get_peaks(valid_pred_peaks)
        
    labels = [0, 1, 2, 3, 4]

    train_cm = confusion_matrix(train_actuals_peaks, train_preds_peaks, labels=labels)
    valid_cm = confusion_matrix(valid_actuals_peaks, valid_preds_peaks, labels=labels)

    train_precision, train_recall = precision_recall(len(labels), train_cm)
    valid_precision, valid_recall = precision_recall(len(labels), valid_cm)

    train_recall_ = train_recall + [0]
    train_cm = np.concatenate((train_cm, np.array(train_precision).reshape(5,1)), axis=1)
    train_cm = np.concatenate((train_cm, np.array(train_recall_).reshape(1,6)), axis=0)
    train_cm = np.nan_to_num(train_cm, nan=0)
    np.set_printoptions(suppress=True, formatter={'float_kind': lambda x: f"{x:0.2f}"})
    valid_recall_ = valid_recall + [0]
    valid_cm = np.concatenate((valid_cm, np.array(valid_precision).reshape(5,1)), axis=1)
    valid_cm = np.concatenate((valid_cm, np.array(valid_recall_).reshape(1,6)), axis=0)
    valid_cm = np.nan_to_num(valid_cm, nan=0)
    np.set_printoptions(suppress=True, formatter={'float_kind': lambda x: f"{x:0.2f}"})

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    disp_train = ConfusionMatrixDisplay(confusion_matrix=train_cm)
    disp_train.plot(ax=axs[0], cmap='Blues', colorbar=False, values_format='0.1f')
    axs[0].set_xlabel('Predicted Peaks')
    axs[0].set_ylabel('Actual Peaks')
    disp_valid = ConfusionMatrixDisplay(confusion_matrix=valid_cm)
    disp_valid.plot(ax=axs[1], cmap='Reds', colorbar=False, values_format='0.1f')
    axs[0].set_xticks([0, 1, 2, 3, 4, 5])
    axs[0].set_xticklabels(["0", "1", "2", "3", "4", "precision"]) 
    axs[0].set_yticks([0, 1, 2, 3, 4, 5])
    axs[0].set_yticklabels(["0", "1", "2", "3", "4", "recall"])
    axs[1].set_xticks([0, 1, 2, 3, 4, 5])
    axs[1].set_xticklabels(["0", "1", "2", "3", "4", "precision"])  
    axs[1].set_yticks([0, 1, 2, 3, 4, 5])
    axs[1].set_yticklabels(["0", "1", "2", "3", "4", "recall"])
    axs[1].set_xlabel('Predicted Peaks')
    axs[1].set_ylabel('Actual Peaks')
    plt.tight_layout()
    plt.savefig(f'{output}_confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()


"""filter data points the first E12 value 
difference between real and predicted is greater than 0.5V"""
def get_first_E12(cell_value):
    if pd.isna(cell_value) or cell_value.strip() == "":
        return None
    try:
        d = ast.literal_eval(cell_value)
        # Check if the dictionary has an 'E12' key and if it is a non-empty list
        if isinstance(d, dict) and 'E12' in d:
            values = d['E12']
            if isinstance(values, list) and len(values) > 0:
                return values[0]
    except Exception as e:
        print(f"Error parsing cell value: {cell_value}\nError: {e}")
    return None
# filter
def diff_is_large(row):
    real_val = row['E12_real_first']
    pred_val = row['E12_pred_first']
    # Ensure both values exist before computing the difference
    if real_val is not None and pred_val is not None:
        return abs(real_val - pred_val) > 0.5
    return False

def extract_site(val):
    d = ast.literal_eval(val)
    return d.get('site')


def get_E12(cell_value):
    value_list = []
    d = ast.literal_eval(cell_value)
    if isinstance(d, dict) and 'E12' in d:
        values = d['E12']
    for value in values:
        value_list.append(value)

    return value_list
def map_sites_to_labels(row, site_column='real_site_values'):
    # Ensure m_lig is a dictionary (convert if necessary)
    m_lig = row['M-Lig_Index']  # adjust this key if your column name is different
    if isinstance(m_lig, str):
        m_lig = ast.literal_eval(m_lig)
    
    sites = row[site_column]
    # If sites is nested, flatten it.
    if sites and isinstance(sites[0], list):
        sites = sites[0]
    
    assigned = []
    for site in sites:
        for k, v in m_lig.items():
            if site == v:  
                assigned.append(k)
                break
    return assigned

def plot_sites(site_csv):
    df = pd.read_csv(site_csv)
    df_plot = df.copy()
    df_plot = df_plot.dropna(subset=['E12_sites_real', 'E12_sites_pred'])
    df['E12_real_first'] = df['E12_sites_real'].apply(get_first_E12)
    df['E12_pred_first'] = df['E12_sites_pred'].apply(get_first_E12)
    df_filtered = df[df.apply(diff_is_large, axis=1)]
    df_plot['real_site_values'] = df_plot['E12_sites_real'].apply(extract_site)
    df_plot['pred_site_values'] = df_plot['E12_sites_pred'].apply(extract_site)
    df_filtered['real_site_values'] = df_filtered['E12_sites_real'].apply(extract_site)
    df_filtered['pred_site_values'] = df_filtered['E12_sites_pred'].apply(extract_site)
    df_plot['E12_real'] = df_plot['E12_sites_real'].apply(get_E12)
    df_plot['E12_pred'] = df_plot['E12_sites_pred'].apply(get_E12)
    df_filtered['E12_real'] = df_filtered['E12_sites_real'].apply(get_E12)
    df_filtered['E12_pred'] = df_filtered['E12_sites_pred'].apply(get_E12)
    df_filtered['real_site_labels'] = df_filtered.apply(map_sites_to_labels, axis=1)
    df_filtered['pred_site_labels'] = df_filtered.apply(map_sites_to_labels, site_column='pred_site_values', axis=1)
    df_plot['real_site_labels'] = df_plot.apply(map_sites_to_labels, axis=1)
    df_plot['pred_site_labels'] = df_plot.apply(map_sites_to_labels, site_column='pred_site_values', axis=1)
    color_map = {
    'M':  '#d8a48f',
    'L1': '#d6ce93',
    'L2': '#8e7dbe',
    'L3': '#f1e3d3',
    'L4': '#f2d0a9'}
    bar_width = 0.4
    def count_labels(label_list):
        counts = {}
        for lbl in label_list:
            counts[lbl] = counts.get(lbl, 0) + 1
        return counts

    def total_count(counts_dict):
        """Sum of all counts in a dict {label: count}."""
        return sum(counts_dict.values())

    num_subplots = 5
    chunk_size = 20  # 20 data points per subplot

    all_labels = list(color_map.keys())


    unique_idxs_all = sorted(df_plot['index'].unique())
    unique_idxs_all = unique_idxs_all[: num_subplots * chunk_size]

    # -- Optional: Pre-scan across ALL data for a global max_ox, max_re --
    max_ox_global = 0
    max_re_global = 0
    for idx in unique_idxs_all:
        df_this = df_plot[df_plot['index'] == idx]
        df_ox = df_this[df_this['Reaction'] == 'oxidation']
        if not df_ox.empty:
            row_ox = df_ox.iloc[0]
            max_ox_global = max(
                max_ox_global,
                total_count(count_labels(row_ox['real_site_labels'])),
                total_count(count_labels(row_ox['pred_site_labels']))
            )
        df_re = df_this[df_this['Reaction'] == 'reduction']
        if not df_re.empty:
            row_re = df_re.iloc[0]
            max_re_global = max(
                max_re_global,
                total_count(count_labels(row_re['real_site_labels'])),
                total_count(count_labels(row_re['pred_site_labels']))
            )


    fig, axs = plt.subplots(num_subplots, 1, figsize=(10, 24), sharey=True)

    for sp_idx in range(num_subplots):
        # Slice out 20 indexes for this subplot
        start = sp_idx * chunk_size
        end = start + chunk_size
        sub_idxs = unique_idxs_all[start:end]

        ax = axs[sp_idx]  # current subplot axis

        x_vals = np.arange(len(sub_idxs))

        # -------------------------
        # PLOT THE DATA FOR sub_idxs
        # -------------------------
        for i, idx in enumerate(sub_idxs):
            df_this = df_plot[df_plot['index'] == idx]

            # -- Oxidation --
            df_ox = df_this[df_this['Reaction'] == 'oxidation']
            if not df_ox.empty:
                row_ox = df_ox.iloc[0]
                real_ox_counts = count_labels(row_ox['real_site_labels'])
                pred_ox_counts = count_labels(row_ox['pred_site_labels'])
                # Compute E12 differences
                e12_diff_ox = None
                if (row_ox['E12_real'] is not None) and (row_ox['E12_pred'] is not None):
                    e12_diff_ox = [r - p for r, p in zip(row_ox['E12_real'], row_ox['E12_pred'])]
            else:
                real_ox_counts = {}
                pred_ox_counts = {}
                e12_diff_ox = None

            # -- Reduction --
            df_re = df_this[df_this['Reaction'] == 'reduction']
            if not df_re.empty:
                row_re = df_re.iloc[0]
                real_re_counts = count_labels(row_re['real_site_labels'])
                pred_re_counts = count_labels(row_re['pred_site_labels'])
                e12_diff_re = None
                if (row_re['E12_real'] is not None) and (row_re['E12_pred'] is not None):
                    e12_diff_re = [r - p for r, p in zip(row_re['E12_real'], row_re['E12_pred'])]
            else:
                real_re_counts = {}
                pred_re_counts = {}
                e12_diff_re = None

            # ---- Oxidation Bars (Real) ----
            bottom = 0
            for lbl in all_labels:
                c = real_ox_counts.get(lbl, 0)
                if c > 0:
                    ax.bar(
                        x=i - bar_width/2,
                        height=c,
                        bottom=bottom,
                        width=bar_width,
                        color=color_map[lbl],
                        edgecolor='black'
                    )
                    bottom += c

            # ---- Oxidation Bars (Predicted) ----
            bottom = 0
            for lbl in all_labels:
                c = pred_ox_counts.get(lbl, 0)
                if c > 0:
                    ax.bar(
                        x=i + bar_width/2,
                        height=c,
                        bottom=bottom,
                        width=bar_width,
                        color=color_map[lbl],
                        edgecolor='black'
                    )
                    bottom += c

            # ---- Reduction Bars (Real) ----
            bottom = 0
            for lbl in all_labels:
                c = real_re_counts.get(lbl, 0)
                if c > 0:
                    ax.bar(
                        x=i - bar_width/2,
                        height=-c,
                        bottom=bottom,
                        width=bar_width,
                        color=color_map[lbl],
                        edgecolor='black'
                    )
                    bottom -= c

            # ---- Reduction Bars (Predicted) ----
            bottom = 0
            for lbl in all_labels:
                c = pred_re_counts.get(lbl, 0)
                if c > 0:
                    ax.bar(
                        x=i + bar_width/2,
                        height=-c,
                        bottom=bottom,
                        width=bar_width,
                        color=color_map[lbl],
                        edgecolor='black'
                    )
                    bottom -= c

            # ---- Plot E12 differences for Oxidation ----
            if e12_diff_ox is not None and len(e12_diff_ox) > 0:
                for j, diff_val in enumerate(e12_diff_ox):
                    y_pos = 0.5 + j  # e.g., 0.5, 1.5, 2.5, ...
                    ax.text(
                        i,
                        y_pos,
                        f"{diff_val:+.1f}",
                        ha='center',
                        va='center',
                        color='red',
                        fontsize=8,
                        fontweight='bold'
                    )

            # ---- Plot E12 differences for Reduction ----
            if e12_diff_re is not None and len(e12_diff_re) > 0:
                for j, diff_val in enumerate(e12_diff_re):
                    y_pos = -0.5 - j  # e.g., -0.5, -1.5, -2.5, ...
                    ax.text(
                        i,
                        y_pos,
                        f"{diff_val:+.1f}",
                        ha='center',
                        va='center',
                        color='red',
                        fontsize=8,
                        fontweight='bold'
                    )

        # ---- Axis Formatting for Each Subplot ----
        ax.set_xticks(x_vals)
        ax.set_xticklabels(sub_idxs)
        ax.set_xlabel('Complex (index)')

        # Set a consistent y-limit across all subplots (optional)
        max_val = max(max_ox_global, max_re_global)
        ax.set_ylim(-max_val - 0.5, max_val + 0.5)

        # Show absolute labels for negative ticks (optional)
        ticks = range(-max_val, max_val + 1)
        ax.set_yticks(ticks)
        labels = [str(abs(t)) if t != 0 else '0' for t in ticks]
        ax.set_yticklabels(labels)
        ax.set_ylabel('Number of Sites')
        ax.grid(axis='y', linestyle='--')

    # ---- Build a Legend Once, Using the Last Axis or a Separate Legend ----
    patches = [mpatches.Patch(color=color_map[l], label=l) for l in all_labels]
    axs[0].legend(handles=patches, title='Site Label', loc='best', bbox_to_anchor=(0.99, 0.38))

    plt.tight_layout()
    plt.savefig('sites_E12.png', dpi=300, bbox_inches='tight')
    plt.show()


# def evaluate_classifier(train_file, valid_file):
#     def process_data(data):
#         def convert_to_argmax(x):
#             # Convert string representation of list to actual list
#             if isinstance(x, str):
#                 x = [int(i) for i in x.strip('[]').split(',')]
#             # Find the index of maximum value
#             max_val = max(x)
#             max_indices = [i for i, val in enumerate(x) if val == max_val]
#             # Return the first max index if there are multiple
#             return max_indices[0]
        
#         actuals = data['Actuals'].apply(convert_to_argmax)
#         predictions = data['Predictions'].apply(convert_to_argmax)
        
#         return np.array(actuals), np.array(predictions)

#     def evaluate_metrics(y_true, y_pred):
#         accuracy = accuracy_score(y_true, y_pred)
#         precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
#         return {
#             'Accuracy': accuracy,
#             'Precision': precision,
#             'Recall': recall,
#             'F1': f1
#         }

#     train_data = pd.read_csv(train_file)
#     valid_data = pd.read_csv(valid_file)    

#     train_actuals, train_predictions = process_data(train_data)
#     valid_actuals, valid_predictions = process_data(valid_data)



#     train_metrics = evaluate_metrics(train_actuals, train_predictions)
#     valid_metrics = evaluate_metrics(valid_actuals, valid_predictions)

#     print("訓練集評估指標:")
#     for metric, value in train_metrics.items():
#         print(f"{metric}: {value:.4f}")

#     print("\n驗證集評估指標:")
#     for metric, value in valid_metrics.items():
#         print(f"{metric}: {value:.4f}")

#     # 計算每個類別的指標
#     print("\n每個類別的詳細指標:")
#     for dataset_name, (y_true, y_pred) in [("訓練集", (train_actuals, train_predictions)), 
#                                         ("驗證集", (valid_actuals, valid_predictions))]:
#         print(f"\n{dataset_name}:")
#         precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)
#         for i, (p, r, f, s) in enumerate(zip(precision, recall, f1, support)):
#             print(f"類別 {i}:")
#             print(f"Precision: {p:.4f}")
#             print(f"Recall: {r:.4f}")
#             print(f"F1-score: {f:.4f}")
#             print(f"Support: {s}")



def evaluate_classifier(train_file, valid_file):
    # 讀取訓練集與驗證集的預測與實際值
    train_df = pd.read_csv(train_file)
    valid_df = pd.read_csv(valid_file)

    # 將字串轉換為 list
    train_actuals = [list(map(int, s.strip('"').split(','))) for s in train_df['Actuals']]
    train_preds = [list(map(int, s.strip('"').split(','))) for s in train_df['Predictions']]
    valid_actuals = [list(map(int, s.strip('"').split(','))) for s in valid_df['Actuals']]
    valid_preds = [list(map(int, s.strip('"').split(','))) for s in valid_df['Predictions']]

    # 將 list 轉換為 1D array
    train_actuals_1d = [item for sublist in train_actuals for item in sublist]
    train_preds_1d = [item for sublist in train_preds for item in sublist]
    valid_actuals_1d = [item for sublist in valid_actuals for item in sublist]
    valid_preds_1d = [item for sublist in valid_preds for item in sublist]

    # 計算訓練集整體指標
    acc_train = accuracy_score(train_actuals_1d, train_preds_1d)
    prec_train = precision_score(train_actuals_1d, train_preds_1d, average='weighted')
    rec_train = recall_score(train_actuals_1d, train_preds_1d, average='weighted')
    f1_train = f1_score(train_actuals_1d, train_preds_1d, average='weighted')

    # 計算驗證集整體指標
    acc_valid = accuracy_score(valid_actuals_1d, valid_preds_1d)
    prec_valid = precision_score(valid_actuals_1d, valid_preds_1d, average='weighted')
    rec_valid = recall_score(valid_actuals_1d, valid_preds_1d, average='weighted')
    f1_valid = f1_score(valid_actuals_1d, valid_preds_1d, average='weighted')

    # 計算每個類別的指標（訓練集）
    unique_classes = sorted(set(train_actuals_1d + train_preds_1d))
    class_metrics_train = {}
    for cls in unique_classes:
        class_metrics_train[cls] = {
            'precision': precision_score(train_actuals_1d, train_preds_1d, labels=[cls], average='micro'),
            'recall': recall_score(train_actuals_1d, train_preds_1d, labels=[cls], average='micro'),
            'f1': f1_score(train_actuals_1d, train_preds_1d, labels=[cls], average='micro')
        }

    # 計算每個類別的指標（驗證集）
    unique_classes = sorted(set(valid_actuals_1d + valid_preds_1d))
    class_metrics_valid = {}
    for cls in unique_classes:
        class_metrics_valid[cls] = {
            'precision': precision_score(valid_actuals_1d, valid_preds_1d, labels=[cls], average='micro'),
            'recall': recall_score(valid_actuals_1d, valid_preds_1d, labels=[cls], average='micro'),
            'f1': f1_score(valid_actuals_1d, valid_preds_1d, labels=[cls], average='micro')
        }

    return  acc_train, prec_train, rec_train, f1_train, acc_valid, prec_valid, rec_valid, f1_valid,
