import json
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader 
from optparse import OptionParser
import os, sys

import sys
from datautils import *

sys.path.append('../')
from models.pretrain_models import *
from utils.trainutils_v2 import *
from utils.chemutils import *


# ============================== #
#      Parsing input options     #
# ============================== # 
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default='')
    parser.add_option("--i_organic", dest="input_organic", default='pka_data.csv')
    parser.add_option("--reaction", dest="reaction", default='reduction')
    parser.add_option("--type", dest="type", default="multiple")  # single or multiple 
    parser.add_option("-t", "--test_size", dest="test_size", type=float, default=0.2)
    parser.add_option("--num_features", dest="num_features", type=int, default=153)   #37- effective charge #131
    parser.add_option("--output_size", dest="output_size", type=int, default=10)
    parser.add_option("--dropout", dest="dropout", type=float, default=0.3)
    parser.add_option("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_option("-e","--num_epochs", dest="num_epochs", type=int, default=300)
    parser.add_option("--lr", dest="lr", type=float, default=0.001)
    parser.add_option("--depth", dest="depth", type=int, default=3)
    parser.add_option("--anneal_rate", dest="anneal_rate", type=float, default=0.9)
    parser.add_option("--model_type", dest="model_type", type=str, default='DMPNN')
    parser.add_option("--device", dest="device", type=str, default='cuda', help='使用的設備：cuda 或 cpu')
    parser.add_option("--label_column", dest="label_column", type=str, default='pka_values', help='標籤欄位名稱')
    opts, args = parser.parse_args()

    print(f"Loading data from: {opts.input_organic}")
    
    # 測試數據加載
    try:
        train_loader, test_loader = dataloader_v2.load_data(
            opts.input_organic, 
            opts.test_size, 
            is_metal=False, 
            features=opts.num_features, 
            label_columns=[opts.label_column]
        )
        print(f"✓ Data loaded successfully!")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        sys.exit(1)

    device = torch.device(opts.device)
    model  = Organic_GRN(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, cla_dim=opts.output_size, depth=opts.depth, dropout=opts.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=opts.anneal_rate)

    print(f"Using device: {device}")
    print("Starting training...")
    for epoch in range(opts.num_epochs):
        model.train()
        total_loss, total_cla_loss, total_reg_loss, count = 0, 0, 0, 0
        for i,batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            loss  = model(batch, device)
            loss.backward()
            
            total_loss += loss.item()
            count += 1
            optimizer.step()

        if count > 0:
            avg_loss = total_loss / count
            print(f"Epoch {epoch+1}/{opts.num_epochs}, Average Loss: {avg_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{opts.num_epochs}, No valid batches")
        
        scheduler.step()

    print("Training completed!")
    
    # 測試推理
    print("Testing inference...")
    model.eval()
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            if i >= 10:  # 只測試前3個batch
                break
            try:
                batch = batch.to(device)
                predictions = model.sample(batch, device)
                print(predictions)
                print(batch.ys)
                print(f"✓ Test batch {i+1}: {len(predictions)} predictions")
            except Exception as e:
                print(f"✗ Error in test batch {i+1}: {e}")

        predictions, targets = collect_matched_predictions_and_targets(model, test_loader, device)
        
        print(f"Collected {len(predictions)} predictions and {len(targets)} targets")
        
        # 繪製parity plot
        print("Creating parity plot...")
        r2, rmse, mae = plot_parity(predictions, targets, "pKa 預測 Parity Plot", "pka_parity_plot.png")
        
        # 繪製殘差圖
        print("Creating residuals plot...")
        plot_residuals(predictions, targets, "pka_residuals_plot.png")
        
        print(f"✓ 圖表已保存！")
        print(f"  R² = {r2:.4f}")
        print(f"  RMSE = {rmse:.4f}")
        print(f"  MAE = {mae:.4f}")


    # save model
    torch.save(model.state_dict(), os.path.join("../checkpoint", "pretrain_model.pkl"))
    print("Model saved!")

    # save config
    config = {
        'input': opts.input_organic,
        'num_features': opts.num_features,
        'output_size': opts.output_size,
        'dropout': opts.dropout,
        'batch_size': opts.batch_size,
        'num_epochs': opts.num_epochs,
        'lr': opts.lr,
        'depth': opts.depth,
        'anneal_rate': opts.anneal_rate,
        'model_type': opts.model_type
    }
    with open(os.path.join(os.getcwd(), "config.json"), 'w') as f:
        json.dump(config, f, indent=4)