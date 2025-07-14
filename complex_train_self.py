import json
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader 
from optparse import OptionParser
import os, sys

from models.model import *
from models.pretrain_models import *
from utils.trainutils_v2 import *
from utils.chemutils import *
from utils.datautils import *

# ============================== #
#      Parsing input options     #
# ============================== # 
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default='data/example_TMCs_data.csv')
    parser.add_option("--reaction", dest="reaction", default='reduction')
    parser.add_option("--type", dest="type", default="multiple")  # single or multiple 
    parser.add_option("-t", "--test_size", dest="test_size", type=float, default=0.2)
    parser.add_option("--num_features", dest="num_features", type=int, default=153)   #37- effective charge #131
    parser.add_option("--output_size", dest="output_size", type=int, default=10)
    parser.add_option("--dropout", dest="dropout", type=float, default=0.3)
    parser.add_option("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_option("-e","--num_epochs", dest="num_epochs", type=int, default=200)
    parser.add_option("--lr", dest="lr", type=float, default=0.0001)
    parser.add_option("--depth1", dest="depth1", type=int, default=3)
    parser.add_option("--depth2", dest="depth2", type=int, default=2)
    parser.add_option("--depth3", dest="depth3", type=int, default=2)
    parser.add_option("--anneal_rate", dest="anneal_rate", type=float, default=0.9)
    parser.add_option("--model_type", dest="model_type", type=str, default='DMPNN')
    parser.add_option("--pretrain", dest="pretrain", default=True)
    parser.add_option("--global_graph", dest="global_graph", default=False)
    parser.add_option("--device", dest="device", type=str, default='cpu', help='使用的設備：cuda 或 cpu')
    parser.add_option("--label_column", dest="label_column", type=str, default='E12', help='標籤欄位名稱')
    opts, args = parser.parse_args()

    print(f"Loading data from: {opts.input}")
    
    # 測試數據加載
    try:
        train_loader, test_loader = dataloader_v2.load_TMCs_data(
            opts.input, 
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
    model  = complex_HGRN(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, cla_output_dim=opts.output_size, depth1=opts.depth1,depth2=opts.depth2,depth3=opts.depth3, dropout=opts.dropout).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opts.num_epochs, eta_min=1e-4)

    # Load Pretrain GCN1 model
    if opts.pretrain:
        pretrained_model      = OGNN_RNN_allmask(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, output_dim=opts.output_size,dropout=opts.dropout).to(device)
        pretrained_checkpoint = torch.load('./checkpoint/model_pretrain_gcn1-1-500.pkl', map_location='cpu')
        
        gcn1_state_dict = {k: v for k, v in pretrained_checkpoint.items() if 'GCN1' in k}
        pretrained_model.load_state_dict(gcn1_state_dict, strict=False)
        pretrained_model.eval()
        model.GCN1.load_state_dict(pretrained_model.GCN1.state_dict())
        print("Pretrain GCN1 loaded successfully")

    print(f"Using device: {device}")
    print("Starting training...")
    for epoch in range(opts.num_epochs):
        model.train()
        total_loss, total_cla_loss, total_reg_loss, count = 0, 0, 0, 0
        for i,batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)
            loss  = model(batch, device, global_graph=opts.global_graph)
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

        for data in test_loader:
            data = data.to(device)
            model.sample(data, device, global_graph=opts.global_graph)

    print("Training completed!")

    # save model
    torch.save(model.state_dict(), os.path.join("../checkpoint", f"TMCs_{opts.label_column}_model.pkl"))
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