import json
import torch
import os,sys
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader   import DataLoader 
from torch.optim.lr_scheduler import ReduceLROnPlateau
from optparse import OptionParser

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
    parser.add_option("-i", "--input", dest="input", default='./data/imbalance/dataset_random_31.csv')
    parser.add_option("-t", "--test_size", dest="test_size", type=float, default=0.2)
    parser.add_option("--num_features", "--num_features", dest="num_features", default=153)
    parser.add_option("--output_size", dest="output_size", default=1)
    parser.add_option("--dropout", dest="dropout", type=float, default=0.2)
    parser.add_option("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_option("--num_epochs", dest="num_epochs", type=int, default=200)
    parser.add_option("--lr", dest="lr", type=float, default=0.0001)
    parser.add_option("--depth1", dest="depth1", type=int, default='3')
    parser.add_option("--depth2", dest="depth2", type=int, default='2')
    parser.add_option("--depth3", dest="depth3", type=int, default='2')
    parser.add_option("--anneal_rate", dest="anneal_rate", type=float, default=0.9)
    parser.add_option("--pretrain", dest="pretrain", default=True)
    opts, args = parser.parse_args()

    train_data, test_loader = load_traintest_dataset(file_path=opts.input, tensorize_fn=tensorize_with_subgraphs, batch_size=opts.batch_size)
    train_loader  = DataLoader(train_data, batch_size=opts.batch_size, shuffle=False)
    random_state = opts.input.split('_')[-1].split('.')[0]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model  = OMGNN_RNN(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, output_dim=opts.output_size, depth1=opts.depth1, depth2=opts.depth2, depth3=opts.depth3, dropout=opts.dropout).to(device)

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

    train_loss_history,train_reg_history,train_cla_history, test_loss_history,test_reg_history,test_cla_history, train_accuracy_history, test_accuracy_history = [], [], [], [], [], [], [], []
    
    # Early stopping parameters
    best_test_loss = float('inf')
    best_model_state = None
    
    for epoch in range(opts.num_epochs):
        model.train()
        total_loss, total_cla_loss, total_reg_loss, count = 0, 0, 0, 0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device)

            loss  = model(batch, device)

            loss.backward(retain_graph=True) 
            total_loss += loss.item()
            count += 1
            optimizer.step()

        train_loss = (total_loss / count)
        print(f"Epoch {epoch}, Train RMSE Loss: {train_loss:.4f}")

        train_loss , train_reg_loss, train_cla_loss, train_accuracy = OM.evaluate_model(model, train_loader, device)
        test_loss  , test_reg_loss , test_cla_loss , test_accuracy  = OM.evaluate_model(model, test_loader, device)
        train_loss_history.append(train_loss)
        train_reg_history.append(train_reg_loss)
        train_cla_history.append(train_cla_loss)
        test_loss_history.append(test_loss)
        test_reg_history.append(test_reg_loss)
        test_cla_history.append(test_cla_loss)
        train_accuracy_history.append(train_accuracy)
        test_accuracy_history.append(test_accuracy)
        
        scheduler.step()

        # Save best model after 200 epochs
        if epoch >= 150:
            if test_loss < best_test_loss:
                best_test_loss = test_loss
                best_model_state = model.state_dict()
                torch.save(best_model_state, os.path.join(f"checkpoint/best_model_{random_state}.pkl"))
                print(f"New best model saved at epoch {epoch} with test loss: {test_loss:.4f}")

    # Load the best model before final evaluation
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with test loss: {best_test_loss:.4f}")


    # Save model
    torch.save(model.state_dict(), os.path.join(f"checkpoint/model_{random_state}.pkl"))
