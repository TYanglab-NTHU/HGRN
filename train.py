import os,sys
import json
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader 
from optparse   import OptionParser

from models.pretrain_models import *
from models.model import *

from utils.trainutils_v2 import *
from utils.chemutils import *
from utils.datautils import *

# ============================== #
#      Parsing input options     #
# ============================== # 
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default='/work/u7069586/E-hGNN/data/old_organo_rp_site_raw1.csv')
    parser.add_option("--pretrain_path", dest="pretrain_path", default='/work/u7069586/OMGNN-OROP/scripts/')
    parser.add_option("--reaction", dest="reaction", default='reduction')
    parser.add_option("--type", dest="type", default="multiple")  # single or multiple 
    parser.add_option("-t", "--test_size", dest="test_size", type=float, default=0.2)
    parser.add_option("--num_features", "--num_features", dest="num_features", default=153)
    parser.add_option("--output_size", dest="output_size", default=1)
    parser.add_option("--dropout", dest="dropout", type=float, default=0.2)
    parser.add_option("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_option("--num_epochs", dest="num_epochs", type=int, default=250)
    parser.add_option("--lr", dest="lr", type=float, default=0.001)
    parser.add_option("--depth1", dest="depth1", type=int, default='3')
    parser.add_option("--depth2", dest="depth2", type=int, default='2')
    parser.add_option("--depth3", dest="depth3", type=int, default='2')
    parser.add_option("--anneal_rate", dest="anneal_rate", type=float, default=0.9)
    parser.add_option("--pretrain", dest="pretrain", default=True)
    opts, args = parser.parse_args()

train_data, test_loader = data_loader(file_path=opts.input,tensorize_fn=tensorize_with_subgraphs,batch_size=opts.batch_size,test_size=opts.test_size)
train_loader  = DataLoader(train_data, batch_size=opts.batch_size, shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model  = OMGNN_RNN(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, output_dim=opts.output_size, depth1=opts.depth1, depth2=opts.depth2, depth3=opts.depth3, dropout=opts.dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=opts.anneal_rate)


"Load Pretrain GCN1 model"
if opts.pretrain:
    pretrained_model      = OGNN_RNN_allmask(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, output_dim=opts.output_size,dropout=opts.dropout).to(device)
    pretrained_checkpoint = torch.load('./checkpoint/model_pretrain_gcn1-1-500.pkl', map_location='cpu')
    
    # 只載入 GCN1 相關的參數
    gcn1_state_dict = {k: v for k, v in pretrained_checkpoint.items() if 'GCN1' in k}
    pretrained_model.load_state_dict(gcn1_state_dict, strict=False)
    pretrained_model.eval()
    model.GCN1.load_state_dict(pretrained_model.GCN1.state_dict())
    print("Pretrain GCN1 loaded successfully")

train_loss_history,train_reg_history,train_cla_history, test_loss_history,test_reg_history,test_cla_history, train_accuracy_history, test_accuracy_history = [], [], [], [], [], [], [], []
for epoch in range(opts.num_epochs):
    model.train()
    total_loss, total_cla_loss, total_reg_loss, count = 0, 0, 0, 0
    for i,batch in enumerate(train_loader):
        optimizer.zero_grad()
        batch = batch.to(device)
        loss  = model(batch, device)

        loss.backward(retain_graph=True) 
        total_loss += loss.item()
        count += 1
        optimizer.step()
    train_loss = (total_loss / count)
    print(f"Epoch {epoch}, Train RMSE Loss: {train_loss:.4f}")

    train_loss , train_reg_loss, train_cla_loss, train_accuracy = OM.evaluate_model(model, train_loader, device, output_file=None)
    test_loss  , test_reg_loss , test_cla_loss , test_accuracy  = OM.evaluate_model(model, test_loader, device,  output_file=None)
    train_loss_history.append(train_loss)
    train_reg_history.append(train_reg_loss)
    train_cla_history.append(train_cla_loss)
    test_loss_history.append(test_loss)
    test_reg_history.append(test_reg_loss)
    test_cla_history.append(test_cla_loss)
    train_accuracy_history.append(train_accuracy)
    test_accuracy_history.append(test_accuracy)
    
    df_loss = pd.DataFrame({'train_loss': train_loss_history, 'test_loss': test_loss_history, 'train_reg_loss': train_reg_history, 'test_reg_loss': test_reg_history, 'train_cla_loss': train_cla_history, 'test_cla_loss': test_cla_history, 'train_cla_accuray':train_accuracy_history, 'test_cla_accuray':test_accuracy_history})
    df_loss.to_csv(os.path.join(os.getcwd(), "loss.csv"))


# evaluate model
OM.evaluate_model(model, train_loader, device, output_file="train_pred_true.csv")
OM.evaluate_model(model, test_loader, device, output_file="valid_pred_true.csv")

torch.save(model.state_dict(), os.path.join(os.getcwd(), "model.pkl"))

# save config
config = {
    'input': opts.input,
    'pretrain_path': opts.pretrain_path,
    'num_features': opts.num_features,
    'output_size': opts.output_size,
    'dropout': opts.dropout,
    'batch_size': opts.batch_size,
    'test_size': opts.test_size,
    'num_epochs': opts.num_epochs,
    'lr': opts.lr,
    'depth': opts.depth1,
    'depth2': opts.depth2,
    'depth3': opts.depth3,
    'anneal_rate': opts.anneal_rate,
    'pretrain': opts.pretrain
}
with open(os.path.join(os.getcwd(), "config.json"), 'w') as f:
    json.dump(config, f, indent=4)