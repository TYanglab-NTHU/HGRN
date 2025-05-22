import json
import torch
import torch.optim.lr_scheduler as lr_scheduler
from torch_geometric.loader import DataLoader 
from optparse import OptionParser
import os, sys

sys.path.append('/work/u7069586/E-hGNN_f/')
from models.pretrain_models import *
from utils.trainutils_v2 import *
from utils.chemutils import *
from utils.datautils import *

# ============================== #
#      Parsing input options     #
# ============================== # 
if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input", default='')
    parser.add_option("--i_organic", dest="input_organic", default='/work/u7069586/E-hGNN/data/one_organic_ip_ea_rp.csv')
    parser.add_option("--i_metal", dest="input_metal", default='/work/u7069586/E-hGNN_f/data/metal_ip_ea_rp.csv')
    parser.add_option("--i_element", dest="input_element", default='/work/u7069586/E-hGNN_f/data/without_metal_ip_ea.csv')
    parser.add_option("--reaction", dest="reaction", default='reduction')
    parser.add_option("--type", dest="type", default="multiple")  # single or multiple 
    parser.add_option("-t", "--test_size", dest="test_size", type=float, default=0.2)
    parser.add_option("--num_features", "--num_features", dest="num_features", default=153)   #37- effective charge #131
    parser.add_option("--output_size", dest="output_size", default=1)
    parser.add_option("--dropout", dest="dropout", type=float, default=0.3)
    parser.add_option("--batch_size", dest="batch_size", type=int, default=1)
    parser.add_option("--num_epochs", dest="num_epochs", type=int, default=500)
    parser.add_option("--lr", dest="lr", type=float, default=0.001)
    parser.add_option("--depth", dest="depth", type=int, default='3')
    parser.add_option("--anneal_rate", dest="anneal_rate", type=float, default=0.9)
    parser.add_option("--model_type", dest="model_type", type=str, default='DMPNN')
    parser.add_option("--device", dest="device", type=str, default='cuda', help='使用的設備：cuda 或 cpu')
    opts, args = parser.parse_args()

print("Loading data...")
train_dataset_metal, _   = OrganicMetal_potential.data_loader(opts.input_metal, opts.test_size, is_metal=True, features=opts.num_features)
train_dataset_element, _ = OrganicMetal_potential.data_loader(opts.input_element, opts.test_size, is_metal=True, features=opts.num_features)
train_dataset_organic, test_dataset_organic = OrganicMetal_potential.data_loader(opts.input_organic, opts.test_size, is_metal=False, features=opts.num_features)
print("Data loaded")
# train_dataset_metal, _   = OrganicMetal_potential.data_loader(opts.input_metal, opts.test_size, is_metal=True, features=opts.num_features)
# train_dataset_element, _ = OrganicMetal_potential.data_loader(opts.input_element, opts.test_size, is_metal=True, features=opts.num_features)
# train_dataset_organic, test_dataset_organic = OrganicMetal_potential.data_loader(opts.input_organic, opts.test_size, is_metal=False, features=opts.num_features)
train_dataset = train_dataset_metal + train_dataset_element + train_dataset_organic 
test_loader   = DataLoader(test_dataset_organic, batch_size=opts.batch_size, shuffle=False)
train_loader  = DataLoader(train_dataset, batch_size=opts.batch_size, shuffle=True)

device = torch.device(opts.device)
print("Creating model...")
model  = OGNN_RNN_allmask(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, output_dim=opts.output_size, depth=opts.depth ,dropout=opts.dropout, model_type=opts.model_type).to(device)
print("Model created")
optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=opts.anneal_rate)

train_loss_history,train_reg_history,train_cla_history, test_loss_history,test_reg_history,test_cla_history, train_accuracy_history, test_accuracy_history = [], [], [], [], [], [], [], []
print("Training model...")
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

    train_loss, train_reg_loss, train_cla_loss, train_accuracy =  OrganicMetal_potential.evaluate_model(model, train_loader, device, output_file=None)
    test_loss,  test_reg_loss,  test_cla_loss,  test_accuracy  =  OrganicMetal_potential.evaluate_model(model, test_loader, device, output_file=None)
    print(f"Epoch {epoch}, Train RMSE Loss: {train_loss:.4f}")
    train_loss_history.append(train_loss)
    test_loss_history.append(test_loss)
    train_reg_history.append(train_reg_loss)
    train_cla_history.append(train_cla_loss)
    test_reg_history.append(test_reg_loss)
    test_cla_history.append(test_cla_loss)
    train_accuracy_history.append(train_accuracy)
    test_accuracy_history.append(test_accuracy)

    df_loss = pd.DataFrame({'train_loss': train_loss_history, 'test_loss': test_loss_history, 'train_reg_loss': train_reg_history, 'test_reg_loss': test_reg_history, 'train_cla_loss': train_cla_history, 'test_cla_loss': test_cla_history, 'train_cla_accuray':train_accuracy_history, 'test_cla_accuray':test_accuracy_history})
    # df_loss = pd.DataFrame({'train_loss': train_loss_history, 'train_reg_loss': train_reg_history, 'train_cla_loss': train_cla_history, 'train_cla_accuray':train_accuracy_history})
    df_loss.to_csv(os.path.join(os.getcwd(), "loss.csv"))

# evaluate model
OrganicMetal_potential.evaluate_model(model, train_loader, device, output_file="train_pred_true.csv")
OrganicMetal_potential.evaluate_model(model, test_loader, device, output_file="valid_pred_true.csv")

# save model
torch.save(model.state_dict(), os.path.join(os.getcwd(), "model.pkl"))

# save config
config = {
    'input': opts.input_organic,
    'input_metal': opts.input_metal,
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

# parity plot
parity_plot("reg_train_pred_true.csv", "reg_valid_pred_true.csv")