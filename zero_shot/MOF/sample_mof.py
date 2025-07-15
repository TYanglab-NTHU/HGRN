import json
import torch
from torch_geometric.data     import Data
from torch_geometric.loader   import DataLoader
from optparse   import OptionParser
import torch.nn.functional as F 

from model import *

from datautils_mof       import *
from periodic_chemutils  import *

import warnings
warnings.filterwarnings("ignore")

# load config.json
try:
    with open( '/config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    config = {}

if __name__ == '__main__':
    parser = OptionParser()

    # use config value as default, if not in config, use hardcoded value as backup
    parser.add_option("-i", "--input", dest="input",
                      default=config.get('input', '../../data/1st_TMCs_E12.csv'))
    parser.add_option("--reaction", dest="reaction",
                      default=config.get('reaction', 'reduction'))
    parser.add_option("--type", dest="type",
                      default=config.get('type', 'multiple')) # single or multiple
    parser.add_option("-t", "--test_size", dest="test_size", type=float,
                      default=float(config.get('test_size', 0.2))) # 注意類型轉換
    parser.add_option("--num_features", dest="num_features", type=int, # 注意類型轉換
                      default=int(config.get('num_features', 153)))
    parser.add_option("--output_size", dest="output_size", type=int, # 注意類型轉換
                      default=int(config.get('output_size', 1)))
    parser.add_option("--dropout", dest="dropout", type=float,
                      default=float(config.get('dropout', 0.3))) # 注意類型轉換
    parser.add_option("--batch_size", dest="batch_size", type=int,
                      default=int(config.get('batch_size', 1))) # 注意類型轉換
    parser.add_option("--num_epochs", dest="num_epochs", type=int,
                      default=int(config.get('num_epochs', 200))) # 注意類型轉換
    parser.add_option("--lr", dest="lr", type=float,
                      default=float(config.get('lr', 0.001))) # 注意類型轉換
    parser.add_option("--depth1", dest="depth1", type=int,
                      default=int(config.get('depth1', 3))) # 注意類型轉換
    parser.add_option("--depth2", dest="depth2", type=int,
                      default=int(config.get('depth2', 2))) # 注意類型轉換
    parser.add_option("--depth3", dest="depth3", type=int,
                      default=int(config.get('depth3', 2))) # 注意類型轉換
    parser.add_option("--anneal_rate", dest="anneal_rate", type=float,
                      default=float(config.get('anneal_rate', 0.9))) # 注意類型轉換
    # 對於布林值，需要更仔細的處理，因為 config.get 可能返回字符串 'True' 或 'False'
    pretrain_default = config.get('pretrain', True)

    opts, args = parser.parse_args()

    dataset = data_loader('./test_mof.csv', 1)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = OMGNN_RNN(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, output_dim=opts.output_size, depth1=opts.depth1, depth2=opts.depth2, depth3=opts.depth3, dropout=opts.dropout).to(device)

    model_path_ = './model.pkl' # 或者從 opts 或 config 獲取
    if os.path.exists(model_path_):
        checkpoint = torch.load(model_path_, map_location=device) # 直接加載到目標 device
        model.load_state_dict(checkpoint)
        model.eval()
        print(f"model weights loaded from {model_path_} successfully.")
    else:
        print(f"Warning: model weights file {model_path_} not found. Model will use randomly initialized weights.")

    for i ,batch in enumerate(loader):
        batch = batch.to(device)
        print(f"{i+1}")
        output = model.sample_no_reaction(batch, device)
        print(f"reduction E12: {output[2]}")
        print(f"oxidation E12: {output[5]}")
