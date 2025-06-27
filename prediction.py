import torch
from   torch_geometric.loader import DataLoader
from   optparse   import OptionParser

import json
import os, sys
sys.path.append('/work/u7069586/E-hGNN_f/')
from utils.datautils import *
from utils.trainutils_v2 import *
from utils.chemutils  import *

from models.model import *

# load config.json
try:
    with open('./config.json', 'r') as f:
        config = json.load(f)
except FileNotFoundError:
    print("error: cannot find config.json file. use default value.")
    config = {} 

if __name__ == '__main__':
    parser = OptionParser()

    # use config value as default, if not, use the hardcoded value as backup
    parser.add_option("-i", "--input", dest="input",
                      default=config.get('input', './data/organo_rp_site_raw1.csv'))
    parser.add_option("--reaction", dest="reaction",
                      default=config.get('reaction', 'reduction'))
    parser.add_option("-t", "--test_size", dest="test_size", type=float,
                      default=float(config.get('test_size', 0.2))) 
    parser.add_option("--num_features", dest="num_features", type=int, 
                      default=int(config.get('num_features', 153)))
    parser.add_option("--output_size", dest="output_size", type=int, 
                      default=int(config.get('output_size', 1)))
    parser.add_option("--dropout", dest="dropout", type=float,
                      default=float(config.get('dropout', 0.3))) 
    parser.add_option("--batch_size", dest="batch_size", type=int,
                      default=int(config.get('batch_size', 1))) 
    parser.add_option("--num_epochs", dest="num_epochs", type=int,
                      default=int(config.get('num_epochs', 200)))
    parser.add_option("--lr", dest="lr", type=float,
                      default=float(config.get('lr', 0.001)))
    parser.add_option("--depth1", dest="depth1", type=int,
                      default=int(config.get('depth1', 3)))
    parser.add_option("--depth2", dest="depth2", type=int,
                      default=int(config.get('depth2', 1)))
    parser.add_option("--depth3", dest="depth3", type=int,
                      default=int(config.get('depth3', 2)))
    parser.add_option("--anneal_rate", dest="anneal_rate", type=float,
                      default=float(config.get('anneal_rate', 0.9))) 
    parser.add_option("--with_reaction", dest="with_reaction", action="store_true",
                    default=config.get('with_reaction', True))
    opts, args = parser.parse_args()


device = torch.device('cpu')
model  = OMGNN_RNN(node_dim=opts.num_features, bond_dim=11, hidden_dim=opts.num_features, output_dim=opts.output_size, depth1=opts.depth1, depth2=opts.depth2, depth3=opts.depth3, dropout=opts.dropout).to(device)

model_path_ = './checkpoint/model.pkl' 
if os.path.exists(model_path_):
    checkpoint = torch.load(model_path_, map_location=device) 
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"model weight loaded from {model_path_}")
else:
    print(f"warning: model weight file {model_path_} not found")


if opts.with_reaction:
    all_data = alldata_loader(file_path=opts.input, tensorize_fn=tensorize_with_subgraphs)
    loader   = DataLoader(all_data, batch_size=opts.batch_size, shuffle=False)
    count = 0
    for data in loader:
        count += 1
        data = data.to(device)
        print(data.name[0][0])
        result = model.sample(data, device)
        print(result)

# if not opts.with_reaction:
#     count = 0
#     for data in loader:
#         count += 1
#         data = data.to(device)
#         print(data.name[0][0])
#         result = model.sample_no_reaction(data, device)
#         print(result)