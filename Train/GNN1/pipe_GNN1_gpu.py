from training_pipe_GNN1_utils import train_GNN1
from pathlib import Path
import argparse
import torch
import json
from visualize import plot_history_GNN3

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Call the train_GNN1 function with the provided arguments
    results = train_GNN1(name = config["name"],
                         datapath_train = config["datapath_train"],
                         datapath_val = config["datapath_val"],
                         n_epochs = config["n_epochs"],
                         encoding_size = config["encoding_size"],
                         GCN_size = config["GCN_size"],
                         mlp_size = config["mlp_size"],
                         edge_size = config["edge_size"],
                         batch_size = config["batch_size"],
                         num_workers = config["num_workers"],
                         feature_position = True, 
                         use_dropout = False, 
                         lr = 0.0001 , 
                         print_bar = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)