from training_pipe_GNN3_utils import train_GNN3
from pathlib import Path
import argparse
import torch
import json
from visualize import plot_history_GNN3

def main(args):
    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Call the train_GNN3 function with the provided arguments
    results = train_GNN3(config["name"], Path(config["datapath_train"]), Path(config["datapath_val"]), 
                         config["n_epochs"], config["encoding_size"], config["GCN_size"], config["edge_size"],
                         feature_position=False, use_dropout=False, lr=0.0001, print_bar=False, 
                         graph_embedding=config["graph_embedding"], mlp_hidden=config["mlp_hidden"], 
                         batch_size=config["batch_size"], modif_accelerate=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)