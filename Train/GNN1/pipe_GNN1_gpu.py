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
    results = train_GNN1(config)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)