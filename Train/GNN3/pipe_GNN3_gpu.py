from training_pipe_GNN3_utils import TrainGNN3
from pathlib import Path
import argparse
import torch
import json
from visualize import plot_history_GNN3
import torch.multiprocessing as mp


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Call the train_GNN3 function with the provided arguments
    mp.set_sharing_strategy('file_system') # Can cause memory leak

    TrainingGNN3 = TrainGNN3(config)
    TrainingGNN3.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)