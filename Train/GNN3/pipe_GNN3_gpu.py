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
    if config['batch_size'] < 256:
        mp.set_sharing_strategy('file_system') # will cause memory  leak 
    else : 
        print("you won't have memory leak but you can have too number of open files")
        mp.set_sharing_strategy('file_descriptor')#will work only if the number of batcj < 1024
    TrainingGNN3 = TrainGNN3(config)
    TrainingGNN3.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)