from training_pipe_GNN1_utils import TrainGNN1
import argparse
import json
import torch.multiprocessing as mp


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    mp.set_sharing_strategy('file_system') # Can cause memory leak
    
    # Call the train_GNN1 function with the provided arguments

    TrainingGNN1 = TrainGNN1(config)
        
    TrainingGNN1.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)