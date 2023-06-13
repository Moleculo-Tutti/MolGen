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
    if config['batch_size'] < 128:
        mp.set_sharing_strategy('file_system') # will cause memory  leak
        print("maybe you will have memory leak")
    else : 
        mp.set_sharing_strategy('file_descriptor')#will work only if the number of batcj < 1024
        print("it can crash if too many file are open in the process")
    # Call the train_GNN1 function with the provided arguments

    TrainingGNN1 = TrainGNN1(config)
        
    TrainingGNN1.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)