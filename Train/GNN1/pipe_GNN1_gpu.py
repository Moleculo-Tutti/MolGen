from training_pipe_GNN1_utils import TrainGNN1
from training_pipe_GNN1_utils_multithread import TrainGNN1_multithread
import argparse
import json
import torch.multiprocessing as mp


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)
    if config['batch_size'] < 256:
        mp.set_sharing_strategy('file_system') # will cause memory  leak 
    else : 
        mp.set_sharing_strategy('file_descriptor')#will work only if the number of batcj < 1024
    # Call the train_GNN1 function with the provided arguments

    if config['use_multithreading_on_epochs']: #not on batches
        TrainingGNN1 = TrainGNN1_multithread(config)
        
    else :
        TrainingGNN1 = TrainGNN1(config)
        
    TrainingGNN1.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)