from training_pipe_GNN1_utils import TrainGNN1
from training_pipe_GNN1_utils_multithread import TrainGNN1_multithread
import argparse
import json

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Call the train_GNN1 function with the provided arguments
    if config['use_multithreading']:
        TrainingGNN1 = TrainGNN1(config)
    else :
        TrainingGNN1 = TrainGNN1_multithread(config)
    TrainingGNN1.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)