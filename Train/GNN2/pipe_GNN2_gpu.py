from training_pipe_GNN2_utils import TrainGNN2
import argparse
import json
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main(args):
    # Load config file
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Call the train_GNN1 function with the provided arguments
    TrainingGNN2 = TrainGNN2(config)
    TrainingGNN2.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    main(args)