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
    # Go to the file of the experience you want to continue and load the best model
    path = Path('experiments/'+args.name_exp)
    # Open the file six_best_epoch.txt and read the number of the last epoch
    with open(path/'six_best_epochs.txt', 'r') as f:
        lines = f.readlines()
        epoch_values = [float(line.split(' ')[1]) for line in lines]
        best_line_index = epoch_values.index(max(epoch_values))
        loss_value = float(lines[best_line_index].split(' ')[-1])

    #load the best model and its adam optimizer
    print('Loading best checkpoint number {} of the epoch {} with a loss of {}'.format(best_line_index, epoch_values[best_line_index], loss_value))

    checkpoint = torch.load(path/'history_training/checkpoint_{}.pt'.format(best_line_index))

    # Call the train_GNN3 function with the provided arguments
    file_path_config = path / 'parameters.json'

    # Ouvrir le fichier JSON et charger la configuration
    with open(file_path_config, "r") as file:
        config = json.load(file)    
    
    # Call the train_GNN3 function with the provided arguments
    mp.set_sharing_strategy('file_system') # Can cause memory leak
    
    TrainingGNN3 = TrainGNN3(config, continue_training= True, checkpoint = checkpoint)
    TrainingGNN3.train()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name_exp', default='my_exp', help='Path to the experience you want to continue')
    args = parser.parse_args()
    main(args)