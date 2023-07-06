import torch
import numpy 

import sys
import os
import json
import gc

from pathlib import Path

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from Model.GNN1 import ModelWithEdgeFeatures as GNN1
from Model.GNN2 import ModelWithEdgeFeatures as GNN2
from Model.GNN3 import ModelWithEdgeFeatures as GNN3
from Model.GNN3 import ModelWithgraph_embedding_modif as GNN3_embedding
from Model.GNN3 import ModelWithgraph_embedding_close_or_not_without_node_embedding as GNN3_closing



def load_model(checkpoint_path, model, optimizer):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def load_model_3(checkpoint_path, model_1, model_2, optimizer_1, optimizer_2):

    checkpoint = torch.load(checkpoint_path)
    model_1.load_state_dict(checkpoint['model_graph_state_dict'])
    model_2.load_state_dict(checkpoint['model_node_state_dict'])
    optimizer_1.load_state_dict(checkpoint['optimizer_graph_state_dict'])
    optimizer_2.load_state_dict(checkpoint['optimizer_node_state_dict'])

    
    return model_1, model_2, optimizer_1, optimizer_2

def get_model_GNN1(config, encoding_size, edge_size):

    return GNN1(in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                hidden_channels_list=config["GCN_size"],
                mlp_hidden_channels=config['mlp_hidden'],
                edge_channels=edge_size, 
                num_classes=encoding_size, 
                use_dropout=config['use_dropout'],
                size_info=config['use_size'],
                max_size=config['max_size'])

def get_model_GNN2(config, encoding_size, edge_size):

    return GNN2(in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                hidden_channels_list=config["GCN_size"],
                mlp_hidden_channels=config['mlp_hidden'],
                edge_channels=edge_size, 
                num_classes=edge_size, 
                size_info=config['use_size'],
                max_size=config['max_size'],
                use_dropout=config['use_dropout'])

def get_model_GNN3_bis(config, encoding_size, edge_size):

    GNN3_1 = GNN3_closing(
                in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                hidden_channels_list=config["GCN_size"],
                mlp_hidden_channels = config['mlp_hidden'],
                edge_channels=edge_size,
                num_classes=1,
                use_dropout=config['use_dropout'],
                size_info=config['use_size'],
                max_size=config['max_size'])
    GNN3_2 = GNN3_embedding(
                in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                hidden_channels_list=config["GCN_size"],
                mlp_hidden_channels = config['mlp_hidden'],
                edge_channels=edge_size,
                num_classes=2,
                use_dropout=config['use_dropout'],
                size_info=config['use_size'],
                max_size=config['max_size'])
    return GNN3_1, GNN3_2



def load_best_models(path):
    with open(path/'six_best_epochs.txt', 'r') as f:
        lines = f.readlines()
        epoch_values = [float(line.split(' ')[1]) for line in lines]
        best_line_index = epoch_values.index(max(epoch_values))
        loss_value = float(lines[best_line_index].split(' ')[-1])
    print('Loading best checkpoint number {} of the epoch {} with a loss of {}'.format(best_line_index, epoch_values[best_line_index], loss_value))
    checkpoint_path = path / 'history_training' / f'checkpoint_{best_line_index}.pt'
    return checkpoint_path


class Model_GNNs():

    def __init__(self,args):
        """
        Initialize the model
        input:
        args: arguments of the training, name of the experiment, training or not, encoding type, kekulization or not

        return: None
        """
            # Read the configs as json 

        experiment_name = args.exp_name

        experiment_path = Path('..') / 'trained_models' / experiment_name

        
        # List the folders in the experiment path

        folders = os.listdir(experiment_path)
        for folder in folders:
            if folder.startswith('GNN1'):
                GNN1_path = experiment_path / folder 
            elif folder.startswith('GNN2'):
                GNN2_path = experiment_path / folder
            elif folder.startswith('GNN3'):
                GNN3_path = experiment_path / folder

        # Read the config as a json

        config1_path = GNN1_path / 'parameters.json'
        config2_path = GNN2_path / 'parameters.json'
        config3_path = GNN3_path / 'parameters.json'

        with open(config1_path, 'r') as f:
            config1 = json.load(f)
        with open(config2_path, 'r') as f:
            config2 = json.load(f)
        with open(config3_path, 'r') as f:
            config3 = json.load(f)
        
        # Open the models with the best loss on the validation set


        GNN1_path = load_best_models(GNN1_path)
        GNN2_path = load_best_models(GNN2_path)
        GNN3_path = load_best_models(GNN3_path)

        print(GNN1_path, GNN2_path, GNN3_path)

        if args.encod == 'charged':
            encoding_size = 13
        elif args.encod == 'polymer':
            encoding_size = 8
        
        if args.keku:
            edge_size = 3
        else:
            edge_size = 4

        self.GNN1 = get_model_GNN1(config1, encoding_size, edge_size)
        self.GNN2 = get_model_GNN2(config2, encoding_size, edge_size)
        self.GNN3_1, self.GNN3_2 = get_model_GNN3_bis(config3, encoding_size, edge_size)

        self.optimizer_GNN1 = torch.optim.Adam(self.GNN1.parameters(), lr=config1["lr"])
        self.optimizer_GNN2 = torch.optim.Adam(self.GNN2.parameters(), lr=config2["lr"])
        self.optimizer_GNN3_1 = torch.optim.Adam(self.GNN3_1.parameters(), lr=config3["lr"])
        self.optimizer_GNN3_2 = torch.optim.Adam(self.GNN3_2.parameters(), lr=config3["lr"])


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.GNN1_model, self.optimizer_GNN1 = load_model(GNN1_path, self.GNN1, self.optimizer_GNN1)
        self.GNN2_model, self.optimizer_GNN2 = load_model(GNN2_path, self.GNN2, self.optimizer_GNN2)
        self.GNN3_1_model, self.GNN3_2_model, self.optimizer_GNN3_1, self.optimizer_GNN3_2 = load_model_3(GNN3_path, self.GNN3_1, self.GNN3_2, self.optimizer_GNN3_1, self.optimizer_GNN3_2)

        self.GNN1_model.to(self.device)
        self.GNN2_model.to(self.device)
        self.GNN3_1_model.to(self.device)
        self.GNN3_2_model.to(self.device)


        self.GNN1_model.eval()
        self.GNN2_model.eval()
        self.GNN3_1_model.eval()
        self.GNN3_2_model.eval()