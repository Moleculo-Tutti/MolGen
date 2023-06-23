import torch
import numpy 

import sys
import os
import json
import gc

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from Model.GNN1 import ModelWithEdgeFeatures as GNN1
from Model.GNN1 import ModelWithNodeConcat as GNN1_node_concat
from Model.GNN2 import ModelWithEdgeFeatures as GNN2
from Model.GNN2 import ModelWithNodeConcat as GNN2_node_concat
from Model.GNN3 import ModelWithEdgeFeatures as GNN3
from Model.GNN3 import ModelWithgraph_embedding_modif as GNN3_embedding


def load_model(checkpoint_path, model):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    return model

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

def get_model_GNN3(config, encoding_size, edge_size):

    if config['graph_embedding']:
        return GNN3_embedding(in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                    hidden_channels_list=config["GCN_size"],
                    mlp_hidden_channels = config['mlp_hidden'],
                    edge_channels=edge_size, 
                    num_classes=edge_size,
                    use_dropout=config['use_dropout'],
                    size_info=config['use_size'],
                    max_size=config['max_size'])

    return GNN3(in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                hidden_channels_list=config["GCN_size"],
                edge_channels=edge_size, 
                use_dropout=config['use_dropout'])


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
        self.name = args.name_experiment
        self.training = args.training

        if args.encod == 'charged':
            encoding_size = 13
        elif args.encod == 'polymer':
            encoding_size = 8
        
        if args.keku:
            edge_size = 3
        else:
            edge_size = 4

        self.experiment_path = Path('..') / 'trained_models' / self.name


        folders = os.listdir(self.experiment_path)
        for folder in folders:
            if folder.startswith('GNN1'):
                GNN1_path = self.experiment_path / folder 
            elif folder.startswith('GNN2'):
                GNN2_path =self.experiment_path / folder
            elif folder.startswith('GNN3'):
                GNN3_path = self.experiment_path / folder

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

        GNN1_model = get_model_GNN1(config1, encoding_size, edge_size)
        GNN2_model = get_model_GNN2(config2, encoding_size, edge_size)
        GNN3_model = get_model_GNN3(config3, encoding_size, edge_size)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.GNN1 = load_model(GNN1_path, GNN1_model)
        self.GNN2 = load_model(GNN2_path, GNN2_model)
        self.GNN3 = load_model(GNN3_path, GNN3_model)

        self.GNN1.to(self.device)
        self.GNN2.to(self.device)
        self.GNN3.to(self.device)

        if self.training:
            self.GNN1.train()
            self.GNN2.train()
            self.GNN3.train()
        else:
            self.GNN1.eval()
            self.GNN2.eval()
            self.GNN3.eval()

