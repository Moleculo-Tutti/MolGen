import torch
import numpy 

import sys
import os
import json
import gc
from path import Path


cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

from DataPipeline.dataset import ZincSubgraphDatasetStep, custom_collate_passive_add_feature_GNN2, custom_collate_GNN2
from Model.GNN2 import ModelWithEdgeFeatures, ModelWithNodeConcat
from Model.GNN1 import ModelWithNodeConcat, ModelWithEdgeFeatures
from Model.GNN3 import ModelWithgraph_embedding_modif
from Model.metrics import pseudo_accuracy_metric, pseudo_recall_for_each_class, pseudo_precision_for_each_class

from FinalPipeline.utils import get_model_GNN1, get_model_GNN2, get_model_GNN3, load_model


def load_best_models(path):
    with open(path/'six_best_epochs.txt', 'r') as f:
        lines = f.readlines()
        epoch_values = [float(line.split(' ')[1]) for line in lines]
        best_line_index = epoch_values.index(max(epoch_values))
        loss_value = float(lines[best_line_index].split(' ')[-1])
    print('Loading best checkpoint number {} of the epoch {} with a loss of {}'.format(best_line_index, epoch_values[best_line_index], loss_value))
    checkpoint_path = path / 'history_training' / f'checkpoint_{best_line_index}.pt'
    return checkpoint_path


class Models_GNN():

    def get_gnn1(self):
        return self.GNN1
    
    def get_gnn2(self):
        return self.GNN2
    
    def get_gnn3(self):
        return self.GNN3
    def __init__(self,args):
        self.name = args.name_experiment
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

        if args.encod == 'charged':
            encoding_size = 13
        elif args.encod == 'polymer':
            encoding_size = 8
        
        if args.keku:
            edge_size = 3
            edge_mapping='kekulized'
        else:
            edge_size = 4
            edge_mapping='aromatic'

        
        self.encoding_size = encoding_size
        self.edge_size = edge_size
        self.encoding_type =args.encod
        self.feature_position = config1["feature_position"]

        self.score_list = config1["score_list"]
        self.desired_score_list = args.desired_score_list


        self.GNN1 = get_model_GNN1(config1, encoding_size, edge_size)
        self.GNN2 = get_model_GNN2(config2, encoding_size, edge_size)
        self.GNN3 = get_model_GNN3(config3, encoding_size, edge_size)

        self.optimizer_GNN1 = torch.optim.Adam(self.GNN1.parameters(), lr=config1["lr"])
        self.optimizer_GNN2 = torch.optim.Adam(self.GNN2.parameters(), lr=config2["lr"])
        self.optimizer_GNN3 = torch.optim.Adam(self.GNN3.parameters(), lr=config3["lr"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.GNN1_model, self.optimizer_GNN1 = load_model(GNN1_path, self.GNN1, self.optimizer_GNN1)
        self.GNN2_model, self.optimizer_GNN2 = load_model(GNN2_path, self.GNN2, self.optimizer_GNN2)
        self.GNN3_model, self.optimizer_GNN3 = load_model(GNN3_path, self.GNN3, self.optimizer_GNN3)

        self.GNN1_model.to(self.device)
        self.GNN2_model.to(self.device)
        self.GNN3_model.to(self.device)

        self.GNN1_model.eval()
        self.GNN2_model.eval()
        self.GNN3_model.eval()
        

    def forward(self, data):
        data = data.to(self.device)
        data = self.GNN1_model(data)
        data = self.GNN2_model(data)
        data = self.GNN3_model(data)
        return data
