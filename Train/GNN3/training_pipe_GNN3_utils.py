import torch
import random

import pandas as pd

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

from torch.optim import AdamW

import torch_geometric.transforms as T

from torch_geometric.data import Batch

from torch.utils.data import DataLoader

from pathlib import Path

from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import numpy as np
import time


import sys
import os

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

from DataPipeline.dataset import ZincSubgraphDatasetStep, custom_collate_GNN3
from Model.GNN3 import ModelWithEdgeFeatures, ModelWithgraph_embedding_modif
from Model.metrics import  pseudo_accuracy_metric_gnn3


def train_one_epoch(loader, model, size_edge, device, optimizer, criterion, epoch_metric, print_bar = False):
    model.train()
    total_loss = 0
    num_correct = 0
    num_output = torch.zeros(size_edge)  # Already on CPU
    num_labels = torch.zeros(size_edge)  # Already on CPU
    total_graphs_processed = 0
    global_cycles_created = 0
    global_well_placed_cycles = 0
    global_well_type_cycles = 0
    global_cycles_missed = 0
    global_cycles_shouldnt_created = 0
    global_num_wanted_cycles = 0

    if print_bar:
        progress_bar = tqdm_notebook(loader, desc="Training", unit="batch")
    else:
        progress_bar = tqdm(loader, desc="Training", unit="batch")
    
    for batch_idx, batch in enumerate(progress_bar):

        data = batch[0].to(device)
        node_labels = batch[1].to(device)
        mask = batch[2].to(device)
        
        optimizer.zero_grad()
        out = model(data)

        # Convert node_labels to class indices
        
        node_labels = node_labels.to(device)
        mask = mask.to(device)

        # Use node_labels_indices with CrossEntropyLoss
        #loss = criterion(out, node_labels, mask)
        loss = criterion(out[mask], node_labels[mask])
    
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs
        # Add softmax to out
        softmax_out = F.softmax(out, dim=1)

        if epoch_metric:

            cycles_created, well_placed_cycles , well_type_cycles, cycles_missed, cycles_shouldnt_created, num_wanted_cycles = pseudo_accuracy_metric_gnn3(data,out,node_labels,mask, edge_size = size_edge)      
            # Calculate metrics and move tensors to CPU
            num_output += torch.sum(softmax_out[mask], dim=0).detach().cpu()
            num_labels += torch.sum(node_labels[mask], dim=0).detach().cpu()
            global_cycles_created +=cycles_created
            global_well_placed_cycles += well_placed_cycles
            global_well_type_cycles += well_type_cycles
            global_cycles_missed += cycles_missed
            global_cycles_shouldnt_created += cycles_shouldnt_created
            global_num_wanted_cycles += num_wanted_cycles
            
            
            loss_value = total_loss / (data.num_graphs * (progress_bar.last_print_n + 1))
            total_graphs_processed += data.num_graphs
            
            denominator =global_cycles_created+global_cycles_shouldnt_created+global_num_wanted_cycles
            if denominator == 0:
                f1_score = 1
            else:
                f1_score = 2 * global_cycles_created / denominator

            if global_cycles_created == 0:
                conditional_precision_placed = 1
            else:
                conditional_precision_placed = global_well_placed_cycles/(global_cycles_created)
            if print_bar:
                progress_bar.set_postfix(loss=loss_value, avg_num_output=num_output / total_graphs_processed, avg_num_labels=num_labels / total_graphs_processed,
                pseudo_precision = global_cycles_created/(global_cycles_created+global_cycles_shouldnt_created),  pseudo_recall = global_cycles_created/global_num_wanted_cycles ,
                pseudo_recall_placed = global_well_placed_cycles/global_num_wanted_cycles, pseudo_recall_type = global_well_type_cycles/global_num_wanted_cycles, 
                conditional_precision_placed = conditional_precision_placed, f1_score = f1_score)
            
    if epoch_metric:
        return (
            total_loss / len(loader.dataset),
            num_output / total_graphs_processed,
            num_labels / total_graphs_processed, 
            global_cycles_created/(global_cycles_created+global_cycles_shouldnt_created), 
            global_cycles_created/global_num_wanted_cycles , 
            global_well_placed_cycles/global_num_wanted_cycles, 
            global_well_type_cycles/global_num_wanted_cycles,
            conditional_precision_placed,
            f1_score)
    
    else:
        return total_loss / len(loader.dataset), None, None, None, None, None, None, None, None


def eval_one_epoch(loader, model, size_edge, device, criterion, print_bar=False, val_metric_size=1):
    model.eval()
    total_loss = 0
    num_correct = 0
    num_output = torch.zeros(size_edge)  # Already on CPU
    num_labels = torch.zeros(size_edge)  # Already on CPU
    total_graphs_processed = 0
    global_cycles_created = 0
    global_well_placed_cycles = 0
    global_well_type_cycles = 0
    global_cycles_missed = 0
    global_cycles_shouldnt_created = 0
    global_num_wanted_cycles = 0

    if print_bar:
        progress_bar = tqdm_notebook(loader, desc="Evaluating", unit="batch")
    else:
        progress_bar = tqdm(loader, desc="Evaluating", unit="batch")

    for i in tqdm(range(val_metric_size)):

        with torch.no_grad():
            for batch_idx, batch in enumerate(progress_bar):
                data = batch[0].to(device)
                node_labels = batch[1].to(device)
                mask = batch[2].to(device)

                out = model(data)

                node_labels = node_labels.to(device)
                mask = mask.to(device)

                loss = criterion(out[mask], node_labels[mask])

                # Add softmax to out
                softmax_out = F.softmax(out, dim=1)

                cycles_created, well_placed_cycles, well_type_cycles, cycles_missed, cycles_shouldnt_created, num_wanted_cycles = pseudo_accuracy_metric_gnn3(
                    data, out, node_labels, mask, edge_size=size_edge)

                num_output += torch.sum(softmax_out[mask], dim=0).detach().cpu()
                num_labels += torch.sum(node_labels[mask], dim=0).detach().cpu()
                global_cycles_created += cycles_created
                global_well_placed_cycles += well_placed_cycles
                global_well_type_cycles += well_type_cycles
                global_cycles_missed += cycles_missed
                global_cycles_shouldnt_created += cycles_shouldnt_created
                global_num_wanted_cycles += num_wanted_cycles

                total_loss += loss.item() * data.num_graphs
                total_graphs_processed += data.num_graphs

    denominator = global_cycles_created + global_cycles_shouldnt_created + global_num_wanted_cycles
    if denominator == 0:
        f1_score = 1
    else:
        f1_score = 2 * global_cycles_created / denominator

    if global_cycles_created == 0:
        conditional_precision_placed = 1
    else:
        conditional_precision_placed = global_well_placed_cycles / (global_cycles_created)

    return (
        total_loss / len(loader.dataset),
        num_output / total_graphs_processed,
        num_labels / total_graphs_processed,
        global_cycles_created / (global_cycles_created + global_cycles_shouldnt_created),
        global_cycles_created / global_num_wanted_cycles,
        global_well_placed_cycles / global_num_wanted_cycles,
        global_well_type_cycles / global_num_wanted_cycles,
        conditional_precision_placed,
        f1_score)



class TrainGNN3():
    def __init__(self, config):
        self.config = config
        self.name = config['name']
        self.datapath_train = config['datapath_train']
        self.datapath_val = config['datapath_val']
        self.n_epochs = config['n_epochs']
        self.GCN_size = config['GCN_size']
        self.mlp_hidden = config['mlp_hidden']
        self.batch_size = config['batch_size']
        self.feature_position = config['feature_position']
        self.use_dropout = config['use_dropout']
        self.lr = config['lr']
        self.graph_embedding = config['graph_embedding']
        self.print_bar = config['print_bar']
        self.num_workers = config['num_workers']
        self.every_epoch_save = config['every_epoch_save']
        self.every_epoch_metric = config['every_epoch_metric']
        self.val_metric_size = config['val_metric_size']
        self.max_size = config['max_size']
        self.size_info = config['use_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Training on {self.device}")

        print(f"Loading data...")
        self.loader_train, self.loader_val, self.model, self.encoding_size, self.edge_size = self.load_data_model()
        print(f"Data loaded")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.training_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector','pseudo_precision', 'pseudo_recall' , 'pseudo_recall_placed', 'pseudo_recall_type','conditionnal_precision_placed', 'f1_score'])
        self.eval_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector','pseudo_precision', 'pseudo_recall' , 'pseudo_recall_placed', 'pseudo_recall_type','conditionnal_precision_placed', 'f1_score'])

        self.prepare_saving()

        # Store the 6 best models
        self.six_best_eval_loss = [(0, float('inf'))] * 6

    def load_data_model(self):
        # Load the data
        dataset_train = ZincSubgraphDatasetStep(self.datapath_train, GNN_type=3, feature_position=self.feature_position)
        dataset_val = ZincSubgraphDatasetStep(self.datapath_val, GNN_type=3, feature_position=self.feature_position)
        
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers, collate_fn=custom_collate_GNN3)
        loader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers, collate_fn=custom_collate_GNN3)

        encoding_size = dataset_train.encoding_size
        edge_size = dataset_train.edge_size

        
        # Load the model
        if self.graph_embedding:
            model = ModelWithgraph_embedding_modif(in_channels = encoding_size + int(self.feature_position), # We increase the input size to take into account the feature position
                                                hidden_channels_list=self.GCN_size,
                                                mlp_hidden_channels=self.mlp_hidden,
                                                edge_channels=edge_size, 
                                                num_classes=edge_size,
                                                use_dropout=self.use_dropout,
                                                size_info=self.size_info,
                                                max_size=self.max_size)
        else:
            model = ModelWithEdgeFeatures(in_channels=encoding_size + int(self.feature_position), # We increase the input size to take into account the feature position
                                        hidden_channels_list=self.GCN_size,
                                        edge_channels=edge_size, 
                                        use_dropout=self.use_dropout,
                                        max_size=self.max_size)
        
        return loader_train, loader_val, model.to(self.device), encoding_size, edge_size
    
    def prepare_saving(self):
        self.directory_path_experience = os.path.join("./experiments", self.name)
        self.directory_path_epochs = os.path.join(self.directory_path_experience,"history_training")

        if not os.path.exists(self.directory_path_experience):
            # Create the directory if it doesn't exist
            os.makedirs(self.directory_path_experience)
            print(f"The '{self.name}' directory has been successfully created in the 'experiments' directory.")
        else:
            # Display a message if the directory already exists
            print(f"The '{self.name}' directory already exists in the 'experiments' directory.")

        if not os.path.exists(self.directory_path_epochs) :
            os.makedirs(self.directory_path_epochs)
        
        file_path = os.path.join(self.directory_path_experience, "parameters.txt")
        

        with open(file_path, "w") as file:
            for param, value in self.config.items():
                # Convert lists to strings if necessary
                if isinstance(value, list):
                    value = ', '.join(str(item) for item in value)
                line = f"{param}: {value}\n"
                file.write(line)
    
    def train(self):

        for epoch in tqdm(range(0, self.n_epochs+1)):
            torch.cuda.empty_cache()
            save_epoch = False
            if epoch % self.every_epoch_metric == 0:
                loss, avg_output_vector, avg_label_vector,  pseudo_precision, pseudo_recall , pseudo_recall_placed, pseudo_recall_type, conditionnal_precision_placed, f1_score = train_one_epoch(
                    loader=self.loader_train,
                    model=self.model,
                    size_edge=self.edge_size,
                    device=self.device,
                    optimizer=self.optimizer,
                    epoch_metric = True,
                    criterion=self.criterion,
                    print_bar = self.print_bar)
                
                self.training_history.loc[epoch] = [epoch, loss, avg_output_vector, avg_label_vector, pseudo_precision, pseudo_recall , pseudo_recall_placed, pseudo_recall_type, conditionnal_precision_placed, f1_score]

                loss, avg_output_vector, avg_label_vector,  pseudo_precision, pseudo_recall , pseudo_recall_placed, pseudo_recall_type, conditionnal_precision_placed, f1_score = eval_one_epoch(
                    loader=self.loader_val,
                    model=self.model,
                    size_edge=self.edge_size,
                    device=self.device,
                    criterion=self.criterion,
                    print_bar = self.print_bar,
                    val_metric_size = self.val_metric_size)
                
                self.eval_history.loc[epoch] = [epoch, loss, avg_output_vector, avg_label_vector, pseudo_precision, pseudo_recall , pseudo_recall_placed, pseudo_recall_type, conditionnal_precision_placed, f1_score]
                
                # Check if the loss is better than one of the 6 best losses (compare only along the second dimension of the tuples)

                if loss < max(self.six_best_eval_loss, key=lambda x: x[1])[1]:
                    # switch the save variable to True
                    save_epoch = True
                    index_max = self.six_best_eval_loss.index(max(self.six_best_eval_loss, key=lambda x: x[1]))
                    self.six_best_eval_loss[index_max] = (epoch, loss)
            
            else:
                loss, _, _, _, _, _, _, _, _ = train_one_epoch(
                    loader=self.loader_train,
                    model=self.model,
                    size_edge=self.edge_size,
                    device=self.device,
                    optimizer=self.optimizer,
                    epoch_metric = False,
                    criterion=self.criterion,
                    print_bar = self.print_bar)
                
                self.training_history.loc[epoch] = [epoch, loss, None, None, None, None, None, None, None, None]
                self.eval_history.loc[epoch] = [epoch, None, None, None, None, None, None, None, None, None]

            if save_epoch:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # Add any other relevant information you want to save here
                }
                epoch_save_file = os.path.join(self.directory_path_epochs, f'checkpoint_{index_max}.pt')
                torch.save(checkpoint, epoch_save_file)

                training_csv_directory = os.path.join(self.directory_path_experience, 'training_history.csv')    
                self.training_history.to_csv(training_csv_directory)

                eval_csv_directory = os.path.join(self.directory_path_experience, 'eval_history.csv')    
                self.eval_history.to_csv(eval_csv_directory)

                # Create a txt file containing the infos about the six best epochs saved 
                six_best_epochs_file = os.path.join(self.directory_path_experience, 'six_best_epochs.txt')
                with open(six_best_epochs_file, 'w') as file:
                    for epoch, loss in self.six_best_eval_loss:
                        file.write(f'Epoch {epoch} with loss {loss}\n')
                    