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
import json
import gc

import sys
import os

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

from DataPipeline.dataset import ZincSubgraphDatasetStep, custom_collate_GNN3_bis
from Model.GNN3 import  ModelWithgraph_embedding_modif, ModelWithgraph_embedding_close_or_not_with_node_embedding, ModelWithgraph_embedding_close_or_not_without_node_embedding
from Model.metrics import  metric_gnn3_bis_graph_level, metric_gnn3_bis_if_cycle


def train_one_epoch(loader, model_node, size_edge, device, optimizer, criterion_node, epoch_metric, print_bar = False, model_graph = None, criterion_graph = None):
    model_node.train()
    model_graph.train()
    total_loss = 0
    total_loss_graph = 0
    total_loss_node = 0
    total_graphs_processed = 0
    global_cycles_predicted= 0
    global_num_wanted_cycles = 0
    global_non_cycles_well_predicted = 0
    global_cycles_well_predicted = 0
    global_well_placed_cycles = 0
    global_well_type_cycles = 0

    progress_bar = tqdm(loader, desc="Training", unit="batch")
    
    for _, batch in enumerate(progress_bar):
        data = batch[0].to(device)
        node_labels = batch[1].to(device)
        mask = batch[2].to(device)
        supposed_close_label = batch[3].to(device) #1 if we close a cycle 0 otherwise
        
        optimizer.zero_grad()
        
        #gnn graph
        close = model_graph(data)
        close_output = torch.sigmoid(close)
        supposed_close_label = supposed_close_label.unsqueeze(1)

        try:
            loss_graph = criterion_graph(close_output, supposed_close_label)
            loss_graph.backward()
            total_loss_graph += loss_graph.item() * data.num_graphs

            #we combine the mask with the supposed_close, if a graph is supposed_closed (no cycle to make) all these nodes are added to the mask
            supposed_close_label_extended = supposed_close_label.repeat_interleave(torch.bincount(data.batch))
            mask = torch.logical_and(mask, supposed_close_label_extended)

            #node in the mask and who have their second value of vector equal to 1
            node_where_closing_label = torch.logical_and(mask, node_labels[:,1] == 1)

            #gnn node
            out = model_node(data)
            prob_which_link = torch.sigmoid(out[:,0])
            num_graph = data.batch.max() + 1
            exp_sum_groups = torch.zeros(num_graph, device=device)
            exp_values = torch.exp(out[:, 1])
            exp_sum_groups.scatter_add_(0, data.batch, exp_values)        
            # Calculer les probabilités softmax par groupe d'indices
            prob_which_neighbour = exp_values / exp_sum_groups[data.batch]

            # Use node_labels_indices with CrossEntropyLoss but without 
            loss_where = criterion_node(prob_which_neighbour[mask], node_labels[mask,1])
            loss_which_type = criterion_node(prob_which_link[node_where_closing_label], node_labels[node_where_closing_label,0])
            loss = loss_where + loss_which_type
        
            loss.backward()
            optimizer.step()
            total_loss_node += loss_where.item() * data.num_graphs + loss_which_type.item() * data.num_graphs
            total_loss += loss_graph.item() * data.num_graphs * data.num_graphs +loss_where.item() * data.num_graphs + loss_which_type.item() * data.num_graphs
        except Exception as e:
            # Generic handler for any other exception
            print('model1_output', close_output)
            print('model2_output', out)
            print('sigmoid_ouput', prob_which_link)
            print('softmax_output', prob_which_neighbour)
            print("An error occurred:", str(e))

        if epoch_metric:
            num_wanted_cycles, cycles_predicted, not_cycles_well_predicted, cycles_well_predicted = metric_gnn3_bis_graph_level(data, close_output, supposed_close_label, device=device)
            cycles_created_at_good_place, good_types_cycles_predicted = metric_gnn3_bis_if_cycle(data, prob_which_link, prob_which_neighbour, node_labels, supposed_close_label, device=device)
 
            total_graphs_processed += data.num_graphs
            global_cycles_predicted += cycles_predicted
            global_num_wanted_cycles += num_wanted_cycles
            global_non_cycles_well_predicted += not_cycles_well_predicted
            global_cycles_well_predicted += cycles_well_predicted
            global_well_placed_cycles += cycles_created_at_good_place
            global_well_type_cycles += good_types_cycles_predicted
            
            del cycles_predicted, num_wanted_cycles, not_cycles_well_predicted, cycles_well_predicted, cycles_created_at_good_place, good_types_cycles_predicted
        
        del data, node_labels, mask, supposed_close_label, node_where_closing_label, out, prob_which_link, num_graph, exp_sum_groups, exp_values, prob_which_neighbour
        del loss_where, loss_which_type, loss , loss_graph, close_output, supposed_close_label_extended


    if (total_graphs_processed == 0):
        total_graphs_processed = 1
    if global_num_wanted_cycles == 0:
        global_num_wanted_cycles = 1
    if global_cycles_predicted == 0:
        global_cycles_predicted = 1
    accuracy_num_cycles = (global_cycles_well_predicted + global_non_cycles_well_predicted) / total_graphs_processed
    precision_num_cycles = global_cycles_well_predicted/ global_cycles_predicted
    recall_num_cycles = global_cycles_well_predicted/ global_num_wanted_cycles  
    accuracy_neighhbor_chosen = global_well_placed_cycles / global_num_wanted_cycles
    accuracy_type_chosen = global_well_type_cycles / global_num_wanted_cycles
    if (precision_num_cycles + recall_num_cycles) == 0:
        f1_score_num_cycles = 0
    else:
        f1_score_num_cycles = 2 * precision_num_cycles * recall_num_cycles / (precision_num_cycles + recall_num_cycles)
    del global_cycles_predicted, global_num_wanted_cycles, global_non_cycles_well_predicted, global_cycles_well_predicted, global_well_placed_cycles, global_well_type_cycles, total_graphs_processed
    if epoch_metric: 
        return (
            total_loss / len(loader.dataset),
            accuracy_num_cycles,
            precision_num_cycles, 
            recall_num_cycles, 
            accuracy_neighhbor_chosen , 
            accuracy_type_chosen, 
            f1_score_num_cycles)
    

    else:
        return total_loss / len(loader.dataset), None, None, None, None, None, None


def eval_one_epoch(loader, model_node, size_edge, device, criterion_node, print_bar=False, val_metric_size=1, model_graph=None, criterion_graph = None):
    model_node.eval()
    model_graph.eval()
    total_loss = 0
    total_loss_graph = 0
    total_loss_node = 0
    total_graphs_processed = 0
    global_cycles_predicted= 0
    global_num_wanted_cycles = 0
    global_non_cycles_well_predicted = 0
    global_cycles_well_predicted = 0
    global_well_placed_cycles = 0
    global_well_type_cycles = 0

    for i in tqdm(range(val_metric_size)):

        for batch_idx, batch in enumerate(loader):
            data = batch[0].to(device)
            node_labels = batch[1].to(device)
            mask = batch[2].to(device)
            supposed_close_label = batch[3].to(device) #1 if we close a cycle 0 otherwise
                    
            #gnn graph
            close = model_graph(data)
            close_output = torch.sigmoid(close)
            supposed_close_label = supposed_close_label.unsqueeze(1)
            loss_graph = criterion_graph(close_output, supposed_close_label)
            loss_graph.backward()
            total_loss_graph += loss_graph.item() * data.num_graphs


            
            #we combine the mask with the supposed_close, if a graph is supposed_closed (no cycle to make) all these nodes are added to the mask
            supposed_close_label_extended = supposed_close_label.repeat_interleave(torch.bincount(data.batch))
            mask = torch.logical_and(mask, supposed_close_label_extended)

            #node in the mask and who have their second value of vector equal to 1
            node_where_closing_label = torch.logical_and(mask, node_labels[:,1] == 1)

            #gnn node
            out = model_node(data)
            prob_which_link = torch.sigmoid(out[:,0])
            num_graph = data.batch.max() + 1
            exp_sum_groups = torch.zeros(num_graph, device=device)
            exp_values = torch.exp(out[:, 1])
            exp_sum_groups.scatter_add_(0, data.batch, exp_values)        
            # Calculer les probabilités softmax par groupe d'indices
            prob_which_neighbour = exp_values / exp_sum_groups[data.batch]

            # Use node_labels_indices with CrossEntropyLoss but without 
            loss_where = criterion_node(prob_which_neighbour[mask], node_labels[mask,1])
            loss_which_type = criterion_node(prob_which_link[node_where_closing_label], node_labels[node_where_closing_label,0])

            total_loss_node += loss_where.item() * data.num_graphs + loss_which_type.item() * data.num_graphs
            total_loss += loss_graph.item() * data.num_graphs * data.num_graphs +loss_where.item() * data.num_graphs + loss_which_type.item() * data.num_graphs
            # Add softmax to out
        
            num_wanted_cycles, cycles_predicted, not_cycles_well_predicted, cycles_well_predicted = metric_gnn3_bis_graph_level(data, close_output, supposed_close_label, device=device)
            cycles_created_at_good_place, good_types_cycles_predicted = metric_gnn3_bis_if_cycle(data, prob_which_link, prob_which_neighbour, node_labels, supposed_close_label, device=device)

            total_graphs_processed += data.num_graphs
            global_cycles_predicted += cycles_predicted
            global_num_wanted_cycles += num_wanted_cycles
            global_non_cycles_well_predicted += not_cycles_well_predicted
            global_cycles_well_predicted += cycles_well_predicted
            global_well_placed_cycles += cycles_created_at_good_place
            global_well_type_cycles += good_types_cycles_predicted
            
            del cycles_predicted, num_wanted_cycles, not_cycles_well_predicted, cycles_well_predicted, cycles_created_at_good_place, good_types_cycles_predicted
            
            del data, node_labels, mask, supposed_close_label, node_where_closing_label, out, prob_which_link, num_graph, exp_sum_groups, exp_values, prob_which_neighbour
            del loss_where, loss_which_type, loss_graph, close_output, supposed_close_label_extended


    if (total_graphs_processed == 0):
        total_graphs_processed = 1
    if global_num_wanted_cycles == 0:
        global_num_wanted_cycles = 1
    if global_cycles_predicted == 0:
        global_cycles_predicted = 1
    accuracy_num_cycles = (global_cycles_well_predicted + global_non_cycles_well_predicted) / total_graphs_processed
    precision_num_cycles = global_cycles_well_predicted / global_cycles_predicted
    recall_num_cycles = global_cycles_well_predicted/ global_num_wanted_cycles  
    accuracy_neighhbor_chosen = global_well_placed_cycles / global_num_wanted_cycles
    accuracy_type_chosen = global_well_type_cycles / global_num_wanted_cycles
    f1_score_num_cycles = 2 * precision_num_cycles * recall_num_cycles / (precision_num_cycles + recall_num_cycles)
    del global_cycles_predicted, global_num_wanted_cycles, global_non_cycles_well_predicted, global_cycles_well_predicted, global_well_placed_cycles, global_well_type_cycles, total_graphs_processed

    return (
            total_loss /( val_metric_size* len(loader.dataset)),
            accuracy_num_cycles,
            precision_num_cycles, 
            recall_num_cycles, 
            accuracy_neighhbor_chosen , 
            accuracy_type_chosen, 
            f1_score_num_cycles)



class TrainGNN3_bis():
    def __init__(self, config, continue_training= False, checkpoint = None):
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
        self.score_list = config['score_list']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.split_two_parts = config['split_two_parts']
        self.node_embedding_for_second_part = config['node_embedding_for_second_part']
        print(f"Training on {self.device}")
        self.continue_training = continue_training

        print(f"Loading data...")
        self.loader_train, self.loader_val, self.model_node, self.encoding_size, self.edge_size,self.model_graph = self.load_data_model()
        print(f"Data loaded")
        self.begin_epoch = 0

        self.optimizer = torch.optim.Adam(self.model_node.parameters(), lr=self.lr)
        if self.continue_training:
            self.model_node.load_state_dict(checkpoint['model_node_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.begin_epoch = checkpoint['epoch']

        #cross entropy loss without softmax
        self.criterion_node = nn.BCELoss()
        self.criterion_graph = nn.BCELoss()

        self.training_history = pd.DataFrame(columns=['epoch', 'loss', 'accuracy_num_cycles', 'precision_num_cycles', 'recall_num_cycles', 'accuracy_neighhbor_chosen' , 'accuracy_type_chosen', 'f1_score_num_cycles'])
        self.eval_history = pd.DataFrame(columns=['epoch', 'loss', 'accuracy_num_cycles', 'precision_num_cycles', 'recall_num_cycles', 'accuracy_neighhbor_chosen' , 'accuracy_type_chosen', 'f1_score_num_cycles'])

        self.prepare_saving()

        # Store the 6 best models
        self.six_best_eval_loss = [(0, float('inf'))] * 6

        if self.continue_training:
            # Open the six best eval loss
            with open(os.path.join(self.directory_path_experience, 'six_best_epochs.txt'), 'r') as f:
                for i in range(6):
                    line = f.readline()
                    epoch, loss = line.split(' with loss ')  # Utilisez ' with loss ' comme séparateur
                    epoch = epoch.split('Epoch ')[1]  # Supprimez 'Epoch ' de la valeur de l'époque
                    self.six_best_eval_loss[i] = (int(epoch), float(loss))

    def load_data_model(self):
        # Load the data
        
        dataset_train = ZincSubgraphDatasetStep(self.datapath_train, GNN_type=4, feature_position=self.feature_position, scores_list=self.score_list)
        dataset_val = ZincSubgraphDatasetStep(self.datapath_val, GNN_type=4, feature_position=self.feature_position, scores_list=self.score_list)

        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers, collate_fn=custom_collate_GNN3_bis)
        loader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers, collate_fn=custom_collate_GNN3_bis)


        encoding_size = dataset_train.encoding_size
        edge_size = dataset_train.edge_size

        # Load the model
            
        if self.node_embedding_for_second_part:
            model_graph = ModelWithgraph_embedding_close_or_not_with_node_embedding(in_channels = encoding_size + int(self.feature_position) + int(len(self.score_list)), # We increase the input size to take into account the feature position
                                        hidden_channels_list=self.GCN_size,
                                        mlp_hidden_channels=self.mlp_hidden,
                                        edge_channels=edge_size,
                                        num_classes=1, #0 if we want to close nothing and 1 if  we close one cycle in the graph
                                        use_dropout=self.use_dropout,
                                        size_info=self.size_info,
                                        max_size=self.max_size,
                                        encoding_size=encoding_size)
        else :
            model_graph =ModelWithgraph_embedding_close_or_not_without_node_embedding(in_channels = encoding_size + int(self.feature_position) + int(len(self.score_list)), # We increase the input size to take into account the feature position
                                        hidden_channels_list=self.GCN_size,
                                        mlp_hidden_channels=self.mlp_hidden,
                                        edge_channels=edge_size,
                                        num_classes=1, #0 if we want to close nothing and 1 if  we close one cycle in the graph
                                        use_dropout=self.use_dropout,
                                        size_info=self.size_info,
                                        max_size=self.max_size,
                                        encoding_size=encoding_size)
            
        model_node = ModelWithgraph_embedding_modif(in_channels = encoding_size + int(self.feature_position) + int(len(self.score_list)), # We increase the input size to take into account the feature position
                                        hidden_channels_list=self.GCN_size,
                                        mlp_hidden_channels=self.mlp_hidden,
                                        edge_channels=edge_size, 
                                        num_classes=edge_size -1, #close with a simple double and which one to close just size 2
                                        use_dropout=self.use_dropout,
                                        size_info=self.size_info,
                                        max_size=self.max_size)

        print("edge size =", edge_size)
        print("modele num classes = ", model_node.num_classes)
        
        return loader_train, loader_val, model_node.to(self.device), encoding_size, edge_size, model_graph.to(self.device)
    
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
        
        file_path = os.path.join(self.directory_path_experience, "parameters.json")

        with open(file_path, "w") as file:
            json.dump(self.config, file)
        
        if self.continue_training:
            # Add the previous training history
            self.training_history = pd.read_csv(os.path.join(self.directory_path_experience, "training_history.csv"), index_col=0)
            self.eval_history = pd.read_csv(os.path.join(self.directory_path_experience, "eval_history.csv"), index_col=0)
            # Keep only the losses above the begin epoch
            self.training_history = self.training_history[self.training_history['epoch'] < self.begin_epoch]
            self.eval_history = self.eval_history[self.eval_history['epoch'] < self.begin_epoch]
    
    def train(self):

        for epoch in tqdm(range(self.begin_epoch, self.n_epochs+1)):
            torch.cuda.empty_cache()
            save_epoch = False
            if epoch % self.every_epoch_metric == 0:
                loss, accuracy_num_cycles, precision_num_cycles, recall_num_cycles, accuracy_neighhbor_chosen , accuracy_type_chosen, f1_score_num_cycles= train_one_epoch(
                    loader=self.loader_train,
                    model_node=self.model_node,
                    size_edge=self.edge_size,
                    device=self.device,
                    optimizer=self.optimizer,
                    epoch_metric = True,
                    criterion_node=self.criterion_node,
                    print_bar = self.print_bar,
                    model_graph = self.model_graph,
                    criterion_graph=self.criterion_graph)
                
                self.training_history.loc[epoch] = [epoch, loss, accuracy_num_cycles, precision_num_cycles, recall_num_cycles, accuracy_neighhbor_chosen , accuracy_type_chosen, f1_score_num_cycles]

                loss, accuracy_num_cycles, precision_num_cycles, recall_num_cycles, accuracy_neighhbor_chosen , accuracy_type_chosen, f1_score_num_cycles = eval_one_epoch(
                    loader=self.loader_val,
                    model_node=self.model_node,
                    size_edge=self.edge_size,
                    device=self.device,
                    criterion_node=self.criterion_node,
                    print_bar = self.print_bar,
                    val_metric_size = self.val_metric_size,
                    model_graph = self.model_graph,
                    criterion_graph=self.criterion_graph)
                
                self.eval_history.loc[epoch] = [epoch,loss, accuracy_num_cycles, precision_num_cycles, recall_num_cycles, accuracy_neighhbor_chosen , accuracy_type_chosen, f1_score_num_cycles]
                
                # Check if the loss is better than one of the 6 best losses (compare only along the second dimension of the tuples)

                if loss < max(self.six_best_eval_loss, key=lambda x: x[1])[1]:
                    # switch the save variable to True
                    save_epoch = True
                    index_max = self.six_best_eval_loss.index(max(self.six_best_eval_loss, key=lambda x: x[1]))
                    self.six_best_eval_loss[index_max] = (epoch, loss)
            
            else:
                loss, _, _, _, _, _, _,  = train_one_epoch(
                    loader=self.loader_train,
                    model_node=self.model_node,
                    size_edge=self.edge_size,
                    device=self.device,
                    optimizer=self.optimizer,
                    epoch_metric = False,
                    criterion_node=self.criterion_node,
                    print_bar = self.print_bar,
                    model_graph = self.model_graph,
                    criterion_graph=self.criterion_graph)
                
                self.training_history.loc[epoch] = [epoch, loss, None, None, None, None, None, None]
                self.eval_history.loc[epoch] = [epoch, None, None, None, None, None, None, None]

            if save_epoch:
                checkpoint = {
                    'epoch': epoch,
                    'model_node_state_dict': self.model_node.state_dict(),
                    'model_graph_state_dict': self.model_graph.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict()}

                epoch_save_file = os.path.join(self.directory_path_epochs, f'checkpoint_{index_max}.pt')
                torch.save(checkpoint, epoch_save_file)

                training_csv_directory = os.path.join(self.directory_path_experience, 'training_history.csv')
                if os.path.exists(training_csv_directory):
                    # If the file already exists, we append the new data
                    self.training_history.to_csv(training_csv_directory, mode='a', header=False)
                else:
                    self.training_history.to_csv(training_csv_directory)   

                eval_csv_directory = os.path.join(self.directory_path_experience, 'eval_history.csv')    
                if os.path.exists(eval_csv_directory):        
                    # If the file already exists, we append the new data
                    self.eval_history.to_csv(eval_csv_directory, mode='a', header=False)
                else:
                    self.eval_history.to_csv(eval_csv_directory)

                # Create a txt file containing the infos about the six best epochs saved 
                six_best_epochs_file = os.path.join(self.directory_path_experience, 'six_best_epochs.txt')
                with open(six_best_epochs_file, 'w') as file:
                    for epoch, loss in self.six_best_eval_loss:
                        file.write(f'Epoch {epoch} with loss {loss}\n')
                del checkpoint, epoch_save_file, six_best_epochs_file, training_csv_directory, eval_csv_directory, file
            del loss
            gc.collect()
                    