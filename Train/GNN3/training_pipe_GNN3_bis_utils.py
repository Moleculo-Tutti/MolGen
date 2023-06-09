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
from Model.GNN3 import  ModelWithgraph_embedding_modif, ModelWithgraph_embedding_close_or_not_with_node_embedding, ModelWithgraph_embedding_close_or_not_without_node_embedding, ModelWithgraph_embedding_modif_gcnconv, ModelWithgraph_embedding_close_or_not_gcnconv
from Model.metrics import  metric_gnn3_bis_graph_level, metric_gnn3_bis_if_cycle, metric_gnn3_bis_if_cycle_actualized


def train_one_epoch(loader, model_node, size_edge, device, optimizer_graph, optimizer_node, criterion_node, epoch_metric, print_bar = False, model_graph = None, criterion_graph = None, criterion_node_softmax = None):
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
        
        optimizer_graph.zero_grad()
        optimizer_node.zero_grad()
        
        #gnn graph
        close = model_graph(data)
        supposed_close_label = supposed_close_label.unsqueeze(1)
        close_sig = torch.sigmoid(close)

        loss_graph = criterion_graph(close, supposed_close_label)
        loss_graph.backward()
        optimizer_graph.step()
        total_loss_graph += loss_graph.item() * data.num_graphs

        #we combine the mask with the supposed_close, if a graph is supposed_closed (no cycle to make) all these nodes are added to the mask
        supposed_close_label_extended = supposed_close_label.repeat_interleave(torch.bincount(data.batch))
        previous_mask = mask.clone()
        mask = torch.logical_and(mask, supposed_close_label_extended)

        #node in the mask and who have their second value of vector equal to 1
        node_where_closing_label = torch.logical_and(mask, node_labels[:,1] == 1)

        #gnn node
        out = model_node(data)

        out_which_link = out[:,0]
        prob_which_link = torch.sigmoid(out_which_link[node_where_closing_label])
        loss_which_type = criterion_node(out_which_link[node_where_closing_label], node_labels[node_where_closing_label,0])


        #oroblem do the softmax on the minimal thing, we separate
        out_which_neighbour = out[:,1]
        labels_where = node_labels[mask,1]
        appartenance_tensor = data.batch[mask]
        begin_indices= torch.cat([torch.tensor([0], device=device), (appartenance_tensor[1:] != appartenance_tensor[:-1]).nonzero().flatten() + 1])
        end_indices = torch.cat([begin_indices[1:],torch.tensor([len(appartenance_tensor)], device=device)])
        lengths = end_indices - begin_indices

        out_which_neighbour_decomposed = torch.split(out_which_neighbour[mask], lengths.tolist())
        labels_where_decomposed = torch.split(labels_where, lengths.tolist())


        """ old version false because cout node not in mask for the softmax
        num_graph = data.batch.max() + 1
        exp_sum_groups = torch.zeros(num_graph, device=device)
        exp_values = torch.exp(out[:, 1])
        exp_sum_groups.scatter_add_(0, data.batch, exp_values)        
        # Calculer les probabilités softmax par groupe d'indices
        prob_which_neighbour = exp_values / exp_sum_groups[data.batch]
        # we should rebatch for 
        log_prob_which_neighbour = torch.log(prob_which_neighbour[mask], dim = 1 )
        we also used NNLoss
        """

        # Use node_labels_indices with CrossEntropyLoss but without softmax
        loss_where = 0
        for i in range(len(out_which_neighbour_decomposed)):
            loss_where += criterion_node_softmax(out_which_neighbour_decomposed[i], labels_where_decomposed[i])

        loss = loss_where + loss_which_type
        loss.backward()
        optimizer_node.step()
        total_loss_node += loss_where.item() * data.num_graphs + loss_which_type.item() * data.num_graphs
        total_loss += loss_graph.item() * data.num_graphs +loss_where.item() * data.num_graphs + loss_which_type.item() * data.num_graphs


        if epoch_metric:
            target_restricted_type = node_labels[node_where_closing_label,0]
            num_wanted_cycles, cycles_predicted, not_cycles_well_predicted, cycles_well_predicted = metric_gnn3_bis_graph_level(data, close_sig, supposed_close_label, device=device)
            cycles_created_at_good_place, good_types_cycles_predicted = metric_gnn3_bis_if_cycle_actualized(prob_which_link, out_which_neighbour_decomposed,target_restricted_type, labels_where_decomposed,device)
 
            total_graphs_processed += data.num_graphs
            global_cycles_predicted += cycles_predicted
            global_num_wanted_cycles += num_wanted_cycles
            global_non_cycles_well_predicted += not_cycles_well_predicted
            global_cycles_well_predicted += cycles_well_predicted
            global_well_placed_cycles += cycles_created_at_good_place
            global_well_type_cycles += good_types_cycles_predicted
            
            del cycles_predicted, num_wanted_cycles, not_cycles_well_predicted, cycles_well_predicted, cycles_created_at_good_place, good_types_cycles_predicted, target_restricted_type
        
        del data, node_labels, mask, supposed_close_label, node_where_closing_label, out, prob_which_link,
        del loss_where, loss_which_type, loss , loss_graph, supposed_close_label_extended, out_which_link, close_sig, close, labels_where
        del out_which_neighbour, out_which_neighbour_decomposed, labels_where_decomposed, lengths, begin_indices, end_indices, appartenance_tensor


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
    del global_cycles_predicted, global_non_cycles_well_predicted, global_cycles_well_predicted, global_well_placed_cycles, global_well_type_cycles, total_graphs_processed
    if epoch_metric: 
        return (
            total_loss / len(loader.dataset),
            total_loss_graph / len(loader.dataset),
            total_loss_node / global_num_wanted_cycles,
            accuracy_num_cycles,
            precision_num_cycles, 
            recall_num_cycles, 
            accuracy_neighhbor_chosen , 
            accuracy_type_chosen, 
            f1_score_num_cycles)
    

    else:
        return total_loss / len(loader.dataset), total_loss_graph / len(loader.dataset), total_loss_node / global_num_wanted_cycles,None, None, None, None, None, None


def eval_one_epoch(loader, model_node, size_edge, device, criterion_node, print_bar=False, val_metric_size=1, model_graph=None, criterion_graph = None,criterion_node_softmax = None):
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
            close_sig = torch.sigmoid(close)
            supposed_close_label = supposed_close_label.unsqueeze(1)
            loss_graph = criterion_graph(close, supposed_close_label)
            loss_graph.backward()
            total_loss_graph += loss_graph.item() * data.num_graphs


            
            #we combine the mask with the supposed_close, if a graph is supposed_closed (no cycle to make) all these nodes are added to the mask
            supposed_close_label_extended = supposed_close_label.repeat_interleave(torch.bincount(data.batch))
            mask = torch.logical_and(mask, supposed_close_label_extended)

            #node in the mask and who have their second value of vector equal to 1
            node_where_closing_label = torch.logical_and(mask, node_labels[:,1] == 1)

            #gnn node
            out = model_node(data)

            out_which_link = out[:,0]
            prob_which_link = torch.sigmoid(out_which_link[node_where_closing_label])
            loss_which_type = criterion_node(out_which_link[node_where_closing_label], node_labels[node_where_closing_label,0])


            #oroblem do the softmax on the minimal thing, we separate
            out_which_neighbour = out[:,1]
            labels_where = node_labels[mask,1]
            appartenance_tensor = data.batch[mask]
            begin_indices= torch.cat([torch.tensor([0], device=device), (appartenance_tensor[1:] != appartenance_tensor[:-1]).nonzero().flatten() + 1])
            end_indices = torch.cat([begin_indices[1:],torch.tensor([len(appartenance_tensor)], device=device)])
            lengths = end_indices - begin_indices
            out_which_neighbour_decomposed = torch.split(out_which_neighbour[mask], lengths.tolist())
            labels_where_decomposed = torch.split(labels_where, lengths.tolist())


            """ old version false because cout node not in mask for the softmax
            num_graph = data.batch.max() + 1
            exp_sum_groups = torch.zeros(num_graph, device=device)
            exp_values = torch.exp(out[:, 1])
            exp_sum_groups.scatter_add_(0, data.batch, exp_values)        
            # Calculer les probabilités softmax par groupe d'indices
            prob_which_neighbour = exp_values / exp_sum_groups[data.batch]
            # we should rebatch for 
            log_prob_which_neighbour = torch.log(prob_which_neighbour[mask], dim = 1 )
            we also used NNLoss
            """

            # Use node_labels_indices with CrossEntropyLoss but without softmax
            loss_where = 0
            for i in range(len(out_which_neighbour_decomposed)):
                loss_where += criterion_node_softmax(out_which_neighbour_decomposed[i], labels_where_decomposed[i])

            total_loss_node += loss_where.item() * data.num_graphs + loss_which_type.item() * data.num_graphs
            total_loss += loss_graph.item() * data.num_graphs * data.num_graphs +loss_where.item() * data.num_graphs + loss_which_type.item() * data.num_graphs
            # Add softmax to out

            target_restricted_type = node_labels[node_where_closing_label,0]
        
            num_wanted_cycles, cycles_predicted, not_cycles_well_predicted, cycles_well_predicted = metric_gnn3_bis_graph_level(data, close_sig, supposed_close_label, device=device)
            cycles_created_at_good_place, good_types_cycles_predicted = metric_gnn3_bis_if_cycle_actualized(prob_which_link, out_which_neighbour_decomposed,target_restricted_type, labels_where_decomposed,device)

            total_graphs_processed += data.num_graphs
            global_cycles_predicted += cycles_predicted
            global_num_wanted_cycles += num_wanted_cycles
            global_non_cycles_well_predicted += not_cycles_well_predicted
            global_cycles_well_predicted += cycles_well_predicted
            global_well_placed_cycles += cycles_created_at_good_place
            global_well_type_cycles += good_types_cycles_predicted
            
            del cycles_predicted, num_wanted_cycles, not_cycles_well_predicted, cycles_well_predicted, cycles_created_at_good_place, good_types_cycles_predicted
            
            del data, node_labels, mask, supposed_close_label, node_where_closing_label, out, prob_which_link
            del loss_where, loss_which_type, loss_graph, supposed_close_label_extended, out_which_link, close, close_sig, labels_where
            del out_which_neighbour, out_which_neighbour_decomposed, labels_where_decomposed, lengths, target_restricted_type, appartenance_tensor, begin_indices, end_indices


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
    del global_cycles_predicted, global_non_cycles_well_predicted, global_cycles_well_predicted, global_well_placed_cycles, global_well_type_cycles, total_graphs_processed

    return (
            total_loss /( val_metric_size* len(loader.dataset)),
            total_loss_graph /( val_metric_size* len(loader.dataset)),
            total_loss_node /global_num_wanted_cycles,
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
        self.use_gcnconv = config['use_gcnconv']
        print(f"Training on {self.device}")
        self.continue_training = continue_training
        self.size_minimum_scaffold = config['size_minimum_scaffold']

        print(f"Loading data...")
        self.loader_train, self.loader_val, self.model_node, self.encoding_size, self.edge_size,self.model_graph = self.load_data_model()
        print(f"Data loaded")
        self.begin_epoch = 0

        self.optimizer_node = torch.optim.Adam(self.model_node.parameters(), lr=self.lr)
        self.optimizer_graph = torch.optim.Adam(self.model_graph.parameters(), lr=self.lr)
        if self.continue_training:
            self.model_node.load_state_dict(checkpoint['model_node_state_dict'])
            self.optimizer_graph.load_state_dict(checkpoint['optimizer_graph_state_dict'])
            self.optimizer_node.load_state_dict(checkpoint['optimizer_node_state_dict'])
            self.begin_epoch = checkpoint['epoch']

        #cross entropy loss without softmax
        self.criterion_node = nn.BCEWithLogitsLoss()
        self.criterion_node_softmax = nn.CrossEntropyLoss()
        self.criterion_graph = nn.BCEWithLogitsLoss()

        self.training_history = pd.DataFrame(columns=['epoch', 'loss', 'loss_graph', 'loss_node', 'accuracy_num_cycles', 'precision_num_cycles', 'recall_num_cycles', 'accuracy_neighhbor_chosen' , 'accuracy_type_chosen', 'f1_score_num_cycles'])
        self.eval_history = pd.DataFrame(columns=['epoch', 'loss', 'loss_graph', 'loss_node', 'accuracy_num_cycles', 'precision_num_cycles', 'recall_num_cycles', 'accuracy_neighhbor_chosen' , 'accuracy_type_chosen', 'f1_score_num_cycles'])

        self.prepare_saving()

        # Store the 6 best models
        self.six_best_eval_loss_graph = [(0, float('inf'))] * 6
        self.six_best_eval_loss_node = [(0, float('inf'))] * 6


        if self.continue_training:
            # Open the six best eval loss
            with open(os.path.join(self.directory_path_experience, 'six_best_epochs_graph.txt'), 'r') as f:
                for i in range(6):
                    line = f.readline()
                    epoch, loss = line.split(' with loss ')  # Utilisez ' with loss ' comme séparateur
                    epoch = epoch.split('Epoch ')[1]  # Supprimez 'Epoch ' de la valeur de l'époque
                    self.six_best_eval_loss_graph[i] = (int(epoch), float(loss))
            with open(os.path.join(self.directory_path_experience, 'six_best_epochs_node.txt'), 'r') as f:
                for i in range(6):
                    line = f.readline()
                    epoch, loss = line.split(' with loss ')
                    epoch = epoch.split('Epoch ')[1]
                    self.six_best_eval_loss_node[i] = (int(epoch), float(loss))

    def load_data_model(self):
        # Load the data
        
        dataset_train = ZincSubgraphDatasetStep(self.datapath_train, GNN_type=4, feature_position=self.feature_position, scores_list=self.score_list, size_minimum_scaffold=self.size_minimum_scaffold)
        dataset_val = ZincSubgraphDatasetStep(self.datapath_val, GNN_type=4, feature_position=self.feature_position, scores_list=self.score_list, size_minimum_scaffold=self.size_minimum_scaffold)

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
            if self.use_gcnconv:
                model_graph = ModelWithgraph_embedding_close_or_not_gcnconv(in_channels = encoding_size + int(self.feature_position) + int(len(self.score_list)), # We increase the input size to take into account the feature position
                                        hidden_channels_list=self.GCN_size,
                                        mlp_hidden_channels=self.mlp_hidden,
                                        edge_channels=edge_size,
                                        num_classes=1, #0 if we want to close nothing and 1 if  we close one cycle in the graph
                                        use_dropout=self.use_dropout,
                                        size_info=self.size_info,
                                        max_size=self.max_size,
                                        encoding_size=encoding_size)
            else:
                model_graph =ModelWithgraph_embedding_close_or_not_without_node_embedding(in_channels = encoding_size + int(self.feature_position) + int(len(self.score_list)), # We increase the input size to take into account the feature position
                                        hidden_channels_list=self.GCN_size,
                                        mlp_hidden_channels=self.mlp_hidden,
                                        edge_channels=edge_size,
                                        num_classes=1, #0 if we want to close nothing and 1 if  we close one cycle in the graph
                                        use_dropout=self.use_dropout,
                                        size_info=self.size_info,
                                        max_size=self.max_size,
                                        encoding_size=encoding_size)
        if self.use_gcnconv:
            model_node = ModelWithgraph_embedding_modif_gcnconv(in_channels = encoding_size + int(self.feature_position) + int(len(self.score_list)), # We increase the input size to take into account the feature position
                                        hidden_channels_list=self.GCN_size,
                                        mlp_hidden_channels=self.mlp_hidden,
                                        edge_channels=edge_size, 
                                        num_classes=edge_size -1, #close with a simple double and which one to close just size 2
                                        use_dropout=self.use_dropout,
                                        size_info=self.size_info,
                                        max_size=self.max_size)
        else: 
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
            save_epoch_graph = False
            save_epoch_node = False
            save_csv = False
            if epoch % self.every_epoch_metric == 0:
                save_csv = True
                loss,loss_graph, loss_node, accuracy_num_cycles, precision_num_cycles, recall_num_cycles, accuracy_neighhbor_chosen , accuracy_type_chosen, f1_score_num_cycles= train_one_epoch(
                    loader=self.loader_train,
                    model_node=self.model_node,
                    size_edge=self.edge_size,
                    device=self.device,
                    optimizer_graph=self.optimizer_graph,
                    optimizer_node=self.optimizer_node,
                    epoch_metric = True,
                    criterion_node=self.criterion_node,
                    print_bar = self.print_bar,
                    model_graph = self.model_graph,
                    criterion_graph=self.criterion_graph,
                    criterion_node_softmax=self.criterion_node_softmax)
                    
                self.training_history.loc[epoch] = [epoch, loss, loss_graph, loss_node, accuracy_num_cycles, precision_num_cycles, recall_num_cycles, accuracy_neighhbor_chosen , accuracy_type_chosen, f1_score_num_cycles]

                loss, loss_graph, loss_node, accuracy_num_cycles, precision_num_cycles, recall_num_cycles, accuracy_neighhbor_chosen , accuracy_type_chosen, f1_score_num_cycles = eval_one_epoch(
                    loader=self.loader_val,
                    model_node=self.model_node,
                    size_edge=self.edge_size,
                    device=self.device,
                    criterion_node=self.criterion_node,
                    print_bar = self.print_bar,
                    val_metric_size = self.val_metric_size,
                    model_graph = self.model_graph,
                    criterion_graph=self.criterion_graph,
                    criterion_node_softmax=self.criterion_node_softmax)
                
                self.eval_history.loc[epoch] = [epoch,loss, loss_graph, loss_node, accuracy_num_cycles, precision_num_cycles, recall_num_cycles, accuracy_neighhbor_chosen , accuracy_type_chosen, f1_score_num_cycles]
                
                # Check if the loss is better than one of the 6 best losses (compare only along the second dimension of the tuples)

                if loss_graph < max(self.six_best_eval_loss_graph, key=lambda x: x[1])[1]:
                    # switch the save variable to True
                    save_epoch_graph = True
                    index_max_graph = self.six_best_eval_loss_graph.index(max(self.six_best_eval_loss_graph, key=lambda x: x[1]))
                    self.six_best_eval_loss_graph[index_max_graph] = (epoch, loss_graph)
                
                if loss_node < max(self.six_best_eval_loss_node, key=lambda x: x[1])[1]:
                    # switch the save variable to True
                    save_epoch_node = True
                    index_max_node = self.six_best_eval_loss_node.index(max(self.six_best_eval_loss_node, key=lambda x: x[1]))
                    self.six_best_eval_loss_node[index_max_node] = (epoch, loss_node)
                
            else:
                loss, loss_graph, loss_node, _, _, _, _, _, _ = train_one_epoch(
                    loader=self.loader_train,
                    model_node=self.model_node,
                    size_edge=self.edge_size,
                    device=self.device,
                    optimizer_graph=self.optimizer_graph,
                    optimizer_node=self.optimizer_node,
                    epoch_metric = False,
                    criterion_node=self.criterion_node,
                    print_bar = self.print_bar,
                    model_graph = self.model_graph,
                    criterion_graph=self.criterion_graph,
                    criterion_node_softmax=self.criterion_node_softmax)
                
                self.training_history.loc[epoch] = [epoch, loss, loss_graph, loss_node, None, None, None, None, None, None]
                self.eval_history.loc[epoch] = [epoch, None, None, None, None, None, None, None, None, None]

            if save_epoch_graph:
                checkpoint = {
                    'epoch': epoch,
                    'model_graph_state_dict': self.model_graph.state_dict(),
                    'optimizer_graph_state_dict': self.optimizer_graph.state_dict()}

                epoch_save_file = os.path.join(self.directory_path_epochs, f'checkpoint_graph{index_max_graph}.pt')
                torch.save(checkpoint, epoch_save_file)
                six_best_epochs_file = os.path.join(self.directory_path_experience, 'six_best_epochs_graph.txt')
                with open(six_best_epochs_file, 'w') as file:
                    for epoch, loss_graph in self.six_best_eval_loss_graph:
                        file.write(f'Epoch {epoch} with loss {loss_graph}\n')
                del checkpoint, epoch_save_file, six_best_epochs_file, file

            if save_csv:
                training_csv_directory = os.path.join(self.directory_path_experience, 'training_history.csv')

                self.training_history.to_csv(training_csv_directory)   

                eval_csv_directory = os.path.join(self.directory_path_experience, 'eval_history.csv')    

                self.eval_history.to_csv(eval_csv_directory)

                # Create a txt file containing the infos about the six best epochs saved 

                del training_csv_directory, eval_csv_directory
            
            if save_epoch_node:
                checkpoint = {
                    'epoch': epoch,
                    'model_node_state_dict': self.model_node.state_dict(),
                    'optimizer_node_state_dict': self.optimizer_node.state_dict()}

                epoch_save_file = os.path.join(self.directory_path_epochs, f'checkpoint_node{index_max_node}.pt')
                torch.save(checkpoint, epoch_save_file)
                six_best_epochs_file = os.path.join(self.directory_path_experience, 'six_best_epochs_node.txt')
                with open(six_best_epochs_file, 'w') as file:
                    for epoch, loss_node in self.six_best_eval_loss_node:
                        file.write(f'Epoch {epoch} with loss {loss_node}\n')
                del checkpoint, epoch_save_file, six_best_epochs_file, file

            del loss, loss_graph, loss_node
            gc.collect()
                        