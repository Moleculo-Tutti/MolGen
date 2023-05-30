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
from Model.GNN3 import ModelWithEdgeFeatures, ModelWithgraph_embedding, ModelWithgraph_embedding_modif
from Model.metrics import  pseudo_accuracy_metric_gnn3


def train_one_epoch(loader, model, size_edge, device, optimizer, criterion, print_bar = False):
    model.train()
    total_loss = 0
    num_correct = 0
    num_output = torch.zeros(size_edge)  # Already on CPU
    num_labels = torch.zeros(size_edge)  # Already on CPU
    total_graphs_processed = 0
    global_cycles_created = 0
    global_well_placed_cycles = 0
    global_well_type_cycles =0
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
    
    
        # Add softmax to out
        softmax_out = F.softmax(out, dim=1)

        cycles_created, well_placed_cycles , well_type_cycles, cycles_missed, cycles_shouldnt_created, num_wanted_cycles = pseudo_accuracy_metric_gnn3(data,out,node_labels,mask)        
        # Calculate metrics and move tensors to CPU
        num_output += torch.sum(softmax_out[mask], dim=0).detach().cpu()
        num_labels += torch.sum(node_labels[mask], dim=0).detach().cpu()
        global_cycles_created +=cycles_created
        global_well_placed_cycles += well_placed_cycles
        global_well_type_cycles += well_type_cycles
        global_cycles_missed += cycles_missed
        global_cycles_shouldnt_created += cycles_shouldnt_created
        global_num_wanted_cycles += num_wanted_cycles
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
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
    


def eval_one_epoch(loader, model, size_edge, device, criterion, print_bar=False):
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
                data, out, node_labels, mask)

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





def train_GNN3(name : str, datapath_train, datapath_val, n_epochs,  encoding_size, GCN_size : list, mlp_size, edge_size = 4, feature_position = True, 
                use_dropout = False, lr = 0.0001 , print_bar = False, graph_embedding = False, mlp_hidden = 512, num_classes = 4, size_info = False, batch_size = 128, modif_accelerate = False, num_workers = 0):


    dataset_train = ZincSubgraphDatasetStep(data_path = datapath_train, GNN_type=3, feature_position=feature_position)
    dataset_val = ZincSubgraphDatasetStep(data_path = datapath_val, GNN_type=3, feature_position=feature_position)
    if feature_position :
        encoding_size += 1
    
    loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers = num_workers, collate_fn=custom_collate_GNN3)
    loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, num_workers = num_workers, collate_fn=custom_collate_GNN3)

    if graph_embedding:
            if modif_accelerate :
                model = ModelWithgraph_embedding_modif(in_channels = encoding_size,hidden_channels_list= GCN_size, mlp_hidden_channels= mlp_hidden, 
                                             edge_channels= edge_size, num_classes= num_classes,size_info= size_info)
            else:
                model = ModelWithgraph_embedding(in_channels = encoding_size,hidden_channels_list= GCN_size, mlp_hidden_channels= mlp_hidden, 
                                             edge_channels= edge_size, num_classes= num_classes,size_info= size_info)
    else:
        model = ModelWithEdgeFeatures(in_channels=encoding_size, hidden_channels_list= GCN_size, edge_channels=edge_size, use_dropout=use_dropout)
    
    
            

    


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Set up the loss function for multiclass 
    criterion = nn.CrossEntropyLoss()
    training_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector','pseudo_precision', 'pseudo_recall' , 'pseudo_recall_placed', 'pseudo_recall_type','conditionnal_precision_placed', 'f1_score'])
    eval_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector','pseudo_precision', 'pseudo_recall' , 'pseudo_recall_placed', 'pseudo_recall_type','conditionnal_precision_placed', 'f1_score'])

    directory_path_experience = os.path.join("./experiments", name)
    directory_path_epochs = os.path.join(directory_path_experience,"history_training")
    if not os.path.exists(directory_path_experience):
        # Create the directory if it doesn't exist
        os.makedirs(directory_path_experience)
        print(f"The '{name}' directory has been successfully created in the 'experiments' directory.")
    else:
        # Display a message if the directory already exists
        print(f"The '{name}' directory already exists in the 'experiments' directory.")

    if not os.path.exists(directory_path_epochs) :
        os.makedirs(directory_path_epochs)
    
    file_path = os.path.join(directory_path_experience, "parameters.txt")
    
    parameters = {
    " type GNN " : "GNN3",
    "datapath_train": datapath_train,
    "datapath_val": datapath_val,
    "n_epochs": n_epochs,
    "encoding_size": encoding_size,
    "GCN_size": GCN_size,
    "mlp_size": mlp_size,
    "edge_size": edge_size,
    "feature_position": feature_position,
    "use_dropout": use_dropout,
    "lr": lr,
    "print_bar": print_bar
    }
    with open(file_path, "w") as file:
        for param, value in parameters.items():
            # Convert lists to strings if necessary
            if isinstance(value, list):
                value = ', '.join(str(item) for item in value)
            line = f"{param}: {value}\n"
            file.write(line)

    #beginning the epoch
    for epoch in range(0, n_epochs+1):
        loss, avg_output_vector, avg_label_vector,  pseudo_precision, pseudo_recall , pseudo_recall_placed, pseudo_recall_type, conditionnal_precision_placed, f1_score = train_one_epoch(
            loader_train, model, edge_size, device, optimizer, criterion, print_bar = print_bar)
        training_history.loc[epoch] = [epoch, loss, avg_output_vector, avg_label_vector,  pseudo_precision, pseudo_recall , pseudo_recall_placed, pseudo_recall_type,conditionnal_precision_placed, f1_score]
        loss, avg_output_vector, avg_label_vector,  pseudo_precision, pseudo_recall , pseudo_recall_placed, pseudo_recall_type,conditionnal_precision_placed, f1_score  = eval_one_epoch(
            loader_val, model, edge_size, device, criterion, print_bar = print_bar)
        eval_history.loc[epoch] = [epoch, loss, avg_output_vector, avg_label_vector,  pseudo_precision, pseudo_recall , pseudo_recall_placed, pseudo_recall_type,conditionnal_precision_placed, f1_score]

    #save the model(all with optimizer step, the loss ) every 5 epochs

        save_every_n_epochs = 10
        if (epoch) % save_every_n_epochs == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # Add any other relevant information you want to save here
            }
            epoch_save_file = os.path.join(directory_path_epochs, f'checkpoint_epoch_{epoch}_{name}.pt')
            torch.save(checkpoint, epoch_save_file)

        training_csv_directory = os.path.join(directory_path_experience, 'training_history.csv')    
        training_history.to_csv(training_csv_directory)

        eval_csv_directory = os.path.join(directory_path_experience, 'eval_history.csv')    
        eval_history.to_csv(eval_csv_directory)
    
