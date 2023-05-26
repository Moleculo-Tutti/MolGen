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


import sys
import os

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

from DataPipeline.dataset import ZincSubgraphDatasetStep, custom_collate_passive_add_feature_GNN2, custom_collate_GNN2
from Model.GNN2 import ModelWithEdgeFeatures
from Model.metrics import pseudo_accuracy_metric, pseudo_recall_for_each_class, pseudo_precision_for_each_class



def train_one_epoch(loader, model, size_edge, device, optimizer, criterion, print_bar = False):
    model.train()
    total_loss = 0
    num_correct = 0
    num_correct_recall = torch.zeros(size_edge)
    num_correct_precision = torch.zeros(size_edge)
    count_per_class_recall = torch.zeros(size_edge)
    count_per_class_precision = torch.zeros(size_edge)
    progress_bar = tqdm_notebook(loader, desc="Training", unit="batch")

    avg_output_vector = np.zeros(size_edge)  # Initialize the average output vector
    avg_label_vector = np.zeros(size_edge)  # Initialize the average label vector
    total_graphs_processed = 0

    

    for batch_idx, batch in enumerate(progress_bar):
        data = batch[0]
        edge_infos = batch[1]
        data = data.to(device)
        optimizer.zero_grad()
        logit_out = model(data)
        edge_infos = edge_infos.to(device)

        loss = criterion(logit_out, edge_infos)
       
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        loss_value = total_loss / (data.num_graphs * (progress_bar.last_print_n + 1))
        
        out = F.softmax(logit_out, dim=1)

        # Collect true labels and predicted labels
        num_correct += pseudo_accuracy_metric(out.detach().cpu(), edge_infos.detach().cpu(), random=True)

        recall_output = pseudo_recall_for_each_class(out.detach().cpu(), edge_infos.detach().cpu(), random=True)
        precision_output = pseudo_precision_for_each_class(out.detach().cpu(), edge_infos.detach().cpu(), random=True)
        num_correct_recall += recall_output[0]
        num_correct_precision += precision_output[0]
        count_per_class_recall += recall_output[1]
        count_per_class_precision += precision_output[1]

        current_avg_output_vector = avg_output_vector / total_graphs_processed
        current_avg_label_vector = avg_label_vector / total_graphs_processed
        
        # Update the average output vector
        avg_output_vector += out.detach().cpu().numpy().mean(axis=0) * data.num_graphs
        avg_label_vector += edge_infos.detach().cpu().numpy().mean(axis=0) * data.num_graphs
        total_graphs_processed += data.num_graphs
        current_avg_output_vector = avg_output_vector / total_graphs_processed
        current_avg_label_vector = avg_label_vector / total_graphs_processed
        avg_correct = num_correct / total_graphs_processed
        avg_correct_recall = num_correct_recall / count_per_class_recall
        avg_correct_precision = num_correct_precision / count_per_class_precision
        avg_f1 = 2 * (avg_correct_recall * avg_correct_precision) / (avg_correct_recall + avg_correct_precision)
        if print_bar:
            progress_bar.set_postfix(loss=loss_value, avg_output_vector=current_avg_output_vector, 
                                 avg_label_vector=current_avg_label_vector, 
                                 avg_correct=avg_correct, num_correct=num_correct, 
                                 total_graphs_processed=total_graphs_processed, 
                                 avg_correct_precision=avg_correct_precision, 
                                 avg_correct_recall=avg_correct_recall, 
                                 avg_f1=avg_f1,
                                 count_per_class_precision=count_per_class_precision,
                                 count_per_class_recall=count_per_class_recall)





    return total_loss / len(loader.dataset), current_avg_label_vector, current_avg_output_vector, avg_correct , avg_correct_precision, avg_correct_recall

def eval_one_epoch(loader, model, size_edge, device, optimizer, criterion):
    model.eval()
    total_loss = 0
    num_correct = 0
    num_correct_recall = torch.zeros(size_edge)
    num_correct_precision = torch.zeros(size_edge)
    count_per_class_recall = torch.zeros(size_edge)
    count_per_class_precision = torch.zeros(size_edge)


    avg_output_vector = np.zeros(size_edge)  # Initialize the average output vector
    avg_label_vector = np.zeros(size_edge)  # Initialize the average label vector
    total_graphs_processed = 0

    

    for batch_idx, batch in enumerate(loader):
        data = batch[0]
        edge_infos = batch[1]
        data = data.to(device)
        optimizer.zero_grad()
        logit_out = model(data)
        edge_infos = edge_infos.to(device)

        loss = criterion(logit_out, edge_infos)
       
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        
        out = F.softmax(logit_out, dim=1)

        # Collect true labels and predicted labels
        num_correct += pseudo_accuracy_metric(out.detach().cpu(), edge_infos.detach().cpu(), random=True)

        recall_output = pseudo_recall_for_each_class(out.detach().cpu(), edge_infos.detach().cpu(), random=True)
        precision_output = pseudo_precision_for_each_class(out.detach().cpu(), edge_infos.detach().cpu(), random=True)
        num_correct_recall += recall_output[0]
        num_correct_precision += precision_output[0]
        count_per_class_recall += recall_output[1]
        count_per_class_precision += precision_output[1]

        current_avg_output_vector = avg_output_vector / total_graphs_processed
        current_avg_label_vector = avg_label_vector / total_graphs_processed
        
        # Update the average output vector
        avg_output_vector += out.detach().cpu().numpy().mean(axis=0) * data.num_graphs
        avg_label_vector += edge_infos.detach().cpu().numpy().mean(axis=0) * data.num_graphs
        total_graphs_processed += data.num_graphs
        current_avg_output_vector = avg_output_vector / total_graphs_processed
        current_avg_label_vector = avg_label_vector / total_graphs_processed
        avg_correct = num_correct / total_graphs_processed
        avg_correct_recall = num_correct_recall / count_per_class_recall
        avg_correct_precision = num_correct_precision / count_per_class_precision




    return total_loss / len(loader.dataset), current_avg_label_vector, current_avg_output_vector, avg_correct , avg_correct_precision, avg_correct_recall






def train_GNN2(name : str, datapath_train, datapath_val, n_epochs,  encoding_size, GCN_size : list, mlp_size, edge_size = 4, feature_position = True, use_dropout = False, lr = 0.0001 , print_bar = False):

    dataset_train = ZincSubgraphDatasetStep(data_path = datapath_train, GNN_type=2)
    dataset_val = ZincSubgraphDatasetStep(data_path = datapath_val, GNN_type=2)
    if feature_position :
        loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, collate_fn=custom_collate_passive_add_feature_GNN2)
        loader_val = DataLoader(dataset_val, batch_size=128, shuffle=True, collate_fn=custom_collate_passive_add_feature_GNN2)
        model = ModelWithEdgeFeatures(in_channels=encoding_size + 1, hidden_channels_list= GCN_size, mlp_hidden_channels=mlp_size, edge_channels=edge_size, num_classes=edge_size, use_dropout=use_dropout)
    else :
        loader_train = DataLoader(dataset_train, batch_size=128, shuffle=True, collate_fn=custom_collate_GNN2)
        loader_val = DataLoader(dataset_val, batch_size=128, shuffle=True, collate_fn=custom_collate_GNN2)
        model = ModelWithEdgeFeatures(in_channels=encoding_size, hidden_channels_list= GCN_size, mlp_hidden_channels=mlp_size, edge_channels=edge_size, num_classes=edge_size, use_dropout=use_dropout)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    # Set up the loss function for multiclass 
    criterion = nn.CrossEntropyLoss()
    training_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector', 'avg_correct', 'precision', 'recall'])
    eval_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector', 'avg_correct', 'precision', 'recall'])

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
        loss, avg_label_vector, avg_output_vector, avg_correct, avg_correct_precision, avg_correct_recall = train_one_epoch(
            loader_train, model, edge_size, device, optimizer, criterion, print_bar = print_bar)
        training_history.loc[epoch] = [epoch, loss, avg_output_vector, avg_label_vector, avg_correct, avg_correct_precision, avg_correct_recall]
        loss, avg_label_vector, avg_output_vector, avg_correct, avg_correct_precision, avg_correct_recall = eval_one_epoch(
            loader_val, model, edge_size, device, optimizer, criterion)
        eval_history.loc[epoch] = [epoch, loss, avg_output_vector, avg_label_vector, avg_correct, avg_correct_precision, avg_correct_recall]

    #save the model(all with optimizer step, the loss ) every 5 epochs

        save_every_n_epochs = 20
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
    