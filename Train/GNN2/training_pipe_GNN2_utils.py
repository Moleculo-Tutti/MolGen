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
from Model.GNN2 import ModelWithEdgeFeatures, ModelWithNodeConcat
from Model.metrics import pseudo_accuracy_metric, pseudo_recall_for_each_class, pseudo_precision_for_each_class

def train_one_epoch(loader, model, size_edge, device, optimizer, criterion, epoch_metric, print_bar = False):
    
    model.train()

    total_loss = 0
    num_correct = 0
    num_correct_recall = torch.zeros(size_edge)
    num_correct_precision = torch.zeros(size_edge)
    count_per_class_recall = torch.zeros(size_edge)
    count_per_class_precision = torch.zeros(size_edge)
    avg_output_vector = np.zeros(size_edge)  # Initialize the average output vector
    avg_label_vector = np.zeros(size_edge)  # Initialize the average label vector
    total_graphs_processed = 0

    if print_bar:
        progress_bar = tqdm_notebook(loader, desc="Training", unit="batch")
    else:
        progress_bar = tqdm(loader, desc="Training", unit="batch")

    for batch_idx, batch in enumerate(progress_bar):

        data = batch[0]
        terminal_node_infos = batch[1]
        data = data.to(device)
        optimizer.zero_grad()
        logit_out = model(data)
        terminal_node_infos = terminal_node_infos.to(device)

        out = F.softmax(logit_out, dim=1)
        loss = criterion(logit_out, terminal_node_infos)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * data.num_graphs

        if epoch_metric:
            # Metrics part
            num_correct += pseudo_accuracy_metric(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)
            recall_output = pseudo_recall_for_each_class(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)
            precision_output = pseudo_precision_for_each_class(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)
            num_correct_recall += recall_output[0]
            num_correct_precision += precision_output[0]
            count_per_class_recall += recall_output[1]
            count_per_class_precision += precision_output[1]
            
            if print_bar:
                loss_value = total_loss / (data.num_graphs * (progress_bar.last_print_n + 1))


            # Update the average output vector
            avg_output_vector += out.detach().cpu().numpy().mean(axis=0) * data.num_graphs
            avg_label_vector += terminal_node_infos.detach().cpu().numpy().mean(axis=0) * data.num_graphs
            total_graphs_processed += data.num_graphs
            current_avg_output_vector = avg_output_vector / total_graphs_processed
            current_avg_label_vector = avg_label_vector / total_graphs_processed
            avg_correct = num_correct / total_graphs_processed
            avg_correct_recall = num_correct_recall / count_per_class_recall
            avg_correct_precision = num_correct_precision / count_per_class_precision
            avg_f1 = 2 * (avg_correct_recall * avg_correct_precision) / (avg_correct_recall + avg_correct_precision)
            if print_bar :
                progress_bar.set_postfix(loss=loss_value, avg_output_vector=current_avg_output_vector, 
                                    avg_label_vector=current_avg_label_vector, 
                                    avg_correct=avg_correct, num_correct=num_correct, 
                                    total_graphs_processed=total_graphs_processed, 
                                    avg_correct_precision=avg_correct_precision, 
                                    avg_correct_recall=avg_correct_recall, 
                                    avg_f1=avg_f1,
                                    count_per_class_precision=count_per_class_precision,
                                    count_per_class_recall=count_per_class_recall)


    if epoch_metric:
        return total_loss / len(loader.dataset), current_avg_label_vector, current_avg_output_vector, avg_correct , avg_correct_precision, avg_correct_recall
    else:
        return total_loss / len(loader.dataset), None, None, None, None, None


def eval_one_epoch(loader, model, edge_size, device, criterion, print_bar = False, val_metric_size = 1):
    model.eval()
    total_loss = 0
    num_correct = 0
    num_correct_recall = torch.zeros(edge_size)
    num_correct_precision = torch.zeros(edge_size)
    count_per_class_recall = torch.zeros(edge_size)
    count_per_class_precision = torch.zeros(edge_size)

    avg_output_vector = np.zeros(edge_size)  # Initialize the average output vector
    avg_label_vector = np.zeros(edge_size)  # Initialize the average label vector
    total_graphs_processed = 0

    for i in tqdm(range(val_metric_size)):

        for batch_idx, batch in enumerate(loader):
            data = batch[0]
            terminal_node_infos = batch[1]
            data = data.to(device)

            logit_out = model(data)
            terminal_node_infos = terminal_node_infos.to(device)

            out = F.softmax(logit_out, dim=1)
            loss = criterion(logit_out, terminal_node_infos)

            # Metrics part
            num_correct += pseudo_accuracy_metric(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)
            recall_output = pseudo_recall_for_each_class(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)
            precision_output = pseudo_precision_for_each_class(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)
            num_correct_recall += recall_output[0]
            num_correct_precision += precision_output[0]
            count_per_class_recall += recall_output[1]
            count_per_class_precision += precision_output[1]
            total_loss += loss.item() * data.num_graphs


            # Update the average output vector
            avg_output_vector += out.detach().cpu().numpy().mean(axis=0) * data.num_graphs
            avg_label_vector += terminal_node_infos.detach().cpu().numpy().mean(axis=0) * data.num_graphs
            total_graphs_processed += data.num_graphs

        current_avg_output_vector = avg_output_vector / total_graphs_processed
        current_avg_label_vector = avg_label_vector / total_graphs_processed
        avg_correct = num_correct / total_graphs_processed
        avg_correct_recall = num_correct_recall / count_per_class_recall
        avg_correct_precision = num_correct_precision / count_per_class_precision

    return total_loss / (val_metric_size * len(loader.dataset)), current_avg_label_vector, current_avg_output_vector, avg_correct , avg_correct_precision, avg_correct_recall






def train_GNN2(name : str, datapath_train, datapath_val, n_epochs,  encoding_size, GCN_size : list, mlp_size, edge_size = 4, feature_position = True, use_dropout = False, lr = 0.0001 , print_bar = False, batch_size = 128, num_workers = 0):

    dataset_train = ZincSubgraphDatasetStep(data_path = datapath_train, GNN_type=2)
    dataset_val = ZincSubgraphDatasetStep(data_path = datapath_val, GNN_type=2)
    if feature_position :
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_passive_add_feature_GNN2, num_workers=num_workers)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_passive_add_feature_GNN2, num_workers=num_workers)
        model = ModelWithEdgeFeatures(in_channels=encoding_size + 1, hidden_channels_list= GCN_size, mlp_hidden_channels=mlp_size, edge_channels=edge_size, num_classes=edge_size, use_dropout=use_dropout)
    else :
        loader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_GNN2, num_workers=num_workers)
        loader_val = DataLoader(dataset_val, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_GNN2, num_workers=num_workers)
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
    for epoch in tqdm(range(0, n_epochs+1)):
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

class TrainGNN2():
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
        self.print_bar = config['print_bar']
        self.num_workers = config['num_workers']
        self.every_epoch_save = config['every_epoch_save']
        self.every_epoch_metric = config['every_epoch_metric']
        self.val_metric_size = config['val_metric_size']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.node_embeddings = config['node_embeddings']
        print(f"Training on {self.device}")

        print(f"Loading data...")
        self.loader_train, self.loader_val, self.model, self.encoding_size, self.edge_size = self.load_data_model()
        print(f"Data loaded")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = nn.CrossEntropyLoss()

        self.training_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector', 'avg_correct', 'precision', 'recall'])
        self.eval_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector', 'avg_correct', 'precision', 'recall'])

        self.prepare_saving()

    def load_data_model(self):
        # Load the data
        dataset_train = ZincSubgraphDatasetStep(self.datapath_train, GNN_type=2, feature_position=self.feature_position)
        dataset_val = ZincSubgraphDatasetStep(self.datapath_val, GNN_type=2, feature_position=self.feature_position)
        
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers, collate_fn=custom_collate_GNN2)
        loader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers, collate_fn=custom_collate_GNN2)

        encoding_size = dataset_train.encoding_size
        edge_size = dataset_train.edge_size

        if self.node_embeddings :
            model = ModelWithNodeConcat(in_channels= encoding_size+ int(self.feature_position), 
                                        hidden_channels_list= self.GCN_size,
                                        mlp_hidden_channels= self.mlp_hidden,
                                        edge_channels= edge_size,
                                        encoding_size= encoding_size,
                                        num_classes=  edge_size)
        
        else :
        # Load the model
            model = ModelWithEdgeFeatures(in_channels=encoding_size + int(self.feature_position), # We increase the input size to take into account the feature position
                                      hidden_channels_list=self.GCN_size, 
                                      mlp_hidden_channels=self.mlp_hidden, 
                                      edge_channels=edge_size, 
                                      use_dropout=self.use_dropout,
                                      num_classes=edge_size)
        
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
            
            if epoch % self.every_epoch_metric == 0:
                loss, avg_label_vector, avg_output_vector, avg_correct, avg_correct_precision, avg_correct_recall = train_one_epoch(
                    loader=self.loader_train,
                    model=self.model,
                    size_edge=self.edge_size,
                    device=self.device,
                    optimizer=self.optimizer,
                    epoch_metric = True,
                    criterion=self.criterion,
                    print_bar = self.print_bar)
                
                self.training_history.loc[epoch] = [epoch, loss, avg_output_vector, avg_label_vector, avg_correct, avg_correct_precision, avg_correct_recall]

                loss, avg_label_vector, avg_output_vector, avg_correct, avg_correct_precision, avg_correct_recall = eval_one_epoch(
                    loader=self.loader_val,
                    model=self.model,
                    edge_size=self.edge_size,
                    device=self.device,
                    criterion=self.criterion,
                    print_bar = self.print_bar,
                    val_metric_size = self.val_metric_size)
                
                self.eval_history.loc[epoch] = [epoch, loss, avg_output_vector, avg_label_vector, avg_correct, avg_correct_precision, avg_correct_recall]
            else:
                loss, _, _, _, _, _ = train_one_epoch(
                    loader=self.loader_train,
                    model=self.model,
                    size_edge=self.edge_size,
                    device=self.device,
                    optimizer=self.optimizer,
                    epoch_metric = False,
                    criterion=self.criterion,
                    print_bar = self.print_bar)
                
                self.training_history.loc[epoch] = [epoch, loss, None, None, None, None, None]
                self.eval_history.loc[epoch] = [epoch, None, None, None, None, None, None]

            if (epoch) % self.every_epoch_save == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    # Add any other relevant information you want to save here
                }
                epoch_save_file = os.path.join(self.directory_path_epochs, f'checkpoint_epoch_{epoch}_{self.name}.pt')
                torch.save(checkpoint, epoch_save_file)

                training_csv_directory = os.path.join(self.directory_path_experience, 'training_history.csv')    
                self.training_history.to_csv(training_csv_directory)

                eval_csv_directory = os.path.join(self.directory_path_experience, 'eval_history.csv')    
                self.eval_history.to_csv(eval_csv_directory)