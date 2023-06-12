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
import json
import gc

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

from DataPipeline.dataset import ZincSubgraphDatasetStep, custom_collate_passive_add_feature_GNN2, custom_collate_GNN2, ZincSubgraphDatasetStep_mutlithread
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






class TrainGNN2():
    def __init__(self, config, continuue_training = False, checkpoint = None):
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
        self.node_embeddings = config['node_embeddings']
        self.max_size = config['max_size']
        self.use_size = config['use_size']
        self.score_list = config['score_list']
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_multithreading = config['use_multithreading']
        self.continue_training = continuue_training

        print(f"Training on {self.device}")

        print(f"Loading data...")
        self.loader_train, self.loader_val, self.model, self.encoding_size, self.edge_size = self.load_data_model()
        print(f"Data loaded")

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.begin_epoch = 0
        if self.continue_training:
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.begin_epoch = checkpoint['epoch']
        self.criterion = nn.CrossEntropyLoss()

        self.training_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector', 'avg_correct', 'precision', 'recall'])
        self.eval_history = pd.DataFrame(columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector', 'avg_correct', 'precision', 'recall'])

        self.prepare_saving()

        # Store the 6 best models
        self.six_best_eval_loss = [(0, float('inf'))] * 6

    def load_data_model(self):
        # Load the data
        if self.use_multithreading:
            dataset_train = ZincSubgraphDatasetStep_mutlithread(self.datapath_train, GNN_type=2, feature_position=self.feature_position, scores_list=self.score_list)
            dataset_val = ZincSubgraphDatasetStep_mutlithread(self.datapath_val, GNN_type=2, feature_position=self.feature_position, scores_list=self.score_list)
        else:
            dataset_train = ZincSubgraphDatasetStep(self.datapath_train, GNN_type=2, feature_position=self.feature_position, scores_list=self.score_list)
            dataset_val = ZincSubgraphDatasetStep(self.datapath_val, GNN_type=2, feature_position=self.feature_position, scores_list=self.score_list)
        
        loader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True, num_workers = self.num_workers, collate_fn=custom_collate_GNN2)
        loader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers = self.num_workers, collate_fn=custom_collate_GNN2)

        encoding_size = dataset_train.encoding_size
        edge_size = dataset_train.edge_size

        if self.node_embeddings :
            model = ModelWithNodeConcat(in_channels= encoding_size+ int(self.feature_position) + int(len(self.score_list)), # We increase the input size to take into account the feature position and the node embeddings (if we use them)
                                        hidden_channels_list= self.GCN_size,
                                        mlp_hidden_channels= self.mlp_hidden,
                                        edge_channels= edge_size,
                                        encoding_size= encoding_size,
                                        num_classes=  edge_size,
                                        max_size= self.max_size)

        
        else :
        # Load the model
            model = ModelWithEdgeFeatures(in_channels=encoding_size + int(self.feature_position) + int(len(self.score_list)), # We increase the input size to take into account the feature position
                                      hidden_channels_list=self.GCN_size, 
                                      mlp_hidden_channels=self.mlp_hidden, 
                                      edge_channels=edge_size, 
                                      use_dropout=self.use_dropout,
                                      num_classes=edge_size,
                                      size_info=self.use_size,
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
        
        file_path = os.path.join(self.directory_path_experience, "parameters.json")
        
        with open(file_path, 'w') as file :
            json.dump(self.config, file)
        
        training_csv_directory = os.path.join(self.directory_path_experience, 'training_history.csv')
        eval_csv_directory = os.path.join(self.directory_path_experience, 'eval_history.csv')
        if os.path.exists(training_csv_directory):
            self.training_history = pd.read_csv(training_csv_directory)
            self.eval_history = pd.read_csv(eval_csv_directory)
        else:
            self.training_history = pd.DataFrame(
                columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector', 'avg_correct', 'precision', 'recall'])
            self.eval_history = pd.DataFrame(
                columns=['epoch', 'loss', 'avg_output_vector', 'avg_label_vector', 'avg_correct', 'precision', 'recall'])
    


    def train(self):

        for epoch in tqdm(range(self.begin_epoch, self.n_epochs+1)):
            torch.cuda.empty_cache()
            save_epoch = False
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
                
                # Check if the loss is better than one of the 6 best losses (compare only along the second dimension of the tuples)

                if loss < max(self.six_best_eval_loss, key=lambda x: x[1])[1]:
                    # switch the save variable to True
                    save_epoch = True
                    index_max = self.six_best_eval_loss.index(max(self.six_best_eval_loss, key=lambda x: x[1]))
                    self.six_best_eval_loss[index_max] = (epoch, loss)
            
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
                if os.path.exists(training_csv_directory):
                    self.training_history.to_csv(training_csv_directory, mode='a', header=False)
                else:
                    self.training_history.to_csv(training_csv_directory)   

                eval_csv_directory = os.path.join(self.directory_path_experience, 'eval_history.csv')    
                if os.path.exists(eval_csv_directory):
                    self.eval_history.to_csv(eval_csv_directory, mode='a', header=False)
                else:
                    self.eval_history.to_csv(eval_csv_directory)

                # Create a txt file containing the infos about the six best epochs saved 
                six_best_epochs_file = os.path.join(self.directory_path_experience, 'six_best_epochs.txt')
                with open(six_best_epochs_file, 'w') as file:
                    for epoch, loss in self.six_best_eval_loss:
                        file.write(f'Epoch {epoch} with loss {loss}\n')
            gc.collect()