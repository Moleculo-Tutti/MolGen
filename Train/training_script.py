import torch
import random

import pandas as pd
import numpy as np

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn

from torch.optim import AdamW

import torch_geometric.transforms as T

from torch_geometric.data import Batch

from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.nn import global_add_pool
from torch_geometric.nn import GraphConv
from torch.utils.data import DataLoader

from sklearn.metrics import mean_squared_error

from pathlib import Path

from tqdm import tqdm


from DataPipeline.dataset import ZincSubgraphDataset, custom_collate
from Model.GNN1 import ModelWithEdgeFeatures
from Model.metrics import pseudo_accuracy_metric, pseudo_recall_for_each_class, pseudo_precision_for_each_class, FocalLoss


def train(model, optimizer, loss_function, loader, device, epoch):
    model.train()
    total_loss = 0
    mse_sum = 0
    num_correct = 0
    num_correct_recall = torch.zeros(10)
    num_correct_precision = torch.zeros(10)
    count_per_class_recall = torch.zeros(10)
    count_per_class_precision = torch.zeros(10)
    progress_bar = tqdm(loader, desc="Training", unit="batch")

    avg_output_vector = np.zeros(10)  # Initialize the average output vector
    avg_label_vector = np.zeros(10)  # Initialize the average label vector
    total_graphs_processed = 0

    

    for batch_idx, batch in enumerate(progress_bar):
        data = batch[0]
        terminal_node_infos = batch[1]
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        terminal_node_infos = terminal_node_infos.to(device)

        loss = loss_function(out, terminal_node_infos)
        num_correct += pseudo_accuracy_metric(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)

        recall_output = pseudo_recall_for_each_class(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)
        precision_output = pseudo_precision_for_each_class(out.detach().cpu(), terminal_node_infos.detach().cpu(), random=True)
        num_correct_recall += recall_output[0]
        num_correct_precision += precision_output[0]
        count_per_class_recall += recall_output[1]
        count_per_class_precision += precision_output[1]
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data.num_graphs
        loss_value = total_loss / (data.num_graphs * (progress_bar.last_print_n + 1))

        # Compute MSE
        mse = mean_squared_error(terminal_node_infos.detach().cpu(), out.detach().cpu())
        mse_sum += mse * data.num_graphs
        mse_value = mse_sum / (data.num_graphs * (progress_bar.last_print_n + 1))

        # Update the average output vector
        avg_output_vector += out.detach().cpu().numpy().mean(axis=0) * data.num_graphs
        avg_label_vector += terminal_node_infos.detach().cpu().numpy().mean(axis=0) * data.num_graphs
        total_graphs_processed += data.num_graphs
        current_avg_output_vector = avg_output_vector / total_graphs_processed
        current_avg_label_vector = avg_label_vector / total_graphs_processed
        avg_correct = num_correct / total_graphs_processed
        avg_correct_recall = num_correct_recall / count_per_class_recall
        avg_correct_precision = num_correct_precision / count_per_class_precision
        progress_bar.set_postfix(loss=loss_value, mse=mse_value, avg_output_vector=current_avg_output_vector, 
                                 avg_label_vector=current_avg_label_vector, 
                                 avg_correct=avg_correct, num_correct=num_correct, 
                                 total_graphs_processed=total_graphs_processed, 
                                 avg_correct_precision=avg_correct_precision, 
                                 avg_correct_recall=avg_correct_recall, 
                                 count_per_class_precision=count_per_class_precision,
                                 count_per_class_recall=count_per_class_recall)


    return total_loss / len(loader.dataset), current_avg_label_vector, current_avg_output_vector, avg_correct, mse_value

def complete_training(model, optimizer, loss_function, loader, device, n_epochs, save=False):

    training_history = pd.DataFrame(columns=['epoch', 'loss', 'mse', 'avg_output_vector', 'avg_label_vector'])

    for epoch in range(1, n_epochs+1):
        loss, avg_label_vector, avg_output_vector, avg_correct, mse = train(loader, epoch)
        training_history = training_history.append({'epoch': epoch, 'loss': loss, 'mse': mse, 'avg_output_vector': avg_output_vector, 'avg_label_vector': avg_label_vector, 'avg_correct': avg_correct}, ignore_index=True)
        #save the model(all with optimizer step, the loss ) every 5 epochs
        
        if save:
            save_every_n_epochs = 5
            if (epoch) % save_every_n_epochs == 0:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    # Add any other relevant information you want to save here
                }
                torch.save(checkpoint, f'checkpoint_epoch_{epoch+1}_{name}.pt')
            
        #save the training history every 10 epochs
        if epoch % 1 == 0:
            training_history.to_csv(f"training_history_{name}.csv", index=False)
        print(f'Epoch: {epoch}, Loss: {loss:.8f}')

def main():
    print('Loading data...')
    datapath = Path('..') / 'DataPipeline/data/preprocessed_graph.pt'
    dataset = ZincSubgraphDataset(data_path = datapath)
    print('Data loaded!')
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, num_workers=0, collate_fn=custom_collate)