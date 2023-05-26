from training_pipe_GNN3_utils import train_GNN3
from pathlib import Path
import argparse
import torch
from visualize import plot_history_GNN3

from training_pipe_GNN3_utils import train_GNN3
from pathlib import Path
import torch
import ast
from visualize import plot_history_GNN3

def main():
    # Prompt user for input
    name = input("Experiment name: ")
    datapath_train = Path('..') / '../DataPipeline/data/preprocessed_graph_train_no_I_Br_P.pt'
    datapath_val = Path('..') / '../DataPipeline/data/preprocessed_graph_val_no_I_Br_P.pt'
    encoding_size = int(input("Encoding size: "))
    GCN_size_str = input("GCN layer sizes (in the format [val1, val2, val3, ...]): ")
    GCN_size = ast.literal_eval(GCN_size_str)
    edge_size = int(input("Edge size: "))
    n_epochs = int(input("Number of epochs: "))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Call the train_GNN3 function with the provided arguments
    results = train_GNN3(name, datapath_train, datapath_val, n_epochs, encoding_size, GCN_size, edge_size,
                         feature_position=False, use_dropout=False, lr=0.0001, print_bar=True)

    # Set the directory path for saving the training history
    dirpath = Path('.') / 'experiments' / name / 'training_history.csv'
    # Call the plot_history_GNN3 function to visualize the training history
    plot_history_GNN3(dirpath)

if __name__ == '__main__':
    main()
