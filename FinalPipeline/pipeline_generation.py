import torch
import torch_geometric
import sys
from tqdm import tqdm
import pandas as pd
from path import Path
import argparse
import json
import os


from utils import GenerationModule, MolGen, tensor_to_smiles



def parse_list_of_floats(input_string):
    return [float(item) for item in input_string.split(',')]

# Convert the graph to smiles

def convert_to_smiles(graph, kekulize=True, encoding_type='charged'):
    smiles = []
    for g in graph:
        smiles.append(tensor_to_smiles(g.x, g.edge_index, g.edge_attr, edge_mapping=kekulize, encoding_type=encoding_type))
    return smiles

def load_best_models(path):
    with open(path/'six_best_epochs.txt', 'r') as f:
        lines = f.readlines()
        epoch_values = [float(line.split(' ')[1]) for line in lines]
        best_line_index = epoch_values.index(max(epoch_values))
        loss_value = float(lines[best_line_index].split(' ')[-1])
    print('Loading best checkpoint number {} of the epoch {} with a loss of {}'.format(best_line_index, epoch_values[best_line_index], loss_value))
    checkpoint_path = path / 'history_training' / f'checkpoint_{best_line_index}.pt'
    return checkpoint_path

def main(args):

    # Read the configs as json 

    experiment_name = args.exp_name

    experiment_path = Path('..') / 'trained_models' / experiment_name

    # List the folders in the experiment path

    folders = os.listdir(experiment_path)
    for folder in folders:
        if folder.startswith('GNN1'):
            GNN1_path = experiment_path / folder 
        elif folder.startswith('GNN2'):
            GNN2_path = experiment_path / folder
        elif folder.startswith('GNN3'):
            GNN3_path = experiment_path / folder

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
    else:
        edge_size = 4

    module = GenerationModule(config1=config1, 
                            config2=config2, 
                            config3=config3, 
                            encoding_size = encoding_size,
                            edge_size = edge_size, 
                            pathGNN1=GNN1_path, 
                            pathGNN2=GNN2_path, 
                            pathGNN3=GNN3_path,
                            encoding_type=args.encod,
                            desired_score_list=args.desired_scores_list)

    graph_batch = module.generate_mol_list(args.n_mols)

    if args.keku:
        edge_mapping='kekulized'
    else:
        edge_mapping='aromatic'

    molecules = convert_to_smiles(graph_batch, kekulize=edge_mapping, encoding_type=args.encod)

    #Save the molecule list in a csv file
    df = pd.DataFrame(molecules, columns=['SMILES'])
    df_name = 'generated_molecules_{0}_{1}_{2}_{3}_{4}.csv'.format(args.n_mols, args.keku, args.encod, args.exp_name, args.desired_scores_list)
    df.to_csv(df_name, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mols', type=int, default=1000, help='Number of molecules to generate')
    parser.add_argument('--keku', type=bool, default=True, help='Kekulize the molecules')
    parser.add_argument('--encod', type=str, default='charged', help='Encoding type')
    parser.add_argument('--exp_name', type=str, default='exp', help='Name of the experience')
    parser.add_argument('--desired_scores_list', type=parse_list_of_floats, default="0.5,0.5,0.5", help='Desired scores for the molecules')

    args = parser.parse_args()
    main(args)
