import torch
import torch_geometric


from utils import GenerationModule, MolGen, tensor_to_smiles
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
from path import Path
import argparse
import json


# Convert the graph to smiles

def convert_to_smiles(graph):
    smiles = []
    for g in graph:
        smiles.append(tensor_to_smiles(g.x, g.edge_index, g.edge_attr, edge_mapping='kekulized', encoding_type='polymer'))
    return smiles

# read the configs
name1 = 'GNN1_charged_kekulized'
epoch1 = 0
name2 = 'GNN2_charged_kekulized'
epoch2 = 0
name3 = 'GNN3_charged_kekulized'
epoch3 = 0
config1_path = Path('..') / 'Train' / 'GNN1' / 'config_GNN1.json'
config2_path = Path('..') / 'Train' / 'GNN2' / 'config_GNN2.json'
config3_path = Path('..') / 'Train' / 'GNN3' / 'config_GNN3.json'

# Read the config as a json

with open(config1_path, 'r') as f:
    config1 = json.load(f)
with open(config2_path, 'r') as f:
    config2 = json.load(f)
with open(config3_path, 'r') as f:
    config3 = json.load(f)

def main(args):

    GNN1_path = Path('.') / 'models/trained_models/GNN1_polymers_checkpoint_4.pt'
    GNN2_path = Path('.') / 'models/trained_models/GNN2_polymers_checkpoint_1.pt'
    GNN3_path = Path('.') / 'models/trained_models/GNN3_polymers_checkpoint_0.pt'

    module = GenerationModule(config1=config1, 
                            config2=config2, 
                            config3=config3, 
                            encoding_size = 8, 
                            edge_size = 3, 
                            pathGNN1=GNN1_path, 
                            pathGNN2=GNN2_path, 
                            pathGNN3=GNN3_path,
                            encoding_type='polymer')

    graph_batch = module.generate_mol_list(args.n_molecules)


    molecules = convert_to_smiles(graph_batch)

    #Save the molecule list in a csv file
    df = pd.DataFrame(molecules, columns=['SMILES'])
    df.to_csv('generated_molecules_{0}_polymers_kekulized.csv'.format(args.n_molecules), index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_molecules', type=int, default=1000, help='Number of molecules to generate')
    args = parser.parse_args()
    main(args)
