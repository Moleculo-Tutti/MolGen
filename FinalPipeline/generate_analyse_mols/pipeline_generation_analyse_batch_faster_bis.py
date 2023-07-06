import torch
import torch_geometric
import sys
from tqdm import tqdm
import pandas as pd
from path import Path
import argparse
import json
import os
from scipy.stats import wasserstein_distance

from rdkit import Chem


cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

from utils_faster import GenerationModuleBatchFasterDouble, tensor_to_smiles
from metrics.utils import canonic_smiles, SA, logP, QED, weight, size, get_n_rings, calculate_novelty, calculate_uniqueness, calculate_validity
from ploting_utils import save_plot_metrics
import concurrent


# All paths 

GENERATED_SAVING_DIR = Path('..') / 'generated_mols' / 'raw'
SCORED_SAVING_DIR = Path('..') / 'generated_mols' / 'scored'
ZINC_DATA_PATH = SCORED_SAVING_DIR / 'zinc_scored_filtered.csv'

SAVE_RESULTS_PATH = Path('..') / 'generate_analyse_mols' / 'results'

def parse_list_of_floats(input_string):
    if input_string == "":
        return []
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

def compute_scores(smiles):
    canonic = canonic_smiles(smiles)
    mol = Chem.MolFromSmiles(canonic)
    if mol is None:
        return None
    return {
        'smiles': canonic,
        'SA': SA(mol),
        'logP': logP(mol),
        'QED': QED(mol),
        'weight': weight(mol),
        'size': size(mol),
        'n_rings': get_n_rings(mol)
    }


import pandas as pd
import numpy as np


def calculate_statistics(data, zinc_data, file_path):
    """
    Calculates and stores statistics for various metrics.
    
    Args:
        data: DataFrame containing the data for the generated molecules.
        zinc_data: DataFrame containing the ZINC dataset.
        file_path: Path to the file to store the statistics.
    """
    
    # Compute validity, uniqueness and novelty
    data_smiles_list = data['smiles'].tolist()
    validity, valid_molecules = calculate_validity(data_smiles_list)
    uniqueness, unique_molecules = calculate_uniqueness(valid_molecules)
    novelty, non_new_mols = calculate_novelty(unique_molecules, zinc_data['smiles'].tolist())
    
    # compute waterstein distance between the generated and the zinc dataset

    # Calculate statistics for the generated dataset
    gen_statistics = {
        'Number of molecules': len(data),
        'Validity': validity,
        'Uniqueness': uniqueness,
        'Novelty': novelty,
        'SA mean': data['SA'].mean(),
        'SA std': data['SA'].std(),
        'logP mean': data['logP'].mean(),
        'logP std': data['logP'].std(),
        'QED mean': data['QED'].mean(),
        'QED std': data['QED'].std(),
        'Weight mean': data['weight'].mean(),
        'Weight std': data['weight'].std(),
        'Size mean': data['size'].mean(),
        'Size std': data['size'].std(),
        'Number of rings mean': data['n_rings'].mean(),
        'Number of rings std': data['n_rings'].std(),
        'Wasserstein distance SA': wasserstein_distance(data['SA'], zinc_data['SA']),
        'Wasserstein distance logP': wasserstein_distance(data['logP'], zinc_data['logP']),
        'Wasserstein distance QED': wasserstein_distance(data['QED'], zinc_data['QED']),
        'Wasserstein distance weight': wasserstein_distance(data['weight'], zinc_data['weight']),
        'Wasserstein distance number of rings': wasserstein_distance(data['n_rings'], zinc_data['n_rings']),
        'Average Wasserstein distance': np.mean([wasserstein_distance(data['SA'], zinc_data['SA']), wasserstein_distance(data['logP'], zinc_data['logP']), wasserstein_distance(data['QED'], zinc_data['QED']), wasserstein_distance(data['weight'], zinc_data['weight'])])
    }
    
    # Calculate statistics for the ZINC dataset
    # Note: This assumes that the ZINC dataset has the same columns (SA, logP, QED, etc.)
    zinc_statistics = {
        'Number of molecules': len(zinc_data),
        'SA mean': zinc_data['SA'].mean(),
        'SA std': zinc_data['SA'].std(),
        'logP mean': zinc_data['logP'].mean(),
        'logP std': zinc_data['logP'].std(),
        'QED mean': zinc_data['QED'].mean(),
        'QED std': zinc_data['QED'].std(),
        'Weight mean': zinc_data['weight'].mean(),
        'Weight std': zinc_data['weight'].std(),
        'Size mean': zinc_data['size'].mean(),
        'Size std': zinc_data['size'].std(),
        'Number of rings mean': zinc_data['n_rings'].mean(),
        'Number of rings std': zinc_data['n_rings'].std()
    }
    
    # Store the statistics in the file
    with open(file_path, 'w') as file:
        for dataset_name, dataset_stats in {'Generated Dataset': gen_statistics, 'ZINC Dataset': zinc_statistics}.items():
            file.write(f'{dataset_name}:\n')
            for stat_name, stat_value in dataset_stats.items():
                file.write(f'    {stat_name}: {stat_value}\n')
            file.write('\n')

    return valid_molecules

def main(args, n_threads=4):

    # Read the configs as json 

    experiment_name = args.exp_name

    experiment_path = Path('..') / '..' / 'trained_models' / experiment_name

    
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

    module = GenerationModuleBatchFasterDouble(config1=config1, 
                            config2=config2, 
                            config3=config3, 
                            encoding_size = encoding_size,
                            edge_size = edge_size, 
                            pathGNN1=GNN1_path, 
                            pathGNN2=GNN2_path, 
                            pathGNN3=GNN3_path,
                            encoding_type=args.encod,
                            batch_size=args.batch_size,
                            desired_score_list=args.desired_scores_list)

    print('Generating molecules...')
    graph_batch = module.generate_mol_list(args.n_mols // args.batch_size, n_threads=args.n_threads)

    if args.keku:
        edge_mapping='kekulized'
    else:
        edge_mapping='aromatic'

    molecules = convert_to_smiles(graph_batch, kekulize=edge_mapping, encoding_type=args.encod)

    #Save the molecule list in a csv file
    df = pd.DataFrame(molecules, columns=['SMILES'])
    df_name = 'generated_molecules_{0}_{1}_{2}_{3}_{4}.csv'.format(args.n_mols, args.keku, args.encod, args.exp_name, args.desired_scores_list)
    df.to_csv(GENERATED_SAVING_DIR / df_name, index=False)

    # Compute the scores
    print('Computing scores...')
    smiles_list = df['SMILES'].tolist()
    # Compute all scores in parallel using ThreadPoolExecutor
    scores_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        for scores in tqdm(executor.map(compute_scores, smiles_list),
                           total=len(smiles_list),
                           desc="Computing scores"):
            if scores is not None:
                scores_list.append(scores)

    # Save data in a new dataframe with the smiles and the scores
    data = pd.DataFrame()
    data['smiles'] = [scores['smiles'] for scores in scores_list]
    data['SA'] = [scores['SA'] for scores in scores_list]
    data['logP'] = [scores['logP'] for scores in scores_list]
    data['QED'] = [scores['QED'] for scores in scores_list]
    data['weight'] = [scores['weight'] for scores in scores_list]
    data['size'] = [scores['size'] for scores in scores_list]
    data['n_rings'] = [scores['n_rings'] for scores in scores_list]

    # Save the scores in a csv file

    scores_name = 'generated_molecules_{0}_{1}_{2}_{3}_{4}_scored.csv'.format(args.n_mols, args.keku, args.encod, args.exp_name, args.desired_scores_list)
    data.to_csv(SCORED_SAVING_DIR / scores_name, index=False)

    # Compute the metrics on the generated dataset
    print('Computing metrics...')
    # Load the zinc dataset

    zinc_scored_data = pd.read_csv(ZINC_DATA_PATH)
    SAVE_RESULTS_PATH = Path('..') / 'generate_analyse_mols' / 'results' / experiment_name / '{0}_{1}_{2}_{3}_{4}_scored.csv'.format(args.n_mols, args.keku, args.encod, args.exp_name, args.desired_scores_list)
    
    # Create the folder to save the results

    if not os.path.exists(SAVE_RESULTS_PATH):
        os.makedirs(SAVE_RESULTS_PATH)

    valid_molecules = calculate_statistics(data, zinc_scored_data, SAVE_RESULTS_PATH / 'statistics.txt')
    valid_mols_df = data[data['smiles'].isin(valid_molecules)]

    # Save plots of the distributions of the scores

    save_plot_metrics(valid_mols_df, zinc_scored_data, 'logP', SAVE_RESULTS_PATH / 'logP.png')
    save_plot_metrics(valid_mols_df, zinc_scored_data, 'SA', SAVE_RESULTS_PATH / 'SA.png')
    save_plot_metrics(valid_mols_df, zinc_scored_data, 'QED', SAVE_RESULTS_PATH / 'QED.png')
    save_plot_metrics(valid_mols_df, zinc_scored_data, 'weight', SAVE_RESULTS_PATH / 'weight.png')
    save_plot_metrics(valid_mols_df, zinc_scored_data, 'size', SAVE_RESULTS_PATH / 'size.png')
    save_plot_metrics(valid_mols_df, zinc_scored_data, 'n_rings', SAVE_RESULTS_PATH / 'n_rings.png')




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_mols', type=int, default=1000, help='Number of molecules to generate')
    parser.add_argument('--keku', type=bool, default=True, help='Kekulize the molecules')
    parser.add_argument('--encod', type=str, default='charged', help='Encoding type')
    parser.add_argument('--exp_name', type=str, default='exp', help='Name of the experience')
    parser.add_argument('--desired_scores_list', type=parse_list_of_floats, default="", help='Desired scores for the molecules')
    parser.add_argument('--n_threads', default=1, type=int, help='Number of threads to use')
    parser.add_argument('--batch_size', default=1000, type=int, help='Batch size to use')
    parser.add_argument('--starting_size', default=1, type=int, help='Starting size of the molecules')

    args = parser.parse_args()
    main(args, args.n_threads)