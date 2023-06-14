import argparse
import pandas as pd
from pathlib import Path
from utils import canonic_smiles, SA, logP, QED, weight, get_n_rings
from tqdm import tqdm
import concurrent.futures

from rdkit import Chem

data_path = Path('..') / 'generated_mols' / 'generated_molecules_1000_True_charged_GNN_scored_logp_[4.0].csv'
output_dir = Path('..') / 'generated_mols' / 'scored' 

smiles_col = 'SMILES'


def compute_metrics(smiles):
    canonic = canonic_smiles(smiles)
    mol = Chem.MolFromSmiles(canonic)
    if mol is None:
        return {
            'smiles': canonic,
            'SA': None,
            'logP': None,
            'QED': None,
            'weight': None,
            'n_rings': None
        }
    return {
        'smiles': canonic,
        'SA': SA(mol),
        'logP': logP(mol),
        'QED': QED(mol),
        'weight': weight(mol),
        'n_rings': get_n_rings(mol)
    }


def main(args, n_threads=4):
    # Load data
    original_data = pd.read_csv(data_path)

    smiles_list = original_data[smiles_col].tolist()

    # Compute all metrics in parallel using ThreadPoolExecutor
    metrics_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_threads) as executor:
        for metrics in tqdm(executor.map(compute_metrics, smiles_list),
                           total=len(smiles_list),
                           desc="Computing metrics"):
            metrics_list.append(metrics)

    # Save data in a new dataframe with the smiles and the metrics
    data = pd.DataFrame()
    data['smiles'] = [metrics['smiles'] for metrics in metrics_list]
    data['SA'] = [metrics['SA'] for metrics in metrics_list]
    data['logP'] = [metrics['logP'] for metrics in metrics_list]
    data['QED'] = [metrics['QED'] for metrics in metrics_list]
    data['weight'] = [metrics['weight'] for metrics in metrics_list]
    data['n_rings'] = [metrics['n_rings'] for metrics in metrics_list]
    
    # Save to output_dir
    data.to_csv(output_dir / 'generated_molecules_1000_True_charged_GNN_scored_logp_[4.0]_scored.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_threads', default=1, type=int, help='Number of threads to use')
    args = parser.parse_args()
    main(args, n_threads=args.n_threads)