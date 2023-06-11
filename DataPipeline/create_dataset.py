import pandas as pd
import torch

from tqdm import tqdm
from path import Path

from rdkit import Chem

from sklearn.model_selection import train_test_split
from preprocessing import process_encode_graph

csv_path = Path("./data/scored_data") / "scored_zinc.csv"

zinc_df = pd.read_csv(csv_path)

def remove_iodine_bromine_phosphorus(df):

    df = df[df['smiles'].str.contains('Br') == False]
    df = df[df['smiles'].str.contains('I') == False]
    df = df[df['smiles'].str.contains('P') == False]
    
    return df


def remove_charged_atoms(df):
    # List to keep track of the indices to be removed
    indices_to_remove = []
    
    # Iterate through the DataFrame
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        smiles = row['smiles']
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:  # invalid SMILES string
            continue

        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == 'C' and atom.GetFormalCharge() < 0) or \
               (atom.GetSymbol() == 'O' and atom.GetFormalCharge() > 0) or \
               (atom.GetSymbol() == 'S' and atom.GetFormalCharge() > 0):
                indices_to_remove.append(index)
                break
                
    # Drop the rows with the indices in indices_to_remove
    return df.drop(indices_to_remove)


def remove_P(df):
    df = df[df['smiles'].str.contains('P') == False]
    return df

def main():
    preprocessed_graph = []
    filtered_df = remove_charged_atoms(remove_P(zinc_df))

    for index, row in tqdm(filtered_df.iterrows(), total=filtered_df.shape[0]):
        scores = {'SA': row['SA'], 'logP': row['logP'], 'QED': row['QED'], 'weight': row['weight'], 'n_rings': row['n_rings']}
        data = process_encode_graph(row['smiles'], optional_scores=scores, encoding_option='charged', kekulize=True)
        preprocessed_graph.append(data)

    # Separate data into train, validation and test sets
    X_train_val, X_test = train_test_split(preprocessed_graph, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_train_val, test_size=0.1111, random_state=42)

    # Save data sets into files
    torch.save(X_train, 'data/preprocessed_graph_train_charged_kekulized_scores.pt')
    torch.save(X_val, 'data/preprocessed_graph_val_charged_kekulized_scores.pt')
    torch.save(X_test, 'data/preprocessed_graph_test_charged_kekulized_scores.pt')

if __name__ == "__main__":
    main()