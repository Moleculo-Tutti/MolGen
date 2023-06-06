import pandas as pd
import torch

from tqdm import tqdm
from path import Path

from rdkit import Chem

from sklearn.model_selection import train_test_split
from preprocessing import process_encode_graph

csv_path = Path("./data") / "rndm_zinc_drugs_clean_3.csv"

zinc_df = pd.read_csv(csv_path)

def remove_iodine_bromine_phosphorus(df):

    df = df[df['smiles'].str.contains('Br') == False]
    df = df[df['smiles'].str.contains('I') == False]
    df = df[df['smiles'].str.contains('P') == False]
    
    return df

def remove_charged_atoms(dataset):
    to_remove = []
    
    for smiles in tqdm(dataset):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:  # invalid SMILES string
            continue

        for atom in mol.GetAtoms():
            if (atom.GetSymbol() == 'C' and atom.GetFormalCharge() < 0) or \
               (atom.GetSymbol() == 'O' and atom.GetFormalCharge() > 0) or \
               (atom.GetSymbol() == 'S' and atom.GetFormalCharge() > 0):
                to_remove.append(smiles)
                break

    return [smiles for smiles in dataset if smiles not in to_remove]


def remove_P(df):
    df = df[df['smiles'].str.contains('P') == False]
    return df

def main():
    preprocessed_graph = []
    filtered_list = remove_charged_atoms(remove_P(zinc_df)['smiles'].to_list())

    for smiles in tqdm(filtered_list):
        data = process_encode_graph(smiles, 'charged', kekulize=True)
        preprocessed_graph.append(data)

    # Separate data into train, validation and test sets
    X_train_val, X_test = train_test_split(preprocessed_graph, test_size=0.1, random_state=42)
    X_train, X_val = train_test_split(X_train_val, test_size=0.1111, random_state=42)

    # Save data sets into files
    torch.save(X_train, 'data/preprocessed_graph_train_charged_kekulized.pt')
    torch.save(X_val, 'data/preprocessed_graph_val_charged_kekulized.pt')
    torch.save(X_test, 'data/preprocessed_graph_test_charged_kekulized.pt')

if __name__ == "__main__":
    main()