import pandas as pd
from rdkit import Chem

def calculate_uniqueness(batch_smiles_valid):
    batch_smiles_uniq = set(batch_smiles_valid)
    uniqueness = len(set(batch_smiles_valid)) / len(batch_smiles_valid) #isometric smiles are done because we use 
    return uniqueness, list(batch_smiles_uniq)


def calculate_novelty(batch_smiles_uniq, zinc_dataset_path):
    # Charger le jeu de données d'entraînement
    zinc = pd.read_csv(zinc_dataset_path)
    
    # Vérifier la nouveauté du lot par rapport au jeu de données d'entraînement
    non_new_molecules = []
    for smiles in batch_smiles_uniq:
        if smiles in zinc['SMILES'].values:
            non_new_molecules.append(smiles)
    
    novelty = 1 - len(non_new_molecules) / len(batch_smiles_uniq)
    return novelty


def calculate_validity(batch_smiles):
    valid_molecules = []
    for smiles in batch_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_molecules.append(smiles)
    
    validity = len(valid_molecules) / len(batch_smiles)
    return validity,valid_molecules
