import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt

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


def plot_molecule_sizes(smiles_series):
    molecule_sizes = []

    # Parcourir chaque SMILES dans la série
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Compter le nombre total d'atomes dans la molécule
            size = mol.GetNumAtoms()
            molecule_sizes.append(size)

    # Tracer l'histogramme des tailles de molécules
    plt.hist(molecule_sizes, bins=20)
    plt.xlabel('Molecule Size')
    plt.ylabel('Frequency')
    plt.title('Molecule Size Distribution')
    plt.show()

def plot_atom_distribution(smiles_series):
    atom_counts = []

    # Parcourir chaque SMILES dans la série
    for smiles in smiles_series:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Compter le nombre d'atomes de chaque type
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            atom_count = dict(pd.Series(atoms).value_counts())
            atom_counts.append(atom_count)

    # Créer un DataFrame à partir des comptages des atomes
    atom_df = pd.DataFrame(atom_counts).fillna(0)

    # Tracer l'histogramme de distribution des atomes
    atom_df.plot(kind='bar', stacked=True)
    plt.xlabel('Molecule')
    plt.ylabel('Atom Count')
    plt.title('Atom Distribution')
    plt.show()

