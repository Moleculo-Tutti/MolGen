import pandas as pd
from rdkit import Chem
import matplotlib.pyplot as plt
from tqdm import tqdm

from rdkit.Chem import QED, Descriptors

def calculate_uniqueness(batch_smiles_valid):
    batch_smiles_uniq = set(batch_smiles_valid)
    uniqueness = len(set(batch_smiles_valid)) / len(batch_smiles_valid) #isometric smiles are done because we use 
    return uniqueness, list(batch_smiles_uniq)


def calculate_novelty(batch_smiles_uniq, zinc_dataset_path):
    # Charger le jeu de données d'entraînement
    zinc = pd.read_csv(zinc_dataset_path)
    
    # Vérifier la nouveauté du lot par rapport au jeu de données d'entraînement
    non_new_molecules = []
    for smiles in tqdm(batch_smiles_uniq):
        if smiles in zinc['smiles'].values:
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
    # xlimit to 50
    plt.xlim(0, 50)
    plt.title('Molecule Size Distribution')
    plt.show()

def plot_atom_distribution_dict(smiles_list):
    """
    Plot the atom distribution of a list of SMILES as an histogram of atom counts for each type of atom
    """
    atom_counts_dict = {}
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
            atom_count = dict(pd.Series(atoms).value_counts())
            for atom in atom_count:
                if atom in atom_counts_dict:
                    atom_counts_dict[atom] += atom_count[atom]
                else:
                    atom_counts_dict[atom] = atom_count[atom]
    
    atom_counts_df = pd.DataFrame.from_dict(atom_counts_dict, orient='index', columns=['count'])
    atom_counts_df.plot(kind='bar', stacked=True)
    plt.xlabel('Atom')
    plt.ylabel('Atom Count')
    plt.title('Atom Distribution')
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



def calculate_scores(smiles_list):
    qed_scores = []
    sa_scores = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            qed_scores.append(QED.qed(mol))
    
    return qed_scores


def plot_scores(qed_scores):
    fig, axs = plt.subplots(2, figsize=(10,8))

    axs[0].hist(qed_scores, bins=20, color='b', alpha=0.7)
    axs[0].set_title('QED scores')
    axs[0].set_xlabel('QED score')
    axs[0].set_ylabel('Count')

    plt.tight_layout()
    plt.show()