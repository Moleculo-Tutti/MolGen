import numpy as np
import pandas as pd
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
import itertools

import os
import sys
cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
parent_parent_dir = os.path.dirname(parent_dir)
sys.path.append(parent_dir)
sys.path.append(parent_parent_dir)

from metrics.SA_Score import sascorer

import scipy.sparse

def canonic_smiles(smiles_or_mol):
    if type(smiles_or_mol) == str:
        mol = Chem.MolFromSmiles(smiles_or_mol)
        if mol is None:
            return smiles_or_mol
    # Remove stereochemistry
    Chem.RemoveStereochemistry(mol)
    return Chem.MolToSmiles(mol, isomericSmiles=False)

def SA(mol):
    """
    Computes RDKit's Synthetic Accessibility score
    """
    return sascorer.calculateScore(mol)

def logP(mol):
    """
    Computes RDKit's logP
    """
    return Chem.Crippen.MolLogP(mol)

def QED(mol):
    """
    Computes RDKit's QED score
    """
    return qed(mol)

def weight(mol):
    """
    Computes molecular weight for given molecule.
    Returns float,
    """
    return Descriptors.MolWt(mol)

def get_n_rings(mol):
    """
    Computes the number of rings in a molecule
    """
    return mol.GetRingInfo().NumRings()

def average_agg_tanimoto(stock_vecs, gen_vecs,
                         batch_size=5000, agg='max',
                         device='cpu', p=1):
    """
    For each molecule in gen_vecs finds closest molecule in stock_vecs.
    Returns average tanimoto score for between these molecules

    Parameters:
        stock_vecs: numpy array <n_vectors x dim>
        gen_vecs: numpy array <n_vectors' x dim>
        agg: max or mean
        p: power for averaging: (mean x^p)^(1/p)
    """
    assert agg in ['max', 'mean'], "Can aggregate only max or mean"
    agg_tanimoto = np.zeros(len(gen_vecs))
    total = np.zeros(len(gen_vecs))
    for j in range(0, stock_vecs.shape[0], batch_size):
        x_stock = torch.tensor(stock_vecs[j:j + batch_size]).to(device).float()
        for i in range(0, gen_vecs.shape[0], batch_size):
            y_gen = torch.tensor(gen_vecs[i:i + batch_size]).to(device).float()
            y_gen = y_gen.transpose(0, 1)
            tp = torch.mm(x_stock, y_gen)
            jac = (tp / (x_stock.sum(1, keepdim=True) +
                         y_gen.sum(0, keepdim=True) - tp)).cpu().numpy()
            jac[np.isnan(jac)] = 1
            if p != 1:
                jac = jac**p
            if agg == 'max':
                agg_tanimoto[i:i + y_gen.shape[1]] = np.maximum(
                    agg_tanimoto[i:i + y_gen.shape[1]], jac.max(0))
            elif agg == 'mean':
                agg_tanimoto[i:i + y_gen.shape[1]] += jac.sum(0)
                total[i:i + y_gen.shape[1]] += jac.shape[0]
    if agg == 'mean':
        agg_tanimoto /= total
    if p != 1:
        agg_tanimoto = (agg_tanimoto)**(1/p)
    return np.mean(agg_tanimoto)

def calculate_uniqueness(batch_smiles_valid):
    batch_smiles_uniq = set(batch_smiles_valid)
    uniqueness = len(set(batch_smiles_valid)) / len(batch_smiles_valid)
    return uniqueness, list(batch_smiles_uniq)


def calculate_novelty(batch_smiles_uniq, zinc_mol_list):
    
    non_new_molecules = []
    for smiles in batch_smiles_uniq:
        if smiles in zinc_mol_list:
            non_new_molecules.append(smiles)
    
    novelty = 1 - len(non_new_molecules) / len(batch_smiles_uniq)
    return novelty, non_new_molecules


def calculate_validity(batch_smiles):
    valid_molecules = []
    for smiles in batch_smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_molecules.append(smiles)
    
    validity = len(valid_molecules) / len(batch_smiles)
    return validity,valid_molecules




def compute_internal_diversity(smiles_list):
    """
    Compute the internal diversity of a list of SMILES strings based on Tanimoto similarity.
    
    Args:
        smiles_list (list of str): List of SMILES strings representing the molecules.
    
    Returns:
        float: The internal diversity of the molecules.
    """
    # Convert SMILES strings to RDKit Mol objects
    molecules = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    # Compute fingerprints for all molecules
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024) for mol in molecules]
    
    # Calculate Tanimoto similarities for each pair of fingerprints
    similarities = []
    for fp1, fp2 in itertools.combinations(fps, 2):
        similarity = DataStructs.TanimotoSimilarity(fp1, fp2)
        similarities.append(similarity)
    
    # Compute the internal diversity as the mean Tanimoto similarity
    internal_diversity = sum(similarities) / len(similarities) if similarities else 0.0
    
    return 1 - internal_diversity

