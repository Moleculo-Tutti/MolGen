import numpy as np
import torch

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.AllChem import GetMorganFingerprintAsBitVect as Morgan
from rdkit.Chem.QED import qed
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import Descriptors
from SA_Score import sascorer

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

def fingerprint(smiles_or_mol, fp_type='maccs', dtype=None, morgan__r=2,
                morgan__n=1024, *args, **kwargs):
    """
    Generates fingerprint for SMILES
    If smiles is invalid, returns None
    Returns numpy array of fingerprint bits

    Parameters:
        smiles: SMILES string
        type: type of fingerprint: [MACCS|morgan]
        dtype: if not None, specifies the dtype of returned array
    """
    fp_type = fp_type.lower()
    molecule = canonic_smiles(smiles_or_mol, *args, **kwargs)
    if molecule is None:
        return None
    if fp_type == 'maccs':
        keys = MACCSkeys.GenMACCSKeys(molecule)
        keys = np.array(keys.GetOnBits())
        fingerprint = np.zeros(166, dtype='uint8')
        if len(keys) != 0:
            fingerprint[keys - 1] = 1  # We drop 0-th key that is always zero
    elif fp_type == 'morgan':
        fingerprint = np.asarray(Morgan(molecule, morgan__r, nBits=morgan__n),
                                 dtype='uint8')
    else:
        raise ValueError("Unknown fingerprint type {}".format(fp_type))
    if dtype is not None:
        fingerprint = fingerprint.astype(dtype)
    return fingerprint


def fingerprints(smiles_mols_array, already_unique=False, *args, **kwargs):
    '''
    Computes fingerprints of smiles np.array/list/pd.Series.
    Inserts np.NaN to rows corresponding to incorrect smiles.
    IMPORTANT: if there is at least one np.NaN, the dtype would be float
    Parameters:
        smiles_mols_array: list/array/pd.Series of smiles or already computed
            RDKit molecules
        already_unique: flag for performance reasons, if smiles array is big
            and already unique. Its value is set to True if smiles_mols_array
            contain RDKit molecules already.
    '''
    if isinstance(smiles_mols_array, pd.Series):
        smiles_mols_array = smiles_mols_array.values
    else:
        smiles_mols_array = np.asarray(smiles_mols_array)
    if not isinstance(smiles_mols_array[0], str):
        already_unique = True

    if not already_unique:
        smiles_mols_array, inv_index = np.unique(smiles_mols_array,
                                                 return_inverse=True)
    
    # Loop through the array and calculate fingerprints
    fps = [fingerprint(smiles_or_mol, *args, **kwargs) for smiles_or_mol in smiles_mols_array]

    length = 1
    for fp in fps:
        if fp is not None:
            length = fp.shape[-1]
            first_fp = fp
            break
    fps = [fp if fp is not None else np.array([np.NaN]).repeat(length)[None, :]
           for fp in fps]
    if scipy.sparse.issparse(first_fp):
        fps = scipy.sparse.vstack(fps).tocsr()
    else:
        fps = np.vstack(fps)
    
    if not already_unique:
        return fps[inv_index]
    return fps


def internal_diversity(gen, n_jobs=1, device='cpu', fp_type='morgan',
                       gen_fps=None, p=1):
    """
    Computes internal diversity as:
    1/|A|^2 sum_{x, y in AxA} (1-tanimoto(x, y))
    """
    if gen_fps is None:
        gen_fps = fingerprints(gen, fp_type=fp_type, n_jobs=n_jobs)
    return 1 - (average_agg_tanimoto(gen_fps, gen_fps,
                                     agg='mean', device=device, p=p)).mean()