import os
import sys
import torch
import argparse
from dataclasses import dataclass

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from DataPipeline.preprocessing import node_encoder, tensor_to_smiles
from generation import Sampling_Path_Batch
from models import Model_GNNs
from RL_trainer import GDCTrainer_path

@dataclass
class Experiment:
    exp_name: str
    encod: str
    keku: bool
    train: bool
    encoding_size: int = 13
    edge_size: int = 3
    encoding_option: str = 'charged'
    compute_lambdas: bool = False

exp = Experiment('GNN_baseline_3_modif', 'charged', True, False, 13, 3, 'charged', False)


from rdkit import Chem
from rdkit.Chem import Descriptors

def logP(smiles_list):
    logP_values = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            logP_values.append(0)
        else: 
            logp = Descriptors.MolLogP(mol)
            if logp > 2.0 and logp < 2.5:
                logP_values.append(1)
            else:
                logP_values.append(0)
    return torch.tensor(logP_values)

def QED(smiles_list):
    qed_values = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            qed_values.append(0)
        else: 
            qed = Descriptors.qed(mol)
            if qed > 0.90:
                qed_values.append(1)
            else:
                qed_values.append(0)
    return torch.tensor(qed_values)




def main(args):
    GNNs_q = Model_GNNs(exp)
    GNNs_a = Model_GNNs(exp)
    GNNs_pi = Model_GNNs(exp)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Module_Gen = Sampling_Path_Batch(GNNs_q, GNNs_a, GNNs_pi, features = {'logP' : logP, 'QED' : QED}, lambdas = torch.Tensor([1.0, 1.0]), device = device, args=exp)

    Trainer = GDCTrainer_path(Module_Gen,
                            features = {'logP' : logP, 'QED' : QED},
                            desired_moments= {'logP' : 1.0, 'QED' : 1.0},
                            q_update_criterion='kld',
                            lr = args.lr,
                            minibatch_size = args.mini_batch_size,
                            batch_size = 1000,
                            min_nabla_lambda = 1e-4,
                            lambdas=torch.Tensor([7.8309, 8.3757])) 
    Trainer.run_steps(args.num_steps, args.num_batches, args.num_minibatches, args.exp_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--num_batches', type=int, default=20)
    parser.add_argument('--num_minibatches', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--mini_batch_size', type=int, default=128)
    args = parser.parse_args()
    main(args)
