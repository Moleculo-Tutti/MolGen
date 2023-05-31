import torch
import torch_geometric


from utils import sample_random_subgraph_ZINC, get_model_GNN1, get_model_GNN2, get_model_GNN3, get_optimizer, one_step, create_torch_graph_from_one_atom, load_model, sample_first_atom
from tqdm import tqdm
from rdkit import Chem
import pandas as pd
from path import Path
import argparse


def tensor_to_smiles(node_features, edge_index, edge_attr):
    # Create an empty editable molecule
    mol = Chem.RWMol()

    # Define atom mapping
    atom_mapping = {
        0: 'C',
        1: 'N',
        2: 'O',
        3: 'F',
        4: 'S',
        5: 'Cl',
    }

    # Add atoms
    for atom_feature in node_features:
        atom_idx = atom_feature[:6].argmax().item()
        atom_symbol = atom_mapping.get(atom_idx)
        atom = Chem.Atom(atom_symbol)
        mol.AddAtom(atom)

    # Define bond type mapping
    bond_mapping = {
        0: Chem.rdchem.BondType.AROMATIC,
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
    }

    # Add bonds
    for start, end, bond_attr in zip(edge_index[0], edge_index[1], edge_attr):
        bond_type_idx = bond_attr[:4].argmax().item()
        bond_type = bond_mapping.get(bond_type_idx)

        # RDKit ignores attempts to add a bond that already exists,
        # so we need to check if the bond exists before we add it
        if mol.GetBondBetweenAtoms(start.item(), end.item()) is None:
            mol.AddBond(start.item(), end.item(), bond_type)

    # Convert the molecule to SMILES
    smiles = Chem.MolToSmiles(mol)

    return smiles


def full_generation(GNN1, GNN2, GNN3, zinc_df):
    graph = create_torch_graph_from_one_atom(sample_first_atom())
    output = torch_geometric.data.Batch.from_data_list([graph])
    queue = [0]
    i = 1
    while queue and i < 100:
        output, queue = one_step(output, queue, GNN1=GNN1, GNN2=GNN2, GNN3=GNN3, device='cuda')
        output = torch_geometric.data.Batch.from_data_list([output])
        i += 1  
    return output, queue

# Function that generate a desired number of molecules
def generate_molecules(n_molecules, GNN1, GNN2, GNN3, zinc_df):
    molecules = []
    for i in tqdm(range(n_molecules)):
        output, queue = full_generation(GNN1, GNN2, GNN3, zinc_df)
        molecules.append(tensor_to_smiles(output.x, output.edge_index, output.edge_attr))
    return molecules


def main(args):

    zinc_path = Path('rndm_zinc_drugs_clean_3.csv')
    zinc_df = pd.read_csv(zinc_path)

    GNN1_path = Path('.') / 'models/trained_models/checkpoint_epoch_247_GNN1.pt'
    GNN2_path = Path('.') / 'models/trained_models/checkpoint_epoch_236_GNN2_test.pt'
    GNN3_path = Path('.') / 'models/trained_models/checkpoint_epoch_110_experience_features_position.pt'

    GNN1 = get_model_GNN1(7)
    GNN2 = get_model_GNN2(7)
    GNN3 = get_model_GNN3(7)

    optimizer_GNN1 = get_optimizer(GNN1, lr=0.0001)
    optimizer_GNN2 = get_optimizer(GNN2, lr=0.0001)
    optimizer_GNN3 = get_optimizer(GNN3, lr=0.0001)

    GNN1, optimizer_GNN1 = load_model(GNN1_path, GNN1, optimizer_GNN1)
    GNN2, optimizer_GNN2 = load_model(GNN2_path, GNN2, optimizer_GNN2)
    GNN3, optimizer_GNN3 = load_model(GNN3_path, GNN3, optimizer_GNN3)

    # Set all models in eval mode

    GNN1.eval()
    GNN2.eval()
    GNN3.eval()

    GNN1.to('cuda')
    GNN2.to('cuda')
    GNN3.to('cuda')


    molecules = generate_molecules(args.n_molecules, GNN1, GNN2, GNN3, zinc_df)

    #Save the molecule list in a csv file
    df = pd.DataFrame(molecules, columns=['SMILES'])
    df.to_csv('generated_molecules_10000.csv', index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_molecules', type=int, default=1000, help='Number of molecules to generate')
    args = parser.parse_args()
    main(args)
