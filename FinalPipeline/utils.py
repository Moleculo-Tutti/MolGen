import random
import torch
import torch_geometric

import torch.nn.functional as F
import sys
import os

from rdkit import Chem
from tqdm import tqdm

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from DataPipeline.preprocessing import process_encode_graph, get_subgraph_with_terminal_nodes_step
from DataPipeline.preprocessing import node_encoder
from Model.GNN1 import ModelWithEdgeFeatures as GNN1
from Model.GNN2 import ModelWithEdgeFeatures as GNN2
from Model.GNN3 import ModelWithEdgeFeatures as GNN3

def tensor_to_smiles(node_features, edge_index, edge_attr):
    # Create an empty editable molecule
    mol = Chem.RWMol()

    # Define atom mapping
    atom_mapping = {
        0: ('C', 0),
        1: ('N', 0),
        2: ('N', 1),
        3: ('N', -1),
        4: ('O', 0),
        5: ('O', -1),
        6: ('F', 0),
        7: ('S', 0),
        8: ('S', -1),
        9: ('Cl', 0),
        10: ('Br', 0),
        11: ('I', 0)
    }

    # Add atoms
    for atom_feature in node_features:
        atom_idx = atom_feature[:12].argmax().item()
        atom_symbol, charge = atom_mapping.get(atom_idx)
        atom = Chem.Atom(atom_symbol)
        atom.SetFormalCharge(charge)
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

def sample_random_subgraph_ZINC(pd_dataframe, start_size):
    indice = random.choice(pd_dataframe.index)
    smiles_str = pd_dataframe.loc[indice, 'smiles']

    torch_graph = process_encode_graph(smiles_str, encoding_option='reduced')
    subgraph_data, terminal_node_info, id_map = get_subgraph_with_terminal_nodes_step(torch_graph, start_size)

    return subgraph_data, terminal_node_info, id_map

def sample_first_atom(encoding = 'reduced'):
    if encoding == 'reduced' or encoding == 'charged':
        prob_dict = {'60': 0.7385023585929047, 
                    '80': 0.1000143018658728, 
                    '70': 0.12239949901813525, 
                    '90': 0.013786373862576426, 
                    '160': 0.017856330814654413,
                    '170': 0.007441135845856433}

    
    return random.choices(list(prob_dict.keys()), weights=list(prob_dict.values()))[0]

def create_torch_graph_from_one_atom(atom, encoding_option='reduced'):
    num_atom = int(atom)

    atom_attribute = node_encoder(num_atom, encoding_option=encoding_option)
    # Create graph
    graph = torch_geometric.data.Data(x=atom_attribute.view(1, -1), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 4)))

    return graph

def load_model(checkpoint_path, model, optimizer):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def get_model_GNN1(config, encoding_size):

    return GNN1(in_channels=encoding_size + int(config['feature_position']),
                hidden_channels_list=config["GCN_size"],
                mlp_hidden_channels=config['mlp_hidden'],
                edge_channels=4, 
                num_classes=encoding_size, 
                use_dropout=config['use_dropout'],
                size_info=config['use_size'])

def get_model_GNN2(config, encoding_size):

    return GNN2(in_channels=encoding_size + int(config['feature_position']), 
                hidden_channels_list=config["GCN_size"],
                mlp_hidden_channels=config['mlp_hidden'],
                edge_channels=4, 
                num_classes=4, 
                use_dropout=config['use_dropout'])

def get_model_GNN3(config, encoding_size):

    return GNN3(in_channels=encoding_size + int(config['feature_position']), 
                hidden_channels_list=config["GCN_size"],
                edge_channels=4, 
                use_dropout=config['use_dropout'])

def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)



def add_edge_or_node_to_graph(graph, initial_node, edge_attr, other_node=None, new_node_attr=None):
    # Convert nodes to tensor
    initial_node = torch.tensor([initial_node], dtype=torch.long)

    # Create new edge attribute

    new_edge_attr = torch.cat([edge_attr, edge_attr], dim=0)

    # If new_node_attr is provided, add a new node
    if new_node_attr is not None:

        # Add new node to graph

        graph.x = torch.cat([graph.x, new_node_attr], dim=0)

        # Adjust other_node to be the new node
        other_node = torch.tensor([graph.x.size(0)-1], dtype=torch.long)
    elif other_node is None:
        raise ValueError("other_node must be provided if no new node is being added")
    else:
        other_node = torch.tensor([other_node], dtype=torch.long)

    # Create new edge
    new_edge = torch.tensor([[initial_node.item(), other_node.item()], 
                             [other_node.item(), initial_node.item()]], dtype=torch.long)
    # Add new edge and its attribute to graph
    graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    graph.edge_attr = torch.cat([graph.edge_attr, new_edge_attr], dim=0)

    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

    return graph

def select_node(tensor):

    # Somme sur les 3 premières dimensions de chaque vecteur
    sum_on_first_three_dims = tensor[:, :3].sum(dim=1)

    # Trouver l'indice du node avec la plus grande somme
    max_index = torch.argmax(sum_on_first_three_dims)

    return tensor[max_index], max_index

def one_step(input_graph, queue : list, GNN1, GNN2, GNN3, device):

    with torch.no_grad():
        current_node = queue[0]

        graph1 = input_graph.clone()
        graph1.x[current_node, -1] = 1
        # add one column to graph1.x

        graph1.x = torch.cat([graph1.x, torch.zeros(graph1.x.size(0), 1)], dim=1)
        
        graph1.x[0:current_node, -1] = 1

        prediction = GNN1(graph1.to(device))

        # Sample next node from prediction

        predicted_node = torch.multinomial(F.softmax(prediction, dim=1), 1).item()
        if predicted_node == 6:
            #Stop 
            queue.pop(0)
            return input_graph, queue

        # Encode next node
        encoded_predicted_node = torch.zeros(prediction.size(), dtype=torch.float)
        encoded_predicted_node[0, predicted_node] = 1

        queue.append(graph1.x.size(0)) # indexing starts at 0

        # GNN2

        graph2 = input_graph.clone()
        graph2.x[current_node, -1] = 1

        # add a zeros to neighbor 
        encoded_predicted_node = torch.cat([encoded_predicted_node, torch.zeros(1, 1)], dim=1)

        graph2.neighbor = encoded_predicted_node
       


        # Add a new column indicating the nodes that have been finalized
        graph2.x = torch.cat([graph2.x, torch.zeros(graph2.x.size(0), 1)], dim=1)
        graph2.x[0:current_node, -1] = 1

        prediction2 = GNN2(graph2.to(device))
        predicted_edge = torch.multinomial(F.softmax(prediction2, dim=1), 1).item()
        encoded_predicted_edge = torch.zeros(prediction2.size(), dtype=torch.float)
        encoded_predicted_edge[0, predicted_edge] = 1
        # GNN3

        new_graph = add_edge_or_node_to_graph(input_graph.clone(), current_node, encoded_predicted_edge, new_node_attr = encoded_predicted_node[:, :-1])
        graph3 = new_graph.clone()
        # Add a one of the last node that are going to possibly bond to another node
        graph3.x[-1, -1] = 1
        #Add a new column indicating the nodes that have been finalized
        graph3.x = torch.cat([graph3.x, torch.zeros(graph3.x.size(0), 1)], dim=1)
        graph3.x[0:current_node, -1] = 1

        mask = torch.cat((torch.zeros(current_node + 1), torch.ones(len(graph3.x) - current_node - 1)), dim=0).bool()
        mask[-1] = False
        graph3.mask = mask
        prediction3 = GNN3(graph3.to(device))
        softmax_prediction3 = F.softmax(prediction3, dim=1)[graph3.mask]
        if softmax_prediction3.size(0) == 0:
            #Stop
            return new_graph, queue
        selected_edge_distribution, max_index = select_node(softmax_prediction3)


        #sample edge
        predicted_edge = torch.multinomial(selected_edge_distribution, 1).item()
        
        if predicted_edge == 3:
            #Stop
            return new_graph, queue
        
        encoded_predicted_edge = torch.zeros(prediction2.size(), dtype=torch.float)
        encoded_predicted_edge[0, predicted_edge] = 1


        output_graph = add_edge_or_node_to_graph(new_graph, graph1.x.size(0), encoded_predicted_edge, other_node=current_node + max_index+1)
        return output_graph, queue

class MolGen():
    def __init__(self, GNN1, GNN2, GNN3, encoding_size, feature_position, device, save_intermidiate_states = False):
        mol_graph = create_torch_graph_from_one_atom(sample_first_atom(), encoding_option='charged')
        self.mol_graph = torch_geometric.data.Batch.from_data_list([mol_graph])
        self.queue = [0]
        self.GNN1 = GNN1
        self.GNN2 = GNN2
        self.GNN3 = GNN3
        self.device = device
        self.feature_position = feature_position
        self.encoding_size = encoding_size

    def one_step(self):
        with torch.no_grad():

            current_node = self.queue[0]
            graph1 = self.mol_graph.clone()

            # add a one to the current node in the last column
            graph1.x[current_node, -1] = 1

            if self.feature_position:
                #add feature position
                graph1.x = torch.cat([graph1.x, torch.zeros(graph1.x.size(0), 1)], dim=1)
                graph1.x[0:current_node, -1] = 1
            prediction = self.GNN1(graph1.to(self.device))
            # Sample next node from prediction
            predicted_node = torch.multinomial(F.softmax(prediction, dim=1), 1).item()
            if predicted_node == self.encoding_size - 1:
                #Stop 
                self.queue.pop(0)
                return
                
            # Encode next node
            encoded_predicted_node = torch.zeros(prediction.size(), dtype=torch.float)
            encoded_predicted_node[0, predicted_node] = 1

        
            self.queue.append(graph1.x.size(0)) # indexing starts from 0

            #GNN2 

            graph2 = self.mol_graph.clone()

            # add a one to the current node in the last column
            graph2.x[current_node, -1] = 1

            if self.feature_position:
                #add feature position
                graph2.x = torch.cat([graph2.x, torch.zeros(graph2.x.size(0), 1)], dim=1)
                graph2.x[0:current_node, -1] = 1

            #add zeros to the neighbor
            encoded_predicted_node = torch.cat([encoded_predicted_node, torch.zeros(1, 1)], dim=1)
            graph2.neighbor = encoded_predicted_node

            prediction2 = self.GNN2(graph2.to(self.device))

            predicted_edge = torch.multinomial(F.softmax(prediction2, dim=1), 1).item()
            encoded_predicted_edge = torch.zeros(prediction2.size(), dtype=torch.float)
            encoded_predicted_edge[0, predicted_edge] = 1

            #GNN3

            # Add the node and the edge to the graph
            new_graph = add_edge_or_node_to_graph(self.mol_graph.clone(), current_node, encoded_predicted_edge, new_node_attr = encoded_predicted_node[:, :-1])
            graph3 = new_graph.clone()

            # Add a one of the last node that are going to possibly bond to another node
            graph3.x[-1, -1] = 1

            if self.feature_position:
                #Add a new column indicating the nodes that have been finalized
                graph3.x = torch.cat([graph3.x, torch.zeros(graph3.x.size(0), 1)], dim=1)
                graph3.x[0:current_node, -1] = 1

            mask = torch.cat((torch.zeros(current_node + 1), torch.ones(len(graph3.x) - current_node - 1)), dim=0).bool()
            mask[-1] = False
            graph3.mask = mask
            prediction3 = self.GNN3(graph3.to(self.device))
            softmax_prediction3 = F.softmax(prediction3, dim=1)[graph3.mask]
            if softmax_prediction3.size(0) == 0:
                #Stop
                self.mol_graph = new_graph
                return

            selected_edge_distribution, max_index = select_node(softmax_prediction3)

             #sample edge
            predicted_edge = torch.multinomial(selected_edge_distribution, 1).item()

            if predicted_edge == 3:
                #Stop
                self.mol_graph = new_graph
                return

            encoded_predicted_edge = torch.zeros(prediction2.size(), dtype=torch.float)
            encoded_predicted_edge[0, predicted_edge] = 1


            output_graph = add_edge_or_node_to_graph(new_graph, graph1.x.size(0), encoded_predicted_edge, other_node=current_node + max_index+1)

            self.mol_graph = output_graph

    def full_generation(self):
        while len(self.queue) > 0:
            self.one_step()




class GenerationModule():
    def __init__(self, config1, config2, config3, encoding_size, pathGNN1, pathGNN2, pathGNN3):
        self.config1 = config1
        self.config2 = config2
        self.config3 = config3
        self.encoding_size = encoding_size
        self.feature_position = config1["feature_position"]
        self.GNN1 = get_model_GNN1(config1, encoding_size)
        self.GNN2 = get_model_GNN2(config2, encoding_size)
        self.GNN3 = get_model_GNN3(config3, encoding_size)

        self.optimizer_GNN1 = torch.optim.Adam(self.GNN1.parameters(), lr=config1["lr"])
        self.optimizer_GNN2 = torch.optim.Adam(self.GNN2.parameters(), lr=config2["lr"])
        self.optimizer_GNN3 = torch.optim.Adam(self.GNN3.parameters(), lr=config3["lr"])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.GNN1_model, self.optimizer_GNN1 = load_model(pathGNN1, self.GNN1, self.optimizer_GNN1)
        self.GNN2_model, self.optimizer_GNN2 = load_model(pathGNN2, self.GNN2, self.optimizer_GNN2)
        self.GNN3_model, self.optimizer_GNN3 = load_model(pathGNN3, self.GNN3, self.optimizer_GNN3)

        self.GNN1_model.to(self.device)
        self.GNN2_model.to(self.device)
        self.GNN3_model.to(self.device)

        self.GNN1_model.eval()
        self.GNN2_model.eval()
        self.GNN3_model.eval()
    
    def generate_mol_list(self, n_mol):
        mol_list = []
        for i in tqdm(range(n_mol)):
            mol = MolGen(self.GNN1_model, self.GNN2_model, self.GNN3_model, self.encoding_size, self.feature_position, self.device)
            mol.full_generation()
            mol_list.append(mol.mol_graph)
        return mol_list

