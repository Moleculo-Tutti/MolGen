import random
import torch
import torch_geometric

import torch.nn.functional as F
import sys
import os

from rdkit import Chem
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor, as_completed
import concurrent

from torch_scatter import scatter_max, scatter_add
from torch_geometric.data import Batch

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from DataPipeline.preprocessing import process_encode_graph, get_subgraph_with_terminal_nodes_step
from DataPipeline.preprocessing import node_encoder
from Model.GNN1 import ModelWithEdgeFeatures as GNN1
from Model.GNN1 import ModelWithNodeConcat as GNN1_node_concat
from Model.GNN2 import ModelWithEdgeFeatures as GNN2
from Model.GNN2 import ModelWithNodeConcat as GNN2_node_concat
from Model.GNN3 import ModelWithEdgeFeatures as GNN3
from Model.GNN3 import ModelWithgraph_embedding_modif as GNN3_embedding

def tensor_to_smiles(node_features, edge_index, edge_attr, edge_mapping = 'aromatic', encoding_type = 'charged'):
    # Create an empty editable molecule
    mol = Chem.RWMol()

    # Define atom mapping
    if encoding_type == 'charged':
        
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

    elif encoding_type == 'polymer':
        atom_mapping = {
            0: ('C', 0),
            1: ('N', 0),
            2: ('O', 0),
            3: ('F', 0),
            4: ('Si', 0),
            5: ('P', 0),
            6: ('S', 0)}

    # Add atoms
    for atom_feature in node_features:
        atom_idx = atom_feature[:12].argmax().item()
        atom_symbol, charge = atom_mapping.get(atom_idx)
        atom = Chem.Atom(atom_symbol)
        atom.SetFormalCharge(charge)
        mol.AddAtom(atom)

    # Define bond type mapping
    if edge_mapping == 'aromatic':
        bond_mapping = {
            0: Chem.rdchem.BondType.AROMATIC,
            1: Chem.rdchem.BondType.SINGLE,
            2: Chem.rdchem.BondType.DOUBLE,
            3: Chem.rdchem.BondType.TRIPLE,
        }
    elif edge_mapping == 'kekulized':
        bond_mapping = {
            0: Chem.rdchem.BondType.SINGLE,
            1: Chem.rdchem.BondType.DOUBLE,
            2: Chem.rdchem.BondType.TRIPLE,
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
    if encoding == 'polymer':
        prob_dict = {'60': 0.7489344573582472,
                    '70': 0.0561389266682314,
                    '80': 0.0678638375933265,
                    '160': 0.08724385192820308,
                    '90': 0.032130486119902095,
                    '140': 0.007666591133009364,
                    '150': 2.184919908044154e-05}

    
    return random.choices(list(prob_dict.keys()), weights=list(prob_dict.values()))[0]

def create_torch_graph_from_one_atom(atom, edge_size, encoding_option='reduced'):
    num_atom = int(atom)

    atom_attribute = node_encoder(num_atom, encoding_option=encoding_option)
    # Create graph
    graph = torch_geometric.data.Data(x=atom_attribute.view(1, -1), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, edge_size)))

    return graph

def load_model(checkpoint_path, model, optimizer):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def get_model_GNN1(config, encoding_size, edge_size):

    return GNN1(in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                hidden_channels_list=config["GCN_size"],
                mlp_hidden_channels=config['mlp_hidden'],
                edge_channels=edge_size, 
                num_classes=encoding_size, 
                use_dropout=config['use_dropout'],
                size_info=config['use_size'],
                max_size=config['max_size'])

def get_model_GNN2(config, encoding_size, edge_size):

    return GNN2(in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                hidden_channels_list=config["GCN_size"],
                mlp_hidden_channels=config['mlp_hidden'],
                edge_channels=edge_size, 
                num_classes=edge_size, 
                size_info=config['use_size'],
                max_size=config['max_size'],
                use_dropout=config['use_dropout'])

def get_model_GNN3(config, encoding_size, edge_size):

    if config['graph_embedding']:
        return GNN3_embedding(in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                    hidden_channels_list=config["GCN_size"],
                    mlp_hidden_channels = config['mlp_hidden'],
                    edge_channels=edge_size, 
                    num_classes=edge_size,
                    use_dropout=config['use_dropout'],
                    size_info=config['use_size'],
                    max_size=config['max_size'])

    return GNN3(in_channels=encoding_size + int(config['feature_position'] + int(len(config['score_list']))),
                hidden_channels_list=config["GCN_size"],
                edge_channels=edge_size, 
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
        assert graph.x.size(1) == new_node_attr.size(1)

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

def select_node(tensor, edge_size):

    # Somme sur les 3 premières dimensions de chaque vecteur
    sum_on_first_dims = tensor[:, :edge_size - 1].sum(dim=1)

    # Trouver l'indice du node avec la plus grande somme
    max_index = torch.argmax(sum_on_first_dims)

    return tensor[max_index], max_index

def add_score_features(subgraph, scores_list, desired_scores_list, GNN_type = 1):

    if scores_list != []:
        assert len(scores_list) == len(desired_scores_list) 
        # Concat the scores to the node features
        for i, score in enumerate(scores_list):
            score_tensor = torch.tensor(desired_scores_list[i], dtype=torch.float).view(1, 1)
            # Duplicate the score tensor to match the number of nodes in the subgraph
            score_tensor = score_tensor.repeat(subgraph.x.size(0), 1)
            subgraph.x = torch.cat([subgraph.x, score_tensor], dim=-1)
        if GNN_type == 2:
            subgraph.neighbor = torch.cat([subgraph.neighbor, torch.zeros((1, len(scores_list)))], dim=-1)
    return subgraph


class MolGen():
    def __init__(self, GNN1, GNN2, GNN3, encoding_size, edge_size, feature_position, device, save_intermidiate_states = False, encoding_option = 'charged', score_list = [], desired_score_list = [], store_scores_3 = False):
        mol_graph = create_torch_graph_from_one_atom(sample_first_atom(encoding_option), edge_size=edge_size, encoding_option=encoding_option)
        self.mol_graph = torch_geometric.data.Batch.from_data_list([mol_graph])
        self.queue = [0]
        self.GNN1 = GNN1
        self.GNN2 = GNN2
        self.GNN3 = GNN3
        self.device = device
        self.feature_position = feature_position
        self.encoding_size = encoding_size
        self.edge_size = edge_size
        self.save_intermidiate_states = save_intermidiate_states
        if save_intermidiate_states:
            self.intermidiate_states = []

        self.score_list = score_list
        self.desired_score_list = desired_score_list
        self.store_scores_3 = store_scores_3
        if self.store_scores_3:
            self.counting_scores_3 = 0
            self.counting_total = 0

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

            graph1 = add_score_features(graph1, self.score_list, self.desired_score_list, GNN_type = 1)

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

            graph2 = add_score_features(graph2, self.score_list, self.desired_score_list, GNN_type = 2)

            assert graph2.x.size(1) == graph2.neighbor.size(1)
            prediction2 = self.GNN2(graph2.to(self.device))

            predicted_edge = torch.multinomial(F.softmax(prediction2, dim=1), 1).item()
            encoded_predicted_edge = torch.zeros(prediction2.size(), dtype=torch.float)
            encoded_predicted_edge[0, predicted_edge] = 1

            # Create a new node that is going to be added to the graph
            new_node = torch.zeros(1, self.encoding_size, dtype=torch.float)
            new_node[0, predicted_node] = 1
            
            #GNN3

            # Add the node and the edge to the graph
            new_graph = add_edge_or_node_to_graph(self.mol_graph.clone(), current_node, encoded_predicted_edge, new_node_attr = new_node)
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

            graph3 = add_score_features(graph3, self.score_list, self.desired_score_list, GNN_type = 3)
            prediction3 = self.GNN3(graph3.to(self.device))
            softmax_prediction3 = F.softmax(prediction3, dim=1)[graph3.mask]
            if softmax_prediction3.size(0) == 0:
                #Stop
                self.mol_graph = new_graph
                return

            if self.store_scores_3 and mask.size(0) > 1:
                # Compute the sum of the two first columns in softmax_prediction3
                sum_first_two_columns = softmax_prediction3[:, 0] + softmax_prediction3[:, 1]
                # Add +1 if there is two scores between 
                if (torch.sum(sum_first_two_columns > 0.5) > 1).item() == 1:
                    print(sum_first_two_columns)
                self.counting_scores_3 += (torch.sum(sum_first_two_columns > 0.5) > 1).item()
                self.counting_total += 1
            selected_edge_distribution, max_index = select_node(softmax_prediction3, edge_size=self.edge_size)

             #sample edge
            predicted_edge = torch.multinomial(selected_edge_distribution, 1).item()

            if predicted_edge == self.edge_size - 1:
                #Stop
                self.mol_graph = new_graph
                return

            encoded_predicted_edge = torch.zeros(prediction2.size(), dtype=torch.float)
            encoded_predicted_edge[0, predicted_edge] = 1


            output_graph = add_edge_or_node_to_graph(new_graph, graph1.x.size(0), encoded_predicted_edge, other_node=current_node + max_index+1)

            self.mol_graph = output_graph

    def full_generation(self):
        max_iter = 300
        i = 0
        while len(self.queue) > 0:
            if i > max_iter:
                break
            self.one_step()
            i += 1

            if self.save_intermidiate_states:
                self.intermidiate_states.append(self.mol_graph.clone())
        
    def is_valid(self):
        if self.edge_size == 3:
            edge_mapping = 'kekulized'
        else:
            edge_mapping = 'aromatic'
        SMILES_str = tensor_to_smiles(self.mol_graph.x, self.mol_graph.edge_index, self.mol_graph.edge_attr, edge_mapping, encoding_type=self.encoding_type)
        mol = Chem.MolFromSmiles(SMILES_str)
        if mol is None:
            return False
        else:
            return True


class GenerationModule():
    def __init__(self, config1, config2, config3, encoding_size, edge_size, pathGNN1, pathGNN2, pathGNN3, checking_mode = False, encoding_type = 'charged', score_list = [], desired_score_list = []):
        self.config1 = config1
        self.config2 = config2
        self.config3 = config3
        self.encoding_size = encoding_size
        self.edge_size = edge_size
        self.encoding_type = encoding_type
        self.feature_position = config1["feature_position"]
        self.checking_mode = checking_mode

        self.score_list = config1["score_list"]
        self.desired_score_list = desired_score_list

        if self.checking_mode:
            self.non_valid_molecules = []

        self.GNN1 = get_model_GNN1(config1, encoding_size, edge_size)
        self.GNN2 = get_model_GNN2(config2, encoding_size, edge_size)
        self.GNN3 = get_model_GNN3(config3, encoding_size, edge_size)

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
    
    def generate_single_molecule(self):
        mol = MolGen(self.GNN1_model,
                     self.GNN2_model,
                     self.GNN3_model,
                     self.encoding_size,
                     self.edge_size,
                     self.feature_position,
                     self.device,
                     save_intermidiate_states=self.checking_mode,
                     encoding_option=self.encoding_type,
                     score_list=self.score_list,
                     desired_score_list=self.desired_score_list,
                     store_scores_3 = True)
        mol.full_generation()
        if self.checking_mode:
            # check validity of the molecule
            if not mol.is_valid():
                self.non_valid_molecules.append(mol.intermidiate_states)
        return mol.mol_graph, mol.counting_scores_3, mol.counting_total

    def generate_mol_list(self, n_mol, n_threads=1):
        mol_list = []
        counting_scores_3_total = 0
        counting_total_total = 0
        if n_threads == 1:
            for i in tqdm(range(n_mol), desc="Generating molecules"):
                mol_graph, counting_scores_3, counting_total = self.generate_single_molecule()
                mol_list.append(mol_graph)
                counting_scores_3_total += counting_scores_3
                counting_total_total += counting_total
        else:

            # Utilize ThreadPoolExecutor to parallelize the task
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Submit tasks to the thread pool
                future_to_mol = {executor.submit(self.generate_single_molecule): i for i in range(n_mol)}
                
                # Collect the results as they become available
                for future in tqdm(as_completed(future_to_mol), total=n_mol, desc="Generating molecules"):
                    mol_graph = future.result()
                    mol_list.append(mol_graph)
        print("Average score 3: ", counting_scores_3_total / counting_total_total)           
        return mol_list
def return_current_nodes_batched(batch_graph, current_nodes_tensor : torch.tensor):

    # Count the number of nodes in each graph up to (but not including) the current node
    node_counts = batch_graph.batch.bincount().cumsum(0)
    offsets = torch.roll(node_counts, shifts=1, dims=0)
    offsets[0] = 0  # The first graph has no offset

    # Add the offsets to the current_nodes indices
    current_nodes_tensor += offsets

    return current_nodes_tensor
        

def set_feature_position(batch_graph, current_nodes_tensor):
    # Add feature position
    zeros_tensor = torch.zeros(batch_graph.x.size(0), 1, device=batch_graph.x.device)
    batch_graph.x = torch.cat([batch_graph.x, zeros_tensor], dim=1)
    current_nodes_expanded = current_nodes_tensor[batch_graph.batch]
    # Create a tensor that contains the cumulative sum of nodes up to the current node in each graph
    cumulative_current_nodes = (torch.arange(batch_graph.x.size(0), device=batch_graph.x.device) < (current_nodes_expanded)).float()
    # Flatten cumulative_current_nodes before using it for indexing
    cumulative_current_nodes = cumulative_current_nodes.view(-1)
    # Set the feature position for each node
    batch_graph.x[:, -1] = cumulative_current_nodes

    return batch_graph

def set_last_nodes(batch_graph):
    
    node_counts = batch_graph.batch.bincount().cumsum(0)
    offsets = torch.roll(node_counts, shifts=1, dims=0)
    offsets[0] = 0  # The first graph has no offset
    last_nodes_each_graph = batch_graph.batch.bincount() - 1 + offsets
    batch_graph.x[last_nodes_each_graph, -1] = 1

    return batch_graph, last_nodes_each_graph


def create_mask(batch_graph, current_nodes_tensor : torch.tensor, last_nodes):
    total_nodes = batch_graph.x.size(0)
    # Expand current_nodes to match the total number of nodes
    expanded_current_nodes = current_nodes_tensor[batch_graph.batch]
    mask = (torch.arange(batch_graph.x.size(0), device=batch_graph.x.device) > (expanded_current_nodes)).float()
    # Create a new mask where the last node of each graph is marked as False
    mask[last_nodes] = False

    # Convert the mask to a boolean mask
    mask = mask.bool()
    
    return mask


def add_nodes_and_edges(graphs, new_nodes, new_edges, current_nodes, mask_gnn2):
    # Iterate over the graphs, new nodes and edges, current nodes and mask simultaneously
    for graph, new_node, new_edge, current_node, add in zip(graphs, new_nodes, new_edges, current_nodes, mask_gnn2):
        # If the mask for this graph is False, do nothing
        if not add:
            continue
      
        # Add the new node to the graph
        graph.x = torch.cat([graph.x, new_node.unsqueeze(0)], dim=0)
        
        # Add an edge from the new node to the current node, and from the current node to the new node
        # to ensure that the graph remains undirected
        new_node_index = graph.x.size(0) - 1  # The index of the new node

        new_edge_indices = torch.tensor([[new_node_index, current_node], [current_node, new_node_index]], device=graph.edge_index.device)
        graph.edge_index = torch.cat([graph.edge_index, new_edge_indices], dim=1)
        
        # Add the new edge attributes to the graph
        graph.edge_attr = torch.cat([graph.edge_attr, new_edge.unsqueeze(0).repeat(2, 1)], dim=0)
    
    return graphs

def select_node_batch(tensor, batch_data, edge_size, mask):
    # Sum on the first dimensions of each vector
    sum_on_first_dims = tensor[:, :edge_size - 1].sum(dim=1)

    # Apply mask to the sum tensor, setting masked values to -inf
    masked_sum = sum_on_first_dims.masked_fill(~mask, float('-inf'))

    # Use scatter_max to find maximum values and their indices for each batch
    max_values, max_indices = scatter_max(masked_sum, batch_data, dim_size=batch_data.max().item() + 1)

    # Check if there is at least one True value in each graph
    count_mask = scatter_add(mask.int(), batch_data, dim_size=batch_data.max().item() + 1)

    # Replace indices where there were no True values in the mask with -1 (or any value you want)
    max_indices = max_indices.where(count_mask > 0, torch.tensor(-1).to(max_indices.device))

    # Sample using the tensor using the multinomial function
    sampled_indices = tensor.multinomial(num_samples=1).squeeze()

    # Replace -1 values in max_indices with the corresponding sampled_indices
    final_indices = torch.where(max_indices != -1, sampled_indices[max_indices], tensor.size(1) - 1)

    return final_indices, max_indices

# Function to check the valence of the nodes in the graph

def check_valence(graph, node):
    # Get the type of the node
    node_type = torch.argmax(graph.x[node])

    # Compute the valence of each node by summing the bond types of all edges connected to the node
    bond_types = torch.argmax(graph.edge_attr, dim=1) + 1  # add 1 to count single bonds as 1, etc.
    valences = torch.bincount(graph.edge_index[0], weights=bond_types)

    # Check if the valence is superior to the limit
    if node_type == 0:
        max_valence = 4
    elif node_type in [1, 2]:
        max_valence = 3
    else:
        max_valence = 6

    if valences[node].item() > max_valence:
        raise ValueError(f"Valence of node {node} is superior to {max_valence}")

    return valences
    
def add_edges_and_attributes(graphs, indices, choices, num_choices, batch, mask):
    # Create a one-hot encoded tensor of shape [num_choices, len(choices)]
    one_hot_choices = torch.zeros(num_choices, choices.size(0), device=choices.device).scatter_(0, choices.unsqueeze(0), 1)

    # Compute cumulative node counts for each graph
    node_counts = torch.bincount(batch)
    node_counts_shifted = torch.cat([torch.tensor([0], device=node_counts.device), node_counts[:-1]])
    cumulative_node_counts = node_counts_shifted.cumsum(0)

    # Compute indices in each graph
    indices_in_each_graph = indices - cumulative_node_counts[batch[indices]]

    for i, graph in enumerate(graphs):
        if mask[i]:
            # Add an edge from the max_index node to the last node
            last_node_index = graph.num_nodes - 1

            if choices[i] != num_choices - 1:  # If the choice is not the last column

                # Add the edge
                new_edge = torch.tensor([[indices_in_each_graph[i], last_node_index], [last_node_index, indices_in_each_graph[i]]], device=graph.edge_index.device)
                graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1) if graph.edge_index is not None else new_edge

                # Add the corresponding edge attribute
                new_attr = one_hot_choices[:, i].unsqueeze(1).t() 
                graph.edge_attr = torch.cat([graph.edge_attr, new_attr, new_attr], dim=0) if graph.edge_attr is not None else new_attr

    return graphs

def sample_first_atom_batch(batch_size, encoding = 'reduced'):
    if encoding == 'reduced' or encoding == 'charged':
        prob_dict = {'60': 0.7385023585929047, 
                    '80': 0.1000143018658728, 
                    '70': 0.12239949901813525, 
                    '90': 0.013786373862576426, 
                    '160': 0.017856330814654413,
                    '170': 0.007441135845856433}
    if encoding == 'polymer':
        prob_dict = {'60': 0.7489344573582472,
                    '70': 0.0561389266682314,
                    '80': 0.0678638375933265,
                    '160': 0.08724385192820308,
                    '90': 0.032130486119902095,
                    '140': 0.007666591133009364,
                    '150': 2.184919908044154e-05}

    atoms = [random.choices(list(prob_dict.keys()), weights=list(prob_dict.values()))[0] for _ in range(batch_size)]
    return atoms

def create_torch_graph_from_one_atom_list(atoms, edge_size, encoding_option='reduced') -> list:
    graphs = []
    for atom in atoms:
        num_atom = int(atom)
        atom_attribute = node_encoder(num_atom, encoding_option=encoding_option)
        # Create graph
        graph = torch_geometric.data.Data(x=atom_attribute.view(1, -1), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, edge_size)))
        graphs.append(graph)

    
    return graphs

class MolGenBatch():
    def __init__(self, GNN1, GNN2, GNN3, encoding_size, edge_size, batch_size, feature_position, device, save_intermidiate_states = False, encoding_option = 'charged', score_list = [], desired_score_list = []):

        self.mol_graphs_list = create_torch_graph_from_one_atom_list(sample_first_atom_batch(batch_size = batch_size, encoding = encoding_option), edge_size=edge_size, encoding_option=encoding_option)
        self.queues = [[0] for _ in range(batch_size)]
        self.batch_size = batch_size
        self.GNN1 = GNN1
        self.GNN2 = GNN2
        self.GNN3 = GNN3
        self.device = device
        self.feature_position = feature_position
        self.encoding_size = encoding_size
        self.edge_size = edge_size
        self.save_intermidiate_states = save_intermidiate_states
        if save_intermidiate_states:
            self.intermidiate_states = []

        self.score_list = score_list
        self.desired_score_list = desired_score_list
        self.finished_molecules = []

    def one_step(self):
        with torch.no_grad():

            # Lists to store the updated queues and current_nodes
            updated_queues = []
            current_nodes = []
            new_mol_graphs_list = []

            # Loop over each queue
            for i, queue in enumerate(self.queues):
                # If a queue is empty, its corresponding molecule is finished
                if not queue:
                    self.finished_molecules.append(self.mol_graphs_list[i])
                    self.batch_size -= 1
                else:
                    updated_queues.append(queue)
                    new_mol_graphs_list.append(self.mol_graphs_list[i])
                    current_nodes.append(queue[0])

            # Replace the original mol_graphs_list with the new one
            self.mol_graphs_list = new_mol_graphs_list
            self.queues = updated_queues

            if len(self.mol_graphs_list) == 0:
                return 
            """
            Prepare the GNN 1 BATCH
            """
            batch_graph12 = Batch.from_data_list(self.mol_graphs_list).to(self.device)
            current_nodes_tensor = torch.tensor(current_nodes, device=self.device, dtype=torch.long)
            current_nodes_batched = return_current_nodes_batched(batch_graph12, current_nodes_tensor) #reindex the nodes to match the batch size
            batch_graph12.x[current_nodes_batched, -1] = 1
            
            if self.feature_position:
                batch_graph12 = set_feature_position(batch_graph12, current_nodes_batched)
            
            batch_graph12 = add_score_features(batch_graph12, self.score_list, self.desired_score_list, GNN_type = 1)
            predictions = self.GNN1(batch_graph12)

            # Apply softmax to prediction
            softmax_predictions = F.softmax(predictions, dim=1)
            # Sample next node from prediction
            predicted_nodes = torch.multinomial(softmax_predictions, num_samples=1)

            # Create a mask for determining which graphs should continue to GNN2
            mask_gnn2 = (predicted_nodes != self.encoding_size - 1).flatten()

            # Handle the stopping condition (where predicted_node is encoding_size - 1)
            stop_mask = (predicted_nodes == self.encoding_size - 1).flatten()
            # Remove the first node from the queues of graphs that have stopped
            self.queues = [queue[1:] if stopped else queue for queue, stopped in zip(self.queues, stop_mask)]

            # Encode next node for the entire batch
            encoded_predicted_nodes = torch.zeros(predictions.size(), device=self.device, dtype=torch.float)
            encoded_predicted_nodes.scatter_(1, predicted_nodes, 1)

            # Add the size of x to the queue for graphs that haven't stopped
            self.queues = [queue + [graph.x.size(0)] if continue_gnn2 else queue
                        for queue, graph, continue_gnn2 in zip(self.queues, self.mol_graphs_list, mask_gnn2)]

            #GNN2 

            if self.feature_position:
                    
                # add zeros to the neighbor
                encoded_predicted_nodes = torch.cat([encoded_predicted_nodes, torch.zeros(self.batch_size, 1).to(encoded_predicted_nodes.device)], dim=1)

            batch_graph12.neighbor = encoded_predicted_nodes

            batch_graph12 = add_score_features(batch_graph12, self.score_list, self.desired_score_list, GNN_type = 2)

            assert batch_graph12.x.size(1) == batch_graph12.neighbor.size(1)
            
            predictions2 = self.GNN2(batch_graph12.to(self.device))
            predicted_edges = torch.multinomial(F.softmax(predictions2, dim=1), num_samples=1)

            encoded_predicted_edges = torch.zeros_like(predictions2, device=self.mol_graphs_list[0].x.device, dtype=torch.float)
            encoded_predicted_edges.scatter_(1, predicted_edges.detach().cpu(), 1)
            
            # Create a new node that is going to be added to the graph for each batch
            new_nodes = torch.zeros(self.batch_size, self.encoding_size, device=self.mol_graphs_list[0].x.device, dtype=torch.float)
            new_nodes.scatter_(1, predicted_nodes.detach().cpu(), 1)
            #GNN3

            # Add the node and the edge to the graph
            new_graph_list = add_nodes_and_edges(self.mol_graphs_list, new_nodes, encoded_predicted_edges, current_nodes, mask_gnn2)
            batch_graph3 = Batch.from_data_list(new_graph_list).to(self.device)
            batch_graph3, last_nodes = set_last_nodes(batch_graph3)
            current_nodes_batched = return_current_nodes_batched(batch_graph3, torch.tensor(current_nodes, device=self.device, dtype=torch.long)) #reindex the nodes to match the batch size
            
            if self.feature_position:
                batch_graph3 = set_feature_position(batch_graph3, current_nodes_batched)
            
            mask = create_mask(batch_graph3, current_nodes_tensor = current_nodes_batched, last_nodes = last_nodes)
            batch_graph3.mask = mask

            batch_graph3 = add_score_features(batch_graph3, self.score_list, self.desired_score_list, GNN_type = 3)

            prediction3 = self.GNN3(batch_graph3)
            
            softmax_prediction3 = F.softmax(prediction3, dim=1)
            edges_predicted, max_indices = select_node_batch(softmax_prediction3, batch_graph3.batch, self.edge_size, mask)
            new_graph_list = add_edges_and_attributes(new_graph_list, max_indices.detach().cpu(), edges_predicted.detach().cpu(), self.edge_size, batch_graph3.batch.detach().cpu(), mask_gnn2)


            self.mol_graphs_list = new_graph_list

    def full_generation(self):
        max_iter = 150
        i = 0
        while len(self.queues) > 0:
            if i > max_iter:
                break
            self.one_step()
            i += 1

            if self.save_intermidiate_states:
                self.intermidiate_states.append(self.mol_graph.clone())
        
    def is_valid(self):
        if self.edge_size == 3:
            edge_mapping = 'kekulized'
        else:
            edge_mapping = 'aromatic'
        SMILES_str = tensor_to_smiles(self.mol_graph.x, self.mol_graph.edge_index, self.mol_graph.edge_attr, edge_mapping, encoding_type=self.encoding_type)
        mol = Chem.MolFromSmiles(SMILES_str)
        if mol is None:
            return False
        else:
            return True


class GenerationModuleBatch():
    def __init__(self, config1, config2, config3, encoding_size, edge_size, pathGNN1, pathGNN2, pathGNN3, checking_mode = False, encoding_type = 'charged', batch_size = 64, score_list = [], desired_score_list = []):
        self.config1 = config1
        self.config2 = config2
        self.config3 = config3
        self.encoding_size = encoding_size
        self.edge_size = edge_size
        self.batch_size = batch_size
        self.encoding_type = encoding_type
        self.feature_position = config1["feature_position"]
        self.checking_mode = checking_mode

        self.score_list = config1["score_list"]
        self.desired_score_list = desired_score_list

        self.num_threads = num_threads

        if self.checking_mode:
            self.non_valid_molecules = []

        self.GNN1 = get_model_GNN1(config1, encoding_size, edge_size)
        self.GNN2 = get_model_GNN2(config2, encoding_size, edge_size)
        self.GNN3 = get_model_GNN3(config3, encoding_size, edge_size)

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
    
    def generate_single_molecule(self):
        mol = MolGenBatch(self.GNN1_model,
                     self.GNN2_model,
                     self.GNN3_model,
                     self.encoding_size,
                     self.edge_size,
                     self.batch_size,
                     self.feature_position,
                     self.device,
                     save_intermidiate_states=self.checking_mode,
                     encoding_option=self.encoding_type,
                     score_list=self.score_list,
                     desired_score_list=self.desired_score_list)
        mol.full_generation()
        if self.checking_mode:
            # check validity of the molecule
            if not mol.is_valid():
                self.non_valid_molecules.append(mol.intermidiate_states)
        return mol.finished_molecules

    def generate_mol_list(self, n_mol, n_threads=1):
        mol_list = []
        if n_threads == 1:
            for i in tqdm(range(n_mol), desc="Generating molecules"):
                mol_graph = self.generate_single_molecule()
                mol_list += mol_graph
        else:

            # Utilize ThreadPoolExecutor to parallelize the task
            with ThreadPoolExecutor(max_workers=n_threads) as executor:
                # Submit tasks to the thread pool
                future_to_mol = {executor.submit(self.generate_single_molecule): i for i in range(n_mol)}
                
                # Collect the results as they become available
                for future in tqdm(as_completed(future_to_mol), total=n_mol, desc="Generating molecules"):
                    mol_graph = future.result()
                    mol_list.append(mol_graph)
                    
        return mol_list
