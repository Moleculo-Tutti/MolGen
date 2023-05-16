import torch
import torch_geometric
import random
from torch_geometric.utils import to_undirected, subgraph
from torch_geometric.data import Data
import matplotlib.pyplot as plt

import networkx as nx
import torch
from torch_geometric.utils import to_networkx


from rdkit import Chem
from rdkit.Chem import rdchem
import torch
from torch_geometric.data import Data



def smiles_to_torch_geometric(smiles):
    """
    Convert a SMILES string into a torch_geometric.data.Data object.

    Args:
    smiles (str): A SMILES string.

    Returns:
    data (torch_geometric.data.Data): A PyTorch Geometric Data object representing the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        atom_features.append(atom.GetAtomicNum())
    x = torch.tensor(atom_features, dtype=torch.float).view(-1, 1)

    # Get bond features and adjacency indices
    bond_features, edge_index = [], []
    for bond in mol.GetBonds():
        bond_type = bond.GetBondTypeAsDouble()
        bond_features.extend([bond_type, bond_type])
        
        src, dest = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index.extend([[src, dest], [dest, src]])

    edge_attr = torch.tensor(bond_features, dtype=torch.float).view(-1, 1)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.num_nodes = len(atom_features)

    return data



def torch_geometric_to_networkx(data):
    """
    Convert a torch_geometric.data.Data object into a networkx.Graph object.

    Args:
    data (torch_geometric.data.Data): A PyTorch Geometric Data object representing the molecule.

    Returns:
    G (networkx.Graph): A NetworkX Graph object representing the molecule.
    """
    # Modify node features to take the argmax, excluding the last element

    copy_data = data.clone()
    if copy_data.x.shape[1] > 1:
        copy_data.x = torch.argmax(copy_data.x[:, :-1], dim=1).unsqueeze(1)

    # Modify edge features to take the argmax
    if copy_data.edge_attr.shape[1] > 1:
        copy_data.edge_attr = torch.argmax(copy_data.edge_attr, dim=1).unsqueeze(1)

    G = to_networkx(copy_data, node_attrs=['x'], edge_attrs=['edge_attr'])

    for i in G.nodes:
        x_attr = G.nodes[i]['x']
        atomic_num = int(x_attr.item()) if hasattr(x_attr, 'item') else int(x_attr)
        G.nodes[i]['atomic_num'] = atomic_num
        del G.nodes[i]['x']

    for i, j in G.edges:
        edge_attr = G.edges[i, j]['edge_attr']
        bond_type = edge_attr.item() if hasattr(edge_attr, 'item') else edge_attr
        G.edges[i, j]['bond_type'] = bond_type
        del G.edges[i, j]['edge_attr']

    return G

def plot_graph(G, show_atom_ids=True, show_atom_types=True, show_edge_types=True, id_map=None, atom_conversion_type='classic', encoding_type = None):
    """
    Plot a NetworkX graph with atom IDs and/or atom types as labels and edge types.

    Args:
    G (networkx.Graph): NetworkX graph representation of the molecule.
    show_atom_ids (bool, optional): Show atom IDs as labels if True. Default is True.
    show_atom_types (bool, optional): Show atom types as labels if True. Default is True.
    show_edge_types (bool, optional): Show edge types as labels if True. Default is True.
    id_map (dict, optional): A mapping from the original atom IDs to new IDs.
    """
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=False, node_size=500)

    if atom_conversion_type == 'classic':
        conversion_atom_num_to_symbol = {1: 'H', 6: 'C', 7: 'N', 8: 'O', 9: 'F', 16: 'S', 35: 'Br', 53: 'I'}
    elif atom_conversion_type == 'onehot':
        if encoding_type == 'all':
            conversion_atom_num_to_symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'P', 5: 'S', 6: 'Cl', 7: 'Br', 8: 'I'}
        elif encoding_type == 'reduced':
            conversion_atom_num_to_symbol = {0: 'C', 1: 'N', 2: 'O', 3: 'F', 4: 'S', 5: 'Cl'}

    # Prepare node labels
    labels = {}
    for node, data in G.nodes(data=True):
        label = ""
        if show_atom_ids:
            label += str(node)
            if id_map and node in id_map:
                label += f"/{id_map[node]}"
        if show_atom_types:
            if show_atom_ids:
                label += ":"
            label += conversion_atom_num_to_symbol[data['atomic_num']]

        labels[node] = label

    nx.draw_networkx_labels(G, pos, labels, font_size=12, font_weight='bold')

    # Prepare and draw edge labels
    if show_edge_types:
        edge_labels = {(i, j): G.edges[i, j]['bond_type'] for i, j in G.edges}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=12, font_weight='bold')

    plt.show()


def get_subgraph(data, indices, id_map):
    """
    Get a subgraph of a torch_geometric graph molecule based on given indices.

    Args:
    data (torch_geometric.data.Data): A PyTorch Geometric Data object representing the molecule.
    indices (list or torch.Tensor): List of node indices to extract as subgraph.

    Returns:
    subgraph_data (torch_geometric.data.Data): Subgraph of the torch_geometric graph molecule.
    """
    if not isinstance(indices, torch.Tensor):
        indices = torch.tensor(indices, dtype=torch.long)

    # Create a dictionary to map the old indices to new indices
    index_map = {old_index: id_map[old_index] for old_index in indices.tolist()}

    # Extract node features
    subgraph_x = torch.zeros(len(index_map), data.x.size(1))
    for i in range(len(indices)):
        subgraph_x[index_map[indices[i].item()]] = data.x[indices[i]]
        
    
    # Extract edges that are connected to the selected nodes
    mask = torch.tensor([src in index_map and tgt in index_map for src, tgt in data.edge_index.t().tolist()]).bool()
    subgraph_edge_index = data.edge_index[:, mask]

    # Relabel the edge indices according to the new node indices
    subgraph_edge_index = torch.tensor([[index_map[src], index_map[tgt]] for src, tgt in subgraph_edge_index.t().tolist()]).t()
    #force the edge index to be int64
    subgraph_edge_index = subgraph_edge_index.type(torch.int64)
    # Extract corresponding edge attributes
    subgraph_edge_attr = data.edge_attr[mask]

    # Create a new torch_geometric data object
    subgraph_data = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr)

    return subgraph_data

def get_subgraph_with_terminal_nodes(data, num_atoms):
    """
    Get a subgraph of a torch_geometric graph molecule based on specific rules.

    Args:
    data (torch_geometric.data.Data): A PyTorch Geometric Data object representing the molecule.
    num_atoms (int): Desired number of atoms in the subgraph.

    Returns:
    subgraph_data (torch_geometric.data.Data): Subgraph of the torch_geometric graph molecule.
    """
    if num_atoms < 1 or num_atoms > data.num_nodes:
        raise ValueError("num_atoms must be between 1 and the number of nodes in the graph.")

    # Randomly select an atom
    start_atom = random.choice(range(data.num_nodes))

    # Initialize the queue and visited set
    queue = [start_atom]
    visited = set()

    id_map = {}
    new_id = 0

    # Breadth-first search
    while queue:
        current = queue.pop(0)
        visited.add(current)

        id_map[current] = new_id
        new_id += 1

        # Add neighbors to the queue
        neighbors = data.edge_index[:, data.edge_index[0] == current][1].tolist()
        random.shuffle(neighbors)
        for neighbor in neighbors:
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)

        # Stop if we've reached the desired number of atoms
        if len(visited) == num_atoms:
            break

    # Get the subgraph with the selected atoms
    subgraph_indices = torch.tensor(list(visited), dtype=torch.long)
    subgraph_data = get_subgraph(data, subgraph_indices, id_map)
    
    external_neighbors = []
    oldest_non_completed = min([i for i in neighbors if i in visited], key = lambda x:id_map[x])
    neighbors_oldest_non_completed = data.edge_index[:, data.edge_index[0] == oldest_non_completed][1].tolist()

        
    for neighbor in neighbors_oldest_non_completed:
        if neighbor not in visited:
            edge_attr_idx = (data.edge_index[0] == oldest_non_completed) & (data.edge_index[1] == neighbor)
            edge_data = data.edge_attr[edge_attr_idx][0]

            external_neighbors_edges = []
            for neighbor2 in data.edge_index[:, data.edge_index[0] == neighbor][1].tolist():
                if neighbor2 in visited and neighbor2 != oldest_non_completed:
                    # Get the index of the edge attribute
                    edge_attr_idx = (data.edge_index[0] == neighbor) & (data.edge_index[1] == neighbor2)
                    # Get the edge attribute
                    edge_attr = data.edge_attr[edge_attr_idx][0]
                    external_neighbors_edges.append((neighbor2, edge_attr))
            external_neighbors.append((neighbor, data.x[neighbor], edge_data, external_neighbors_edges))

    terminal_node_info = (oldest_non_completed, external_neighbors)

    return subgraph_data, terminal_node_info, id_map

def get_subgraph_with_terminal_nodes_step(data, num_steps, impose_edges=False):
    """
    Get a subgraph of a torch_geometric graph molecule based on specific rules.

    Args:
    data (torch_geometric.data.Data): A PyTorch Geometric Data object representing the molecule.
    num_atoms (int): Desired number of atoms in the subgraph.

    Returns:
    subgraph_data (torch_geometric.data.Data): Subgraph of the torch_geometric graph molecule.
    """
    if impose_edges == False:
        if num_steps < 1 or num_steps > data.num_nodes*2:
            raise ValueError("num_atoms must be between 1 and 2 times the number of nodes in the graph.")
    if impose_edges == True:
        if num_steps < 1 or num_steps > data.num_nodes:
            raise ValueError("num_atoms must be between 1 and the number of nodes in the graph.")

    # Randomly select an atom
    start_atom = random.choice(range(data.num_nodes))

    queue = [start_atom]
    subgraph_atoms = [start_atom]
    count_num_steps = 1
    visited = set()

    id_map = {}
    new_id = 0

    id_map[start_atom] = new_id
    new_id += 1

    # Breadth search 

    while queue or count_num_steps <= num_steps:

        current = queue.pop(0)

        # Add neighbors to the queue
        neighbors = data.edge_index[:, data.edge_index[0] == current][1].tolist()
        #reduce nighbors list not in subgraph 
        neighbors = [i for i in neighbors if i not in subgraph_atoms]
        random.shuffle(neighbors)

        if len(neighbors) + count_num_steps >= num_steps:
            predicted = neighbors[:num_steps - count_num_steps]
            for neighbor in neighbors[num_steps - count_num_steps:]:
                id_map[neighbor] = new_id
                new_id += 1
                subgraph_atoms.append(neighbor)
            external_neighbors = []
            for neighbor in predicted:
                #get edge attributes
                edge_attr_idx = (data.edge_index[0] == current) & (data.edge_index[1] == neighbor)
                edge_data = data.edge_attr[edge_attr_idx][0]
                #get external neighbors
                external_neighbors_edges = []
                for neighbor2 in data.edge_index[:, data.edge_index[0] == neighbor][1].tolist():
                    if neighbor2 in subgraph_atoms and neighbor2 != current:
                        # Get the index of the edge attribute
                        edge_attr_idx = (data.edge_index[0] == neighbor) & (data.edge_index[1] == neighbor2)
                        # Get the edge attribute
                        edge_attr = data.edge_attr[edge_attr_idx][0]
                        external_neighbors_edges.append((neighbor2, edge_attr))
                external_neighbors.append((neighbor, data.x[neighbor], edge_data, external_neighbors_edges))
            
            terminal_node_infos = (current, external_neighbors)
            subgraph_indices = torch.tensor(list(subgraph_atoms), dtype=torch.long)
            subgraph_data = get_subgraph(data, subgraph_indices, id_map)
            return subgraph_data, terminal_node_infos, id_map


        for neighbor in neighbors:
            queue.append(neighbor)
            subgraph_atoms.append(neighbor)
            id_map[neighbor] = new_id
            new_id += 1
            count_num_steps += 1   

        visited.add(current)
        if impose_edges == False:
            #For the edges prediction we only count the number of atoms added
            count_num_steps += 1

        # Stop if we've reached the desired number of steps
        if count_num_steps == num_steps:
            # we predict a stop in terminal_node_infos
            terminal_node_infos = (current, [])
            subgraph_indices = torch.tensor(list(subgraph_atoms), dtype=torch.long)
            subgraph_data = get_subgraph(data, subgraph_indices, id_map)
            return subgraph_data, terminal_node_infos, id_map

    raise ValueError("The number of steps is too high for the graph.")    

    
def node_encoder(atom_num : float, encoding_option = 'all') -> torch.Tensor:
    """
    Encode the atom number into a one-hot vector.
    """
    #Verify that encoding_option is among the allowed values
    if encoding_option not in ['all', 'reduced']:
        raise ValueError("encoding_option must be either 'all' or 'reduced'.")
    
    if encoding_option == 'all':
        atom_mapping = {6: 0, 7: 1, 8: 2, 9: 3, 15: 4, 16: 5, 17: 6, 35: 7, 53: 8}
        size = 9
    elif encoding_option == 'reduced':
        atom_mapping = {6: 0, 7: 1, 8: 2, 9: 3, 16: 4, 17: 5}
        size = 6

    # Initialize the one-hot vector
    one_hot = torch.zeros(size + 1)

    # Encode the atom number
    if int(atom_num) in atom_mapping:
        one_hot[atom_mapping[int(atom_num)]] = 1
    else:
        raise ValueError("Atom number not in mapping.")
    
    return one_hot

def edge_encoder(bond_type : float) -> torch.Tensor:
    """
    Encode the bond type into a one-hot vector.
    """
    # Initialize the one-hot vector
    one_hot = torch.zeros(4)

    # Encode the bond type
    if  1.4 < bond_type < 1.6:
        one_hot[0] = 1
    else:
        bond_type = int(bond_type)
        one_hot[bond_type] = 1
    
    return one_hot

def process_encode_graph(smiles, encoding_option = 'all') -> torch_geometric.data.Data:

    """
    Take a SMILES string and encode it into a torch_geometric.data.Data object.
    One hot encode the node and edge features.

    Args:
    smiles (str): SMILES string of the molecule.
    encoding_option (str): Option for which dataset to use. Must be either 'all' or 'reduced'. Used in node_encoder.
    """

    data = smiles_to_torch_geometric(smiles)

    node_features = data.x
    edge_attr = data.edge_attr

    #encode the node features with the function node encoder one atom at a time
    node_features = torch.stack([node_encoder(atom, encoding_option) for atom in node_features])
    
    #encore the edge features with the function edge encoder one bond at a time
    edge_attr = torch.stack([edge_encoder(bond) for bond in edge_attr])

    #create new data object with the encoded node and edge features

    encoded_data = Data(x=node_features, edge_attr=edge_attr, edge_index=data.edge_index)

    return encoded_data


def encode_graph_data(graph, terminal_node_info, node_encoder, edge_encoder):
    """
    Encode the graph data into a format that can be used by the neural network.
    DEPRECATED I THINK
    """

    # Encode the node features
    node_features = torch.zeros(graph.num_nodes, 10)
    for i in range(graph.num_nodes):
        node_features[i] = node_encoder(graph.x[i])
        if i == terminal_node_info[0]:
            node_features[i][9] = 1
        
    
    # Encode the edge features
    edge_features = torch.zeros(graph.num_edges, 4)
    for i in range(graph.num_edges):
        edge_features[i] = edge_encoder(graph.edge_attr[i])

    encoded_graph = Data(x=node_features, edge_index=graph.edge_index, edge_attr=edge_features)

    return encoded_graph

