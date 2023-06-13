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
import gc


def smiles_to_torch_geometric(smiles, charge=False, kekulize=False):
    """
    Convert a SMILES string into a torch_geometric.data.Data object.

    Args:
    smiles (str): A SMILES string.
    kekulize (bool): If True, kekulize the molecule before conversion.

    Returns:
    data (torch_geometric.data.Data): A PyTorch Geometric Data object representing the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Kekulize if necessary
    if kekulize:
        try:
            Chem.Kekulize(mol)
        except:
            raise ValueError("Failed to kekulize SMILES string")

    # Get atom features
    atom_features = []
    for atom in mol.GetAtoms():
        if charge:
            atom_charge = atom.GetFormalCharge()
        else:
            atom_charge = 0
        atom_features.append(atom.GetAtomicNum()*10 + atom_charge)
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
        elif encoding_type == 'charged':
            conversion_atom_num_to_symbol = {0: 'C', 1: 'N', 2: 'N+', 3: 'N-', 4:'O', 5:'O-', 6:'F', 8:'S', 9:'S-', 10:'Cl', 11:'Br', 12:'I'}

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
        if num_steps < 1 or num_steps >= data.num_nodes*2:
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

        if len(neighbors) + count_num_steps > num_steps:
            predicted = neighbors[num_steps - count_num_steps:]
            for neighbor in neighbors[:num_steps - count_num_steps]:
                id_map[neighbor] = new_id
                new_id += 1
                subgraph_atoms.append(neighbor)
                count_num_steps += 1
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

            # Garbage collection
            del queue, subgraph_atoms, visited
            return subgraph_data, terminal_node_infos, id_map

        for neighbor in neighbors:
            queue.append(neighbor)
            subgraph_atoms.append(neighbor)
            id_map[neighbor] = new_id
            new_id += 1
            count_num_steps += 1 

        # Stop if we've reached the desired number of steps
        if count_num_steps >= num_steps and impose_edges == False:
            # we predict a stop in terminal_node_infos
            terminal_node_infos = (current, [])
            subgraph_indices = torch.tensor(list(subgraph_atoms), dtype=torch.long)
            subgraph_data = get_subgraph(data, subgraph_indices, id_map)

            # Garbage collection
            del queue, subgraph_atoms, visited
            return subgraph_data, terminal_node_infos, id_map  
        

        visited.add(current)

        if impose_edges == False:
            #For the edges prediction we only count the number of atoms added
            count_num_steps += 1



    raise ValueError("The number of steps is too high for the graph.")    

    
def node_encoder(atom_num : float, encoding_option = 'all') -> torch.Tensor:
    """
    Encode the atom number into a one-hot vector.
    """
    #Verify that encoding_option is among the allowed values
    if encoding_option not in ['all', 'reduced', 'charged', 'polymer']:
        raise ValueError("encoding_option must be either 'all' or 'reduced' or 'charged'.")
    
    if encoding_option == 'all':
        atom_mapping = {60: 0, 70: 1, 80: 2, 90: 3, 150: 4, 160: 5, 170: 6, 350: 7, 530: 8}
        size = 9
    elif encoding_option == 'reduced':
        atom_mapping = {60: 0, 70: 1, 80: 2, 90: 3, 160: 4, 170: 5}
        size = 6
    elif encoding_option == 'charged':
        atom_mapping = {60: 0, 70:1, 71: 2, 69: 3, 80:4, 79:5, 90:6, 160:7, 159:8, 170:9, 350:10, 530:11}
        size = 12
    elif encoding_option == 'polymer':
        atom_mapping = {60: 0, 70: 1, 80: 2, 90: 3, 140: 4, 150: 5, 160: 6}
        size = 7


    # Initialize the one-hot vector
    one_hot = torch.zeros(size + 1)

    # Encode the atom number
    if int(atom_num) in atom_mapping:
        one_hot[atom_mapping[int(atom_num)]] = 1
    else:
        raise ValueError("Atom number {num} not in mapping.".format(num = int(atom_num)))
    
    return one_hot

def edge_encoder(bond_type : float, kekulize=False) -> torch.Tensor:
    """
    Encode the bond type into a one-hot vector.
    
    Args:
    bond_type (float): The bond type as a float (1.0 for single, 2.0 for double, 3.0 for triple, 1.5 for aromatic).
    kekulize (bool): If True, encode the bond type into a 3-dimensional vector (without aromatic bonds).

    Returns:
    one_hot (torch.Tensor): A one-hot vector representing the bond type.
    """
    # Initialize the one-hot vector
    if kekulize:
        one_hot = torch.zeros(3)
    else:
        one_hot = torch.zeros(4)

    # Encode the bond type
    if not kekulize and 1.4 < bond_type < 1.6:
        one_hot[0] = 1
    else:
        bond_type = int(bond_type)
        one_hot[bond_type - 1] = 1

    return one_hot

def process_encode_graph(smiles, encoding_option='all', kekulize=False, optional_scores = None) -> torch_geometric.data.Data:
    """
    Take a SMILES string and encode it into a torch_geometric.data.Data object.
    One hot encode the node and edge features.

    Args:
    smiles (str): SMILES string of the molecule.
    encoding_option (str): Option for which dataset to use. Must be either 'all' or 'reduced'. Used in node_encoder.
    kekulize (bool): If True, kekulize the molecule before conversion and encode the bond type into a 3-dimensional vector.

    Returns:
    encoded_data (torch_geometric.data.Data): A PyTorch Geometric Data object representing the molecule with encoded node and edge features.
    """
    charge = False
    if encoding_option == 'charged':
        charge = True
    data = smiles_to_torch_geometric(smiles, charge=charge, kekulize=kekulize)

    node_features = data.x
    edge_attr = data.edge_attr

    # Encode the node features with the function node_encoder one atom at a time
    node_features = torch.stack([node_encoder(atom, encoding_option) for atom in node_features])

    # Encode the edge features with the function edge_encoder one bond at a time
    edge_attr = torch.stack([edge_encoder(bond, kekulize=kekulize) for bond in edge_attr])

    # Create new data object with the encoded node and edge features
    encoded_data = Data(x=node_features, edge_attr=edge_attr, edge_index=data.edge_index)

    # Add the optional scores
    if optional_scores is not None:
        for key, value in optional_scores.items():
            encoded_data.__setattr__(key, value)

    return encoded_data

