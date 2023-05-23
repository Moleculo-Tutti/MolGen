import random
import torch

from MolGen.DataPipeline.preprocessing import process_encode_graph, get_subgraph_with_terminal_nodes_step
from MolGen.Model.GNN1 import ModelWithEdgeFeatures as GNN1
from MolGen.Model.GNN2 import ModelWithEdgeFeatures as GNN2
from MolGen.Model.GNN3 import ModelWithEdgeFeatures as GNN3


def sample_random_subgraph_ZINC(pd_dataframe, start_size):
    indice = random.choice(pd_dataframe.index)
    smiles_str = pd_dataframe.loc[indice, 'smiles']

    torch_graph = process_encode_graph(smiles_str, encoding_option='reduced')
    subgraph_data, terminal_node_info, id_map = get_subgraph_with_terminal_nodes_step(torch_graph, start_size)

    return subgraph_data, terminal_node_info, id_map

def sample_first_atom():
    prob_dict = {'C': 0.7385023585929047, 
                 'O': 0.1000143018658728, 
                 'N': 0.12239949901813525, 
                 'F': 0.013786373862576426, 
                 'S': 0.017856330814654413,
                 'Cl': 0.007441135845856433}
    
    return random.choices(list(prob_dict.keys()), weights=list(prob_dict.values()))[0]


def load_model(checkpoint_path, model, optimizer):

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def get_model_GNN1(encoding_size):
    return GNN1(in_channels=encoding_size + 1, 
                hidden_channels_list=[64, 128, 256, 512, 512], 
                mlp_hidden_channels=512, 
                edge_channels=4, 
                num_classes=encoding_size, 
                use_dropout=False)

def get_model_GNN2(encoding_size):
    return GNN2(in_channels=encoding_size, 
                hidden_channels_list=[64, 128, 256, 512, 512], 
                mlp_hidden_channels=512, 
                edge_channels=4, 
                num_classes=4, 
                use_dropout=False)

def get_model_GNN3(encoding_size):
    return GNN3(in_channels=encoding_size, 
                hidden_channels_list=[64, 128, 128, 64, 32, 5], 
                edge_channels=4, 
                use_dropout=False)

def get_optimizer(model, lr):
    return torch.optim.Adam(model.parameters(), lr=lr)



def add_edge_or_node_to_graph(graph, initial_node, edge_attr, new_node_attr=None):
    # Convert nodes to tensor
    initial_node = torch.tensor([initial_node], dtype=torch.long)

    # Create new edge attribute
    new_edge_attr = torch.tensor([edge_attr, edge_attr], dtype=torch.float)

    # If new_node_attr is provided, add a new node
    if new_node_attr is not None:
        # Create new node
        new_node = torch.tensor([new_node_attr], dtype=torch.float)

        # Add new node to graph
        graph.x = torch.cat([graph.x, new_node], dim=0)

        # Adjust other_node to be the new node
        other_node = torch.tensor([graph.x.size(0)-1], dtype=torch.long)
    else:
        raise ValueError("other_node must be provided if no new node is being added")

    # Create new edge
    new_edge = torch.tensor([[initial_node.item(), other_node.item()], 
                             [other_node.item(), initial_node.item()]], dtype=torch.long)

    # Add new edge and its attribute to graph
    graph.edge_index = torch.cat([graph.edge_index, new_edge], dim=1)
    graph.edge_attr = torch.cat([graph.edge_attr, new_edge_attr], dim=0)

    return graph



