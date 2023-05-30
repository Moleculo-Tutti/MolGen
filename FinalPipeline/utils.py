import random
import torch
import torch_geometric

import torch.nn.functional as F
import sys
import os

cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from DataPipeline.preprocessing import process_encode_graph, get_subgraph_with_terminal_nodes_step
from DataPipeline.preprocessing import node_encoder
from Model.GNN1 import ModelWithEdgeFeatures as GNN1
from Model.GNN2 import ModelWithEdgeFeatures as GNN2
from Model.GNN3 import ModelWithEdgeFeatures as GNN3


def sample_random_subgraph_ZINC(pd_dataframe, start_size):
    indice = random.choice(pd_dataframe.index)
    smiles_str = pd_dataframe.loc[indice, 'smiles']

    torch_graph = process_encode_graph(smiles_str, encoding_option='reduced')
    subgraph_data, terminal_node_info, id_map = get_subgraph_with_terminal_nodes_step(torch_graph, start_size)

    return subgraph_data, terminal_node_info, id_map

def sample_first_atom():
    prob_dict = {'6': 0.7385023585929047, 
                 '8': 0.1000143018658728, 
                 '7': 0.12239949901813525, 
                 '9': 0.013786373862576426, 
                 '16': 0.017856330814654413,
                 '17': 0.007441135845856433}
    
    return random.choices(list(prob_dict.keys()), weights=list(prob_dict.values()))[0]

def create_torch_graph_from_one_atom(atom):
    num_atom = int(atom)

    atom_attribute = node_encoder(num_atom, encoding_option='reduced')
    # Create graph
    graph = torch_geometric.data.Data(x=atom_attribute.view(1, -1), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, 4)))

    return graph

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
                use_dropout=False,
                size_info=False)

def get_model_GNN2(encoding_size):
    return GNN2(in_channels=encoding_size, 
                hidden_channels_list=[64, 128, 256, 512, 512], 
                mlp_hidden_channels=512, 
                edge_channels=4, 
                num_classes=4, 
                use_dropout=False)

def get_model_GNN3(encoding_size):
    return GNN3(in_channels=encoding_size + 1, 
                hidden_channels_list=[32, 64, 128, 256, 128, 64, 32, 5],
                edge_channels=4, 
                use_dropout=False)

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

    return graph

def select_node(tensor):
    
    # Vérifier que le tenseur est de dimension 2 et que la deuxième dimension est de taille 4
    assert len(tensor.shape) == 2 and tensor.shape[1] == 5, "Le tenseur doit être de dimension 2 et la deuxième dimension doit être de taille 4"

    # Somme sur les 4 premières dimensions de chaque vecteur
    sum_on_first_three_dims = tensor[:, :4].sum(dim=1)

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

        graph2.neighbor = encoded_predicted_node
        prediction2 = GNN2(graph2.to(device))
        predicted_edge = torch.multinomial(F.softmax(prediction2, dim=1), 1).item()
        encoded_predicted_edge = torch.zeros(prediction2.size(), dtype=torch.float)
        encoded_predicted_edge[0, predicted_edge] = 1
        # GNN3

        new_graph = add_edge_or_node_to_graph(input_graph.clone(), current_node, encoded_predicted_edge, new_node_attr = encoded_predicted_node)
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
        
        if predicted_edge == 4:
            #Stop
            return new_graph, queue
        
        encoded_predicted_edge = torch.zeros(prediction2.size(), dtype=torch.float)
        encoded_predicted_edge[0, predicted_edge] = 1


        output_graph = add_edge_or_node_to_graph(new_graph, graph1.x.size(0), encoded_predicted_edge, other_node=current_node + max_index+1)
        return output_graph, queue