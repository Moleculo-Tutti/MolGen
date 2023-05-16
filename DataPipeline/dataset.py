import torch 
import random

import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Batch

import zstandard as zstd
from io import BytesIO

from tqdm import tqdm
import os


from DataPipeline.preprocessing import get_subgraph_with_terminal_nodes, get_subgraph_with_terminal_nodes_step

def add_node_feature_based_on_position(data):
    # Find the index of the 'current_atom' for each graph in the batch
    current_atom_indices = (data.x[:, 6] == 1).nonzero(as_tuple=True)[0]
    
    # Initialize a zero tensor to store the new feature
    new_feature = torch.zeros(data.num_nodes, 1, device=data.x.device)

    # Calculate cumulative sums of node counts for each graph
    cumsum_node_counts = data.batch.bincount().cumsum(dim=0)
    
    for i in range(data.num_graphs):
        if i == 0:
            # For the first graph, start_index should be 0
            start_index = 0
        else:
            # For the subsequent graphs, start_index is the end_index of the previous graph
            start_index = cumsum_node_counts[i-1]
        
        # end_index is the cumulative sum of node counts up to the current graph
        end_index = cumsum_node_counts[i]
        
        # Find the index of the 'current_atom' within this graph
        current_atom_index = current_atom_indices[(current_atom_indices >= start_index) & (current_atom_indices < end_index)][0]

        # Set the new feature to 1 for nodes before the 'current_atom'
        new_feature[start_index:current_atom_index] = 1

    # Concatenate the new feature to the node features
    data.x = torch.cat([data.x, new_feature], dim=-1)

    return data


class ZincSubgraphDataset(Dataset):

    def __init__(self, data_path):
        self.data_list = torch.load(data_path)
        self.encoding_size = self.data_list[0].x.size(1)
        print('Dataset encoded with size {}'.format(self.encoding_size))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        preprocessed_graph = self.data_list[idx]

        assert preprocessed_graph.x.size(1) == self.encoding_size, 'Encoding size mismatch'
   
        mol_size = len(preprocessed_graph.x)
        num_atoms = random.choice(range(3, mol_size + 1))
        subgraph, terminal_nodes, id_map = get_subgraph_with_terminal_nodes(preprocessed_graph, num_atoms)

        subgraph.x[id_map[terminal_nodes[0]]][self.encoding_size - 1] = 1

        #get the embedding of all the first element of terminal_nodes[1] and make them into a list to take the mean, if terminal_nodes[1] empty make torch.zeros(10)
        label_gnn1 = torch.zeros(self.encoding_size)
        neighbor_atom_list = [neighbor[1] for neighbor in terminal_nodes[1]]

        if len(neighbor_atom_list) != 0:
            label_gnn1 += torch.mean(torch.stack(neighbor_atom_list, dim=0), dim=0)
        else:
            label_gnn1 += torch.tensor([0] * (self.encoding_size - 1) + [1])

        subgraph.y = label_gnn1
        
        #id_chosen = np.random.randint(len(terminal_nodes[1]))
        #subgraph.neighbor = terminal_nodes[1][id_chosen][1]

        #subgraph.edge_neighbor = terminal_nodes[1][id_chosen][2]

        return subgraph, terminal_nodes


class ZincSubgraphDatasetStep(Dataset):

    def __init__(self, data_path, GNN_type : str):
        self.data_list = torch.load(data_path)
        self.encoding_size = self.data_list[0].x.size(1)
        self.GNN_type = GNN_type
        if GNN_type >= 2:
            self.impose_edges = True
        else:
            self.impose_edges = False
        print('Dataset encoded with size {}'.format(self.encoding_size))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        preprocessed_graph = self.data_list[idx]

        assert preprocessed_graph.x.size(1) == self.encoding_size, 'Encoding size mismatch'
   
        mol_size = len(preprocessed_graph.x)
        if self.GNN_type == 1:
            num_steps = random.choice(range(3, 2*mol_size + 1))
        if self.GNN_type >= 2:
            num_steps = random.choice(range(3, mol_size + 1))
        subgraph, terminal_nodes, id_map = get_subgraph_with_terminal_nodes_step(preprocessed_graph, num_steps, impose_edges=self.impose_edges)
        subgraph.x[id_map[terminal_nodes[0]]][self.encoding_size - 1] = 1

        #get the embedding of all the first element of terminal_nodes[1] and make them into a list to take the mean, if terminal_nodes[1] empty make torch.zeros(10)
        label_gnn1 = torch.zeros(self.encoding_size)
        neighbor_atom_list = [neighbor[1] for neighbor in terminal_nodes[1]]

        if len(neighbor_atom_list) != 0:
            label_gnn1 += torch.mean(torch.stack(neighbor_atom_list, dim=0), dim=0)
        else:
            label_gnn1 += torch.tensor([0] * (self.encoding_size - 1) + [1])

        subgraph.y = label_gnn1
        
        if self.GNN_type == 2:
            id_chosen = np.random.randint(len(terminal_nodes[1]))
            subgraph.neighbor = terminal_nodes[1][id_chosen][1]
            subgraph.edge_neighbor = terminal_nodes[1][id_chosen][2]

        if self.GNN_type == 3:
            
            id_chosen = np.random.randint(len(terminal_nodes[1]))
            neighbor = terminal_nodes[1][id_chosen][1]
            edge_neighbor_attr = terminal_nodes[1][id_chosen][2]

            #add neighbor and edge_neighbor to the graph
            subgraph.x = torch.cat([subgraph.x, neighbor.unsqueeze(0)], dim=0)

            node1 = id_map[terminal_nodes[0]]
            node2 = len(subgraph.x) - 1

            add_edge_index = torch.tensor([[node1, node2], [node2, node1]], dtype=torch.long)
            subgraph.edge_index = torch.cat([subgraph.edge_index, add_edge_index], dim=1)
            # add double edge_attribute 
            subgraph.edge_attr = torch.cat([subgraph.edge_attr, edge_neighbor_attr.unsqueeze(0), edge_neighbor_attr.unsqueeze(0)], dim=0)


            node_features_label = torch.zeros(len(subgraph.x), 5)
        

            if len(terminal_nodes[1][id_chosen][3]) != 0:
                cycle_edge_features = [cycle_neighbor[1] for cycle_neighbor in terminal_nodes[1][id_chosen][3]]
            
            else:
                cycle_edge_features = []
            #subgraph.cycle_edge_features = cycle_edge_features_tensor
            return subgraph, cycle_edge_features

        return subgraph


def custom_collate(batch):
    sg_data_list = [item for item in batch]
    terminal_nodes_info_list = [item.y for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    terminal_nodes_info_tensor = torch.stack(terminal_nodes_info_list, dim=0)
    return sg_data_batch, terminal_nodes_info_tensor

def custom_collate_passive_add_feature(batch):
    sg_data_list = [item for item in batch]
    terminal_nodes_info_list = [item.y for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    terminal_nodes_info_tensor = torch.stack(terminal_nodes_info_list, dim=0)

    feature_sg_data_batch = add_node_feature_based_on_position(sg_data_batch)
    return feature_sg_data_batch, terminal_nodes_info_tensor

def custom_collate_GNN1(batch):
    sg_data_list = [item for item in batch]
    terminal_nodes_info_list = [item.y for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    terminal_nodes_info_tensor = torch.stack(terminal_nodes_info_list, dim=0)
    return sg_data_batch, terminal_nodes_info_tensor

def custom_collate_GNN2(batch):
    sg_data_list = [item for item in batch]
    terminal_nodes_info_list = [item.edge_neighbor for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    terminal_nodes_info_tensor = torch.stack(terminal_nodes_info_list, dim=0)
    return sg_data_batch, terminal_nodes_info_tensor

def custom_collate_GNN3(batch):

    sg_data_list = [item[0] for item in batch]
    terminal_nodes_info_list = [item[1] for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    
    return sg_data_batch, terminal_nodes_info_list

def load_compressed_batch(file_path):
    with open(file_path, 'rb') as f:
        dctx = zstd.ZstdDecompressor()
        decompressor = dctx.decompressobj()
        decompressed_data = decompressor.decompress(f.read())
        decompressed_buffer = BytesIO(decompressed_data)
        graph_batch = torch.load(decompressed_buffer)

    return graph_batch

def load_all_and_merge(number_reference_dict, data_dir):
    """
    Args:
        number_reference_dict: dictionary with keys as atom letters and values as the reference number batch to load
    """
    all_graphs = []
    for key, value in tqdm(number_reference_dict.items()):
        subdir_list = os.listdir(data_dir / key)
        #get the path of the batch with the reference number among the subdir_list
        for subdir in subdir_list:
            if value == subdir.split('_')[-1].split('.')[0] and '100000' in subdir:
                batch_path = data_dir / key / subdir
                print('loader the batch named {}'.format(batch_path))
                break

        graph_batch = load_compressed_batch(batch_path)
        all_graphs += graph_batch.to_data_list()
    
    #shuffling the list of graphs
    random.shuffle(all_graphs)

    return all_graphs

class ZincPreloadDataset(Dataset):
    def __init__(self, number_reference_dict, data_dir):
        self.data_list = load_all_and_merge(number_reference_dict, data_dir)
        self.encoding_size = self.data_list[0].x.size(1)
        print('Dataset encoded with size {}'.format(self.encoding_size))
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):

        return self.data_list[idx]