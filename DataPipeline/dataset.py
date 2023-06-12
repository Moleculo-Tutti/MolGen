import torch 
import random

import numpy as np

from torch.utils.data import Dataset
from torch_geometric.data import Batch

import zstandard as zstd
from io import BytesIO

from tqdm import tqdm
import os
from multiprocessing import Manager

import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')

from DataPipeline.preprocessing import get_subgraph_with_terminal_nodes_step

def add_node_feature_based_on_position(data):

    tensor_size = data.x.size(1)

    # Find the index of the 'current_atom' for each graph in the batch
    current_atom_indices = (data.x[:, tensor_size - 1] == 1).nonzero(as_tuple=True)[0]
    
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

class ZincSubgraphDatasetStep(Dataset):

    def __init__(self, data_path, GNN_type : str, feature_position : bool = False, scores_list : list = []):
        self.data_list = torch.load(data_path)
        self.encoding_size = self.data_list[0].x.size(1)
        self.edge_size = self.data_list[0].edge_attr.size(1)
        self.GNN_type = GNN_type
        self.feature_position = feature_position
        self.scores_list = scores_list
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
            num_steps = random.choice(range(1, 2*mol_size))
        if self.GNN_type >= 2:
            num_steps = random.choice(range(1, mol_size))
        subgraph, terminal_nodes, id_map = get_subgraph_with_terminal_nodes_step(preprocessed_graph, num_steps, impose_edges=self.impose_edges)

        if self.GNN_type == 1:

            subgraph.x[id_map[terminal_nodes[0]]][self.encoding_size - 1] = 1 #add a one on the current atom (terminal_nodes[0])

            #get the embedding of all the first element of terminal_nodes[1] and make them into a list to take the mean, if terminal_nodes[1] empty make torch.zeros(10)
            label_gnn1 = torch.zeros(self.encoding_size)
            neighbor_atom_list = [neighbor[1] for neighbor in terminal_nodes[1]]

            if len(neighbor_atom_list) != 0:
                label_gnn1 += torch.mean(torch.stack(neighbor_atom_list, dim=0), dim=0) #if there are some neighbors, we compute the average of their labels
            else:
                label_gnn1 += torch.tensor([0] * (self.encoding_size - 1) + [1]) #if there are no neighbors, we put a one on the last position to indicate a stop

            if self.feature_position:
                
                feature_position_tensor = torch.zeros(subgraph.x.size(0), 1)
                feature_position_tensor[0:id_map[terminal_nodes[0]]] = 1
                subgraph.x = torch.cat([subgraph.x, feature_position_tensor], dim=1)
            
            if self.scores_list != []:
                # Concat the scores to the node features
                for score_name in self.scores_list:
                    score_tensor = torch.tensor(preprocessed_graph[score_name], dtype=torch.float32)
                    # Duplicate the score tensor to match the number of nodes in the subgraph
                    score_tensor = score_tensor.repeat(subgraph.x.size(0), 1)
                    subgraph.x = torch.cat([subgraph.x, score_tensor], dim=-1)

            subgraph.y = label_gnn1
        
        if self.GNN_type == 2:
            
            subgraph.x[id_map[terminal_nodes[0]]][self.encoding_size - 1] = 1 #add a one on the current atom (terminal_nodes[0])

            id_chosen = np.random.randint(len(terminal_nodes[1])) # we sample a random neighbor
            subgraph.neighbor = terminal_nodes[1][id_chosen][1] # we add the neighbor sampled to the graph
            subgraph.edge_neighbor = terminal_nodes[1][id_chosen][2] # we add the edge corresponding to the neighbor sampled to the graph

            if self.feature_position:

                feature_position_tensor = torch.zeros(subgraph.x.size(0), 1)
                feature_position_tensor[0:id_map[terminal_nodes[0]]] = 1
                subgraph.neighbor = torch.cat([subgraph.neighbor, torch.zeros(1)], dim=0)
                subgraph.x = torch.cat([subgraph.x, feature_position_tensor], dim=1)

            if self.scores_list != []:
                # Concat the scores to the node features
                for score_name in self.scores_list:
                    score_tensor = torch.tensor(preprocessed_graph[score_name], dtype=torch.float32)
                    # Duplicate the score tensor to match the number of nodes in the subgraph
                    score_tensor = score_tensor.repeat(subgraph.x.size(0), 1)
                    subgraph.x = torch.cat([subgraph.x, score_tensor], dim=-1)
                    subgraph.neighbor = torch.cat([subgraph.neighbor, torch.zeros(score_tensor.size(1))], dim=0)
                    

        if self.GNN_type == 3:

            id_chosen = np.random.randint(len(terminal_nodes[1])) #we sample a random neighbor
            neighbor = terminal_nodes[1][id_chosen][1]

            # Put a one to indicate the last node generated before adding it to the graph
            neighbor[-1] = 1
            edge_neighbor_attr = terminal_nodes[1][id_chosen][2] # we get the attribute of the edge

            #add neighbor and edge_neighbor to the graph
            subgraph.x = torch.cat([subgraph.x, neighbor.unsqueeze(0)], dim=0)

            node1 = id_map[terminal_nodes[0]]
            node2 = len(subgraph.x) - 1

            add_edge_index = torch.tensor([[node1, node2], [node2, node1]], dtype=torch.long)
            subgraph.edge_index = torch.cat([subgraph.edge_index, add_edge_index], dim=1)
            # add double edge_attribute 
            subgraph.edge_attr = torch.cat([subgraph.edge_attr, edge_neighbor_attr.unsqueeze(0), edge_neighbor_attr.unsqueeze(0)], dim=0)


            node_features_label = torch.zeros(len(subgraph.x), self.edge_size) #there is no triple bond for closing the cycle

            # put ones in the last column of the node_features_label for the terminal node (put stop everywhere)
            node_features_label[:, -1] = 1

            if len(terminal_nodes[1][id_chosen][3]) != 0:
                for cycle_neighbor in terminal_nodes[1][id_chosen][3]:
                    node_features_label[id_map[cycle_neighbor[0]]][:self.edge_size - 1] = cycle_neighbor[1][:self.edge_size - 1]
                    node_features_label[id_map[cycle_neighbor[0]]][-1] = 0

            mask = torch.cat((torch.zeros(node1 + 1), torch.ones(len(subgraph.x) - node1 - 1)), dim=0).bool()
            mask[-1] = False

            if self.feature_position:
                #Concatenate the mask to the node_features_label
                opposite_mask = torch.logical_not(mask)
                opposite_mask[-1] = False
                opposite_mask[node1] = False
                # we add the opposite of the mask to the node features that correspond to the feature_position
                subgraph.x = torch.cat((subgraph.x, opposite_mask.unsqueeze(1)), dim=1) 

            if self.scores_list != []:
                # Concat the scores to the node features
                for score_name in self.scores_list:
                    score_tensor = torch.tensor(preprocessed_graph[score_name], dtype=torch.float32)
                    # Duplicate the score tensor to match the number of nodes in the subgraph
                    score_tensor = score_tensor.repeat(subgraph.x.size(0), 1)
                    subgraph.x = torch.cat([subgraph.x, score_tensor], dim=-1)
            

            subgraph.cycle_label = node_features_label
            subgraph.mask = mask
            subgraph.terminal_node_info = terminal_nodes


        return subgraph
    


class ZincSubgraphDatasetStep_mutlithread(Dataset):

    def __init__(self, data_path, GNN_type : str, feature_position : bool = False, scores_list : list = []):
        self.manager = Manager()
        self.data_list = self.manager.Array(torch.load(data_path))
        self.encoding_size = self.data_list[0].x.size(1)
        self.edge_size = self.data_list[0].edge_attr.size(1)
        self.GNN_type = GNN_type
        self.feature_position = feature_position
        self.scores_list = scores_list
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
            num_steps = random.choice(range(1, 2*mol_size))
        if self.GNN_type >= 2:
            num_steps = random.choice(range(1, mol_size))
        subgraph, terminal_nodes, id_map = get_subgraph_with_terminal_nodes_step(preprocessed_graph, num_steps, impose_edges=self.impose_edges)

        if self.GNN_type == 1:

            subgraph.x[id_map[terminal_nodes[0]]][self.encoding_size - 1] = 1 #add a one on the current atom (terminal_nodes[0])

            #get the embedding of all the first element of terminal_nodes[1] and make them into a list to take the mean, if terminal_nodes[1] empty make torch.zeros(10)
            label_gnn1 = torch.zeros(self.encoding_size)
            neighbor_atom_list = [neighbor[1] for neighbor in terminal_nodes[1]]

            if len(neighbor_atom_list) != 0:
                label_gnn1 += torch.mean(torch.stack(neighbor_atom_list, dim=0), dim=0) #if there are some neighbors, we compute the average of their labels
            else:
                label_gnn1 += torch.tensor([0] * (self.encoding_size - 1) + [1]) #if there are no neighbors, we put a one on the last position to indicate a stop

            if self.feature_position:
                
                feature_position_tensor = torch.zeros(subgraph.x.size(0), 1)
                feature_position_tensor[0:id_map[terminal_nodes[0]]] = 1
                subgraph.x = torch.cat([subgraph.x, feature_position_tensor], dim=1)
            
            if self.scores_list != []:
                # Concat the scores to the node features
                for score_name in self.scores_list:
                    score_tensor = torch.tensor(preprocessed_graph[score_name], dtype=torch.float32)
                    # Duplicate the score tensor to match the number of nodes in the subgraph
                    score_tensor = score_tensor.repeat(subgraph.x.size(0), 1)
                    subgraph.x = torch.cat([subgraph.x, score_tensor], dim=-1)

            subgraph.y = label_gnn1
        
        if self.GNN_type == 2:
            
            subgraph.x[id_map[terminal_nodes[0]]][self.encoding_size - 1] = 1 #add a one on the current atom (terminal_nodes[0])

            id_chosen = np.random.randint(len(terminal_nodes[1])) # we sample a random neighbor
            subgraph.neighbor = terminal_nodes[1][id_chosen][1] # we add the neighbor sampled to the graph
            subgraph.edge_neighbor = terminal_nodes[1][id_chosen][2] # we add the edge corresponding to the neighbor sampled to the graph

            if self.feature_position:

                feature_position_tensor = torch.zeros(subgraph.x.size(0), 1)
                feature_position_tensor[0:id_map[terminal_nodes[0]]] = 1
                subgraph.neighbor = torch.cat([subgraph.neighbor, torch.zeros(1)], dim=0)
                subgraph.x = torch.cat([subgraph.x, feature_position_tensor], dim=1)

            if self.scores_list != []:
                # Concat the scores to the node features
                for score_name in self.scores_list:
                    score_tensor = torch.tensor(preprocessed_graph[score_name], dtype=torch.float32)
                    # Duplicate the score tensor to match the number of nodes in the subgraph
                    score_tensor = score_tensor.repeat(subgraph.x.size(0), 1)
                    subgraph.x = torch.cat([subgraph.x, score_tensor], dim=-1)
                    subgraph.neighbor = torch.cat([subgraph.neighbor, torch.zeros(score_tensor.size(1))], dim=0)
                    

        if self.GNN_type == 3:

            id_chosen = np.random.randint(len(terminal_nodes[1])) #we sample a random neighbor
            neighbor = terminal_nodes[1][id_chosen][1]

            # Put a one to indicate the last node generated before adding it to the graph
            neighbor[-1] = 1
            edge_neighbor_attr = terminal_nodes[1][id_chosen][2] # we get the attribute of the edge

            #add neighbor and edge_neighbor to the graph
            subgraph.x = torch.cat([subgraph.x, neighbor.unsqueeze(0)], dim=0)

            node1 = id_map[terminal_nodes[0]]
            node2 = len(subgraph.x) - 1

            add_edge_index = torch.tensor([[node1, node2], [node2, node1]], dtype=torch.long)
            subgraph.edge_index = torch.cat([subgraph.edge_index, add_edge_index], dim=1)
            # add double edge_attribute 
            subgraph.edge_attr = torch.cat([subgraph.edge_attr, edge_neighbor_attr.unsqueeze(0), edge_neighbor_attr.unsqueeze(0)], dim=0)


            node_features_label = torch.zeros(len(subgraph.x), self.edge_size) #there is no triple bond for closing the cycle

            # put ones in the last column of the node_features_label for the terminal node (put stop everywhere)
            node_features_label[:, -1] = 1

            if len(terminal_nodes[1][id_chosen][3]) != 0:
                for cycle_neighbor in terminal_nodes[1][id_chosen][3]:
                    node_features_label[id_map[cycle_neighbor[0]]][:self.edge_size - 1] = cycle_neighbor[1][:self.edge_size - 1]
                    node_features_label[id_map[cycle_neighbor[0]]][-1] = 0

            mask = torch.cat((torch.zeros(node1 + 1), torch.ones(len(subgraph.x) - node1 - 1)), dim=0).bool()
            mask[-1] = False

            if self.feature_position:
                #Concatenate the mask to the node_features_label
                opposite_mask = torch.logical_not(mask)
                opposite_mask[-1] = False
                opposite_mask[node1] = False
                # we add the opposite of the mask to the node features that correspond to the feature_position
                subgraph.x = torch.cat((subgraph.x, opposite_mask.unsqueeze(1)), dim=1) 

            if self.scores_list != []:
                # Concat the scores to the node features
                for score_name in self.scores_list:
                    score_tensor = torch.tensor(preprocessed_graph[score_name], dtype=torch.float32)
                    # Duplicate the score tensor to match the number of nodes in the subgraph
                    score_tensor = score_tensor.repeat(subgraph.x.size(0), 1)
                    subgraph.x = torch.cat([subgraph.x, score_tensor], dim=-1)
            

            subgraph.cycle_label = node_features_label
            subgraph.mask = mask
            subgraph.terminal_node_info = terminal_nodes


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

def custom_collate_passive_add_feature_GNN2(batch):
    
    # add a column of zeros to the neighbor of the batch

    for item in batch:
        item.neighbor = torch.cat([item.neighbor, torch.zeros(1)], dim=0)
    
    sg_data_list = [item for item in batch]
    
    terminal_nodes_info_list = [item.edge_neighbor for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    terminal_nodes_info_tensor = torch.stack(terminal_nodes_info_list, dim=0)

    feature_sg_data_batch = add_node_feature_based_on_position(sg_data_batch)
    return feature_sg_data_batch, terminal_nodes_info_tensor


def custom_collate_GNN2(batch):
    sg_data_list = [item for item in batch]
    terminal_nodes_info_list = [item.edge_neighbor for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    terminal_nodes_info_tensor = torch.stack(terminal_nodes_info_list, dim=0)
    return sg_data_batch, terminal_nodes_info_tensor

def custom_collate_GNN3(batch):

    sg_data_list = [item for item in batch]
    cycle_label_list = [item.cycle_label for item in batch]
    mask_list = [item.mask for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    cycle_label_tensor = torch.cat(cycle_label_list, dim=0)
    mask_tensor = torch.cat(mask_list, dim=0)
    
    return sg_data_batch, cycle_label_tensor, mask_tensor



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