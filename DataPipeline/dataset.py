import torch 
import random

from torch.utils.data import Dataset
from torch_geometric.data import Batch

import zstandard as zstd
from io import BytesIO

from tqdm import tqdm
import os


from DataPipeline.preprocessing import get_subgraph_with_terminal_nodes



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
        
        id_chosen = np.random.randint(len(terminal_nodes[1]))
        subgraph.neighbor = terminal_nodes[1][id_chosen][1]

        subgraph.edge_neighbor = terminal_nodes[1][id_chosen][2]

        return subgraph
    

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