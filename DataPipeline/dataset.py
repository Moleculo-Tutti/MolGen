import torch 
import random

from torch.utils.data import Dataset
from torch_geometric.data import Batch


from DataPipeline.preprocessing import get_subgraph_with_terminal_nodes



class ZincSubgraphDataset(Dataset):

    def __init__(self, data_path):
        self.data_list = torch.load(data_path)


    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        preprocessed_graph = self.data_list[idx]
        
        mol_size = len(preprocessed_graph.x)
        num_atoms = random.choice(range(3, mol_size + 1))
        subgraph, terminal_nodes, id_map = get_subgraph_with_terminal_nodes(preprocessed_graph, num_atoms)

        subgraph.x[id_map[terminal_nodes[0]]][9] = 1

        #get the embedding of all the first element of terminal_nodes[1] and make them into a list to take the mean, if terminal_nodes[1] empty make torch.zeros(10)
        label_gnn1 = torch.zeros(10)
        neighbor_atom_list = [neighbor[1] for neighbor in terminal_nodes[1]]

        if len(neighbor_atom_list) != 0:
            label_gnn1 += torch.mean(torch.stack(neighbor_atom_list, dim=0), dim=0)
        else:
            label_gnn1 += torch.tensor([0,0,0,0,0,0,0,0,0,1])

        subgraph.y = label_gnn1

        return subgraph
    

def custom_collate(batch):
    sg_data_list = [item for item in batch]
    terminal_nodes_info_list = [item.y for item in batch]

    sg_data_batch = Batch.from_data_list(sg_data_list)
    terminal_nodes_info_tensor = torch.stack(terminal_nodes_info_list, dim=0)
    return sg_data_batch, terminal_nodes_info_tensor