import random
from MolGen.DataPipeline.preprocessing import process_encode_graph, get_subgraph_with_terminal_nodes_step


def sample_random_subgraph_ZINC(pd_dataframe, start_size):
    indice = random.choice(pd_dataframe.index)
    smiles_str = pd_dataframe.loc[indice, 'smiles']

    torch_graph = process_encode_graph(smiles_str)
    subgraph_data, terminal_node_info, id_map = get_subgraph_with_terminal_nodes_step(torch_graph, start_size)

    return subgraph_data, terminal_node_info, id_map

