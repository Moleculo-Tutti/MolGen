import torch
import torch_geometric
import numpy as np
import torch.nn.functional as F
import random
import os, sys
from dataclasses import dataclass

from torch_geometric.data import Batch, Data


cwd = os.getcwd()
parent_dir = os.path.dirname(cwd)
sys.path.append(parent_dir)

from DataPipeline.preprocessing import node_encoder, tensor_to_smiles
from models import Model_GNNs




def return_current_nodes_batched(current_node_tensor, graph_batch):
    # Get the unique graph ids
    batch = graph_batch.batch
    unique_graph_ids = torch.unique(batch)
    # Create a 2D mask that shows where each graph's nodes are located in batch
    mask = batch[None, :] == unique_graph_ids[:, None]
    # Compute the cumulative sum of the mask along the second dimension
    cumulative_mask = mask.cumsum(dim=1)
    # Now, for each graph, the nodes are numbered from 1 to N (or 0 to N-1 if we subtract 1)
    node_indices_per_graph = cumulative_mask - 1
    # But we only want the indices of certain nodes (specified by current_node_tensor)
    # So we create a mask that is True where the node index equals the current node index for the graph
    current_node_mask = node_indices_per_graph == current_node_tensor[:, None]
    # The result is the indices in batch where current_node_mask is True
    # Find the arg of the first True in each row
    result = torch.argmax(current_node_mask.int(), dim=1)
    return result

def set_last_nodes(batch, last_prediction_size, encoding_size):
    # Reset the current node column
    batch.x[:, encoding_size - 1] = 0
    # Set the last nodes to 1
    batch.x[batch.x.shape[0] - last_prediction_size:, encoding_size - 1] = 1
    return batch

def increment_feature_position(batch, current_nodes_batched, stop_mask, encoding_size):

    stopped_current_nodes = current_nodes_batched[stop_mask]

    batch.x[stopped_current_nodes, encoding_size] = 1

    return batch




def create_mask(batch_graph, current_nodes_tensor : torch.tensor, last_prediction_size, encoding_size):
    # Create a mask for the current nodes tensor    
    feature_postion = batch_graph.x[:, encoding_size]
    mask = torch.logical_not(feature_postion.bool())
    # Set the last nodes to False
    mask[batch_graph.x.shape[0] - last_prediction_size:] = False
    # Set the current nodes to False
    mask[current_nodes_tensor] = False

    return mask



def select_node_batch(prediction, batch_data, edge_size, mask):

    # Sum on the first dimensions of each vector
    sum_on_first_dims = prediction[:, :edge_size - 1].sum(dim=1)
    unique_graph_ids = torch.unique(batch_data)
    expanded_sum = sum_on_first_dims[None, :].expand(unique_graph_ids.shape[0], -1)
    
    # Create a 2D mask that shows where each graph's nodes are located in batch
    mask_location = batch_data[None, :] == unique_graph_ids[:, None]
    # Apply the mask to each row of mask_location
    mask = mask_location * mask[None, :]
    # Apply mask to the sum tensor, setting masked values to -inf
    masked_sum = expanded_sum.masked_fill(~mask,float ('-inf'))
    #Find the max value in each row
    max_indices = torch.argmax(masked_sum, dim=1)
    # Create a count mask that counts how many True values there are in each row
    count_mask = mask.sum(dim=1)

    # Replace indices where there were no True values in the mask with -1 (or any value you want)
    minus_one = torch.tensor([-1], device=max_indices.device)
    max_indices = max_indices.where(count_mask > 0, minus_one)
    # Sample using the tensor using the multinomial function
    sampled_indices = prediction.multinomial(num_samples=1).squeeze()
    # Replace -1 values in max_indices with the corresponding sampled_indices
    final_indices = torch.where(max_indices != -1, sampled_indices[max_indices], prediction.size(1) - 1)

    return final_indices, max_indices


def select_option_batch(choice_input_softmax, sigmoid_input):
    # Sample with a multinomial distribution  
    choice_sampled = torch.multinomial(choice_input_softmax, num_samples=1)

    # Get the sigmoid values for the chosen nodes
    chosen_sigmoid_values = sigmoid_input[choice_sampled.squeeze()]
    # Generate random numbers for comparison
    random_numbers = torch.rand(chosen_sigmoid_values.shape, device=chosen_sigmoid_values.device)
    # Binary decision based on the sigmoid value: If the random number is less than the sigmoid value, choose 1
    decision = (random_numbers < chosen_sigmoid_values).to(torch.long)

    return choice_sampled, decision

def compute_softmax_GNN3(choice_input, batch_data, mask):
    unique_graph_ids = torch.unique(batch_data)
    mask_location = batch_data[None, :] == unique_graph_ids[:, None]

    # Multiply the softmax and the mask location
    choice_masked = choice_input[None, :] * mask_location
    choice_masked = choice_masked * mask[None, :]
    choice_masked = choice_masked.masked_fill(torch.logical_or(~mask_location, ~mask[None, :]), float('-1e38'))

    # Apply softmax to the masked tensor
    choice_softmax = F.softmax(choice_masked, dim=1)
    return choice_softmax
    

def add_nodes_and_edges_batch(batch, new_nodes, new_edges, current_nodes_batched, mask):
    device = batch.x.device
    # Add one zero to the end of the new_nodes
    new_nodes = torch.cat([new_nodes, torch.zeros(new_nodes.size(0), 1, device=new_nodes.device)], dim=1)
    # Add new nodes to the node attributes, considering the mask
    new_nodes_masked = new_nodes[mask]

    batch.x = torch.cat([batch.x, new_nodes_masked], dim=0)

    # Create new edges between the current nodes and the new nodes in a bidirectional manner
    num_new_nodes = new_nodes_masked.shape[0]
    new_edge_indices = torch.stack([current_nodes_batched[mask], torch.arange(batch.num_nodes - num_new_nodes, batch.num_nodes, device=device)], dim=0)
    new_edge_indices = torch.cat([new_edge_indices, new_edge_indices.flip(0)], dim=1)  # Making the edges bidirectional

    # Adding the new edges to the edge_index
    batch.edge_index = torch.cat([batch.edge_index, new_edge_indices], dim=1)

    # Create new edge attributes and mask out the entries where mask is False
    new_edge_attrs = new_edges[mask].repeat(2, 1)

    # Adding the new edge attributes to the edge attributes
    batch.edge_attr = torch.cat([batch.edge_attr, new_edge_attrs], dim=0)

    # Update the batch.batch tensor
    new_batch_entries = torch.arange(new_nodes.shape[0], device=device)
    batch.batch = torch.cat([batch.batch, new_batch_entries[mask]])
    #check_valence(batch, torch.arange(batch.num_nodes - num_new_nodes, batch.num_nodes, device=device).tolist())
    #check_valence(batch, current_nodes_batched[mask].tolist())
    # Return the updated batch
    return batch

def add_edges_and_attributes(batch, edges_predicted, indices, mask, stopping_mask):

    num_new_edges = edges_predicted.shape[0]
    mask_edge_predicted = edges_predicted[mask]
    mask_indices = indices[mask]
    
    num_new_nodes = torch.sum(stopping_mask)
    last_indices = torch.arange(batch.num_nodes - num_new_nodes, batch.num_nodes, device=batch.x.device)

    last_nodes_batch = torch.full(stopping_mask.shape, -1, device=batch.x.device)

    last_nodes_batch[stopping_mask] = last_indices
    new_edges_indices = torch.stack([mask_indices, last_nodes_batch[mask]], dim=0)
    new_edges_indices = torch.cat([new_edges_indices, new_edges_indices.flip(0)], dim=1)  # Making the edges bidirectional

    # Adding the new edges to the edge_index
    batch.edge_index = torch.cat([batch.edge_index, new_edges_indices], dim=1)

    # Create new edge attributes and mask out the entries where mask is False
    new_edge_attrs = mask_edge_predicted.repeat(2, 1)

    # Adding the new edge attributes to the edge attributes
    batch.edge_attr = torch.cat([batch.edge_attr, new_edge_attrs], dim=0)

    #check_valence(batch, mask_indices.tolist())
    #check_valence(batch, last_indices.tolist())

    return batch

def extract_all_graphs(batch):
    all_graphs = []
    nb_graphs = batch.batch.max().item() + 1

    for i in range(nb_graphs):
        # Create a mask of booleans
        mask = batch.batch == i
        
        # Extract all the node features that correspond to the i-th graph
        subgraph_x = batch.x[mask]
        # Create a mapping of the corresponding indices from the big graph to the individual graph

        indices_mapping = {j.item(): k for k, j in enumerate(torch.where(mask)[0])}
        mapping_func = np.vectorize(indices_mapping.get)

        # Extract all the edges that correspond to the i-th graph
        edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]

        if edge_mask.sum() == 0:
            subgraph_edge_index = torch.tensor([], dtype=torch.long)
        else:
            subgraph_edge_index = torch.tensor(mapping_func(batch.edge_index[:, edge_mask].cpu().numpy()), dtype=torch.long)

        # Extract all the edge features that correspond to the i-th graph

        
        if batch.edge_attr is not None:
            subgraph_edge_attr = batch.edge_attr[edge_mask]
        else:
            subgraph_edge_attr = None

        # Construct the subgraph
        subgraph = Data(x=subgraph_x, edge_index=subgraph_edge_index, edge_attr=subgraph_edge_attr)
        # Append the subgraph to the list
        all_graphs.append(subgraph)

    return all_graphs


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

def create_torch_graph_from_one_atom_batch(atoms, edge_size, encoding_option='reduced') -> list:
    graphs = []
    for atom in atoms:
        num_atom = int(atom)
        atom_attribute = node_encoder(num_atom, encoding_option=encoding_option)
        # Create graph
        # Increase the size of atom_attribute by one 
        atom_attribute = torch.cat((atom_attribute, torch.zeros(1)), dim=0)

        graph = torch_geometric.data.Data(x=atom_attribute.view(1, -1), edge_index=torch.empty((2, 0), dtype=torch.long), edge_attr=torch.empty((0, edge_size)))
        graphs.append(graph)
            
    
    return Batch.from_data_list(graphs)


class Sampling_Path_Batch():
    def __init__(self, GNNs_Models_q, GNNs_Models_a, GNNs_Models_pi, features, lambdas, device, batch_size, args):
        """
        Initialize the sampling path batch
        input:
        GNNs_Models_q: list of the GNNs models for the q function
        GNNs_Models_a: list of the GNNs models for the a function
        GNNs_Models_pi: list of the GNNs models for the pi function
        features: list of the features of the molecules
        lambdas: list of the lambdas of the molecules
        batch_size: size of the batch
        device: device to use for the computation

        return: None
        """
        self.GNNs_Models_q = GNNs_Models_q
        self.GNNs_Models_a = GNNs_Models_a
        self.GNNs_Models_pi = GNNs_Models_pi
        self.features = features
        self.lambdas = lambdas.to(device)
        self.batch_size = batch_size
        self.device = device

        self.encoding_size = args.encoding_size
        self.encoding_option = args.encoding_option
        self.edge_size = args.edge_size
        self.compute_lambdas = args.compute_lambdas

        batch_mol_graph = create_torch_graph_from_one_atom_batch(sample_first_atom_batch(batch_size = batch_size, encoding = self.encoding_option), edge_size=self.edge_size, encoding_option=self.encoding_option)
        self.batch_mol_graph = batch_mol_graph.to(device) # Encoded in size 14 for the feature position
        self.queues = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.node_counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        self.finished_mask = torch.zeros(batch_size, dtype=torch.bool, device=device)
        # Float 64 for the q_value, a_value and pi_value
        # Requires grad for the pi_value

        self.a_value = torch.ones(batch_size, dtype=torch.float64, device=device)
        self.q_value = torch.ones(batch_size, dtype=torch.float64, device=device)
        if args.compute_lambdas:
            self.pi_value = torch.ones(batch_size, dtype=torch.float64, device=device)
        else:
            self.pi_value = torch.ones(batch_size, dtype=torch.float64, device=device, requires_grad=True)

    def one_step(self):

        with torch.no_grad():

            current_nodes = self.queues.clone()
            
            # Get the molecules that are finished which means that the number in the queue is superior to the number of nodes in the molecule
            old_finished_mask = self.finished_mask
            self.finished_mask = current_nodes > self.node_counts

            # test if no molecule has changed from finished to unfinished
            if torch.any(old_finished_mask & ~self.finished_mask):
                print('error')
                raise ValueError('error')
            # If all the mask is True, then all the molecules are finished
            if torch.all(self.finished_mask):
                return 
            """
            Prepare the GNN 1 BATCH
            """
            current_nodes[self.finished_mask] = current_nodes[self.finished_mask] - 1
            current_nodes_batched = return_current_nodes_batched(current_nodes, self.batch_mol_graph) #reindex the nodes to match the batch size
            
            self.batch_mol_graph.x[: , self.encoding_size - 1] = 0
            # Do you -1 for the current nodes that are finished

            self.batch_mol_graph.x[current_nodes_batched, self.encoding_size - 1] = 1

            # Add score features if needed



            q_predictions = self.GNNs_Models_q.GNN1_model(self.batch_mol_graph)
            a_predictions = self.GNNs_Models_a.GNN1_model(self.batch_mol_graph)
            pi_predictions = self.GNNs_Models_pi.GNN1_model(self.batch_mol_graph)

            
            # Apply softmax to prediction
            q_softmax_predictions = F.softmax(q_predictions, dim=1)
            a_softmax_predictions = F.softmax(a_predictions, dim=1)
            pi_softmax_predictions = F.softmax(pi_predictions, dim=1)

            # Sample next node from prediction
            predicted_nodes = torch.multinomial(q_softmax_predictions, num_samples=1)

            # Get the q, a and pi values for the predicted_node 
            q_value = q_softmax_predictions[torch.arange(self.batch_size), predicted_nodes.flatten()]
            a_value = a_softmax_predictions[torch.arange(self.batch_size), predicted_nodes.flatten()]
            pi_value = pi_softmax_predictions[torch.arange(self.batch_size), predicted_nodes.flatten()]

            # Actualize the q, a, pi and use the finished mask to only actualize the values of the molecules that are not finished
            self.q_value = self.q_value * torch.max(self.finished_mask.float(), q_value)
            self.a_value = self.a_value * torch.max(self.finished_mask.float(), a_value)
            self.pi_value = self.pi_value * torch.max(self.finished_mask.float(), pi_value)

            # Create a mask for determining which graphs should continue to GNN2
            mask_gnn2 = (predicted_nodes != self.encoding_size - 1).flatten()

            # Handle the stopping condition (where predicted_node is encoding_size - 1)
            stop_mask = (predicted_nodes == self.encoding_size - 1).flatten()
                
            # Increment the node count for graphs that haven't stopped and that are not finished
            self.node_counts = self.node_counts + torch.logical_and(~stop_mask, ~self.finished_mask).long()

            # Increment the queue for graphs that have been stopped and that are not finished
            
            self.queues = self.queues + torch.logical_and(stop_mask, ~self.finished_mask).long()

            # Increment the feature position for graphs that have been stopped

            self.batch_mol_graph = increment_feature_position(self.batch_mol_graph, current_nodes_batched, stop_mask, self.encoding_size)

            # Encode next node for the entire batch
            encoded_predicted_nodes = torch.zeros(q_predictions.size(), device=self.device, dtype=torch.float)
            encoded_predicted_nodes.scatter_(1, predicted_nodes, 1)

            #GNN2 
                    
            # add zeros to the neighbor because of the feature position
            encoded_predicted_nodes = torch.cat([encoded_predicted_nodes, torch.zeros(self.batch_size, 1).to(encoded_predicted_nodes.device)], dim=1)

            self.batch_mol_graph.neighbor = encoded_predicted_nodes

            
            q_predictions_2 = self.GNNs_Models_q.GNN2_model(self.batch_mol_graph)
            a_predictions_2 = self.GNNs_Models_a.GNN2_model(self.batch_mol_graph)
            pi_predictions_2 = self.GNNs_Models_pi.GNN2_model(self.batch_mol_graph)

            # Apply softmax to prediction
            q_softmax_predictions_2 = F.softmax(q_predictions_2, dim=1)
            a_softmax_predictions_2 = F.softmax(a_predictions_2, dim=1)
            pi_softmax_predictions_2 = F.softmax(pi_predictions_2, dim=1)

            predicted_edges = torch.multinomial(q_softmax_predictions_2, num_samples=1)

            # Get the q, a and pi values for the predicted_node
            q_value = q_softmax_predictions_2[torch.arange(self.batch_size), predicted_edges.flatten()]
            a_value = a_softmax_predictions_2[torch.arange(self.batch_size), predicted_edges.flatten()]
            pi_value = pi_softmax_predictions_2[torch.arange(self.batch_size), predicted_edges.flatten()]



            # Actualize the q, a, pi and use the mask to only actualize the graphs that have not stopped by getting the max of the stop mask, the finis
            self.q_value = self.q_value * torch.max(torch.logical_or(stop_mask, self.finished_mask).float(), q_value)
            self.a_value = self.a_value * torch.max(torch.logical_or(stop_mask, self.finished_mask).float(), a_value)
            self.pi_value = self.pi_value * torch.max(torch.logical_or(stop_mask, self.finished_mask).float(), pi_value)

        
            encoded_predicted_edges = torch.zeros_like(q_predictions_2, device=self.device, dtype=torch.float)
            encoded_predicted_edges.scatter_(1, predicted_edges, 1)
            
            # Create a new node that is going to be added to the graph for each batch
            new_nodes = torch.zeros(self.batch_size, self.encoding_size, device=self.device, dtype=torch.float)
            new_nodes.scatter_(1, predicted_nodes, 1)

            #GNN3

            # Add the node and the edge to the graph
            self.batch_mol_graph = add_nodes_and_edges_batch(self.batch_mol_graph, new_nodes, encoded_predicted_edges, current_nodes_batched, torch.logical_and(mask_gnn2, ~self.finished_mask))
            

            self.batch_mol_graph = set_last_nodes(self.batch_mol_graph, torch.sum(torch.logical_and(mask_gnn2, ~self.finished_mask), dim=0), self.encoding_size)       
            mask = create_mask(self.batch_mol_graph, current_nodes_tensor = current_nodes_batched, last_prediction_size=torch.sum(torch.logical_and(mask_gnn2, ~self.finished_mask), dim=0), encoding_size=self.encoding_size)
            self.batch_mol_graph.mask = mask
        

            q_prediction_closing = self.GNNs_Models_q.GNN3_1_model(self.batch_mol_graph)
            a_prediction_closing = self.GNNs_Models_a.GNN3_1_model(self.batch_mol_graph)
            pi_prediction_closing = self.GNNs_Models_pi.GNN3_1_model(self.batch_mol_graph)
            
            q_sigmoid_prediction_3 = torch.sigmoid(q_prediction_closing).flatten()
            a_sigmoid_prediction_3 = torch.sigmoid(a_prediction_closing).flatten()
            pi_sigmoid_prediction_3 = torch.sigmoid(pi_prediction_closing).flatten()

            random_number = torch.rand(q_sigmoid_prediction_3.shape, device=self.device)

            closing_mask = (random_number < q_sigmoid_prediction_3).flatten() 

            unique_graph_ids = torch.unique(self.batch_mol_graph.batch)
            mask_location = self.batch_mol_graph.batch[None, :] == unique_graph_ids[:, None]

            mask_location =  mask_location * mask[None, :]
            # Create a mask if all the lines of mask_location are false then put false in the closing mask
            mask_location = torch.sum(mask_location, dim=1)


            # Get the q, a and pi values for the predicted_node. If the closing mask is true, the value is sigmoid, otherwise it is 1-sigmoid
            q_values = torch.where(closing_mask, q_sigmoid_prediction_3, 1 - q_sigmoid_prediction_3)
            a_values = torch.where(closing_mask, a_sigmoid_prediction_3, 1 - a_sigmoid_prediction_3)
            pi_values = torch.where(closing_mask, pi_sigmoid_prediction_3, 1 - pi_sigmoid_prediction_3)


            # Actualize the q, a, pi and use the mask to only actualize the graphs that have not stopped by getting the max of the stop mask, the finis
            self.q_value = self.q_value * torch.max(torch.logical_or(stop_mask, self.finished_mask).float(), q_values)
            self.a_value = self.a_value * torch.max(torch.logical_or(stop_mask, self.finished_mask).float(), a_values)
            self.pi_value = self.pi_value * torch.max(torch.logical_or(stop_mask, self.finished_mask).float(), pi_values)
            

            q_chosing_prediction = self.GNNs_Models_q.GNN3_2_model(self.batch_mol_graph)
            a_chosing_prediction = self.GNNs_Models_a.GNN3_2_model(self.batch_mol_graph)
            pi_chosing_prediction = self.GNNs_Models_pi.GNN3_2_model(self.batch_mol_graph)
            
            
            q_choice_input = q_chosing_prediction[:, 1]
            q_sigmoid_input = q_chosing_prediction[:, 0]
            a_choice_input = a_chosing_prediction[:, 1]
            a_sigmoid_input = a_chosing_prediction[:, 0]
            pi_choice_input = pi_chosing_prediction[:, 1]
            pi_sigmoid_input = pi_chosing_prediction[:, 0]

            # Apply sigmoid 

            q_sigmoid_input = torch.sigmoid(q_sigmoid_input)
            a_sigmoid_input = torch.sigmoid(a_sigmoid_input)
            pi_sigmoid_input = torch.sigmoid(pi_sigmoid_input)

            # Compute softmax 

            q_softmax_input = compute_softmax_GNN3(q_choice_input, self.batch_mol_graph.batch, mask)
            a_softmax_input = compute_softmax_GNN3(a_choice_input, self.batch_mol_graph.batch, mask)
            pi_softmax_input = compute_softmax_GNN3(pi_choice_input, self.batch_mol_graph.batch, mask)


            choosen_indexes, decision = select_option_batch(q_softmax_input, q_sigmoid_input)

            choosen_indexes = choosen_indexes.squeeze()
            
            # Get the q, a and pi values for the predicted_node. Based on the choosen index.

            # Extract the choosen input from sigmoid

            q_extracted_sigmoid = q_sigmoid_input[choosen_indexes].squeeze()
            a_extracted_sigmoid = a_sigmoid_input[choosen_indexes].squeeze()
            pi_extracted_sigmoid = pi_sigmoid_input[choosen_indexes].squeeze()

            decision = decision.squeeze()

            graph_indices = torch.arange(q_softmax_input.shape[0], device=self.device)
            
            q_values = (q_softmax_input[graph_indices, choosen_indexes].squeeze()) * torch.where(decision == 1, q_extracted_sigmoid, 1 - q_extracted_sigmoid)
            a_values = (a_softmax_input[graph_indices, choosen_indexes].squeeze()) * torch.where(decision == 1, a_extracted_sigmoid, 1 - a_extracted_sigmoid)
            pi_values = (pi_softmax_input[graph_indices, choosen_indexes].squeeze()) * torch.where(decision == 1, pi_extracted_sigmoid, 1 - pi_extracted_sigmoid)
            
            # Actualize the q, a, pi and use the mask to only actualize the graphs that have not stopped by getting the max of the stop mask, the finis
            total_mask = torch.logical_or(torch.logical_or(~closing_mask, stop_mask), self.finished_mask)

            self.q_value = self.q_value * torch.max(total_mask.float(), q_values)
            self.a_value = self.a_value * torch.max(total_mask.float(), a_values)
            self.pi_value = self.pi_value * torch.max(total_mask.float(), pi_values)


            encoded_edges_predicted = torch.zeros((decision.shape[0], self.edge_size), device=self.device, dtype=torch.float)

            encoded_edges_predicted.scatter_(1, decision.unsqueeze(1), 1)

            total_mask = torch.logical_and(torch.logical_and(closing_mask, mask_gnn2), ~self.finished_mask)

            self.mol_graphs_list = add_edges_and_attributes(self.batch_mol_graph, encoded_edges_predicted, choosen_indexes.flatten(), total_mask, torch.logical_and(mask_gnn2, ~self.finished_mask))


    def full_generation(self):
        max_iter = 150
        i = 0
        while torch.all(self.finished_mask) == False:
            if i > max_iter:
                break
            self.one_step()
            i += 1
    
    def convert_to_smiles(self):
        graph_list = extract_all_graphs(self.mol_graphs_list)
        smiles_list = []
        for g in graph_list:
            smiles_list.append(tensor_to_smiles(g.x, g.edge_index, g.edge_attr, edge_mapping='kekulized', encoding_type='charged'))
        
        self.smiles_list = smiles_list

    def compute_features(self):
        all_features_values = torch.zeros(self.batch_size, len(self.features), device=self.device)
        for i, fn in enumerate(self.features.values()):
            all_features_values[:, i] = fn(self.smiles_list)
        self.all_features_values = all_features_values
                
    def get_exponents(self):
        exponents = torch.exp(torch.matmul(self.all_features_values, self.lambdas))
        return exponents