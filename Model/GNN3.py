import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv



class CustomMessagePassingLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels):
        super(CustomMessagePassingLayer, self).__init__(aggr='add')
        self.lin = nn.Linear(2*in_channels + edge_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        # Add self-loops
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        
        # Propagate and apply ReLU activation
        x = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return F.relu(x)

    def message(self, x_i, x_j, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=1)
        return self.lin(x)
    


class ModelWithEdgeFeatures(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, edge_channels, num_classes, use_dropout=True, use_batchnorm=True, size_info = False, max_size = 40):
        torch.manual_seed(12345)
        super(ModelWithEdgeFeatures, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.num_classes = num_classes
        self.max_size = max_size

        self.message_passing_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        prev_channels = in_channels
        for hidden_channels in hidden_channels_list:
            self.message_passing_layers.append(
                CustomMessagePassingLayer(prev_channels, hidden_channels, edge_channels)
            )
            if self.use_batchnorm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            prev_channels = hidden_channels


    def forward(self, data):
        x, edge_index, edge_attr, batch, mask = data.x, data.edge_index, data.edge_attr, data.batch, data.mask

        for message_passing_layer, batch_norm_layer in zip(self.message_passing_layers, self.batch_norm_layers):
            x = message_passing_layer(x, edge_index, edge_attr)
            if self.use_batchnorm  and message_passing_layer != self.message_passing_layers[-1]:
                x = batch_norm_layer(x)
            # Put a ReLU activation after each layer but not after the last one
            if message_passing_layer != self.message_passing_layers[-1]:
                x = F.relu(x)
            if self.use_dropout and  message_passing_layer != self.message_passing_layers[-1]:
                x = F.dropout(x, training=self.training)

        return x
    


class ModelWithgraph_embedding_modif(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels,edge_channels, num_classes, use_dropout=True, use_batchnorm=True, size_info = False, max_size = 40):
        torch.manual_seed(12345)
        super(ModelWithgraph_embedding_modif, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.num_classes = num_classes
        self.max_size = max_size

        self.message_passing_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        prev_channels = in_channels
        for hidden_channels in hidden_channels_list:
            self.message_passing_layers.append(
                CustomMessagePassingLayer(prev_channels, hidden_channels, edge_channels)
            )
            if self.use_batchnorm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            prev_channels = hidden_channels
        if self.size_info:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1]*2 + 1, mlp_hidden_channels)
        else:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1]*2, mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, self.num_classes)
        


    def forward(self, data):
        x, edge_index, edge_attr, batch, mask = data.x, data.edge_index, data.edge_attr, data.batch, data.mask

        for message_passing_layer, batch_norm_layer in zip(self.message_passing_layers, self.batch_norm_layers):
            x = message_passing_layer(x, edge_index, edge_attr)
            if self.use_batchnorm:
                x = batch_norm_layer(x)

            # Put a ReLU activation after each layer but not after the last one
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
        
        graph_embedd = global_add_pool(x, batch)


        if self.size_info:
            # Concatenate size of each graph of the batch 
            num_nodes_per_graph = torch.bincount(data.batch).view(-1, 1).float()
            # Normalize num_node 
            num_nodes_per_graph = num_nodes_per_graph / self.max_size

            graph_embedd = torch.cat((graph_embedd, num_nodes_per_graph), dim=1)
        
        graph_embed_rep = graph_embedd[batch]  # Répéter l'encodage pour chaque nœud du graphe

        # Two-layer MLP for classification

        #maybe there is an error with the deice where it has been projected

        new_input = torch.cat((x,graph_embed_rep), dim = 1)
        node_out = F.relu(self.fc1(new_input))
        if self.use_dropout:
            node_out = F.dropout(node_out, training=self.training)
        out = self.fc2(node_out)


        return out 
    


    
class ModelWithgraph_embedding_close_or_not_with_node_embedding(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels,edge_channels, num_classes, use_dropout=True, use_batchnorm=True, size_info = False, max_size = 40, encoding_size = 13):
        torch.manual_seed(12345)
        super(ModelWithgraph_embedding_close_or_not_with_node_embedding, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.num_classes = num_classes
        self.max_size = max_size
        self.encoding_size = encoding_size

        self.message_passing_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        prev_channels = in_channels
        for hidden_channels in hidden_channels_list:
            self.message_passing_layers.append(
                CustomMessagePassingLayer(prev_channels, hidden_channels, edge_channels)
            )
            if self.use_batchnorm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            prev_channels = hidden_channels

        dim_concat = 0
        for dim in hidden_channels_list :
            dim_concat += dim
        if self.size_info:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1] + 1 + dim_concat, mlp_hidden_channels)
        else:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1]+ dim_concat, mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, self.num_classes)
        


    def forward(self, data):
        x, edge_index, edge_attr, batch, mask = data.x, data.edge_index, data.edge_attr, data.batch, data.mask

        node_embeddings = []
        specified_nodes = torch.nonzero(x[:,  self.encoding_size-1] == 1).squeeze()
        for message_passing_layer, batch_norm_layer in zip(self.message_passing_layers, self.batch_norm_layers):
            x = message_passing_layer(x, edge_index, edge_attr)
            if self.use_batchnorm:
                x = batch_norm_layer(x)

            # Put a ReLU activation after each layer but not after the last one
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
            node_embedding = x[specified_nodes]
            node_embeddings.append(node_embedding)
        
        x = global_add_pool(x, batch)
        
        if self.size_info:
            # Concatenate size of each graph of the batch 
            num_nodes_per_graph = torch.bincount(data.batch).view(-1, 1).float()
            # Normalize num_node 
            num_nodes_per_graph = num_nodes_per_graph / self.max_size

            x = torch.cat((x, num_nodes_per_graph), dim=1)

        node_embeddings = torch.cat(node_embeddings, dim=1)
        x = torch.cat((x, node_embeddings), dim=1)
        # Two-layer MLP for classification
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x        



   
class ModelWithgraph_embedding_close_or_not_without_node_embedding(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels,edge_channels, num_classes, use_dropout=True, use_batchnorm=True, size_info = False, max_size = 40, encoding_size = 13):
        torch.manual_seed(12345)
        super(ModelWithgraph_embedding_close_or_not_without_node_embedding, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.num_classes = num_classes
        self.max_size = max_size
        self.encoding_size = encoding_size

        self.message_passing_layers = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        prev_channels = in_channels
        for hidden_channels in hidden_channels_list:
            self.message_passing_layers.append(
                CustomMessagePassingLayer(prev_channels, hidden_channels, edge_channels)
            )
            if self.use_batchnorm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            prev_channels = hidden_channels


        if self.size_info:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1] + 1 , mlp_hidden_channels)
        else:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1], mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, self.num_classes)
        


    def forward(self, data):
        x, edge_index, edge_attr, batch, mask = data.x, data.edge_index, data.edge_attr, data.batch, data.mask

        for message_passing_layer, batch_norm_layer in zip(self.message_passing_layers, self.batch_norm_layers):
            x = message_passing_layer(x, edge_index, edge_attr)
            if self.use_batchnorm:
                x = batch_norm_layer(x)

            # Put a ReLU activation after each layer but not after the last one
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
        
        x = global_add_pool(x, batch)
        
        if self.size_info:
            # Concatenate size of each graph of the batch 
            num_nodes_per_graph = torch.bincount(data.batch).view(-1, 1).float()
            # Normalize num_node 
            num_nodes_per_graph = num_nodes_per_graph / self.max_size

            x = torch.cat((x, num_nodes_per_graph), dim=1)

        # Two-layer MLP for classification
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x        




   
class ModelWithgraph_embedding_close_or_not_gcnconv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels,edge_channels, num_classes, use_dropout=True, use_batchnorm=True, size_info = False, max_size = 40, encoding_size = 13):
        torch.manual_seed(12345)
        super(ModelWithgraph_embedding_close_or_not_gcnconv, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.num_classes = num_classes
        self.max_size = max_size
        self.encoding_size = encoding_size

        
        self.message_passing_layers_simple_bound = torch.nn.ModuleList()
        self.message_passing_layers_double_bound = torch.nn.ModuleList()
        self.message_passing_layers_triple_bound = torch.nn.ModuleList()
        self.message_passing_layers_identity = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        prev_channels = in_channels

        for hidden_channels in hidden_channels_list:
            #create a layer that concatenate 4 GCNconv at each layer, one for each kind of edge

            self.message_passing_layers_simple_bound.append(GCNConv(prev_channels, hidden_channels, bias=True, normalize=True, add_self_loops=False))
            self.message_passing_layers_double_bound.append(GCNConv(prev_channels, hidden_channels, bias=True, normalize=True, add_self_loops=False))
            self.message_passing_layers_triple_bound.append(GCNConv(prev_channels, hidden_channels, bias=True, normalize=True, add_self_loops=False))
            self.message_passing_layers_identity.append(torch.nn.Linear(prev_channels, hidden_channels)) #will replace the add_self_loops=False pour le reste
            if self.use_batchnorm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            prev_channels = hidden_channels


        if self.size_info:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1] + 1 , mlp_hidden_channels)
        else:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1], mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, self.num_classes)
        


    def forward(self, data):
        x, edge_index, edge_attr, batch, mask = data.x, data.edge_index, data.edge_attr, data.batch, data.mask

        for i in range(len(self.message_passing_layers_simple_bound)):
            message_passing_layer_simple_bound = self.message_passing_layers_simple_bound[i]
            message_passing_layer_double_bound = self.message_passing_layers_double_bound[i]
            message_passing_layer_triple_bound = self.message_passing_layers_triple_bound[i]
            message_passing_layer_identity = self.message_passing_layers_identity[i]
            batch_norm_layer = self.batch_norm_layers[i]
            mask_simple_bound = edge_attr[:, 0] == 1
            x_simple_bound = message_passing_layer_simple_bound(x, edge_index[:, mask_simple_bound])
            mask_double_bound = edge_attr[:, 1] == 1
            x_double_bound = message_passing_layer_double_bound(x, edge_index[:, mask_double_bound])
            mask_triple_bound = edge_attr[:, 2] == 1
            x_triple_bound = message_passing_layer_triple_bound(x, edge_index[:, mask_triple_bound])
            x_identity = message_passing_layer_identity(x)
            #do the relu before aggregating with add
            x_simple_bound = F.relu(x_simple_bound)
            x_double_bound = F.relu(x_double_bound)
            x_triple_bound = F.relu(x_triple_bound)
            x_identity = F.relu(x_identity)

            x = torch.sum(torch.stack([x_simple_bound, x_double_bound, x_triple_bound, x_identity], dim=1), dim=1)
            if self.use_batchnorm:
                x = batch_norm_layer(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
                
        x = global_add_pool(x, batch)
        
        if self.size_info:
            # Concatenate size of each graph of the batch 
            num_nodes_per_graph = torch.bincount(data.batch).view(-1, 1).float()
            # Normalize num_node 
            num_nodes_per_graph = num_nodes_per_graph / self.max_size

            x = torch.cat((x, num_nodes_per_graph), dim=1)

        # Two-layer MLP for classification
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x        




class ModelWithgraph_embedding_modif_gcnconv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels,edge_channels, num_classes, use_dropout=True, use_batchnorm=True, size_info = False, max_size = 40):
        torch.manual_seed(12345)
        super(ModelWithgraph_embedding_modif_gcnconv, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.num_classes = num_classes
        self.max_size = max_size

        self.message_passing_layers_simple_bound = torch.nn.ModuleList()
        self.message_passing_layers_double_bound = torch.nn.ModuleList()
        self.message_passing_layers_triple_bound = torch.nn.ModuleList()
        self.message_passing_layers_identity = torch.nn.ModuleList()
        self.batch_norm_layers = torch.nn.ModuleList()
        prev_channels = in_channels

        for hidden_channels in hidden_channels_list:
            #create a layer that concatenate 4 GCNconv at each layer, one for each kind of edge

            self.message_passing_layers_simple_bound.append(GCNConv(prev_channels, hidden_channels, bias=True, normalize=True, add_self_loops=False))
            self.message_passing_layers_double_bound.append(GCNConv(prev_channels, hidden_channels, bias=True, normalize=True, add_self_loops=False))
            self.message_passing_layers_triple_bound.append(GCNConv(prev_channels, hidden_channels, bias=True, normalize=True, add_self_loops=False))
            self.message_passing_layers_identity.append(torch.nn.Linear(prev_channels, hidden_channels)) #will replace the add_self_loops=False pour le reste
            if self.use_batchnorm:
                self.batch_norm_layers.append(torch.nn.BatchNorm1d(hidden_channels))
            prev_channels = hidden_channels

        if self.size_info:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1]*2 + 1, mlp_hidden_channels)
        else:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1]*2, mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, self.num_classes)
        


    def forward(self, data):
        x, edge_index, edge_attr, batch, mask = data.x, data.edge_index, data.edge_attr, data.batch, data.mask

        for i in range(len(self.message_passing_layers_simple_bound)):
            message_passing_layer_simple_bound = self.message_passing_layers_simple_bound[i]
            message_passing_layer_double_bound = self.message_passing_layers_double_bound[i]
            message_passing_layer_triple_bound = self.message_passing_layers_triple_bound[i]
            message_passing_layer_identity = self.message_passing_layers_identity[i]
            batch_norm_layer = self.batch_norm_layers[i]
            mask_simple_bound = edge_attr[:, 0] == 1
            x_simple_bound = message_passing_layer_simple_bound(x, edge_index[:, mask_simple_bound])
            mask_double_bound = edge_attr[:, 1] == 1
            x_double_bound = message_passing_layer_double_bound(x, edge_index[:, mask_double_bound])
            mask_triple_bound = edge_attr[:, 2] == 1
            x_triple_bound = message_passing_layer_triple_bound(x, edge_index[:, mask_triple_bound])
            x_identity = message_passing_layer_identity(x)
            #do the relu before aggregating with add
            x_simple_bound = F.relu(x_simple_bound)
            x_double_bound = F.relu(x_double_bound)
            x_triple_bound = F.relu(x_triple_bound)
            x_identity = F.relu(x_identity)

            x = torch.sum(torch.stack([x_simple_bound, x_double_bound, x_triple_bound, x_identity], dim=1), dim=1)
            if self.use_batchnorm:
                x = batch_norm_layer(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)

        graph_embedd = global_add_pool(x, batch)


        if self.size_info:
            # Concatenate size of each graph of the batch 
            num_nodes_per_graph = torch.bincount(data.batch).view(-1, 1).float()
            # Normalize num_node 
            num_nodes_per_graph = num_nodes_per_graph / self.max_size

            graph_embedd = torch.cat((graph_embedd, num_nodes_per_graph), dim=1)
        
        graph_embed_rep = graph_embedd[batch]  # Répéter l'encodage pour chaque nœud du graphe

        # Two-layer MLP for classification

        #maybe there is an error with the deice where it has been projected

        new_input = torch.cat((x,graph_embed_rep), dim = 1)
        node_out = F.relu(self.fc1(new_input))
        if self.use_dropout:
            node_out = F.dropout(node_out, training=self.training)
        out = self.fc2(node_out)


        return out 