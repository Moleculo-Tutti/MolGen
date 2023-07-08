import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.nn import GCNConv


class CustomMessagePassingLayerOld(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels):
        super(CustomMessagePassingLayerOld, self).__init__(aggr='add')
        self.lin = nn.Linear(2*in_channels + edge_channels, out_channels)
    
    def forward(self, x, edge_index, edge_attr):
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr):
        x = torch.cat([x_i, x_j, edge_attr], dim=1)
        return F.relu(self.lin(x))


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
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels, edge_channels, num_classes=10, use_dropout=True, use_batchnorm=True, size_info = True, max_size = 40):
        torch.manual_seed(12345)
        super(ModelWithEdgeFeatures, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.size_max = max_size

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
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1] + 1, mlp_hidden_channels)
        else:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1], mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        for message_passing_layer, batch_norm_layer in zip(self.message_passing_layers, self.batch_norm_layers):
            x = message_passing_layer(x, edge_index, edge_attr)
            if self.use_batchnorm:
                x = batch_norm_layer(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)

        # Aggregation function to obtain graph embedding
        x = global_add_pool(x, batch)
        
        if self.size_info:
            # Concatenate size of each graph of the batch 
            num_nodes_per_graph = torch.bincount(data.batch).view(-1, 1).float()
            # Normalize num_node 
            num_nodes_per_graph = num_nodes_per_graph / self.size_max
            x = torch.cat((x, num_nodes_per_graph), dim=1)

        # Two-layer MLP for classification
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x
    

class ModelWithNodeConcat(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels, edge_channels, encoding_size, num_classes=10, use_dropout=True, use_batchnorm=True, size_info = True, max_size = 40):
        torch.manual_seed(12345)
        super(ModelWithNodeConcat, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.encoding_size = encoding_size
        self.size_max = max_size

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
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        node_embeddings = []
        specified_nodes = torch.nonzero(x[:, self.encoding_size-1] == 1).squeeze() 
        for message_passing_layer, batch_norm_layer in zip(self.message_passing_layers, self.batch_norm_layers):
            x = message_passing_layer(x, edge_index, edge_attr)
            if self.use_batchnorm:
                x = batch_norm_layer(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
            #because the attention is the last on of encoding size
            node_embedding = x[specified_nodes]
            node_embeddings.append(node_embedding)
        # Aggregation function to obtain graph embedding
        x = global_add_pool(x, batch)
        
        if self.size_info:
            # Concatenate size of each graph of the batch 
            num_nodes_per_graph = torch.bincount(data.batch).view(-1, 1).float()
            # Normalize num_node 
            num_nodes_per_graph = num_nodes_per_graph / self.size_max

            x = torch.cat((x, num_nodes_per_graph), dim=1)

        node_embeddings = torch.cat(node_embeddings, dim=1)
        x = torch.cat((x, node_embeddings), dim=1)
        # Two-layer MLP for classification
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x
    


class ModelWithEdgeFeatures_conv(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels, edge_channels, num_classes=10, use_dropout=True, use_batchnorm=True, size_info = True, max_size = 40):
        super(ModelWithEdgeFeatures_conv, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm
        self.size_info = size_info
        self.size_max = max_size

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
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1] + 1, mlp_hidden_channels)
        else:
            self.fc1 = torch.nn.Linear(hidden_channels_list[-1], mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

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

            x = torch.sum((x_simple_bound, x_double_bound, x_triple_bound, x_identity), dim=1)

            if self.use_batchnorm:
                x = batch_norm_layer(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)

        # Aggregation function to obtain graph embedding
        x = global_add_pool(x, batch)
        
        if self.size_info:
            # Concatenate size of each graph of the batch 
            num_nodes_per_graph = torch.bincount(data.batch).view(-1, 1).float()
            # Normalize num_node 
            num_nodes_per_graph = num_nodes_per_graph / self.size_max
            x = torch.cat((x, num_nodes_per_graph), dim=1)

        # Two-layer MLP for classification
        x = F.relu(self.fc1(x))
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        x = self.fc2(x)

        return x