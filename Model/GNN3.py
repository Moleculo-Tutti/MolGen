import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops


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
    def __init__(self, in_channels, hidden_channels_list, edge_channels, use_dropout=True, use_batchnorm=True):
        torch.manual_seed(12345)
        super(ModelWithEdgeFeatures, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

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
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
        
        return x
    

class ModelWithhraph_embedding also(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, edge_channels, use_dropout=True, use_batchnorm=True):
        torch.manual_seed(12345)
        super(ModelWithEdgeFeatures, self).__init__()

        self.use_dropout = use_dropout
        self.use_batchnorm = use_batchnorm

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
            if self.use_batchnorm:
                x = batch_norm_layer(x)
            # Put a ReLU activation after each layer but not after the last one
            if message_passing_layer != self.message_passing_layers[-1]:
                x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)
        
        return x
    



    

