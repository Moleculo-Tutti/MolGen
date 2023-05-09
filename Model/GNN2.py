import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_add_pool
from torch_geometric.utils import add_self_loops
from GNN1 import CustomMessagePassingLayer


class ModelWithEdgeFeatures(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels_list, mlp_hidden_channels, edge_channels, num_classes=10, use_dropout=True, use_batchnorm=True):
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

        self.fc1 = torch.nn.Linear(hidden_channels_list[-1]+num_classes, mlp_hidden_channels)
        self.fc2 = torch.nn.Linear(mlp_hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch, neighbor = data.x, data.edge_index, data.edge_attr, data.batch, data.neighbor

        for message_passing_layer, batch_norm_layer in zip(self.message_passing_layers, self.batch_norm_layers):
            x = message_passing_layer(x, edge_index, edge_attr)
            if self.use_batchnorm:
                x = batch_norm_layer(x)
            x = F.relu(x)
            if self.use_dropout:
                x = F.dropout(x, training=self.training)

        # Aggregation function to obtain graph embedding
        x = global_add_pool(x, batch)
        
        out =  torch.cat([x, neighbor], dim=1)
        # Two-layer MLP for classification
        out = F.relu(self.fc1(out))
        if self.use_dropout:
            out = F.dropout(out, training=self.training)
        out = self.fc2(out)

        return F.softmax(out, dim=1)