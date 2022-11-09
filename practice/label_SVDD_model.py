import torch
import numpy as np
import torch.nn as nn
# from torch import relu
# from torch_geometric.nn import GCNConv
from dgl.nn.pytorch.conv import GraphConv

class label_SVDD(nn.Module):

    def __init__(self, input_dim, hiddle_dim = 16, embedding_dim = 16,number_class = 2, hiddle_layer = 2, dropout = 0.2) -> None:
        super(label_SVDD, self).__init__()
        
        module = []
        for layer in range(hiddle_layer):
            if layer == 0:
                module.append(GraphConv(input_dim, hiddle_dim, allow_zero_in_degree=True))
            elif layer == hiddle_layer - 1:
                module.append(GraphConv(hiddle_dim, embedding_dim, allow_zero_in_degree=True))
            else:
                module.append(GraphConv(hiddle_dim, hiddle_dim, allow_zero_in_degree=True))
        self.conv_list = nn.Sequential(*module)
        self.dropout = nn.Dropout(dropout, self.training)
        self.relu = nn.ReLU(inplace=False)

        self.linear = nn.Linear(embedding_dim, number_class)

    def forward(self, graph, feature):
        for i, layer in enumerate(self.conv_list):
            if i != len(self.conv_list)-1:
                feature = layer(graph, feature)
                # feature = self.relu(feature)
                feature = self.dropout(feature)
            else:
                feature = layer(graph, feature)
        out = self.linear(feature)
        return feature, out

        