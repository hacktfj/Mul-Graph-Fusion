import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv

class GCN(nn.Module):

    def __init__(self, in_dim, hid_dim, out_dim, graph):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels=in_dim, out_channels=hid_dim)
        self.conv2 = GCNConv(in_channels=hid_dim, out_channels=out_dim)
        self.edge_index = graph.edge_index

    def forward(self, x):
        x = self.conv1(x, self.edge_index)
        h = F.dropout(F.relu(x), 0.3, self.training)
        x = self.conv2(h, self.edge_index)
        return x, h