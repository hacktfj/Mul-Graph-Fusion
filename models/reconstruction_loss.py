import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj

class Reconstruction_loss(nn.Module):
    def __init__(self, h_feats, graph):
        super(Reconstruction_loss, self).__init__()

        # split the number of layers for the encoder and decoders
        self.graph = graph
        self.attr_decoder = GCNConv(in_channels=h_feats, out_channels=graph.x.shape[1])
        self.struct_decoder = GCNConv(in_channels=h_feats, out_channels=h_feats)
        self.edge_index = graph.edge_index
        self.s = to_dense_adj(self.edge_index)[0]
        self.alpha = torch.std(self.s).detach() / (torch.std(graph.x).detach() + torch.std(self.s).detach())

    def forward(self, h):
        # decode feature matrix
        x_ = self.attr_decoder(h, self.edge_index)
        # print (x_.shape)
        # decode adjacency matrix
        h_ = self.struct_decoder(h, self.edge_index)
        # print (h_.shape)
        s_ = torch.matmul(h_, h_.T)
        # s_ = h_ @ h_.T
        # return reconstructed matrices
        score = self.reco_loss_func(self.graph.x, x_, self.s, s_)
        loss = torch.mean(score)
        return loss

    def reco_loss_func(self, x, x_, s, s_):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(s - s_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = self.alpha * attribute_errors \
                + (1 - self.alpha) * structure_errors
        return score