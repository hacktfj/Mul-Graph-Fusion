import torch
import torch.nn as nn
import torch.nn.functional as F
from pygod.models.basic_nn import GCN

class Reconstruction_loss(nn.Module):
    def __init__(self,
                in_dim,
                hid_dim,
                out_dim,
                decoder_layers,
                dropout,
                act,graph):
        super(Reconstruction_loss, self).__init__()

        # split the number of layers for the encoder and decoders
        self.attr_decoder = GCN(in_channels=in_dim,
                                hidden_channels=hid_dim,
                                num_layers=decoder_layers,
                                out_channels=out_dim,
                                dropout=dropout,
                                act=act)

        self.struct_decoder = GCN(in_channels=in_dim,
                                hidden_channels=hid_dim,
                                num_layers=decoder_layers,
                                out_channels=out_dim,
                                dropout=dropout,
                                act=act)
        self.edge_index = graph.edge_index

    def forward(self, h):
        # decode feature matrix
        x_ = self.attr_decoder(h, self.edge_index)
        # print (x_.shape)
        # decode adjacency matrix
        h_ = self.struct_decoder(h, self.edge_index)
        # print (h_.shape)
        s_ = h_ @ h_.T
        # return reconstructed matrices
        return x_, s_, h

def reco_loss_func(x, x_, s, s_):
    # attribute reconstruction loss
    diff_attribute = torch.pow(x - x_, 2)
    attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

    # structure reconstruction loss
    diff_structure = torch.pow(s - s_, 2)
    structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

    score = alpha * attribute_errors \
            + (1 - alpha) * structure_errors
    return score