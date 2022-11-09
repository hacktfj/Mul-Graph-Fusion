import torch
import torch.nn as nn
from .readout import AvgReadout
from .discriminator import Discriminator
# import sys
# sys.path.append("models/")

class DCI_loss(nn.Module):
    def __init__(self, hidden_dim, device):
        super(DCI_loss, self).__init__()
        self.device = device
        self.read = AvgReadout()
        self.sigm = nn.Sigmoid()
        self.disc = Discriminator(hidden_dim)

    def forward(self, h_1, h_2, msk, samp_bias1, samp_bias2, cluster_info, cluster_num):
        loss = 0
        batch_size = 1
        criterion = nn.BCEWithLogitsLoss()
        for i in range(cluster_num):
            node_idx = cluster_info[i]

            h_1_block = torch.unsqueeze(h_1[node_idx], 0)
            c_block = self.read(h_1_block, msk)
            c_block = self.sigm(c_block)
            h_2_block = torch.unsqueeze(h_2[node_idx], 0)

            lbl_1 = torch.ones(batch_size, len(node_idx))
            lbl_2 = torch.zeros(batch_size, len(node_idx))
            lbl = torch.cat((lbl_1, lbl_2), 1).to(self.device)

            ret = self.disc(c_block, h_1_block, h_2_block, samp_bias1, samp_bias2)
            loss_tmp = criterion(ret, lbl)
            loss += loss_tmp

        return loss / cluster_num