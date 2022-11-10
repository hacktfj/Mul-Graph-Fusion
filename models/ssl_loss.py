import torch
import torch.nn as nn
# import sys
# sys.path.append("models/")

class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = torch.unsqueeze(c, 1)
        c_x = c_x.expand_as(h_pl)

        sc_1 = torch.squeeze(self.f_k(h_pl, c_x), 2)
        sc_2 = torch.squeeze(self.f_k(h_mi, c_x), 2)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits

# Applies an average on seq, of shape (batch, nodes, features)
# While taking into account the masking of msk
class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk):
        if msk is None:
            return torch.mean(seq, 1)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 1) / torch.sum(msk)

class SSL_loss(nn.Module):
    def __init__(self, hidden_dim, device):
        super(SSL_loss, self).__init__()
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