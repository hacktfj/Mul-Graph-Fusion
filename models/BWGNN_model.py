import sympy
import scipy
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import dgl.function as fn

import warnings
warnings.filterwarnings("ignore")

class PolyConv(nn.Module):
    def __init__(self,
                 in_feats, 
                 out_feats,
                 theta,
                 activation=F.leaky_relu,
                 lin=False,
                 bias=False):
        super(PolyConv, self).__init__()
        self._theta = theta
        self._k = len(self._theta)
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.activation = activation
        self.linear = nn.Linear(in_feats, out_feats, bias)
        self.lin = lin
        
        # self.reset_parameters()
        # self.linear2 = nn.Linear(out_feats, out_feats, bias)

    def reset_parameters(self):
        if self.linear.weight is not None:
            init.xavier_uniform_(self.linear.weight)
        if self.linear.bias is not None:
            init.zeros_(self.linear.bias)

    def forward(self, graph, feat):
        def unnLaplacian(feat, D_invsqrt, graph):
            """ Operation D^-1/2 A D^-1/2 * Feat
                However, inverse computation for saving time!!!
                which means no need matric computation and only simple mul solved."""
            # feat_temp = feat * D_invsqrt
            # up the speed but memory no way
            graph.ndata['h'] = feat * D_invsqrt
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
            # feat_temp = spmm(to_dense_adj(graph.edge_index).reshape(node_num,-1), feat_temp)
            return feat - graph.ndata.pop('h') * D_invsqrt
        
        with graph.local_scope():
            D_invsqrt = torch.pow(graph.in_degrees().float().clamp(
                min=1), -0.5).unsqueeze(-1).to(feat.device)
            # D_invsqrt = torch.pow(degree(graph.edge_index[0], graph.num_nodes).float().clamp(
            #     min=1), -0.5).unsqueeze(-1).to(feat.device)
            h = self._theta[0]*feat
            for k in range(1, self._k):
                # out = MessagePassing().propagate(edge_index=edge_index,x=feat.to(device),edge_weight = edge_weight)
                # transform GCN laplacian to symmetric normalized laplacian.
                feat = unnLaplacian(feat, D_invsqrt, graph)
                h += self._theta[k]*feat
        if self.lin:
            h = self.linear(h)
            h = self.activation(h)
        return h


def calculate_theta2(d):
    thetas = []
    x = sympy.symbols('x')
    for i in range(d+1):
        f = sympy.poly((x/2) ** i * (1 - x/2) ** (d-i) / (scipy.special.beta(i+1, d+1-i)))
        coeff = f.all_coeffs()
        inv_coeff = []
        for i in range(d+1):
            inv_coeff.append(float(coeff[d-i]))
        thetas.append(inv_coeff)
    return thetas


class BWGNN(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False):
        super(BWGNN, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0])
        for conv in self.conv:
            # 改变 GCN 的聚合方式，设计自己的带通滤波器
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h_embedding = self.act(h)
        h = self.linear4(h_embedding)
        return h
        # return h, h_embedding

class BWGNN_em(nn.Module):
    def __init__(self, in_feats, h_feats, num_classes, graph, d=2, batch=False):
        super(BWGNN_em, self).__init__()
        self.g = graph
        self.thetas = calculate_theta2(d=d)
        self.conv = []
        for i in range(len(self.thetas)):
            if not batch:
                self.conv.append(PolyConv(h_feats, h_feats, self.thetas[i], lin=False))
        self.linear = nn.Linear(in_feats, h_feats)
        self.linear2 = nn.Linear(h_feats, h_feats)
        self.linear3 = nn.Linear(h_feats*len(self.conv), h_feats)
        self.linear4 = nn.Linear(h_feats, num_classes)
        self.act = nn.ReLU()
        self.d = d

    def forward(self, in_feat):
        h = self.linear(in_feat)
        h = self.act(h)
        h = self.linear2(h)
        h = self.act(h)
        h_final = torch.zeros([len(in_feat), 0]).to(in_feat.device)
        for conv in self.conv:
            # 改变 GCN 的聚合方式，设计自己的带通滤波器
            h0 = conv(self.g, h)
            h_final = torch.cat([h_final, h0], -1)
            # print(h_final.shape)
        h = self.linear3(h_final)
        h_embedding = self.act(h)
        h = self.linear4(h_embedding)
        # return h
        return h, h_embedding