import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GCNConv
from torch.nn.functional import cross_entropy
from utils import anomaly_weight
from sklearn.metrics import roc_auc_score

class Label_loss(nn.Module):
    
    def __init__(self, h_feats, number_class, graph, anomaly_type="none") -> None:
        """graph is pyg graph. Use graph for a_weight and mask
        anomaly type: optional min or syn. none for the single mode
        """
        super().__init__()
        self.a_weight = anomaly_weight(graph)
        self.device = graph.x.device
        self.graph = graph
        self.type = anomaly_type
        self.linear = GCNConv(h_feats, number_class) if self.type == "min" else nn.Linear(h_feats, number_class)
        # self.linear = nn.Linear(h_feats, number_class)

    
    def forward(self, hiddle):
        logits = self.linear(hiddle, self.graph.edge_index) if self.type == "min" else self.linear(hiddle)
        train_loss = cross_entropy(logits[self.graph.train_mask], self.graph.y[self.graph.train_mask], weight=torch.tensor([1.0, self.a_weight],device=self.device))
        return train_loss

    def logit_for_tsne(self, hiddle):
        logits = self.linear(hiddle, self.graph.edge_index) if self.type == "min" else self.linear(hiddle)
        return logits

    def val_loss_auc(self, hiddle):
        logits = self.linear(hiddle, self.graph.edge_index) if self.type == "min" else self.linear(hiddle)
        val_loss = F.cross_entropy(logits[self.graph.val_mask], self.graph.y[self.graph.val_mask], weight=torch.tensor([1.0, self.a_weight],device=self.device))
        probs = logits.softmax(1)
        auc = roc_auc_score(self.graph.y[self.graph.val_mask].cpu().numpy(), probs[self.graph.val_mask][:,1].detach().cpu().numpy()) if self.graph.y.is_cuda else \
            roc_auc_score(self.graph.y[self.graph.val_mask].numpy(), probs[self.graph.val_mask][:,1].detach().numpy())
        # return logits to compute auc
        return val_loss, auc

    def test_auc(self, hiddle):
        logits = self.linear(hiddle, self.graph.edge_index) if self.type == "min" else self.linear(hiddle)
        probs = logits.softmax(1)
        auc = roc_auc_score(self.graph.y[self.graph.test_mask].cpu().numpy(), probs[self.graph.test_mask][:,1].detach().cpu().numpy()) if self.graph.y.is_cuda else \
            roc_auc_score(self.graph.y[self.graph.test_mask].numpy(), probs[self.graph.test_mask][:,1].detach().numpy())
        # return logits to compute auc
        return auc
        