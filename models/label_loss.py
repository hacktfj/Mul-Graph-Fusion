import sys
sys.path.append("..")
import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy
from utils import anomaly_weight

class Label_loss(nn.Module):
    
    def __init__(self, h_feats, number_class, graph) -> None:
        """graph is pyg graph. Use graph for a_weight and mask
        """
        super().__init__()
        self.a_weight = anomaly_weight(graph)
        self.device = graph.x.device
        self.graph = graph
        self.linear = nn.Linear(h_feats, number_class)
    
    def forward(self, hiddle):
        logits = self.linear(hiddle)
        train_loss = cross_entropy(logits[self.graph.train_mask], self.graph.y[self.graph.train_mask], weight=torch.tensor([1.0, self.a_weight],device=self.device))
        return train_loss
        