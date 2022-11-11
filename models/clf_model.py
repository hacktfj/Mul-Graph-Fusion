import sys
sys.path.append("..")
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import anomaly_weight
from sklearn.metrics import roc_auc_score

class Classifier(nn.Module):
    
    def __init__(self, h_feats, number_class, graph) -> None:
        """graph is pyg graph. graph here to get the train_mask, val_mask and test_mask
        """
        super().__init__()
        self.a_weight = anomaly_weight(graph)
        self.device = graph.x.device
        self.linear = nn.Linear(h_feats, number_class)
        self.graph = graph
    
    def train_loss(self, hiddle):
        logits = self.linear(hiddle)
        train_loss = F.cross_entropy(logits[self.graph.train_mask], self.graph.y[self.graph.train_mask], weight=torch.tensor([1.0, self.a_weight],device=self.device))
        # return logits to compute auc
        return train_loss

    def forward(self, hiddle):
        return self.train_loss(hiddle)

    
    def val_loss_auc(self, hiddle):
        logits = self.linear(hiddle)
        val_loss = F.cross_entropy(logits[self.graph.val_mask], self.graph.y[self.graph.val_mask], weight=torch.tensor([1.0, self.a_weight],device=self.device))
        probs = logits.softmax(1)
        auc = roc_auc_score(self.graph.y[self.graph.val_mask].cpu().numpy(), probs[self.graph.val_mask][:,1].detach().cpu().numpy()) if self.graph.y.is_cuda else \
            roc_auc_score(self.graph.y[self.graph.val_mask].numpy(), probs[self.graph.val_mask][:,1].detach().numpy())
        # return logits to compute auc
        return val_loss, auc

    def test_auc(self, hiddle):
        logits = self.linear(hiddle)
        probs = logits.softmax(1)
        auc = roc_auc_score(self.graph.y[self.graph.test_mask].cpu().numpy(), probs[self.graph.test_mask][:,1].detach().cpu().numpy()) if self.graph.y.is_cuda else \
            roc_auc_score(self.graph.y[self.graph.test_mask].numpy(), probs[self.graph.test_mask][:,1].detach().numpy())
        # return logits to compute auc
        return auc
    


    