import sys
sys.path.append("..") # Adds higher directory to python modules path.
import torch
import numpy as np
import torch.nn as nn
from torch.nn import init
from torch.optim import Adam
import torch.nn.functional as F
import dgl.function as fn
from utils import *
from dataset import pyg_to_dgl
from early_stop import EarlyStopping
from models.BWGNN_model import BWGNN_em
from models.GAT_model import GAT
from models.GCN_model import GCN
from models.GIN_model import GIN

import warnings
warnings.filterwarnings("ignore")

model_list = ["lof", "oc-svm", "hbos", "bwgnn", "gin", "gat", "gcn", "dominate", "dci"]
class Baseline(nn.Module):

    def __init__(self, model_name, in_feats, h_feats, num_classes, graph, lr = 5e-3, epochs=50, verbose=1):
        """graph: pyg_graph, if dgl_graph is needed, just change here. 
        model type: detail in list of model_list
        verbose: little or many verbose
        """
        super().__init__()
        self.model = None
        self.optimizer = None
        self.a_weight = anomaly_weight(graph)
        self.model_name = model_name
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.graph = graph
        self.lr = lr
        self.device = graph.x.device
        self.epochs = epochs
        self.verbose = verbose
        if model_name == "lof":
            pass
        elif model_name == "oc-svm":
            pass
        elif model_name == "hbos":
            pass
        elif model_name == "bwgnn":
            dgl_graph = pyg_to_dgl(graph)
            self.model = BWGNN_em(in_feats, h_feats, num_classes, dgl_graph)
            self.optimizer = Adam(self.model.parameters(), lr = self.lr)
        elif model_name == "gin":
            self.model = GIN(in_feats, h_feats, num_classes, self.graph)
            self.optimizer = Adam(self.model.parameters(), lr = self.lr)
        elif model_name == "gat":
            self.model = GAT(in_feats, h_feats, num_classes, self.graph)
            self.optimizer = Adam(self.model.parameters(), lr = self.lr)
        elif model_name == "gcn":
            self.model = GCN(in_feats, h_feats, num_classes, self.graph)
            self.optimizer = Adam(self.model.parameters(), lr = self.lr)
        elif model_name == "dominate":
            pass
        elif model_name == "dci":
            pass
    
    def fit(self):
        if self.model_name == "lof":
            pass
        elif self.model_name == "oc-svm":
            pass
        elif self.model_name == "hbos":
            pass
        elif self.model_name == "bwgnn":
            best_auc = self.train(self.model, self.optimizer, self.graph, epochs=self.epochs)
            auc = self.test_auc(self.model, self.graph)
            if self.verbose:
                print (f"Test auc: {auc}; Val best auc: {best_auc}")
            return auc, best_auc
        elif self.model_name == "gin":
            best_auc = self.train(self.model, self.optimizer, self.graph, epochs=self.epochs)
            auc = self.test_auc(self.model, self.graph)
            if self.verbose:
                print (f"Test auc: {auc}; Val best auc: {best_auc}")
            return auc, best_auc
        elif self.model_name == "gat":
            best_auc = self.train(self.model, self.optimizer, self.graph, epochs=self.epochs)
            auc = self.test_auc(self.model, self.graph)  
            if self.verbose:
                print (f"Test auc: {auc}; Val best auc: {best_auc}")
            return auc, best_auc
        elif self.model_name == "gcn":
            best_auc = self.train(self.model, self.optimizer, self.graph, epochs=self.epochs)
            auc = self.test_auc(self.model, self.graph)
            if self.verbose:
                print (f"Test auc: {auc}; Val best auc: {best_auc}")
            return auc, best_auc
        elif self.model_name == "dominate":
            pass

    def train(self, model, optimizer, data, epochs, patience=20):
        best_val_auc = 0
        early_stop = EarlyStopping(patience=patience)
        for epoch in range(epochs):
            model.train()
            logits, _ = model(data.x)
            train_loss = cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=torch.tensor([1.0, self.a_weight],device=self.device))
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            model.eval()
            logits, _ = model(data.x)
            val_loss = cross_entropy(logits[data.val_mask], data.y[data.val_mask], weight=torch.tensor([1.0, self.a_weight],device=self.device))
            probs = logits.softmax(1)
            auc = roc_auc_score(data.y[data.val_mask].cpu().numpy(), probs[data.val_mask][:,1].detach().cpu().numpy()) if data.y.is_cuda else \
                roc_auc_score(data.y[data.val_mask].numpy(), probs[data.val_mask][:,1].detach().numpy())
            if auc >= best_val_auc:
                best_val_auc = auc
            early_stop(val_loss, model)
            if early_stop.early_stop == True:
                print ("Early stopping")
                break
            if self.verbose:
                print (f"Epoch {epoch+1}/{epochs}: val_loss: {val_loss}, val_auc: {auc}")
        print (f"Best val auc: {best_val_auc}")
        return best_val_auc
        

    def test_auc(self, model, data):
        model.eval()
        logits, _ = model(data.x)
        probs = logits.softmax(1)
        auc = roc_auc_score(data.y[data.test_mask].cpu().numpy(), probs[data.test_mask][:,1].detach().cpu().numpy()) if data.y.is_cuda else \
                roc_auc_score(data.y[data.test_mask].numpy(), probs[data.test_mask][:,1].detach().numpy())
        print (f"Final Test Auc: {auc}")
        return auc
    
        