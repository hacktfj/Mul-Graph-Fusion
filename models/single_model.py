"""Single Model for Two Stages Learning.
"""
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
from sklearn.cluster import KMeans
from early_stop import EarlyStopping
from models.BWGNN_model import BWGNN_em
from models.GAT_model import GAT
from models.GCN_model import GCN
from models.GIN_model import GIN
from models.label_loss import Label_loss
from models.reconstruction_loss import Reconstruction_loss
from models.ssl_loss import SSL_loss
from models.clf_model import Classifier

import warnings
warnings.filterwarnings("ignore")

model_list = ["bwgnn", "gin", "gat", "gcn"] # four methods
loss_oriented_list = ["label_oriented", "reconstruction_oriented", "ssl_oriented"]
class Single_model(nn.Module):

    def __init__(self, model_name, loss_oriented, in_feats, h_feats, num_classes, graph, lr = 5e-3, epochs_fit=50, epochs_fine=50, verbose=1) -> None:
        """model name for training and loss_oriented decide which task to learn on the specify model.
        model1,optimizer1 is None for label and reconstruction oriented
        """
        super().__init__()
        self.model = None
        self.model1 = None
        self.optimizer = None
        self.optimizer1 = None
        self.loss = None
        self.loss_optimizer = None
        self.clf = Classifier(h_feats, num_classes, graph)
        self.clf_optimizer = Adam(self.clf.parameters(), lr = lr)

        self.a_weight = anomaly_weight(graph)
        self.model_name = model_name
        self.loss_oriented = loss_oriented
        self.in_feats = in_feats
        self.h_feats = h_feats
        self.num_classes = num_classes
        self.graph = graph
        self.lr = lr
        self.device = graph.x.device
        self.epochs_fit = epochs_fit
        self.epochs_fine = epochs_fine
        self.verbose = verbose
        if model_name == "bwgnn":
            # lr = 1e-2
            dgl_graph = pyg_to_dgl(graph).to(device) if self.device == torch.device("cuda") else pyg_to_dgl(graph)
            self.model = BWGNN_em(in_feats, h_feats, num_classes, dgl_graph)
            # self.optimizer1 = Adam(self.model1.parameters(), lr = lr) if loss_oriented == "ssl_oriented" else None
            # self.optimizer = Adam(self.model.parameters(), lr = lr)
            self.model1 = BWGNN_em(in_feats, h_feats, num_classes, dgl_graph) if loss_oriented == "ssl_oriented" else None
        elif model_name == "gin":
            self.model = GIN(in_feats, h_feats, num_classes, self.graph)
            self.model1 = GIN(in_feats, h_feats, num_classes, self.graph) if loss_oriented == "ssl_oriented" else None
        elif model_name == "gat":
            self.model = GAT(in_feats, h_feats, num_classes, self.graph)
            self.model1 = GAT(in_feats, h_feats, num_classes, self.graph) if loss_oriented == "ssl_oriented" else None
        elif model_name == "gcn":
            self.model = GCN(in_feats, h_feats, num_classes, self.graph)
            self.model1 = GCN(in_feats, h_feats, num_classes, self.graph) if loss_oriented == "ssl_oriented" else None
        self.optimizer = Adam(self.model.parameters(), lr = self.lr)
        self.optimizer1 = Adam(self.model1.parameters(), lr = self.lr) if loss_oriented == "ssl_oriented" else None
        
        if loss_oriented == "label_oriented":
            self.loss = Label_loss(self.h_feats, self.num_classes, self.graph)
        elif loss_oriented == "reconstruction_oriented":
            self.loss = Reconstruction_loss(self.h_feats, self.graph)
        elif loss_oriented == "ssl_oriented":
            self.loss = SSL_loss(self.h_feats, self.device)
        self.loss_optimizer = Adam(self.loss.parameters(), lr = self.lr)
    
    def fit(self, patience=20, recluster_interval = 10):
        """patience for all loss function.
        recluster for ssl_oriented
        """
        if self.loss_oriented == "label_oriented" or self.loss_oriented == "reconstruction_oriented":
            early_stop = EarlyStopping(patience=patience)
            for epoch in range(self.epochs_fit):
                self.model.train()
                self.loss.train()
                _, hiddle = self.model(self.graph.x)

                train_loss = self.loss(hiddle)
                self.optimizer.zero_grad()
                self.loss_optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.loss_optimizer.step()

                early_stop(train_loss, self.model)
                if early_stop.early_stop == True:
                    print ("Early stopping")
                    break
                if self.verbose:
                    print (f"Stage one, Model name: {self.model_name}, loss oriented: {self.loss_oriented}; Epoch {epoch}/{self.epochs_fit}, Training loss: {train_loss}")
        elif self.loss_oriented == "ssl_oriented":
            kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(self.graph.x.to(torch.device("cpu")))
            ss_label = kmeans.labels_
            cluster_info = [list(np.where(ss_label==i)[0]) for i in range(self.num_classes)]
            idx = np.random.permutation(self.graph.x.shape[0])
            shuf_feats = self.graph.x[idx, :]
            early_stop = EarlyStopping(patience=patience)

            for epoch in range(self.epochs_fit):
                self.model.train()
                self.model1.train()
                self.loss.train()
                _, hiddle1 = self.model(self.graph.x)
                _, hiddle2 = self.model1(shuf_feats)
                train_loss = self.loss(hiddle1, hiddle2, None, None, None, cluster_info, self.num_classes)
                
                self.optimizer.zero_grad()
                self.optimizer1.zero_grad()
                self.loss_optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.optimizer1.step()
                self.loss_optimizer.step()

                # re-clustering
                if epoch % recluster_interval == 0:
                    self.model.eval()
                    _, hiddle1 = self.model(self.graph.x)
                    kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(hiddle1.detach().cpu().numpy())
                    ss_label = kmeans.labels_
                    cluster_info = [list(np.where(ss_label==i)[0]) for i in range(self.num_classes)]
                
                # record
                early_stop(train_loss, self.model)
                if early_stop.early_stop == True:
                    print ("Early stopping")
                    break
                if self.verbose:
                    print (f"Stage one, Model name: {self.model_name}, loss oriented: {self.loss_oriented}; Epoch {epoch+1}/{self.epochs_fit}, Training loss: {train_loss}")

    def fine_tuning(self, patience=20):
        if self.loss_oriented == "label_oriented" or self.loss_oriented == "reconstruction_oriented" or self.loss_oriented == "ssl_oriented":
            early_stop = EarlyStopping(patience=patience)
            best_val_auc = 0
            for epoch in range(self.epochs_fine):
                self.model.train()
                self.clf.train()
                _, hiddle = self.model(self.graph.x)

                train_loss = self.clf(hiddle)
                self.optimizer.zero_grad()
                self.clf_optimizer.zero_grad()
                train_loss.backward()
                self.optimizer.step()
                self.clf_optimizer.step()
                if self.verbose:
                    print (f"Stage two, Model name: {self.model_name}, loss oriented: {self.loss_oriented}; Epoch {epoch}/{self.epochs_fine}, Training loss: {train_loss}")
                
                self.model.eval()
                self.clf.eval()
                _, hiddle = self.model(self.graph.x)
                val_loss, val_auc = self.clf.val_loss_auc(hiddle)
                if val_auc >= best_val_auc:
                    best_val_auc = val_auc
                early_stop(val_loss, self.model)
                if early_stop.early_stop == True:
                    print ("Early stopping")
                    break
                if self.verbose:
                    print (f"Stage two, Model name: {self.model_name}, loss oriented: {self.loss_oriented}; Epoch {epoch+1}/{self.epochs_fine}, Val auc: {val_auc}")
            self.model.eval()
            self.clf.eval()
            _, hiddle = self.model(self.graph.x)
            test_auc = self.clf.test_auc(hiddle)
            return test_auc, best_val_auc

        