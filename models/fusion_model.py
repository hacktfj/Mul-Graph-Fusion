"""Fusion model with the ssl oriented for now. Function fit, fine-tuning and fusion feature.
"""
"""Single Model for Two Stages Learning.
"""
from pickletools import optimize
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
class Fusion_model(nn.Module):

    def __init__(self, model_name_list, loss_oriented, in_feats, h_feats, num_classes, graph, fusion_strategy=0, lr = 5e-3, epochs_fit=50, epochs_fine=50, verbose=1) -> None:
        """model name for training and loss_oriented decide which task to learn on the specify model.
        model1,optimizer1 is None for label and reconstruction oriented
        fusion_strategy == 0: no fusion.
        fusion_strategy == 1: fusion feature.
        fusion_strategy == 2: fusion view feature. __init__ with the learnable balanced weight and optimizer.
        fusion_strategy == 3: fusion feature and view feature. __init__ with the learnable balanced weight and optimizer.
        and more fusion strategy for later experiments.
        """
        super().__init__()
        self.model_list = []
        self.model1_list = []
        self.optimizer_list = []
        self.optimizer1_list = []
        self.loss = None
        self.loss_optimizer = None
        self.model_len = len(model_name_list)
        self.clf = Classifier(h_feats*self.model_len, num_classes, graph)
        self.clf_optimizer = Adam(self.clf.parameters(), lr = lr)
        self.fusion_strategy = fusion_strategy
        self.balance_weight = None
        self.balance_optimizer = None

        if fusion_strategy == 2 or fusion_strategy == 3:
            self.balance_weight = [nn.Parameter(torch.ones([self.model_len], dtype=torch.float32, requires_grad=True))]
            # b_weight = [[0.0, param1, param2]]
            self.balance_optimizer = Adam(self.balance_weight, lr = 1e-2, weight_decay=5e-2)    

        self.a_weight = anomaly_weight(graph)
        self.model_name_list = model_name_list
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

        for model_name in self.model_name_list:
            if model_name == "bwgnn":
                dgl_graph = pyg_to_dgl(graph)
                self.model_list.append(BWGNN_em(in_feats, h_feats, num_classes, dgl_graph))
                # self.optimizer = Adam(self.model.parameters(), lr = lr)
                self.model1_list.append(BWGNN_em(in_feats, h_feats, num_classes, dgl_graph) if loss_oriented == "ssl_oriented" else None)
                # self.optimizer1 = Adam(self.model1.parameters(), lr = lr) if loss_oriented == "ssl_oriented" else None
            elif model_name == "gin":
                self.model_list.append(GIN(in_feats, h_feats, num_classes, self.graph))
                self.model1_list.append(GIN(in_feats, h_feats, num_classes, self.graph) if loss_oriented == "ssl_oriented" else None)
            elif model_name == "gat":
                self.model_list.append(GAT(in_feats, h_feats, num_classes, self.graph))
                self.model1_list.append(GAT(in_feats, h_feats, num_classes, self.graph) if loss_oriented == "ssl_oriented" else None)
            elif model_name == "gcn":
                self.model_list.append(GCN(in_feats, h_feats, num_classes, self.graph))
                self.model1_list.append(GCN(in_feats, h_feats, num_classes, self.graph) if loss_oriented == "ssl_oriented" else None)
            self.optimizer_list.append(Adam(self.model_list[-1].parameters(), lr = self.lr))
            self.optimizer1_list.append(Adam(self.model1_list[-1].parameters(), lr = self.lr) if loss_oriented == "ssl_oriented" else None)
            
            
        if loss_oriented == "label_oriented":
            self.loss = Label_loss(self.h_feats*self.model_len, self.num_classes, self.graph)
        elif loss_oriented == "reconstruction_oriented":
            self.loss = Reconstruction_loss(self.h_feats*self.model_len, self.graph)
        elif loss_oriented == "ssl_oriented":
            self.loss = SSL_loss(self.h_feats*self.model_len, self.device)
        self.loss_optimizer = Adam(self.loss.parameters(), lr = self.lr)
    
    def fusion_feature(self, hiddle_list, fusion_strategy):
        """what does the fusion means detail.
        fusion_strategy == 0: no fusion.
        fusion_strategy == 1: fusion feature.
        fusion_strategy == 2: fusion view feature.
        fusion_strategy == 3: fusion feature and view feature.
        and more fusion strategy for later experiments.

        Return:
            return the handled hiddle features: fusion feature at the specify strategy.
        """
        hiddle = None
        if fusion_strategy == 0:
            hiddle = torch.concat(hiddle_list, axis=1)
        elif fusion_strategy == 1:
            hiddle = torch.concat(hiddle_list, axis=1)
            hiddle = zero2one((1-cosine_distance(hiddle.T)).mean(axis=0))*hiddle
        elif fusion_strategy == 2:
            hiddle_list_temp = []
            for pos, hiddle in enumerate(hiddle_list):
                hiddle_list_temp.append(self.balance_weight[0][pos]*hiddle)
            hiddle = torch.concat(hiddle_list_temp, axis=1)
        elif fusion_strategy == 3:
            hiddle_list_temp = []
            for pos, hiddle in enumerate(hiddle_list):
                hiddle_list_temp.append(self.balance_weight[0][pos]*hiddle)
            hiddle = torch.concat(hiddle_list_temp, axis=1)
            hiddle = zero2one((1-cosine_distance(hiddle.T)).mean(axis=0))*hiddle
        return hiddle

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
            kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(self.graph.x)
            ss_label = kmeans.labels_
            cluster_info = [list(np.where(ss_label==i)[0]) for i in range(self.num_classes)]
            idx = np.random.permutation(self.graph.x.shape[0])
            shuf_feats = self.graph.x[idx, :]
            early_stop = EarlyStopping(patience=patience)

            for epoch in range(self.epochs_fit):
                print (f"balance weight: {self.balance_weight}")
                self.loss.train()
                hiddle1_list = []
                hiddle2_list = []
                for model in self.model_list:
                    model.train()
                    _, hiddle1 = model(self.graph.x)
                    hiddle1_list.append(hiddle1)
                for model1 in self.model1_list:
                    model1.train()
                    _, hiddle2 = model1(shuf_feats)
                    hiddle2_list.append(hiddle2)
                
                fusion_hiddle1, fusion_hiddle2 = self.fusion_feature(hiddle1_list, fusion_strategy=self.fusion_strategy), self.fusion_feature(hiddle2_list, fusion_strategy=self.fusion_strategy)
                train_loss = self.loss(fusion_hiddle1, fusion_hiddle2, None, None, None, cluster_info, self.num_classes)
                
                [optimizer.zero_grad() for optimizer in self.optimizer_list]
                [optimizer.zero_grad() for optimizer in self.optimizer1_list]
                if self.fusion_strategy == 2 or self.fusion_strategy == 3:
                    self.balance_optimizer.zero_grad()
                self.loss_optimizer.zero_grad()
                train_loss.backward()
                [optimizer.step() for optimizer in self.optimizer_list]
                [optimizer.step() for optimizer in self.optimizer1_list]
                if self.fusion_strategy == 2 or self.fusion_strategy == 3:
                    self.balance_optimizer.step()
                self.loss_optimizer.step()

                # re-clustering
                if epoch % recluster_interval == 0:
                    hiddle1_list = []
                    for model in self.model_list:
                        model.eval()
                        _, hiddle1 = model(self.graph.x)
                        hiddle1_list.append(hiddle1)
                    fusion_hiddle1 = self.fusion_feature(hiddle1_list, fusion_strategy=self.fusion_strategy)
                    kmeans = KMeans(n_clusters=self.num_classes, random_state=0).fit(fusion_hiddle1.detach().cpu().numpy())
                    ss_label = kmeans.labels_
                    cluster_info = [list(np.where(ss_label==i)[0]) for i in range(self.num_classes)]
                
                # record
                early_stop(train_loss)
                if early_stop.early_stop == True:
                    print ("Early stopping")
                    break
                if self.verbose:
                    print (f"Stage one, Model list name: {self.model_name_list}, loss oriented: {self.loss_oriented}; Epoch {epoch}/{self.epochs_fit}, Training loss: {train_loss}")

    def fine_tuning(self, patience=20):
        if self.loss_oriented == "label_oriented" or self.loss_oriented == "reconstruction_oriented" or self.loss_oriented == "ssl_oriented":
            early_stop = EarlyStopping(patience=patience)
            best_val_auc = 0
            for epoch in range(self.epochs_fine):
                self.clf.train()
                hiddle_list = []
                for model in self.model_list:
                    model.train()
                    _, hiddle = model(self.graph.x)
                    hiddle_list.append(hiddle)
                hiddle = self.fusion_feature(hiddle_list, self.fusion_strategy)

                train_loss = self.clf(hiddle)

                [optimizer.zero_grad() for optimizer in self.optimizer_list]
                if self.fusion_strategy == 2 or self.fusion_strategy == 3:
                    self.balance_optimizer.zero_grad()
                self.clf_optimizer.zero_grad()
                train_loss.backward()
                [optimizer.step() for optimizer in self.optimizer_list]
                if self.fusion_strategy == 2 or self.fusion_strategy == 3:
                    self.balance_optimizer.step()
                self.clf_optimizer.step()
                if self.verbose:
                    print (f"Stage two, Model list name: {self.model_name_list}, loss oriented: {self.loss_oriented}; Epoch {epoch}/{self.epochs_fine}, Training loss: {train_loss}")
                
                self.clf.eval()
                hiddle_list = []
                for model in self.model_list:
                    model.eval()
                    _, hiddle = model(self.graph.x)
                    hiddle_list.append(hiddle)
                hiddle = self.fusion_feature(hiddle_list, self.fusion_strategy)

                val_loss, val_auc = self.clf.val_loss_auc(hiddle)
                if val_auc >= best_val_auc:
                    best_val_auc = val_auc
                early_stop(val_loss)
                if early_stop.early_stop == True:
                    print ("Early stopping")
                    break
                if self.verbose:
                    print (f"Stage two, Model name: {self.model_name_list}, loss oriented: {self.loss_oriented}; Epoch {epoch}/{self.epochs_fine}, Val auc: {val_auc}")
            self.clf.eval()
            hiddle_list = []
            for model in self.model_list:
                model.eval()
                _, hiddle = model(self.graph.x)
                hiddle_list.append(hiddle)
            hiddle = self.fusion_feature(hiddle_list, self.fusion_strategy)
            test_auc = self.clf.test_auc(hiddle)
            return test_auc, best_val_auc
