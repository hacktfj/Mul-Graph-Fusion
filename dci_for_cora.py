import torch
import numpy as np
import torch.nn as nn
from models.GAT_model import GAT
from models.GCN_model import GCN
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from dataset import pyg_dataset, pyg_to_dgl
from early_stop import EarlyStopping
from models.BWGNN_model import BWGNN_em
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans
from torch.nn import Linear
from dataset import pyg_dataset, pyg_to_dgl
from models.dci import DCI_loss
from utils import *

torch.manual_seed(21)
device = torch.device("cpu")

data_name = "cora"

# run five times to get mean and std for test. best performance for val.
test_auc_mean = []
test_auc_std = []
val_auc_best = []
test_list = []
val_list = []

np.random.seed(2)
# dataset preparing
data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="min").dataset
dgl_data = pyg_to_dgl(data)
data = data.to(device)
dgl_data = dgl_data.to(device)
a_weight = anomaly_weight(data)

epochs = 1000
hid_dim = 64
number_class = 2
recluster_interval = 10
kmeans = KMeans(n_clusters=number_class, random_state=0).fit(data.x)
ss_label = kmeans.labels_
cluster_info = [list(np.where(ss_label==i)[0]) for i in range(number_class)]
idx = np.random.permutation(data.x.shape[0])
shuf_feats = data.x[idx, :]

gcn_model1 = GCN(data.x.shape[1], hid_dim, number_class, data).to(device)
gcn_model2 = GCN(data.x.shape[1], hid_dim, number_class, data).to(device)
loss_dci = DCI_loss(hid_dim, device)
gcn_optimizer = Adam([{"params": gcn_model1.parameters(), "lr": 1e-3},\
                        {"params": gcn_model2.parameters(), "lr": 1e-3}])

for epoch in range(epochs):
    gcn_model1.train()
    gcn_model2.train()
    _, hid1 = gcn_model1(data.x)
    _, hid2 = gcn_model2(shuf_feats)
    train_loss = loss_dci(hid1, hid2, None, None, None, cluster_info, number_class)

    gcn_optimizer.zero_grad()
    train_loss.backward()
    gcn_optimizer.step()
    print (f"epoch: {epoch + 1}/{epochs}, loss: {train_loss}")
    # re-clustering
    if epoch % recluster_interval == 0:
        _, emb = gcn_model1(data.x)
        kmeans = KMeans(n_clusters=number_class, random_state=0).fit(emb.detach().cpu().numpy())
        ss_label = kmeans.labels_
        cluster_info = [list(np.where(ss_label==i)[0]) for i in range(number_class)]

cls_model = Linear(in_features=hid_dim, out_features=number_class)
cls_optimizer = Adam(cls_model.parameters(), lr = 5e-3)
epochs = 100

for epoch in range(epochs):
    cls_model.train()
    gcn_model1.eval()

    _, hid = gcn_model1(data.x)
    logits = cls_model(hid, data.edge_index)
    train_loss = cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=torch.tensor([1.0, a_weight],device=device))
    
    cls_model.zero_grad()
    train_loss.backward()
    cls_optimizer.step()

    cls_model.eval()
    logits = cls_model(hid, data.edge_index)
    val_loss = cross_entropy(logits[data.val_mask], data.y[data.val_mask], weight=torch.tensor([1.0, a_weight],device=device))
    probs = logits.softmax(1)
    auc = roc_auc_score(data.y[data.val_mask].cpu().numpy(), probs[data.val_mask][:,1].detach().cpu().numpy()) if data.y.is_cuda else \
            roc_auc_score(data.y[data.val_mask].numpy(), probs[data.val_mask][:,1].detach().numpy())
    print (f"Epoch {epoch+1}/{epochs}: val_loss: {val_loss}, val_auc: {auc}")


