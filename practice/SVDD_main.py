"""Reduce a number class from alleviating for loop. Meanwhile, loss 5% around accuracy for the unjoined loss dependency.
Therefore, trainning with auxilary label-oriented task may be helpful.
"""
import time
import torch
import numpy as np
import torch.nn as nn
from early_stop import EarlyStopping
from torch.optim import Adam
from SVDD_model import SVDD
from dataset import pyg_dataset, pyg_to_dgl
from sklearn.metrics import roc_auc_score

def anomaly_score(node_embedding, c):
    # anomaly score of an instance is calculated by 
    # square Euclidean distance between the node embedding and the center c
    return torch.sum((node_embedding - c) ** 2)

def nor_loss(node_embedding_list, c):
    # normal loss is calculated by mean squared Euclidian distance of 
    # the normal node embeddings to hypersphere center c 
    s = 0
    num_node = node_embedding_list.size()[0]
    for i in range(num_node):
        s = s + anomaly_score(node_embedding_list[i], c)
    return s/num_node

# def AUC_loss(anomaly_node_emb, normal_node_emb, c):
#     # AUC_loss encourages the score of anomaly instances to be higher than those of normal instances
#     s = 0
#     num_anomaly_node = anomaly_node_emb.size()[0]
#     num_normal_node = normal_node_emb.size()[0]
#     for i in range(num_anomaly_node):
#         for j in range(num_normal_node):
#             s1 = anomaly_score(anomaly_node_emb[i], c)
#             s2 = anomaly_score(normal_node_emb[j], c)
#             s = s + torch.sigmoid(s1 - s2)
#     return s/(num_anomaly_node * num_normal_node) # check devide by zero

def AUC_loss(anomaly_node_emb, normal_node_emb, c):
    # AUC_loss encourages the score of anomaly instances to be higher than those of normal instances
    s = 0
    num_anomaly_node = anomaly_node_emb.size()[0]
    num_normal_node = normal_node_emb.size()[0]
    s2_list = []
    for j in range(num_normal_node):
        s2_list.append(anomaly_score(normal_node_emb[j], c))
    for i in range(num_anomaly_node):
            s1 = anomaly_score(anomaly_node_emb[i], c)
            s = s + torch.sigmoid(s1 - torch.tensor(s2_list)).sum()
    return s/(num_anomaly_node * num_normal_node) # check devide by zero

def objecttive_loss(anomaly_node_emb, normal_node_emb, c, regularizer=1):
    Nloss = nor_loss(normal_node_emb, c)
    AUCloss = AUC_loss(anomaly_node_emb, normal_node_emb, c)
    loss = Nloss - regularizer * AUCloss
    return loss


# load the dataset
graph = pyg_dataset(dataset_name="cora", dataset_spilt=[0.064,0.3,0.3], anomaly_type="min").dataset
dgl_graph = pyg_to_dgl(graph)

# train_normal mask and train_anomaly mask
train_anomaly = [bool(graph.y[i] & graph.train_mask[i]) for i in range(len(graph.train_mask))] 
train_normal = [bool((~graph.y[i]) & graph.train_mask[i]) for i in range(len(graph.train_mask))]
val_anomaly = [bool(graph.y[i] & graph.val_mask[i]) for i in range(len(graph.val_mask))]
val_normal = [bool((~graph.y[i]) & graph.val_mask[i]) for i in range(len(graph.val_mask))]
test_anomaly = [bool(graph.y[i] & graph.test_mask[i]) for i in range(len(graph.test_mask))]
test_normal = [bool((~graph.y[i]) & graph.test_mask[i]) for i in range(len(graph.test_mask))]


features = graph.x
edge_index = graph.edge_index
true_label = graph.y
node_features = graph.num_node_features
AUC_regularizer = 1
n_embedding = 32
hiddle_dim = 32
hiddle_layer = 2

model = SVDD(input_dim = node_features, hiddle_dim=hiddle_dim, number_class= n_embedding, hiddle_layer= hiddle_layer, dropout=0.5)
lr = 1e-2
epochs = 50
weight_decay = 5e-4
optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
early_stopping = EarlyStopping(patience = 20)

# trainning
for epoch in range(epochs):
    start = time.time()
    model.train()
    node_embedding = model(dgl_graph,features)
    if epoch % 10 == 0:
        center = model(dgl_graph, features)[train_normal].detach().mean(0)
    
    # import IPython
    # IPython.embed()
    # exit()
    # SVDD loss function
    loss_train = objecttive_loss(node_embedding[train_anomaly], node_embedding[train_normal], center, AUC_regularizer)
    model.zero_grad()
    loss_train.backward()
    optimizer.step()

    model.eval()
    node_embedding = model(dgl_graph, features)

    loss_val = objecttive_loss(node_embedding[val_anomaly], node_embedding[val_normal], center, AUC_regularizer)
    val_anomaly_score = [anomaly_score(embedding, center) for embedding in node_embedding[graph.val_mask].cpu().detach()]
    auc = roc_auc_score(graph.y[graph.val_mask], val_anomaly_score)
    
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {}'.format(loss_train.item()),
          'loss_val: {}'.format(loss_val.item()),
          'roc_auc: {:.4f}'.format(auc),
          'time: {:.4f}s'.format(time.time() - start))    
    early_stopping(loss_val, model)
    if early_stopping.early_stop:
        print("Early stopping")
        break

test_anomaly_score = [anomaly_score(embedding, center) for embedding in node_embedding[graph.test_mask].cpu().detach()]
auc = roc_auc_score(graph.y[graph.test_mask], test_anomaly_score)

print(f"Test Auc: {auc}")

test_anomaly_score = [anomaly_score(embedding, center) for embedding in node_embedding[graph.val_mask].cpu().detach()]
auc = roc_auc_score(graph.y[graph.val_mask], test_anomaly_score)

print(f"Val Auc: {auc}")