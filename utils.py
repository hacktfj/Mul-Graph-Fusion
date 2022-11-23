import torch
import numpy as np
from torch.nn.functional import cross_entropy
from early_stop import EarlyStopping
from sklearn.metrics import roc_auc_score


def anomaly_weight(data):
    return (data.y == 0).sum() / (data.y == 1).sum()

def train_for_GCN(model, optimizer, data, weight, epochs):
    best_val_auc = 0
    early_stop = EarlyStopping(patience=20)
    for epoch in range(epochs):
        model.train()
        logits, _ = model(data.x)
        train_loss = cross_entropy(logits[data.train_mask], data.y[data.train_mask], weight=torch.tensor([1.0, weight]))
        model.zero_grad()
        train_loss.backward()
        optimizer.step()
        
        model.eval()
        logits, _= model(data.x)
        val_loss = cross_entropy(logits[data.val_mask], data.y[data.val_mask], weight=torch.tensor([1.0, weight]))
        probs = logits.softmax(1)
        auc = roc_auc_score(data.y[data.val_mask].numpy(), probs[data.val_mask][:,1].detach().numpy())
        
        if auc >= best_val_auc:
            best_val_auc = auc

        early_stop(val_loss, model)
        if early_stop.early_stop == True:
            print ("Early stopping")
            break
        print (f"Epoch {epoch+1}/{epochs}: val_loss: {val_loss}, val_auc: {auc}")
    model.eval()
    logits, hid= model(data.x)
    probs = logits.softmax(1)
    auc = roc_auc_score(data.y[data.test_mask].numpy(), probs[data.test_mask][:,1].detach().numpy())
    print (f"Final Test Auc: {auc}")
    return hid, auc, best_val_auc


def cosine_distance(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    # .clamp(min=eps)
    # 0-1
    return (torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)).absolute()

# 越无关，权重越大
def zero2one(feature):
    """Input: feature must be a 1d numpy array
    """
    # feature = np.array(feature)
    min = feature.min()
    max = feature.max()
    return (feature - min)/(max-min)

def feature_normalize(feature, axis=1, eps=1e-10):
    """2D array feature to row normalize"""
    mean = None
    std = None
    if axis == 1:
        mean = feature.mean(axis=axis).reshape(-1,1)
        std = feature.std(axis=axis).reshape(-1,1)
    elif axis == 0:
        mean = feature.mean(axis=axis).reshape(1,-1)
        std = feature.std(axis=axis).reshape(1,-1)
    return (feature - mean) / (std + eps)

def feature_fusion(feature_list, weight,strategy=0):
    """been detached feature list."""
    feature_list_ = []
    # weight = len(feature_list)*torch.softmax(*weight, dim = 0)
    if strategy == 0:
        for i, feature in enumerate(feature_list):
            feature_list_.append(weight[0][i] * feature_normalize(feature,axis=0))
        hiddle = torch.concat(feature_list_, axis=1)
        return hiddle
    else:
        pass

def read_param2per(file_path):
    line_array = []
    with open(file_path,"r") as f:
        f_line_list = f.readlines()
        for f_line in f_line_list:
            f_num = f_line.split(" ")
            f_num = np.array(f_num).astype(float)
            line_array.append(f_num)
        line_array = np.array(line_array)
    return line_array

def read_single():
    pass
def read_fusion():
    pass

if __name__ == "__main__":
    param2per = read_param2per("./result/param2performance_weibo_min_gatbw.txt")
    print (param2per[:,2].reshape(11,10))
