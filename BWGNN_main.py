import dgl
import time
import torch
import argparse
import numpy as np
from BWGNN_model import BWGNN
from dataset import pyg_dataset, pyg_to_dgl
import torch.nn.functional as F
from early_stop import EarlyStopping, plot_earlystop
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, roc_auc_score, precision_score, confusion_matrix
 

import warnings
warnings.filterwarnings("ignore")

def train(model, g, args):
    features = g.x
    labels = g.y
    index = list(range(len(labels)))
    # if dataset_name == 'amazon':
    #     index = list(range(3305, len(labels)))

    idx_train, idx_rest, y_train, y_rest = train_test_split(index, labels[index], stratify=labels[index],
                                                            train_size=args.train_ratio,
                                                            random_state=2, shuffle=True)
    idx_valid, idx_test, y_valid, y_test = train_test_split(idx_rest, y_rest, stratify=y_rest,
                                                            test_size=0.67,
                                                            random_state=2, shuffle=True)
    train_mask = torch.zeros([len(labels)]).bool()
    val_mask = torch.zeros([len(labels)]).bool()
    test_mask = torch.zeros([len(labels)]).bool()

    train_mask[idx_train] = 1
    val_mask[idx_valid] = 1
    test_mask[idx_test] = 1
    train_losses = []
    val_losses = []
    print('train/dev/test samples: ', train_mask.sum().item(), val_mask.sum().item(), test_mask.sum().item())
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    best_f1, final_tf1, final_trec, final_tpre, final_tmf1, final_tauc = 0., 0., 0., 0., 0., 0.

    weight = (1-labels[train_mask]).sum().item() / labels[train_mask].sum().item()
    # IPython.embed()
    # exit()
    print('cross entropy weight: ', weight)
    early_stopping = EarlyStopping(patience=20)
    time_start = time.time()
    
    for e in range(args.epoch):
        model.train()
        model.to()
        features = features.to()
        labels = labels.to()

        start = time.time()
        logits = model(features)
        end = time.time()
        print (f"model training takes {end - start} seconds")

        start = time.time()
        loss = F.cross_entropy(logits[train_mask], labels[train_mask], weight=torch.tensor([1., weight]))
        train_losses.append(loss.detach().item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        end = time.time()
        print (f"model optimization {end - start} seconds")
        model.eval()
        loss_val = F.cross_entropy(logits[val_mask], labels[val_mask], weight=torch.tensor([1., weight]))
        val_losses.append(loss_val.detach().item())
        early_stopping(loss_val, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

        probs = logits.softmax(1)
        # 在验证集中选择最好的threshold，来求f1 score
        f1, thres = get_best_f1(labels[val_mask], probs[val_mask])
        preds = np.zeros_like(labels)
        preds[probs[:, 1] > thres] = 1

        trec = recall_score(labels[val_mask], preds[val_mask])
        tpre = precision_score(labels[val_mask], preds[val_mask])
        tauc = roc_auc_score(labels[val_mask], probs[val_mask][:, 1].detach().numpy())
        if best_f1 < f1:
            best_f1 = f1
        print('Epoch {}/{}, loss: {:.4f}, val f1: {:.4f}, val recall: {:.4f}, val precision: {:.4f}, val auc: {:.4f},   (best {:.4f})'
        .format(e, args.epoch, loss, f1, trec, tpre, tauc, best_f1))

    model.eval()
    probs = logits.softmax(1)
    # 在测试集中选择最好的threshold，来求f1 score
    f1, thres = get_best_f1(labels[test_mask], probs[test_mask])
    
    preds = np.zeros_like(labels)
    preds[probs[:, 1] > thres] = 1
    preds[probs[:, 1] < thres] = 0
    trec = recall_score(labels[test_mask], preds[test_mask])
    tpre = precision_score(labels[test_mask], preds[test_mask])
    tauc = roc_auc_score(labels[test_mask], probs[test_mask][:, 1].detach().numpy())
    time_end = time.time()

    print('time cost: ', time_end - time_start, 's')
    print('Test: THRESHOLD {:.2f}, REC {:.2f} PRE {:.2f} F1 {:.2f} AUC {:.2f}'.format(thres*100, trec*100,
                                                                     tpre*100, f1*100, tauc*100))
    return f1, tauc, train_losses,val_losses

# threshold adjusting for best macro f1
def get_best_f1(labels, probs):
    best_f1, best_thre = 0, 0
    for thres in np.linspace(0.05, 0.95, 19):
        preds = np.zeros_like(labels)
        preds[probs[:,1] > thres] = 1
        mf1 = f1_score(labels, preds, average='macro')
        if mf1 > best_f1:
            best_f1 = mf1
            best_thre = thres
    return best_f1, best_thre

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='BWGNN')
    parser.add_argument("--dataset", type=str, default="cora",
                        help="Dataset for this model (yelp/amazon/weibo/reddit)")
    parser.add_argument("--train_ratio", type=float, default=0.4, help="Training ratio")
    parser.add_argument("--hid_dim", type=int, default=64, help="Hidden layer dimension")
    parser.add_argument("--order", type=int, default=2, help="Order C in Beta Wavelet")
    parser.add_argument("--homo", type=int, default=1, help="1 for BWGNN(Homo) and 0 for BWGNN(Hetero)")
    parser.add_argument("--epoch", type=int, default=100, help="The max number of epochs")
    parser.add_argument("--run", type=int, default=1, help="Running times")

    args, known = parser.parse_known_args()
    dataset_name = args.dataset
    homo = args.homo
    order = args.order
    h_feats = args.hid_dim
    # graph = pyg_dataset(dataset_name, dataset_spilt=[0.1,0.2,0.2], anomaly_type="min").dataset
    graph = pyg_dataset(dataset_name, dataset_spilt=[0.1,0.2,0.2], anomaly_type="syn", anomaly_ratio=0.1).dataset
    # graph = pyg_dataset(dataset_name, dataset_spilt=[0.1,0.2,0.2]).dataset
    in_feats = graph.num_node_features
    num_classes = 2
    
    graph_dgl = pyg_to_dgl(graph)
    if args.run == 1:
        if homo:
            model = BWGNN(in_feats, h_feats, num_classes, graph_dgl, d=order)
        _, _, train_loss, val_loss = train(model, graph, args)
        
        #  plot early stop curve
        plot_earlystop(train_loss, val_loss)
        # visualize the loss as the network trained

    else:
        final_mf1s, final_aucs = [], []
        count = 0
        for tt in range(args.run):
            if homo:
                model = BWGNN(in_feats, h_feats, num_classes, graph_dgl, d=order)
            mf1, auc, train_loss, val_loss = train(model, graph, args)
            #  plot early stop curve
            plot_earlystop(train_loss, val_loss, count)
            count = count + 1
            final_mf1s.append(mf1)
            final_aucs.append(auc)
        final_mf1s = np.array(final_mf1s)
        final_aucs = np.array(final_aucs)
        
        print('MF1-mean: {:.2f}, MF1-std: {:.2f}, MF1-best: {:.2f}, AUC-mean: {:.2f}, AUC-std: {:.2f}, AUC-best: {:.2f}'.format(100 * np.mean(final_mf1s),
                                                                                            100 * np.std(final_mf1s), 100 * np.max(final_mf1s),
                                                                100 * np.mean(final_aucs), 100 * np.std(final_aucs), 100 * np.max(final_aucs)))
