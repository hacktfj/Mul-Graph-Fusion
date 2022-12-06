from torch_geometric.data import Data
from models.baseline_model import Baseline
import os
import sys
import torch
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from numpy import percentile
import matplotlib.pyplot as plt
import matplotlib.font_manager
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
import numpy as np
from dataset import pyg_dataset
from sklearn.metrics import roc_auc_score
from dataset import pyg_to_dgl
from models.fusion_model import Fusion_model
# Define the number of inliers and outliers
n_samples = 200
outliers_fraction = 0.25
clusters_separation = [0]
in_feats = 2
h_feats = 16
num_classes = 2

# Compare given detectors under given settings
# Initialize the data
xx, yy = np.meshgrid(np.linspace(-7, 7, 100), np.linspace(-7, 7, 100))
n_inliers = int((1. - outliers_fraction) * n_samples)
n_outliers = int(outliers_fraction * n_samples)
ground_truth = np.zeros(n_samples, dtype=int)
ground_truth[-n_outliers:] = 1

# Data generation
X1 = 0.3 * np.random.randn(n_inliers // 2, 2)
X2 = 0.3 * np.random.randn(n_inliers // 2, 2)
X = np.r_[X1, X2]
# Add outliers
X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]

position=0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
graph = Data(x=torch.Tensor(X), edge_index=torch.LongTensor([[0],[0]]),y=torch.LongTensor(ground_truth),train_mask=position,val_mask=position,test_mask=position).to(device)

random_state = 42
# Define nine outlier detection tools to be compared
classifiers = {
    'Histogram-base Outlier Detection (HBOS)': HBOS(
        contamination=outliers_fraction),
    'Isolation Forest': IForest(contamination=outliers_fraction,
                                random_state=random_state),
    'Local Outlier Factor (LOF)':
        LOF(n_neighbors=35, contamination=outliers_fraction),
    'One-class SVM (OCSVM)': OCSVM(contamination=outliers_fraction),
    'Graph Convolutional Network (GCN)': Baseline("gcn",in_feats, h_feats, num_classes, graph).to(device),
    'Graph Isomorphism Network (GIN)': Baseline("gin",in_feats, h_feats, num_classes, graph).to(device),
    'Graph Attention Network (GAT)': Baseline("gat",in_feats, h_feats, num_classes, graph).to(device),
    'Multi-view Graph Anomaly Detection (Our Mul-GAD)': Fusion_model(["gat","gcn",], "label_oriented", in_feats, h_feats, num_classes, graph, fusion_strategy=3, epochs_fit=75, verbose=0, anomaly_type="min").to(device)
}

# Show all detectors
for i, clf in enumerate(classifiers.keys()):
    print('Model', i + 1, clf)

# Fit the models with the generated data and
# compare model performances
for i, offset in enumerate(clusters_separation):
    np.random.seed(42)

    # Fit the model
    plt.figure(figsize=(15, 8))
    for i, (clf_name, clf) in enumerate(classifiers.items()):
        print()
        print(i + 1, 'fitting', clf_name)
        Z = None
        threshold = None
        if "GCN" in clf_name or "GIN" in clf_name or "GAT" in clf_name or "Mul-GAD" in clf_name:
            logits = clf.fit_for_tsne()
            scores_pred = logits.softmax(1)[:,1] * -1
            threshold = percentile(scores_pred.detach().cpu().numpy(), 100 * outliers_fraction)
            data = Data(x=torch.Tensor(np.c_[xx.ravel(), yy.ravel()]), edge_index=torch.LongTensor([[0],[0]]),y=torch.LongTensor(ground_truth),train_mask=position,val_mask=position,test_mask=position).to(device)
            # data = pyg_to_dgl(data) if clf_name == "BWGNN" else data 
            Z = clf.decision_for_tsne(data).softmax(1)[:,1] * -1
            Z = Z.detach().cpu().numpy()
        else:
            # fit the data and tag outliers
            clf.fit(X)
            scores_pred = clf.decision_function(X) * -1
            threshold = percentile(scores_pred, 100 * outliers_fraction)
            # plot the levels lines and the points
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
        
        Z = Z.reshape(xx.shape)
        subplot = plt.subplot(2, 4, i + 1)
        subplot.contourf(xx, yy, Z, levels=np.linspace(Z.min(), threshold, 7),
                         cmap=plt.cm.Blues_r)
        # a = subplot.contour(xx, yy, Z, levels=[threshold],
        #                     linewidths=2, colors='red')
        subplot.contourf(xx, yy, Z, levels=[threshold, Z.max()],
                         colors='orange')
        b = subplot.scatter(X[:-n_outliers, 0], X[:-n_outliers, 1], c='white',
                            s=20, edgecolor='k')
        c = subplot.scatter(X[-n_outliers:, 0], X[-n_outliers:, 1], c='black',
                            s=20, edgecolor='k')
        subplot.axis('tight')
        subplot.legend(
            [
                # a.collections[0],
                b, c],
            [
                # 'learned decision function', 
                'normal', 'abnormal'],
            prop=matplotlib.font_manager.FontProperties(size=10),
            loc='lower right')
        subplot.set_xlabel("%d. %s" % (i + 1, clf_name))
        subplot.set_xlim((-7, 7))
        subplot.set_ylim((-7, 7))
    plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
    plt.suptitle("Decision Boundary for Varying Algorithms")
plt.savefig(str('./result/tnse.eps'), bbox_inches="tight", format="eps")
plt.show()