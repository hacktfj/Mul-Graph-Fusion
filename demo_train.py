import torch
import torch.nn as nn
from models.GAT_model import GAT
from torch.optim import Adam
from torch.nn.functional import cross_entropy
from dataset import pyg_dataset, pyg_to_dgl
from early_stop import EarlyStopping
from models.BWGNN_model import BWGNN_em
from torch_geometric.utils import to_dense_adj
from sklearn.metrics import roc_auc_score
from pygod.models import DOMINANT
from dataset import pyg_dataset, pyg_to_dgl
from utils import anomaly_weight, train_for_GCN, feature_fusion


# 使用五折验证得到比较可信的auc
data1 = pyg_dataset(dataset_name="cora", dataset_spilt=[0.4,0.2,0.3], anomaly_type="syn", anomaly_ratio=0.1).dataset
data2 = pyg_dataset(dataset_name="cora", dataset_spilt=[0.4,0.2,0.3], anomaly_type="min", anomaly_ratio=0.1).dataset
data_list = [data1, data2]
for data in data_list:
    dgl_data = pyg_to_dgl(data)
    a_weight = anomaly_weight(data)

    # data = pyg_dataset(dataset_name="cora", dataset_spilt=[0.4,0.2,0.2], anomaly_type="min").dataset
    model = DOMINANT(verbose=True, gpu=-1, epoch=5, lr=1e-3)
    model = model.fit(data)

    x_, s_, hid_dom = model.model(data.x, data.edge_index)

    s = to_dense_adj(data.edge_index)[0]
    score = model.loss_func(data.x,x_,s,s_)
    score = score.detach().cpu().numpy()
    outlier_scores = model.decision_function(data)
    test_auc_do = roc_auc_score(data.y[data.test_mask].numpy(), score[data.test_mask])
    print('Final Test AUC:', test_auc_do)
    print ("--------------------- The Embedding of Dominate have done!!! ------------------")


    number_class = 2
    hid_dim = 64
    BWGNN_model = BWGNN_em(data.x.shape[1], 64, number_class, dgl_data)
    BW_optimizer = Adam(BWGNN_model.parameters(), lr = 1e-4)
    epochs = 100
    hid_bw, test_auc_bw, best_auc_bw = train_for_GCN(BWGNN_model, BW_optimizer, data, a_weight, epochs)
    print ("--------------------- The Embedding of BWGNN have done!!! ------------------")


    hid_dim = 64
    edge_index = data.edge_index
    gat_model = GAT(data.x.shape[1], 64, number_class, data)
    GAT_optimizer = Adam(gat_model.parameters(), lr = 1e-3)
    epochs = 100
    hid_gat, test_auc_gat, best_auc_gat = train_for_GCN(gat_model, GAT_optimizer, data, a_weight, epochs)
    print ("--------------------- The Embedding of GAT have done!!! ------------------")


    import torch.nn.functional as F
    from pygod.models.basic_nn import GCN
    from torch_geometric.utils import to_dense_adj
    class DOMINANT_recon(nn.Module):
        def __init__(self,
                    in_dim,
                    hid_dim,
                    out_dim,
                    decoder_layers,
                    dropout,
                    act):
            super(DOMINANT_recon, self).__init__()

            # split the number of layers for the encoder and decoders
            self.attr_decoder = GCN(in_channels=in_dim,
                                    hidden_channels=hid_dim,
                                    num_layers=decoder_layers,
                                    out_channels=out_dim,
                                    dropout=dropout,
                                    act=act)

            self.struct_decoder = GCN(in_channels=in_dim,
                                    hidden_channels=hid_dim,
                                    num_layers=decoder_layers,
                                    out_channels=out_dim,
                                    dropout=dropout,
                                    act=act)

        def forward(self, h, edge_index):
            # decode feature matrix
            x_ = self.attr_decoder(h, edge_index)
            # print (x_.shape)
            # decode adjacency matrix
            h_ = self.struct_decoder(h, edge_index)
            # print (h_.shape)
            s_ = h_ @ h_.T
            # return reconstructed matrices
            return x_, s_, h

    def reco_loss_func(x, x_, s, s_):
        # attribute reconstruction loss
        diff_attribute = torch.pow(x - x_, 2)
        attribute_errors = torch.sqrt(torch.sum(diff_attribute, 1))

        # structure reconstruction loss
        diff_structure = torch.pow(s - s_, 2)
        structure_errors = torch.sqrt(torch.sum(diff_structure, 1))

        score = alpha * attribute_errors \
                + (1 - alpha) * structure_errors
        return score

    s = to_dense_adj(data.edge_index)[0]
    alpha = torch.std(s).detach() / (torch.std(data.x).detach() + torch.std(s).detach())
    print (f"Alpha: {alpha}")

    feature_list = [hid_dom.detach(), hid_bw.detach(), hid_gat.detach()]
    # l_weight = [nn.Parameter(torch.tensor(zero2one((1-cosine_distance(hiddle.T)).sum(axis=0))*20, dtype=torch.float32, requires_grad=True))]
    # l_weight = [nn.Parameter(torch.randn([hiddle.shape[-1]], dtype=torch.float32, requires_grad=True))]
    l_weight = [nn.Parameter(torch.ones([hid_dom.shape[-1]*3], dtype=torch.float32, requires_grad=True))]
    b_weight = [nn.Parameter(torch.ones([3], dtype=torch.float32, requires_grad=True))]
    optimizer_ = Adam(l_weight, lr = 5e-2, weight_decay=5e-2)
    optimizer_b = Adam(b_weight, lr = 1e-1, weight_decay=5e-2)
    hiddle = feature_fusion(feature_list, b_weight)

    decode_layer = 1
    # domin_recon_model = DOMINANT_recon(hiddle.shape[1], hid_dim, data.x.shape[1], decode_layer, dropout=0.3, act= F.relu)
    domin_recon_model = DOMINANT_recon(hiddle.shape[1], data.x.shape[1], data.x.shape[1], decode_layer, dropout=0.3, act= F.relu)
    optimizer = Adam(domin_recon_model.parameters(), lr = 5e-3, weight_decay=5e-4)
    early_stop = EarlyStopping(patience=20)
    epochs = 50
    best_auc_fu = 0
    print (b_weight)

    for epoch in range(epochs):
        hiddle = feature_fusion(feature_list, b_weight)
        # hiddle = torch.concat((b_weight[0][0] * feature_normalize(GAT_hid.detach(),axis=0), b_weight[0][1] * feature_normalize(BW_hid.detach(),axis=0), b_weight[0][2] * feature_normalize(hid_dominate.detach(),axis=0)), axis=1)
        # hiddle = zero2one((1-cosine_distance(hiddle.T)).mean(axis=0))*hiddle
        hiddle_ = torch.mul(hiddle, hiddle.shape[1]*torch.softmax(*l_weight, dim = 0))
        # hiddle_ = torch.mul(hiddle, *l_weight)

        domin_recon_model.train()
        x_, s_, hid  = domin_recon_model(hiddle_, data.edge_index)
        
        nodes_loss = reco_loss_func(data.x, x_, s, s_)
        train_loss = nodes_loss.mean()
        nodes_loss_numpy = nodes_loss.detach().numpy()

        optimizer.zero_grad()
        optimizer_.zero_grad()
        optimizer_b.zero_grad()
        train_loss.backward()
        optimizer.step()
        optimizer_.step()
        optimizer_b.step()
        print (b_weight)

        val_auc = roc_auc_score(data.y[data.val_mask].numpy(), nodes_loss_numpy[data.val_mask])
        if val_auc >= best_auc_fu:
            best_auc_fu = val_auc
        early_stop(train_loss, domin_recon_model)
        if early_stop.early_stop == True:
            print ("Early stopping")
            break
        print (f"Epoch {epoch+1}/{epochs}: loss: {train_loss}, val_auc: {val_auc}")

    print (b_weight)
    domin_recon_model.eval()
    x_, s_, hid  = domin_recon_model(hiddle_, data.edge_index)
    nodes_loss = reco_loss_func(data.x, x_, s, s_)
    train_loss = nodes_loss.mean()
    nodes_loss_numpy = nodes_loss.detach().numpy()

    test_auc_fu = roc_auc_score(data.y[data.test_mask].numpy(), nodes_loss_numpy[data.test_mask])
    print (f"Final Test Auc: {test_auc_fu}")

    print (f"DOMINANT TEST AUC: {test_auc_do}; BWGNN TEST AUC: {test_auc_bw}; GAT TEST AUC: {test_auc_gat};")
    print (f"DOMINANT BEST VAL AUC: unkonwn; BWGNN BEST VAL AUC: {best_auc_bw}; GAT BEST VAL AUC: {best_auc_gat};")
    print (f"Fusion Model TEST AUC {test_auc_fu}; BEST VAL AUC: {best_auc_fu}")