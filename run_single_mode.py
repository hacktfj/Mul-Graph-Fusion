import torch
import numpy as np
from dataset import pyg_dataset, pyg_to_dgl
from models.single_model import Single_model

torch.manual_seed(21)
np.random.seed(2)
device = torch.device("cuda")

model_list = ["bwgnn", "gin", "gcn", "gat"]
data_list = ["pubmed","amazon_computer","amazon_photo","weibo","books"]
# data_list = ["fraud_amazon",] 3*3* 7*2 dataset
# loss_oriented_list = ["label_oriented", "reconstruction_oriented", "ssl_oriented"]
loss_oriented_list = ["label_oriented", "reconstruction_oriented", "ssl_oriented"]
run_times = range(5)
hid_dim = 64
number_class = 2

for data_name in data_list:
    if data_name in ["pubmed","amazon_computer","amazon_photo","weibo","books",]:
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="min").dataset.to(device)
        for model_name in model_list:
            for loss_oriented in loss_oriented_list:
                test_auc_list = []
                best_auc_list = []
                for run in run_times:
                    model = Single_model(model_name, loss_oriented, data.x.shape[1], hid_dim, number_class, data, verbose=0).to(device)
                    model.fit()
                    test_auc,best_auc = model.fine_tuning()
                    best_auc_list.append(best_auc)
                    test_auc_list.append(test_auc)
                print (f"Single Mode: model name {model_name}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type min; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
                with open("result/singlemode_for_all_dataset.txt",'a') as f:
                    f.write(f"Single Mode: model name {model_name}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type min; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n") 

    if data_name in ["pubmed","amazon_computer","amazon_photo"]:
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="syn").dataset.to(device)
        for model_name in model_list:
            for loss_oriented in loss_oriented_list:
                test_auc_list = []
                best_auc_list = []
                for run in run_times:
                    model = Single_model(model_name, loss_oriented, data.x.shape[1], hid_dim, number_class, data, verbose=0).to(device)
                    model.fit()
                    test_auc,best_auc = model.fine_tuning()
                    best_auc_list.append(best_auc)
                    test_auc_list.append(test_auc)
                print (f"Single Mode: model name {model_name}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type syn; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
                with open("result/singlemode_for_all_dataset.txt",'a') as f:
                    f.write(f"Single Mode: model name {model_name}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type syn; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n")