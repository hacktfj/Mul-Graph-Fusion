import torch
import numpy as np
from dataset import pyg_dataset, pyg_to_dgl
from models.fusion_model import Fusion_model

torch.manual_seed(21)
np.random.seed(10)


device = torch.device("cuda")
# data_list = ["cora","citeseer","pubmed","weibo","reddit"]
data_list = ["reddit"]
# loss_oriented_list = ["label_oriented", "reconstruction_oriented", "ssl_oriented"]
loss_oriented_list = ["label_oriented", "ssl_oriented"]
run_times = range(5)
hid_dim = 64

number_class = 2
data = None
record_string = ""
# change the model list and the loss oriented
model_list_list = [["bwgnn","gat","gin"]]

for data_name in data_list:
    if data_name == "cora" or data_name == "citeseer" or data_name == "pubmed" or data_name == "amazon_computer" or data_name == "amazon_photo":
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="min").dataset.to(device)
        for model_list in model_list_list:
            for loss_oriented in loss_oriented_list:
                test_auc_list = []
                best_auc_list = []
                for run in run_times:
                    model = Fusion_model(model_list, loss_oriented, data.x.shape[1], hid_dim, number_class, data, fusion_strategy=3, epochs_fit=30, epochs_fine=30, verbose=0).to(device)
                    model.fit()
                    test_auc,best_auc = model.fine_tuning()
                    best_auc_list.append(best_auc)
                    test_auc_list.append(test_auc)
                print (f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type min; fusion strategy: {3}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
                with open("result/fusionmode_for_all_dataset.txt",'a') as f:
                    f.write(f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type min; fusion strategy: {3}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n") 

        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="syn", anomaly_ratio=0.2).dataset.to(device)
        for model_list in model_list_list:
            for loss_oriented in loss_oriented_list:
                test_auc_list = []
                best_auc_list = []
                for run in run_times:
                    model = Fusion_model(model_list, loss_oriented, data.x.shape[1], hid_dim, number_class, data, fusion_strategy=3, epochs_fit=30, epochs_fine=30, verbose=0).to(device)
                    model.fit()
                    test_auc,best_auc = model.fine_tuning()
                    best_auc_list.append(best_auc)
                    test_auc_list.append(test_auc)
                print (f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type syn; fusion strategy: {3}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
                with open("result/fusionmode_for_all_dataset.txt",'a') as f:
                    f.write(f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type syn; fusion strategy: {3}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n")
    else:
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="syn").dataset.to(device)
        for model_list in model_list_list:
            for loss_oriented in loss_oriented_list:
                test_auc_list = []
                best_auc_list = []
                for run in run_times:
                    model = Fusion_model(model_list, loss_oriented, data.x.shape[1], hid_dim, number_class, data, fusion_strategy=3, epochs_fit=30, epochs_fine=30, verbose=0).to(device)
                    model.fit()
                    test_auc,best_auc = model.fine_tuning()
                    best_auc_list.append(best_auc)
                    test_auc_list.append(test_auc)
                print (f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type syn; fusion strategy: {3}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
                with open("result/fusionmode_for_all_dataset.txt",'a') as f:
                    f.write(f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type syn; fusion strategy: {3}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n")            
# with open("result/singlemode_for_all_dataset.txt",'a') as f:
#     f.write(record_string) 