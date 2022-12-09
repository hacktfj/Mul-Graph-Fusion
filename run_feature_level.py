
import torch
import numpy as np
from dataset import pyg_dataset, pyg_to_dgl
from models.fusion_model import Fusion_model

torch.manual_seed(21)
np.random.seed(2)
device = torch.device("cuda")
loss_oriented = "label_oriented"
hid_dim = 64
number_class = 2

data_list = ["pubmed","amazon_computer","amazon_photo","weibo","books",]
# data_list = ["weibo","books"]
fusion_strategy_list = range(2)
run_times = range(5)
model_list_list = [["gat","gin"],["bwgnn","gat"],["bwgnn","gin"],["bwgnn","gat","gin"]]
# model_list_list = [["bwgnn","gat"]]

for data_name in data_list:
    if data_name in ["pubmed","amazon_computer","amazon_photo","weibo","books",]:
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="min").dataset.to(device)
        for model_list in model_list_list:
            for fusion_strategy in fusion_strategy_list:
                test_auc_list = []
                best_auc_list = []
                for run in run_times:
                    np.random.seed(2*run)
                    model = Fusion_model(model_list, loss_oriented, data.x.shape[1], hid_dim, number_class, data, fusion_strategy=fusion_strategy, epochs_fit=100, verbose=0, anomaly_type="none").to(device)
                    model.fit()
                    test_auc,best_auc = model.no_fine_tuning()
                    best_auc_list.append(best_auc)
                    test_auc_list.append(test_auc)
                print (f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type min; fusion strategy: {fusion_strategy}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
                with open("result/feature_level_for_all_dataset.txt",'a') as f:
                    f.write(f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type min; fusion strategy: {fusion_strategy}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n") 
    
    if data_name in ["pubmed","amazon_computer","amazon_photo"]:
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="syn").dataset.to(device)
        for model_list in model_list_list:
            for fusion_strategy in fusion_strategy_list:
                test_auc_list = []
                best_auc_list = []
                for run in run_times:
                    model = Fusion_model(model_list, loss_oriented, data.x.shape[1], hid_dim, number_class, data, fusion_strategy=fusion_strategy, epochs_fit=100, verbose=0, anomaly_type="none").to(device)
                    model.fit()
                    test_auc,best_auc = model.no_fine_tuning()
                    best_auc_list.append(best_auc)
                    test_auc_list.append(test_auc)
                print (f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type syn; fusion strategy: {fusion_strategy}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
                with open("result/feature_level_for_all_dataset.txt",'a') as f:
                    f.write(f"Fusion Mode: model list {model_list}; dataset {data_name}; loss_oriented {loss_oriented}; anomaly type min; fusion strategy: {fusion_strategy}; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n") 