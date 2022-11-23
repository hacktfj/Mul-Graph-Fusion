import torch
import numpy as np
from dataset import pyg_dataset, pyg_to_dgl
from models.baseline_model import Baseline

torch.manual_seed(21)
np.random.seed(2)

device = torch.device("cuda")
model_list = ["bwgnn", "gin", "gat", "gcn"]
# data_list = ["cora","citeseer","pubmed","weibo","reddit"]
data_list = ["reddit"]
run_times = range(5)
hid_dim = 64
number_class = 2
data = None
record_string = ""
for data_name in data_list:
    if data_name == "cora" or data_name == "citeseer" or data_name == "pubmed":
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="min").dataset.to(device)
        for model_name in model_list:
            test_auc_list = []
            best_auc_list = []
            for run in run_times:
                model = Baseline(model_name, data.x.shape[1], hid_dim, number_class, data, verbose=0).to(device)
                test_auc,best_auc =  model.fit()
                best_auc_list.append(best_auc)
                test_auc_list.append(test_auc)
            print (f"Baseline {model_name}; dataset {data_name}; anomaly type min; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
            record_string = record_string + f"Baseline {model_name}; dataset {data_name}; anomaly type min; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n"
        
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="syn", anomaly_ratio=0.2).dataset.to(device)
        for model_name in model_list:
            test_auc_list = []
            best_auc_list = []
            for run in run_times:
                model = Baseline(model_name, data.x.shape[1], hid_dim, number_class, data, verbose=0).to(device)
                test_auc,best_auc =  model.fit()
                best_auc_list.append(best_auc)
                test_auc_list.append(test_auc)
            print (f"Baseline {model_name}; dataset {data_name}; anomaly type syn; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
            record_string = record_string + f"Baseline {model_name}; dataset {data_name}; anomaly type syn; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n"
    else:
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="syn").dataset.to(device)
        for model_name in model_list:
            test_auc_list = []
            best_auc_list = []
            for run in run_times:
                model = Baseline(model_name, data.x.shape[1], hid_dim, number_class, data, verbose=0).to(device)
                test_auc,best_auc =  model.fit()
                best_auc_list.append(best_auc)
                test_auc_list.append(test_auc)
            print (f"Baseline {model_name}; dataset {data_name}; anomaly type syn; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}")
            record_string = record_string + f"Baseline {model_name}; dataset {data_name}; anomaly type syn; test auc mean {np.array(test_auc_list).mean()}; test auc std {np.array(test_auc_list).std()}; best val auc mean {np.array(best_auc_list).mean()}\n"
with open("result/baseline_for_all_dataset.txt",'a') as f:
    f.write(record_string) 
