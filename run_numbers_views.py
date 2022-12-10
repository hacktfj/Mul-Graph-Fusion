import torch
import random
import math
import numpy as np
from dataset import pyg_dataset, pyg_to_dgl
from models.fusion_model import Fusion_model

torch.manual_seed(21)
np.random.seed(2)
device = torch.device("cuda")
hid_dim = 64
number_class = 2
loss_oriented = "label_oriented"

data_list = ["pubmed","amazon_computer","amazon_photo","weibo","books",]
run_times = range(5)
model_dict = ["gcn","gin","gat","bwgnn"]
# model_list_list = [["gcn"],["gat"],["gat","gin"],["bwgnn","gat"],["bwgnn","gin"],["bwgnn","gat","gin"],["gcn","gat","gin"],["bwgnn","gcn","gat","gin"]]

def random_sample():
    model_list_list = []
    model_list_list.append(random.sample(model_dict, k=1))
    model_list_list.append(random.sample(model_dict, k=2))
    model_list_list.append(random.sample(model_dict, k=3))
    model_list_list.append(random.sample(model_dict, k=4))
    return model_list_list

def index_times_mean(cur_list, k, interval = 4):
    nums = len(cur_list)
    result = 0
    for i in range(nums):
        if i % interval == k:
            result += cur_list[i]
    return result / (nums/interval)

def index_times_std(cur_list, k, interval = 4):
    nums = len(cur_list)
    result = 0
    mean = index_times_mean(cur_list, k, interval=interval)
    for i in range(nums):
        if i % interval == k:
            result += ((cur_list[i] - mean)*(cur_list[i] - mean))
    return math.sqrt(result / (nums/interval))

for data_name in data_list:
    if data_name in ["pubmed","amazon_computer","amazon_photo","weibo","books",]:
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="min").dataset.to(device)
        test_auc_list = []
        best_auc_list = []
        for run in run_times:
            model_list_list = random_sample()
            # print (model_list_list)
            for model_list in model_list_list:
                np.random.seed(2*run)
                model = Fusion_model(model_list, loss_oriented, data.x.shape[1], hid_dim, number_class, data, fusion_strategy=3, epochs_fit=75, verbose=0, anomaly_type="none").to(device)
                model.fit()
                test_auc,best_auc = model.no_fine_tuning()
                best_auc_list.append(best_auc)
                test_auc_list.append(test_auc)
        method_numbers = len(model_dict)
        for i in range(method_numbers):
            print (f"model number {i+1}; dataset {data_name}; anomaly type min; test auc mean {index_times_mean(test_auc_list,k=i)}; test auc std {index_times_std(test_auc_list, k=i)}; best val auc mean {index_times_mean(best_auc_list, k=i)}")
            with open("result/numbers_views_for_all_dataset.txt",'a') as f:
                f.write(f"model number {i+1}; dataset {data_name}; anomaly type min; test auc mean {index_times_mean(test_auc_list,k=i)}; test auc std {index_times_std(test_auc_list, k=i)}; best val auc mean {index_times_mean(best_auc_list, k=i)}\n") 
    
    if data_name in ["pubmed","amazon_computer","amazon_photo"]:
        data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3], anomaly_type="syn").dataset.to(device)
        test_auc_list = []
        best_auc_list = []
        for run in run_times:
            model_list_list = random_sample()
            # print (model_list_list)
            for model_list in model_list_list:
                np.random.seed(2*run)
                model = Fusion_model(model_list, loss_oriented, data.x.shape[1], hid_dim, number_class, data, fusion_strategy=3, epochs_fit=75, verbose=0, anomaly_type="none").to(device)
                model.fit()
                test_auc,best_auc = model.no_fine_tuning()
                best_auc_list.append(best_auc)
                test_auc_list.append(test_auc)
        method_numbers = len(model_dict)
        for i in range(method_numbers):
            print (f"model number {i+1}; dataset {data_name}; anomaly type syn; test auc mean {index_times_mean(test_auc_list,k=i)}; test auc std {index_times_std(test_auc_list, k=i)}; best val auc mean {index_times_mean(best_auc_list, k=i)}")
            with open("result/numbers_views_for_all_dataset.txt",'a') as f:
                f.write(f"model number {i+1}; dataset {data_name}; anomaly type syn; test auc mean {index_times_mean(test_auc_list,k=i)}; test auc std {index_times_std(test_auc_list, k=i)}; best val auc mean {index_times_mean(best_auc_list, k=i)}\n") 