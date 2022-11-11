import torch
from dataset import pyg_dataset, pyg_to_dgl
from models.baseline_model import Baseline, model_list
from models.single_model import Single_model, loss_oriented_list
from models.fusion_model import Fusion_model

torch.manual_seed(21)

data_name = "reddit"
data = pyg_dataset(dataset_name=data_name, dataset_spilt=[0.4,0.29,0.3]).dataset
model_list = ["bwgnn","gat","gin","gcn"]
model_list = ["bwgnn","gat","gin",]
# model_list = ["bwgnn","gat"]
# model_list = ["bwgnn"]
# model_list = ["bwgnn"]
# loss_oriented_list = [loss_oriented_list[-1]]
# for loss_oriented in loss_oriented_list:
#     for model_name in model_list:
#         model = Single_model(model_name, loss_oriented ,data.x.shape[1], 32, 2, data, epochs_fit=20, epochs_fine=30, verbose=0)
#         model.fit()
#         auc, best_auc = model.fine_tuning()
#         print (f"Dataset {data_name}, method {model_name}; loss oriented: {loss_oriented}; Test auc: {auc}; Best val auc: {best_auc}")

# for model_name in model_list:
#     # for fusion_strategy in range(4):
#     model = Single_model(model_name, "ssl_oriented" ,data.x.shape[1], 32, 2, data, epochs_fit=50, epochs_fine=50, verbose=0)
#     model.fit()
#     auc, best_auc = model.fine_tuning()
#     print (f"Dataset {data_name}, method {model_name}; fusion strategy: {3} loss oriented: ssl_oriented; Test auc: {auc}; Best val auc: {best_auc}")

model = Fusion_model(model_list, "ssl_oriented" ,data.x.shape[1], 64, 2, data, fusion_strategy=3, epochs_fit=30, epochs_fine=30, verbose=1)
model.fit()
auc, best_auc = model.fine_tuning()
print (f"Dataset {data_name}, method fusion model; fusion strategy: {3} loss oriented: ssl_oriented; Test auc: {auc}; Best val auc: {best_auc}")