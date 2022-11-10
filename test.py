from dataset import pyg_dataset, pyg_to_dgl
from models.baseline_model import Baseline, model_list
from models.single_model import Single_model

data = pyg_dataset(dataset_name="cora", dataset_spilt=[0.4,0.29,0.3]).dataset
# model_list = ["bwgnn","gat","gin","gcn"]
# for model_name in model_list:
model = Single_model("bwgnn", "label_oriented",data.x.shape[1], 64, 2, data, epochs=100, verbose=1)
model.fit()
auc, best_auc = model.fine_tuning()
print (f"Dataset reddit: ; Test auc: {auc}; Best val auc: {best_auc}")