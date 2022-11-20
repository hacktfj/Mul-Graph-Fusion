import math
import time
from typing import Optional, Tuple, Union
import torch
import dgl
from dgl.data import FraudYelpDataset, FraudAmazonDataset
from torch_geometric.datasets import Planetoid, Amazon, Flickr, KarateClub
from torch_geometric.data import Data
import torch_geometric.transforms as T
from pygod.generator import gen_contextual_outliers,gen_structural_outliers
from sklearn.model_selection import train_test_split
from pygod.utils import load_data
import warnings
import pandas

dataset_ava_list = ["karate", "cora", "citeseer", "pubmed", "amazon_computer", "flickr", "Weibo", "reddit", "fraud_amazon", "fraud_yelp"]

class pyg_dataset():
    def __init__(self, dataset_name: str = "cora", dataset_spilt: Union[list,Tuple] = [0.6,0.2,0.2],anomaly_type: Optional[str] = None, anomaly_ratio: float=0.05, transform: bool = True) -> None:
        '''Dataset for symthetic and organic anomaly dataset. `The unified
        pyg_dataset` makes a piece of cake for handling and managing data.

        Args:
            anomaly_type: `None` means organic anomaly dataset. `"syn"` represent syntheic contextual and structural anomaly
                and `"min"` means the the min class as the anomalies
            dataset_spilt: list or tuple. The first term is `training ratio`, second `validation ratio`, last term `test ratio`. all sum equals to 1.
                stuctural and contexture are all 50% percent.
        
        Stats:
            Inject anomaliers dataset: Cora, Citeseer, Pubmed, Flickr and Amazon Computers
            Organic anomaliers dataset: Weibo, Reddit, FraudYelp and FraudAmazon. 
        '''
        self.dataset_name = dataset_name
        self.dataset_spilt = dataset_spilt
        self.anomaly_type = anomaly_type
        self.anomaly_ratio = anomaly_ratio
        self.node_nums = None
        self.anomaly_num = None
        self.dataset = None
        
        self.transform = T.NormalizeFeatures() if transform == True else None

        if self.dataset_name.lower() == "cora" or self.dataset_name.lower() == "citeseer" or self.dataset_name.lower() == "pubmed":
            self.dataset = Planetoid(f"./data/{self.dataset_name.lower()}",f"{self.dataset_name.lower()}",transform=self.transform)[0]
        elif self.dataset_name.lower() == "karate":
            karate = KarateClub().data
            position = 0
            self.dataset = Data(x=karate.x, edge_index=karate.edge_index,y=karate.y,train_mask=position,val_mask=position,test_mask=position)
        elif self.dataset_name.lower() == "amazon_computer":
            amazon = Amazon(f"./data/amazon",f"Computers",transform=self.transform)[0]
            position = 1
            self.dataset = Data(x=amazon.x, edge_index=amazon.edge_index,y=amazon.y,train_mask=position,val_mask=position,test_mask=position)
        elif self.dataset_name.lower() == "flickr":
            self.dataset = Flickr(f"./data/flickr",transform=self.transform)[0]
        elif self.dataset_name.lower() == "reddit" or self.dataset_name.lower() == "weibo":
            temp = load_data(f"{self.dataset_name.lower()}",f"./data/{self.dataset_name.lower()}")
            position = 1
            self.dataset = Data(x=temp.x, edge_index=temp.edge_index,y=torch.tensor(temp.y, dtype=torch.long),train_mask=position,val_mask=position,test_mask=position)
        elif self.dataset_name.lower() == "fraud_yelp":
            graph = FraudYelpDataset("data/fraud_yelp")[0]
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
            self.dataset = Data(x=graph.ndata['feature'],y=graph.ndata["label"].to(torch.long),edge_index=torch.stack(graph.edges(),dim=0),train_mask=graph.ndata["train_mask"].to(torch.bool),val_mask=graph.ndata["val_mask"].to(torch.bool),test_mask=graph.ndata["test_mask"].to(torch.bool))
        elif self.dataset_name.lower() == "fraud_amazon":
            graph = FraudAmazonDataset("data/fraud_amazon")[0]
            graph = dgl.to_homogeneous(graph, ndata=['feature', 'label', 'train_mask', 'val_mask', 'test_mask'])
            graph = dgl.add_self_loop(graph)
            self.dataset = Data(x=graph.ndata['feature'],y=graph.ndata["label"].to(torch.long),edge_index=torch.stack(graph.edges(),dim=0),train_mask=graph.ndata["train_mask"].to(torch.bool),val_mask=graph.ndata["val_mask"].to(torch.bool),test_mask=graph.ndata["test_mask"].to(torch.bool))
        else:
            warnings.WarningMessage(f"Dataset wrong, {self.dataset_name}s are not considered in the experiment; \
                {dataset_ava_list} is availble")

        self.node_nums = self.dataset.num_nodes
        num_nodes = self.node_nums
        idx_train, idx_test = train_test_split(list(range(num_nodes)),train_size=self.dataset_spilt[0])
        idx_val, idx_test = train_test_split(idx_test,train_size=self.dataset_spilt[1]/(1 - self.dataset_spilt[0]))
        idx_test, _ = train_test_split(idx_test,train_size=self.dataset_spilt[2]/(1 - self.dataset_spilt[0] - self.dataset_spilt[1]))
        
        train_mask = torch.zeros([num_nodes]).bool()
        val_mask = torch.zeros([num_nodes]).bool()
        test_mask = torch.zeros([num_nodes]).bool()
        train_mask[idx_train] = 1
        val_mask[idx_val] = 1
        test_mask[idx_test] = 1
        self.dataset.train_mask = train_mask
        self.dataset.val_mask = val_mask
        self.dataset.test_mask = test_mask

        if anomaly_type is not None:
            if anomaly_type == "syn":
                warnings.warn(f"Anomaly is syn and anomaly rate is conformed to the aforementioned setting")
                print (f"anomaly syntheic is on processing")
                start = time.time()
                nums = self.anomaly_ratio*num_nodes
                clique = int (math.sqrt(nums/2))
                self.dataset, yc = gen_contextual_outliers(self.dataset,n=int(nums/2),k=50)
                self.dataset, ys = gen_structural_outliers(self.dataset,m=clique,n=clique)
                self.dataset.y = yc.logical_or(ys).to(torch.long)
                end = time.time()
                self.anomaly_num = nums
                print (f"using {(end-start):.2f} seconds")
            elif anomaly_type == "min":
                warnings.warn(f"Anomaly is min class of dataset and anomaly rate is not conformed to setting")
                y_pan = pandas.Series(self.dataset.y)
                min_class = y_pan.value_counts().argmin()
                min_num = y_pan.value_counts().min()
                self.dataset.y = torch.tensor(y_pan.apply(lambda x: 1 if x == min_class else 0).to_numpy(),dtype = torch.long)
                self.anomaly_ratio = min_num/num_nodes
                self.anomaly_num = min_num
        else:
            warnings.warn(f"Anomaly is organic and anomaly rate is not conformed to setting")
            self.anomaly_ratio = self.dataset.y.sum()/num_nodes
            self.anomaly_type = "useless due to organic anomaly"
            self.anomaly_num = self.dataset.y.sum()
            
        
    def __str__(self) -> str:
        return f"dataset_name: {self.dataset_name};\n dataset_spilt: {self.dataset_spilt};\n anomaly_type: {self.anomaly_type};\n\
 anomaly_ratio: {self.anomaly_ratio*100:.2f}%;\n anomaly_nums: {self.anomaly_num};\n node_nums: {self.node_nums};\n transform: {self.transform}"
    
    def __repr__(self) -> str:
        return f"dataset_name: {self.dataset_name};\n dataset_spilt: {self.dataset_spilt};\n anomaly_type: {self.anomaly_type};\n\
 anomaly_ratio: {self.anomaly_ratio*100:.2f}%;\n anomaly_nums: {self.anomaly_num};\n node_nums: {self.node_nums};\n transform: {self.transform}"

def pyg_to_dgl(pyg_dataset):
    graph_dgl = dgl.DGLGraph().to(pyg_dataset.x.device)
    graph_dgl.add_nodes(pyg_dataset.x.shape[0])
    graph_dgl.ndata['x'] = pyg_dataset.x
    graph_dgl.add_edges(*pyg_dataset.edge_index)
    return graph_dgl

# if __name__ == "main":
#     dataset_list = ["Cora","Citeseer","Pubmed","Amazon_computer","Flickr","Weibo","Reddit", "Fraud_amazon", "Fraud_yelp"]
#     for data_name in dataset_list:
#         dataset = pyg_dataset(data_name,dataset_spilt=[0.6,0.2,0.2], anomaly_ratio=0.05)
#         print (dataset.dataset.edge_index.shape)
#         print (dataset.dataset.x.shape)