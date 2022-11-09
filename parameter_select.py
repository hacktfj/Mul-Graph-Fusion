from typing import Union


defaule_par_dict = dict({
    "lr": 1e-3,
    "hiddle_layer": 2,
    "hiddle_dim": 16,
    "weight_decay": 5e-4,
    "dropout": 0.2,
    "train_radio": 0.4
})
par_dict = dict({
    "lr": [1e-4, 1e-3 ,1e-2],
    "hiddle_layer": [1,2,3],
    "hiddle_dim": [8, 16, 32, 64],
    "weight_decay": [5e-4, 5e-3],
    "dropout": [0.2 ,0.5 ,0.8],
    "train_radio": [0.1, 0.4, 0.7]
})
class Parameter:
    # parameter too many to select, use the control varible method to choose.
    def __init__(self, lr: Union[list,tuple] = [1e-4, 1e-3 ,1e-2], hiddle_layer: Union[list,tuple] = [1,2,3], hiddle_dim: Union[list,tuple] = [8, 16, 32, 64], weight_decay: Union[list,tuple] = [5e-4, 5e-3],dropout: Union[list,tuple] = [0.2 ,0.5 ,0.8], train_radio: Union[list,tuple] = [0.1, 0.4, 0.7]) -> None:
        """Include the common-used parameter for GNN, we can also change it whatever we want.
            Args:
                'train radio': the rest is separated equally for validation and test.
                3 * 3 * 4 * 2 * 3 * 3 = 648 次
        """
        self.hiddle_layer = hiddle_layer
        self.hiddle_dim = hiddle_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout
        self.train_radio = train_radio


    def __str__(self) -> str:
        print (f"Parameter settting:\n hiddle_layer: {self.hiddle_layer};\n hiddle_dim: {self.hiddle_dim};\n learning rate: {self.lr};\
\n Adam weight decay: {self.weight_decay};\n dropout: {self.dropout};\n train ratio: {self.train_radio}")

    def __repr__(self) -> str:
        print (f"Parameter settting:\n hiddle_layer: {self.hiddle_layer};\n hiddle_dim: {self.hiddle_dim};\n learning rate: {self.lr};\
\n Adam weight decay: {self.weight_decay};\n dropout: {self.dropout};\n train ratio: {self.train_radio}")
