import torch
from torch_geometric.datasets.tu_dataset import  TUDataset

class IndexedTUDataset(TUDataset):
    def __getitem__(self, idx):
        if isinstance(idx, int):
            return self._get_single(idx)
        elif isinstance(idx, (list, torch.Tensor)):
            return [self._get_single(i) for i in idx]
        else:
            raise TypeError("Index must be int, list or tensor")

    def _get_single(self, idx):
        if isinstance(idx, torch.Tensor):
            idx = idx.item()
        data = super().__getitem__(idx)
        data.GlobalGraphIdx = torch.tensor(idx, dtype=torch.long)
        return data

def DatasetVars(name = "PROTEINS",teacherName="GIN"):
    if teacherName=="GIN":
        if "IMDB-BINARY" in name:
            return {"num_classes": 2,"num_features": 65,
                    "teacherPath":"GNNmodels/IMDB-BINARY/GIN/hidden_64_dropout_0.0_num_layers_5/",
                    "hidden":64,
                    "dropout":0.0,
                    "num_layers":5}
        elif "IMDB-MULTI" in name:
            return {"num_classes": 3,"num_features": 65,
                    "teacherPath":"GNNmodels/IMDB-MULTI/GIN/hidden_64_dropout_0.5_num_layers_3/",
                    "hidden":64,
                    "dropout":0.5,
                    "num_layers":3}
        elif "REDDIT-BINARY" in name:
            return {"num_classes": 2,"num_features": 65,
                    "teacherPath":"GNNmodels/REDDIT-BINARY/GIN/hidden_16_dropout_0.0_num_layers_5/",
                    "hidden":16,
                    "dropout":0.0,
                    "num_layers":5}
        elif name == "PROTEINS":
            return {"num_classes": 2,"num_features": 3,
                    "teacherPath":"GNNmodels/PROTEINS/GIN/hidden_64_dropout_0.0_num_layers_5/",
                    "hidden":64,
                    "dropout":0.0,
                    "num_layers":5}
        elif name == "ENZYMES":
            return {"num_classes": 6,"num_features": 3,
                    "teacherPath":"GNNmodels/ENZYMES/GIN/GIN_hidden_dims_64_num_layers3_dropout_0.0_pooling_attention/",
                    "hidden":64,
                    "dropout":0.0,
                    "num_layers":3}
        elif name == "NCI1":
            return {"num_classes": 2,"num_features": 37,
                    "teacherPath":"GNNmodels/NCI1/GIN/hidden_64_dropout_0.0_num_layers_5/",
                    "hidden":64,
                    "dropout":0.0,
                    "num_layers":5}
        elif name=="BZR":
            return {"num_classes": 2,"num_features": 53,
                    "teacherPath":"GNNmodels/BZR/GIN/hidden_64_dropout_0.0_num_layers_3_hops_2_batch_size_32/",
                    "hidden":64,
                    "dropout":0.0,
                    "num_layers":3}
        elif name=="DD":
            return {"num_classes": 2,"num_features": 89,
                    "teacherPath":"GNNmodels/DD/GIN/hidden_64_dropout_0.0_num_layers_5_hops_2_batch_size_32/",
                    "hidden":64,
                    "dropout":0.0,
                    "num_layers":5}
        elif name=="MUTAG":
            return {"num_classes": 2,"num_features": 7,
                    "teacherPath":"GNNmodels/MUTAG/GIN/GIN_hidden_dims_64_num_layers3_dropout_0.0_pooling_attention/",
                    "hidden":64,
                    "dropout":0.0,
                    "num_layers":3}
        elif name=="COLLAB":
            return {"num_classes": 3, "num_features": 0,
                    "teacherPath": "GNNmodels/COLLAB/GIN/GIN_hidden_dims_64_num_layers3_dropout_0.0_pooling_attention/",
                    "hidden": 64,
                    "dropout": 0.0,
                    "num_layers": 3}

