import argparse
import warnings
import time
import torch
import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (EarlyStopping,
                                         ModelCheckpoint)
from torch_geometric.loader import DataLoader
from torch_geometric.datasets.tu_dataset import  TUDataset
from termcolor import colored
from Dataset import DatasetVars
from models import *
from pl_models import *
from train_utils import (get_free_gpu,
                         NaNStopping)
from data_utils import k_fold_without_validation


warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str,default="PROTEINS")
    parser.add_argument('--dataset_index',type=int,default=0)
    parser.add_argument('--Teacher_name',type=str,default="GIN")
    parser.add_argument('--hidden_dim', type=int,default=32)
    parser.add_argument('--num_layers', type=int,default=4)
    parser.add_argument('--pooling_method', type=str,default="attention")
    parser.add_argument('--max_epochs', type=int,default=300)
    parser.add_argument('--num_classes', type=int,default=10)
    parser.add_argument('--dropout', type=float,default=0.)
    parser.add_argument('--lr', type=float,default=8e-3)
    parser.add_argument('--weight_decay', type=float,default=1e-7)
    parser.add_argument('--batch_size', type=int,default=32)
    parser.add_argument('--device_id', type=int,default=-1)
    parser.add_argument('--seed', type=int,default=1)
    parser.add_argument('--numWorkers', type=int,default=2)
    parser.add_argument('--useLaPE',action='store_true',default=False)
    args = parser.parse_args()
    args = vars(args)
    print ("args: ", args)


    """get GPU"""
    if torch.cuda.is_available():
        device = get_free_gpu(5)
    else:
        device = None
    if args['device_id']>=0:
        device = args['device_id']
    print (colored(f"using device: {device}",'red','on_yellow'))
    args['device_id'] = device

    torch.manual_seed(args['seed'])

    random_id = int(np.random.choice(5000000)) # only for showing exp
    args["random_id"] = random_id
    dataset_name = args["dataset"]
    args['dataset_name'] = dataset_name

    s= time.time()

    print (colored(f"loading dataset: {dataset_name}",'red','on_yellow'))

    dataset_idx = args['dataset_index']
    dataset_path = f'data/withAdditionalAttr/'

    tu_dataset = TUDataset(root='data/raw/', name=dataset_name)
    train_indices, test_indices = k_fold_without_validation(tu_dataset, 10)
    train_indices = train_indices[dataset_idx]
    test_indices = test_indices[dataset_idx]
    train_dataset = tu_dataset[train_indices]
    test_dataset = tu_dataset[test_indices]


    t = DatasetVars(dataset_name)
    args['num_classes'] = t['num_classes']
    args["node_dim"] = t["num_features"]
    if args["useLaPE"]:
        args["node_dim"] = t["num_features"] + 20
    else:
        args["node_dim"] = t["num_features"]

    args['pyg_dataset']=test_dataset


    print (test_dataset[0])
    if args["Teacher_name"]=="GIN":
        gnn = GIN(**args)
        model_saving_path = f"GNNmodels/{dataset_name}/GIN/hidden_dims_{args['hidden_dim']}_num_layers{args['num_layers']}_dropout_{args['dropout']}_pooling_{args['pooling_method']}/fold_index_{args['dataset_index']}/"
        result_saving_path = f"GNN_results/{dataset_name}/GIN/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}/fold_index_{args['dataset_index']}/"

    elif args["Teacher_name"]=="GCN":
        gnn = GCN(**args)
        model_saving_path = f"GNNmodels/{dataset_name}/GCN/hidden_dims_{args['hidden_dim']}_num_layers{args['num_layers']}_dropout_{args['dropout']}_pooling_{args['pooling_method']}/fold_index_{args['dataset_index']}/"
        result_saving_path = f"GNN_results/{dataset_name}/GCN/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}/fold_index_{args['dataset_index']}/"


    args["model"] = gnn

    args["model_saving_path"] = model_saving_path
    args["result_saving_path"] = result_saving_path

    train_dloader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=True,
                                   num_workers=args["numWorkers"])
    test_dloader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=args["numWorkers"])
    args['train_loader'] = train_dloader

    args['test_loader'] = test_dloader

    pl_model = GNN_plModel(**args)
    print ("model loaded")

    """check point"""
    checkpoint_callback = ModelCheckpoint(
        monitor='valid_acc',
        dirpath=model_saving_path,
        filename='best_model-{epoch:02d}-{valid_acc:.4f}',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )
    """early stopping"""
    early_stop_callback = EarlyStopping(
       monitor='valid_acc',
       min_delta=0.00,
       patience=30,
       verbose=False,
       mode='max'
    )
    trainer = pl.Trainer(default_root_dir=f'saved_models/{dataset_name}/GIN/',max_epochs=120,accelerator='cpu' if device is None else 'gpu',devices=1 if device is None else [device],callbacks=[NaNStopping(),early_stop_callback,checkpoint_callback],enable_progress_bar=True,logger=False)
    trainer.fit(model=pl_model, train_dataloaders=train_dloader, val_dataloaders=test_dloader)
    torch.cuda.empty_cache()
