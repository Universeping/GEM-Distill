import warnings
import argparse
import torch
import time
import os
import numpy as np
from termcolor import colored
from glob import glob
from torch_geometric.datasets.tu_dataset import  TUDataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from models import *
from pl_models import *
from Dataset import DatasetVars
from Transforms import (
                   DropEdge,
                   RandomPathTransform,
                   TeacherModelTransform,)

from data_utils import k_fold_without_validation

from train_utils import (get_free_gpu,NaNStopping)

warnings.filterwarnings("ignore")
torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='MuGSI')

"""add arguments"""

parser.add_argument('--dataset', type=str,default="IMDB-MULTI")
parser.add_argument('--dataset_index', type=int,default=0)
parser.add_argument('--num_hops', type=int,default=1)
parser.add_argument('--use_edge_feats',action='store_false',default=True)
parser.add_argument('--hidden_dim', type=int,default=32)
parser.add_argument('--num_layers', type=int,default=4)
parser.add_argument('--out_dim', type=int,default=32)
parser.add_argument('--num_classes', type=int,default=10)
parser.add_argument('--dropout', type=float,default=0.)
parser.add_argument('--first_layer_dropout', type=float,default=0.)
parser.add_argument('--pooling_method', type=str,default='sum')
parser.add_argument('--Khop_pooling_method', type=str,default='attention')
parser.add_argument('--lr', type=float,default=8e-3)
parser.add_argument('--lr_patience', type=int,default=30)
parser.add_argument('--gamma', type=float,default=0.5)
parser.add_argument('--weight_decay', type=float,default=0.0)
parser.add_argument('--batch_size', type=int,default=32)
parser.add_argument('--device_id', type=int,default=-1)
parser.add_argument('--max_epochs', type=int,default=300)
parser.add_argument('--numWorkers', type=int,default=2)

parser.add_argument('--use_KD',action='store_true',default=False)
parser.add_argument('--KD_name', type=str,default="NULL")
parser.add_argument('--studentModelName', type=str,default="MLP")
parser.add_argument('--teacherModelName', type=str,default="GIN")
parser.add_argument('--teacherFileIndex', type=int,default=0)
parser.add_argument('--useSoftLabel',action='store_true',default=False)
parser.add_argument('--useNodeSim',action='store_true',default=False)
parser.add_argument('--useNodeFeatureAlign',action='store_true',default=False)
parser.add_argument('--useClusterMatching',action='store_true',default=False)
parser.add_argument('--clusterAlgo', type=str,default="louvain")
parser.add_argument('--useRandomWalkConsistency',action='store_true',default=False)
parser.add_argument('--useDropoutEdge',action='store_true',default=False)
parser.add_argument('--useMixUp',action='store_true',default=False)
parser.add_argument('--useGraphPooling',action='store_true',default=False)
parser.add_argument('--useNCE',action='store_true',default=False)

parser.add_argument('--softLabelReg', type=float,default=1e-1)
parser.add_argument('--nodeSimReg', type=float,default=1e-3)
parser.add_argument('--NodeFeatureReg', type=float,default=1e-1)
parser.add_argument('--ClusterMatchingReg', type=float,default=1e-1)
parser.add_argument('--RandomWalkConsistencyReg', type=float,default=1e-1)
parser.add_argument('--graphPoolingReg', type=float,default=1e-1)
parser.add_argument('--pathLength', type=int,default=8)
parser.add_argument('--MixUpReg', type=float,default=0.)
parser.add_argument('--NCEReg', type=float,default=0.)
parser.add_argument('--seed', type=int,default=1)

# for additional attributes
parser.add_argument('--use_AdditionalAttr',action='store_false',default=True)
parser.add_argument('--useLaPE',action='store_true',default=False)

args = parser.parse_args()
args = vars(args)
for k in args:
    print (colored.colored(f"{k}: {args[k]}",'red','on_white'))


"""get GPU"""
if torch.cuda.is_available():
    device = get_free_gpu(9) # more than 10GB free GPU memory
else:
    device = None
if args['device_id']>=0:
    device = args['device_id']
print (colored.colored(f"using device: {device}",'red','on_yellow'))
args['device_id'] = device

torch.manual_seed(args["seed"])

random_id = int(np.random.choice(5000000)) # only for showing exp
args["random_id"] = random_id
dataset_name = args['dataset']
args['dataset_name'] = dataset_name


t = DatasetVars(dataset_name,teacherName=args['teacherModelName'])
args['num_classes'] = t['num_classes']
if args["useLaPE"]:
    args["node_dim"] = t["num_features"]+20
else:
    args["node_dim"] = t["num_features"]

model_saving_path = f"best_models/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_KDname_{args['KD_name']}/fold_index_{args['dataset_index']}/"
loss_saving_path = f"Los0sCurves/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}_KD_method_{args['KD_name']}/fold_index_{args['dataset_index']}/"
args["model_saving_path"] = model_saving_path
args["loss_saving_path"] = loss_saving_path


if args['use_KD']:
    result_saving_path = f"KD_results/{dataset_name}/{args['studentModelName']}_{args['teacherModelName']}/pathLength_{args['pathLength']}_hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}_nodeSim_{args['nodeSimReg']}_KD_method_{args['KD_name']}/fold_index_{args['dataset_index']}/"
    args["model_saving_path"] = model_saving_path
    args["result_saving_path"] = result_saving_path

else:
    if args["studentModelName"] == "GIN" or args["studentModelName"] == "GCN":
        args["model_saving_path"] = model_saving_path
    else:
        args["model_saving_path"] = None
    args["result_saving_path"] = f"KD_results/{dataset_name}/{args['studentModelName']}_{args['teacherModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}_noKD/fold_index_{args['dataset_index']}/"

"""load model"""

if args['studentModelName'] == "MLP":
    print (colored.colored(f"using MLP as student model",'red','on_yellow'))
    student_model = MLP(**args)
    args["model"] = student_model

elif args['studentModelName'] == "GIN":
    print (colored.colored(f"using GIN as student model",'red','on_yellow'))
    tmp_dataset = TUDataset(root=f'data/raw/',name=dataset_name)
    args["pyg_dataset"] = tmp_dataset
    student_model = GIN(**args)
    args["model"] = student_model

elif args['studentModelName'] == "GCN":
    print (colored.colored(f"using GCN as student model",'red','on_yellow'))
    tmp_dataset = TUDataset(root=f'data/raw/',name=dataset_name)
    args["pyg_dataset"] = tmp_dataset
    student_model = GCN(**args)
    args["model"] = student_model




if args["use_KD"]:
    raw_dataset = TUDataset(root=f"data/raw/",name=dataset_name)
    t = DatasetVars(dataset_name,teacherName=args['teacherModelName'])
    hidden,dropout,num_layers,num_classes = t["hidden"],t["dropout"],t["num_layers"],t["num_classes"]
    args["teacher_hidden_dim"] = hidden
    if args['teacherModelName'] == "GIN":
        teacherModel = GIN(hidden_dim=hidden,dropout=dropout,num_layers=num_layers,num_classes=num_classes,pyg_dataset=raw_dataset,dataset_name=dataset_name)
        TmodelPath = t["teacherPath"] + f"fold_index_{args['dataset_index']}/"
    elif args['teacherModelName'] == "GCN":
        teacherModel = GCN(hidden_dim=hidden,dropout=dropout,num_layers=num_layers,num_classes=num_classes,pyg_dataset=raw_dataset,pooling_method="attention",dataset_name=dataset_name)
        TmodelPath = t["teacherPath"] + f"fold_index_{args['dataset_index']}/"
        print (colored.colored(f"teacherModel:{TmodelPath}",'red'))

    print (TmodelPath)
    TmodelPath = glob(os.path.join(TmodelPath,'*.pt'))[0]
    print (f"teacherModel path:{TmodelPath}")
    if device>=0:
        # qq = torch.load(TmodelPath,map_location=torch.device(f"cuda:{device}"))
        # print (qq["pred.weight"].shape)
        teacherModel.load_state_dict(torch.load(TmodelPath,map_location=torch.device(f"cuda:{device}")))
    else:
        assert "No Free GPUs"
    teacherModel.eval()

    """set transforms"""
    transforms = []
    if args['useDropoutEdge']:
        transforms.append(DropEdge(p=0.07))
    if args['useRandomWalkConsistency']:
        transforms.append(RandomPathTransform(path_length=args["pathLength"]))
    if args['useMixUp']:
        args['teacherModel']=teacherModel

    transforms.append(TeacherModelTransform(teacherModel, use_clustering=args["useClusterMatching"],
                                            cluster_algo=args["clusterAlgo"]))
    transforms = Compose(transforms)

"""train"""
s= time.time()
print (colored.colored(f"loading dataset: {dataset_name}",'red','on_yellow'))

dataset_idx = args['dataset_index']
dataset_path = f'data/withAdditionalAttr/'

if args["use_KD"]:
    tu_dataset = TUDataset(root=dataset_path,name=dataset_name,transform=transforms)
else:
    tu_dataset = TUDataset(root='data/raw/',name=dataset_name)
train_indices, test_indices = k_fold_without_validation(tu_dataset,10)
train_indices = train_indices[dataset_idx]
test_indices = test_indices[dataset_idx]
train_dataset = tu_dataset[train_indices]
test_dataset = tu_dataset[test_indices]



t = time.time()
print (t-s, f"seconds used to load dataset {dataset_name}")


# for real-use
train_dloader = DataLoader(train_dataset,batch_size=args['batch_size'],shuffle=True,num_workers=args["numWorkers"])
#valid_dloader = DataLoader(val_dataset,batch_size=500,shuffle=False, num_workers=4 if torch.cuda.is_available() else 0)
test_dloader = DataLoader(test_dataset,batch_size=100,shuffle=False,num_workers=args["numWorkers"])
args['test_loader'] = test_dloader
args['train_loader'] = train_dloader


pl_model = pl_Model(**args)

"""early stopping"""
early_stop_callback = EarlyStopping(
   monitor='valid_acc',
   min_delta=0.00,
   patience=60,
   verbose=False,
   mode='max'
)

if args["use_KD"]:
    trainer = pl.Trainer(default_root_dir=f'saved_models/{dataset_name}/{args["studentModelName"]}/',max_epochs=args["max_epochs"],accelerator='cpu' if device is None else 'gpu',devices=1 if device is None else [device],enable_progress_bar=True,logger=False,callbacks=[NaNStopping(),early_stop_callback],enable_checkpointing=False)
    trainer.fit(model=pl_model, train_dataloaders=train_dloader, val_dataloaders=test_dloader)
else:
    trainer = pl.Trainer(default_root_dir=f'saved_models/{dataset_name}/{args["studentModelName"]}/',max_epochs=args["max_epochs"],accelerator='cpu' if device is None else 'gpu',devices=1 if device is None else [device],enable_progress_bar=True,logger=False,callbacks=[NaNStopping(),early_stop_callback],enable_checkpointing=False)
    trainer.fit(model=pl_model, train_dataloaders=train_dloader, val_dataloaders=test_dloader)
results = trainer.test(model=pl_model, dataloaders=test_dloader)
torch.cuda.empty_cache()