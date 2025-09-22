import warnings
import glob
import argparse
from models import *
from pl_models import GEMDistill_plModel
from torch_geometric.datasets.tu_dataset import  TUDataset
from torch_geometric.transforms import Compose
from torch_geometric.loader import DataLoader
from Dataset import IndexedTUDataset,DatasetVars
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from Transforms import (
                   RandomPathTransform,
                   TeacherModelTransform,)
from train_utils import (get_free_gpu,NaNStopping,)
from data_utils import (k_fold_without_validation)




warnings.filterwarnings('ignore')

torch.cuda.empty_cache()

parser = argparse.ArgumentParser(description='GEMDistill')

"""add arguments"""
parser.add_argument('--dataset', type=str,default="PROTEINS")
parser.add_argument('--dataset_index', type=int,default=0)
parser.add_argument('--seed', type=int,default=1)
parser.add_argument('--lr', type=float,default=8e-3)
parser.add_argument('--lr_patience', type=int,default=30)
parser.add_argument('--weight_decay', type=float,default=0.0)
parser.add_argument('--batch_size', type=int,default=32)
parser.add_argument('--device_id', type=int,default=-1)
parser.add_argument('--max_epochs', type=int,default=300)
parser.add_argument('--use_edge_feats',action='store_false',default=True)
parser.add_argument('--num_hops', type=int,default=1)
parser.add_argument('--studentModelName', type=str,default="MLP")
parser.add_argument('--hidden_dim', type=int,default=64)
parser.add_argument('--num_layers', type=int,default=3)
parser.add_argument('--out_dim', type=int,default=32)
parser.add_argument('--num_classes', type=int,default=10)
parser.add_argument('--dropout', type=float,default=0.)
parser.add_argument('--pooling_method', type=str,default='attention')
parser.add_argument('--teacherModelName', type=str,default="GIN")
parser.add_argument('--use_KD',action='store_true',default=False)
parser.add_argument('--KD_name', type=str,default="NULL")
parser.add_argument('--numWorkers', type=int,default=2)
parser.add_argument('--pathLength', type=int,default=8)

"""for additional attributes"""
parser.add_argument('--useLaPE',action='store_true',default=False)
parser.add_argument('--use_AdditionalAttr',action='store_false',default=True)

parser.add_argument('--useNodeKD',action='store_true',default=False)
parser.add_argument('--NodeKDReg', type=float,default=0.2)
parser.add_argument('--useSubgraphKD',action='store_true',default=False)
parser.add_argument('--SubgraphKDReg', type=float,default=0.1)
parser.add_argument('--communityAlgo', type=str,default="louvain")
parser.add_argument('--useGraphPooling',action='store_true',default=False)
parser.add_argument('--graphPoolingReg', type=float,default=10)
parser.add_argument('--useGraphKD',action='store_true',default=False)
parser.add_argument('--graphKDReg', type=float,default=10)
parser.add_argument('--kd_order', type=str, default='G,S,N')

"""Ensemble arguments"""
#students model = GEMDistill
parser.add_argument("--lamb", type=float, default=0.1)
parser.add_argument("--tau", type=float, default=1.0)
parser.add_argument('--useDifferentHidden',action='store_true',default=False)
parser.add_argument('--hidden_dim_G', type=int,default=32)
parser.add_argument('--hidden_dim_S', type=int,default=32)
parser.add_argument('--hidden_dim_N', type=int,default=32)
parser.add_argument("--beta", type=float, default=1)

"""GraphKD arguments"""
parser.add_argument("--K_neighbors", type=int, default=5)
parser.add_argument("--G_alpha", type=float, default=5)
parser.add_argument("--G_beta", type=float, default=0.25)


args = parser.parse_args()
args = vars(args)
for k in args:
    print(colored.colored(f"{k}: {args[k]}", 'red', 'on_white'))


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

model_saving_path = f"best_models/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}/fold_index_{args['dataset_index']}/"
loss_saving_path = f"Los0sCurves/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}_KD_method_{args['KD_name']}/fold_index_{args['dataset_index']}/"
args["model_saving_path"] = model_saving_path
args["loss_saving_path"] = loss_saving_path

#Use KD
result_saving_path = f"KD_results/{dataset_name}/{args['studentModelName']}_{args['teacherModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_hops_{args['num_hops']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}_KD_method_{args['KD_name']}_useHardAware_{args['useGraphKD']}_neighbors_{args['K_neighbors']}/fold_index_{args['dataset_index']}/"
args["model_saving_path"] = model_saving_path
args["result_saving_path"] = result_saving_path

"""load model"""
if args['studentModelName'] == "MLP":
    print (colored.colored(f"using MLP as student model",'red','on_yellow'))
    student_model = MLP(**args)
    args["model"] = student_model

elif args['studentModelName'] == "GEMDistill":
    print(colored.colored(f"using GEMDistill as student model",'red','on_yellow'))
    tmp_dataset = TUDataset(root=f'data/raw/',name=dataset_name)
    args["pyg_dataset"] = tmp_dataset
    student_model = GEMDistill(**args)
    args["model"] = student_model

#use KD load pretained teacher model
raw_dataset = TUDataset(root=f"data/raw/",name=dataset_name)
t = DatasetVars(dataset_name,teacherName=args['teacherModelName'])
hidden,dropout,num_layers,num_classes = t["hidden"],t["dropout"],t["num_layers"],t["num_classes"]
args["teacher_hidden_dim"] = hidden
if args['teacherModelName'] == "GIN":
    teacherModel = GIN(hidden_dim=hidden, dropout=dropout, num_layers=num_layers, num_classes=num_classes,
                       pyg_dataset=raw_dataset, dataset_name=dataset_name)
    TmodelPath = t["teacherPath"] + f"fold_index_{args['dataset_index']}/"
elif args['teacherModelName'] == "GCN":
    teacherModel = GCN(hidden_dim=hidden, dropout=dropout, num_layers=num_layers, num_classes=num_classes,
                       pyg_dataset=raw_dataset, pooling_method="attention", dataset_name=dataset_name)
    TmodelPath = t["teacherPath"] + f"fold_index_{args['dataset_index']}/"
    print(colored.colored(f"teacherModel:{TmodelPath}", 'red'))

print (TmodelPath)
TmodelPath = glob.glob(os.path.join(TmodelPath,'*.pt'))[0]
print (f"teacherModel path:{TmodelPath}")
if device>=0:
    teacherModel.load_state_dict(torch.load(TmodelPath,map_location=torch.device(f"cuda:{device}")))
else:
    assert "No Free GPUs"
teacherModel.eval()


"""dataset transforms"""
transforms = []

if args['useRandomWalkConsistency']:
    transforms.append(RandomPathTransform(path_length=args["pathLength"]))

transforms.append(TeacherModelTransform(teacherModel, use_clustering=args["useSubgraphKD"],
                                            cluster_algo=args["communityAlgo"]))
transforms = Compose(transforms)


"""get train val test dataset"""
s = time.time()
print (colored.colored(f"loading dataset: {dataset_name}",'red','on_yellow'))

dataset_idx = args['dataset_index']
dataset_path = f'data/withAdditionalAttr/' if args['studentModelName'] != "GA-MLP" else f'data/GA_MLP/'
tu_dataset = IndexedTUDataset(root=dataset_path, name=dataset_name, transform=transforms)

args["full_dataset"] = tu_dataset

train_indices, test_indices = k_fold_without_validation(tu_dataset,10)
train_indices = train_indices[dataset_idx]
test_indices = test_indices[dataset_idx]
train_dataset = tu_dataset[train_indices]
test_dataset = tu_dataset[test_indices]

args[("train_dataset")] = train_dataset


t = time.time()
print (t-s, f"seconds used to load dataset {dataset_name}")
print (test_dataset[0])

train_dloader = DataLoader(train_dataset,batch_size=args['batch_size'],shuffle=True,num_workers=args["numWorkers"])#modify shuffle False
test_dloader = DataLoader(test_dataset,batch_size=100,shuffle=False,num_workers=args["numWorkers"])
args['test_loader'] = test_dloader

args['parsed_kd_order'] = [item.strip().upper() for item in args['kd_order'].split(',')]

pl_model = GEMDistill_plModel(**args)

"""early stopping"""
early_stop_callback = EarlyStopping(
   monitor='valid_acc',
   min_delta=0.00,
   patience=60,
   verbose=False,
   mode='max'
)

trainer = pl.Trainer(default_root_dir=f'saved_models/{dataset_name}/{args["studentModelName"]}/',max_epochs=args["max_epochs"],accelerator='cpu' if device is None else 'gpu',devices=1 if device is None else [device],enable_progress_bar=True,logger=False,callbacks=[NaNStopping(),early_stop_callback],enable_checkpointing=False)
trainer.fit(model=pl_model, train_dataloaders=train_dloader, val_dataloaders=test_dloader)
torch.cuda.empty_cache()


