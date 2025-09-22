import warnings
import argparse
from torch_geometric.datasets.tu_dataset import  TUDataset
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import EarlyStopping
import pytorch_lightning as pl
from models import *
from Dataset import DatasetVars

from Transforms import DropEdge
from data_utils import  k_fold_without_validation
from train_utils import (NaNStopping,
                         get_free_gpu)
warnings.filterwarnings("ignore")


# provide a parser for the command line
parser = argparse.ArgumentParser()
# add augument for string arguments

parser.add_argument('--use_AdditionalAttr', action='store_true', default=False)
parser.add_argument('--use_KD', action='store_true', default=False)
parser.add_argument('--useLaPE', action='store_true', default=False)
parser.add_argument('--dataset', type=str, default="IMDB-MULTI")
parser.add_argument('--hidden_dim', type=int, default=32)
parser.add_argument('--out_dim', type=int, default=32)
parser.add_argument('--num_classes', type=int, default=10)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--first_layer_dropout', type=float, default=0.)
parser.add_argument('--pooling_method', type=str, default='attention')
parser.add_argument('--lr', type=float, default=8e-3)
parser.add_argument('--lr_patience', type=int, default=30)
parser.add_argument('--weight_decay', type=float, default=1e-7)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--device_id', type=int, default=-1)
parser.add_argument('--max_epochs', type=int, default=150)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--dataset_index', type=int,default=0)
parser.add_argument('--numWorkers', type=int,default=2)
parser.add_argument('--teacherModelName', type=str,default="GIN")
parser.add_argument('--studentModelName', type=str,default="MLP")
parser.add_argument('--num_layers', type=int,default=4)
parser.add_argument('--gamma', type=float,default=0.5)
parser.add_argument('--num_hops', type=int,default=1)


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
    dataset_path = f'data/withAdditionalAttr/'
else:
    args["node_dim"] = t["num_features"]
    dataset_path = f'data/raw/'

model_saving_path = f"best_models/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_batch_size_{args['batch_size']}_KDname_NoKD/fold_index_{args['dataset_index']}/"
loss_saving_path = f"Los0sCurves/{dataset_name}/{args['studentModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}_KD_method_NoKD/fold_index_{args['dataset_index']}/"
args["model_saving_path"] = model_saving_path
args["loss_saving_path"] = loss_saving_path

args["model_saving_path"] = None
args["result_saving_path"] = f"KD_results/{dataset_name}/{args['studentModelName']}_{args['teacherModelName']}/hidden_{args['hidden_dim']}_dropout_{args['dropout']}_num_layers_{args['num_layers']}_batch_size_{args['batch_size']}_useLaPE_{args['useLaPE']}_noKD/fold_index_{args['dataset_index']}/"

print(colored.colored(f"using MLP as student model", 'red', 'on_yellow'))
student_model = MLP_Ada(**args)
args["model"] = student_model

s= time.time()
print (colored.colored(f"loading dataset: {dataset_name}",'red','on_yellow'))

dataset_idx = args['dataset_index']


tu_dataset = TUDataset(root=dataset_path, name=dataset_name)

train_indices, test_indices = k_fold_without_validation(tu_dataset,10)
train_indices = train_indices[dataset_idx]
test_indices = test_indices[dataset_idx]
train_dataset = tu_dataset[train_indices]
test_dataset = tu_dataset[test_indices]


t = time.time()
print (t-s, f"seconds used to load dataset {dataset_name}")


# for real-use
train_dloader = DataLoader(train_dataset,batch_size=args['batch_size'],shuffle=True,num_workers=args["numWorkers"])
test_dloader = DataLoader(test_dataset,batch_size=100,shuffle=False,num_workers=args["numWorkers"])
args['test_loader'] = test_dloader
args['train_loader'] = train_dloader

pl_model = pl_models(**args)


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