import torch
import torch.nn as nn
import time
import tqdm
import termcolor as colored
import torch.nn.functional as F
import torch.optim as optim
import concurrent.futures
import copy
import numpy as np
from collections import defaultdict
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torch.nn import (Sequential,
                      Linear,
                      ReLU,
                      BatchNorm1d as BN,)

from torch_geometric.nn import (global_add_pool,
                                global_mean_pool,
                                AttentionalAggregation)
from torch_geometric.data import Data
from torch_geometric.nn import (GINConv,
                                GINEConv,
                                GCNConv,
                                MessagePassing)

from torch_geometric.utils import (add_self_loops,
                                   degree)
from train_utils import (calc_KL_divergence,
                         calc_node_similarity,
                         nodeFeatureAlignment,
                         calculate_conditional_probabilities,
                         calculate_kl_loss,
                         fast_cosine_sim_matrix,
                         calculate_conditional_probabilities_gemd,
                         calculate_kl_loss_check,
                         extract_teacher_emb,
                         extract_teacher_pred,
                         extract_graph_subset_emb,
                         extract_graph_subset,
                         extract_graph_vitual_edge_KNN,
                         cal_HG_graph_embloss,
                         compute_entropy)

from data_utils import (save_model_weights,
                        write_results_to_file)


def IniGEMWeights(count,dataset):
    alpha_vals = torch.ones(count)
    graph_weights = torch.ones(len(dataset))
    graph_weights /=  graph_weights.sum()
    return alpha_vals, graph_weights

def GEMD_update_weights(
       combine_loss ,logits_s, logits_t, graphs_weights,idx , beta, Classes
):
    criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)
    with torch.no_grad():
        out_s = logits_s.log_softmax(dim=1)
        out_t = logits_t.log_softmax(dim=-1)
        kl_loss = criterion(out_s, out_t).sum(1)
        loss = kl_loss + combine_loss
        errors = 1 - torch.exp(-beta * loss)  # torch.sigmoid(loss)
        #errors [batch_size] to 1
        error = torch.sum(graphs_weights[idx] * errors) / torch.sum(graphs_weights[idx])
        error = error + 1e-16
        alpha = max( (torch.log((1 - error) / error )), 1e-16)+ torch.log(torch.tensor(Classes - 1.0, device=error.device))
        #If an error occurs, please replace it with this line of code
        #alpha = max((torch.log((1 - error) / error)), 1e-16)
        graphs_weights[idx] = graphs_weights[idx] * torch.exp(alpha * errors)
        graphs_weights[idx] /= graphs_weights[idx].sum()

    return graphs_weights, alpha

class GEMDistill_plModel(pl.LightningModule):
    def __init__(self, model, lr,
                 weight_decay, lr_patience ,
                 model_saving_path,
                 test_loader=None,use_KD= False,**kwargs):
        super(GEMDistill_plModel, self).__init__()
        """tarining and validation parameters"""
        self.save_hyperparameters(ignore=["model", "val_loader", "test_loader", "teacherModel", "pyg_dataset", "full_dataset", "train_dataset"])

        self.model = model
        self.lr = lr
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.num_classes = kwargs["num_classes"]
        self.dataset = kwargs["dataset"]
        self.fold_index = kwargs['dataset_index']
        self.batch_size = kwargs['batch_size']
        self.teacherModel = kwargs.get("teacherModel", None)

        self.acc = Accuracy(top_k=1, task='multiclass', num_classes=kwargs['num_classes'])
        self.test_dataset = test_loader

        self.dropout = kwargs['dropout']
        self.first_layer_dropout = 0.
        self.hidden_dim = kwargs['hidden_dim']
        self.random_id = kwargs['random_id']
        self.model_saving_path = model_saving_path
        self.loss_saving_path = kwargs["loss_saving_path"]
        self.result_saving_path = kwargs["result_saving_path"]
        self.device_id = kwargs["device_id"]

        self.kl_div = nn.KLDivLoss(reduction="none", log_target=True)
        self.nll = nn.NLLLoss()
        self.train_acc = []
        self.val_acc = []
        self.record_acc = []
        self.per_epoch_loss = []
        self.loss_cache = []
        self.best_valid_acc = -1.
        self.test_acc = 0.
        self.max_epochs = kwargs['max_epochs']
        self.seed = kwargs['seed']
        self.best_valid_epoch = 0

        self.train_start_time = None
        self.train_end_time = None
        self.inference_time = []
        self.infer_start = 0.
        self.infer_end = 0.

        self.teacher = kwargs["teacherModelName"]
        self.stu = kwargs["studentModelName"]
        self.dataset_name = kwargs["dataset"]
        self.dataset = kwargs["pyg_dataset"]
        self.expId = f"expId_{self.random_id}"
        self.use_KD = use_KD

        self.useGraphKD = kwargs.get('useGraphKD', False)
        self.graphKDgReg = kwargs.get('graphKDReg', 0.0)
        self.useGraphPooling = kwargs.get('useGraphPooling', False)
        self.graphPoolingReg = kwargs.get('graphPoolingReg', 0.0)
        self.useNodeKD = kwargs.get('useNodeKD',False)
        self.NodeKDReg = kwargs.get('NodeKDReg', 0.0)
        self.useSubgraphKD = kwargs.get('useSubgraphKD',False)
        self.SubgraphKDReg = kwargs.get('SubgraphKDReg', 0.0)
        self.communityAlgo = kwargs.get('communityAlgo', "louvain")
        self.kd_order = kwargs['parsed_kd_order']


        self.use_AdditionalAttr = kwargs.get("use_AdditionalAttr", False)
        if kwargs["useDifferentHidden"]:
            self.linear_proj = nn.Linear(2 * kwargs["hidden_dim_G"], kwargs['teacher_hidden_dim'])
        else:
            self.linear_proj = nn.Linear(2 * kwargs["hidden_dim"], kwargs['teacher_hidden_dim'])

        """GEMD """
        self.lamb = kwargs["lamb"]
        self.tau = kwargs["tau"]
        self.beta = kwargs["beta"]
        self.ModelCount = 3
        self.alpha_per_batch = {model_idx: [] for model_idx in range(self.ModelCount)} #use store every batch alpha for per sub_model
        self.alpha_vals, self.graph_weights = IniGEMWeights(self.ModelCount, self.dataset)

        """Hard aware"""
        self.train_dataset = kwargs["train_dataset"]
        self.full_dataset = kwargs["full_dataset"]
        self.k_neighbors = kwargs["K_neighbors"]
        self.G_alpha = kwargs["G_alpha"]
        self.G_beta= kwargs["G_beta"]
        self.graph_edge_idx_list = extract_graph_vitual_edge_KNN(self.train_dataset,self.full_dataset, self.k_neighbors, self.device_id)
        self.teacherEmblist = extract_teacher_emb(self.full_dataset).to(self.device_id)
        self.teacherPredlist = extract_teacher_pred(self.full_dataset).to(self.device_id)
        self.graph_edge_idx = self.graph_edge_idx_list[0]
        self.graph_edge_weight = torch.ones((self.graph_edge_idx.shape[1])).view(-1, 1).to(self.device_id)
        self.edge_prob = torch.zeros(self.graph_edge_idx_list[0].shape[1], device=self.device_id) #self.graph_edge_idx[0].shape[1] == num_edge
        self.stuPredlist = torch.zeros(len(self.full_dataset), self.num_classes).to(self.device_id)





    def forward(self, data):
        x = self.model(data)
        return x

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=8e-3, weight_decay=self.weight_decay)
        # We will reduce the learning rate

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=self.lr_patience,
                                                         min_lr=5e-7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_acc'}
        # return {'optimizer':optimizer,'lr_scheduler':scheduler}


    def on_train_epoch_start(self):
        self.train_start_time = time.time()
        if self.current_epoch < 1:
            noise = torch.distributions.Beta(self.G_beta, self.G_beta).sample(
                (self.graph_edge_weight.shape[0],)).view(-1, 1).to(self.device_id)
            self.graph_edge_weight = self.graph_edge_weight * noise
        else:
            a = self.G_alpha / (100 ** ((self.current_epoch - 1) / (self.max_epochs - 1)))
            self.edge_prob = extract_graph_subset(self.teacherPredlist, self.teacherEmblist, self.stuPredlist, self.graph_edge_idx_list[1], a)
            sampling_mask = torch.bernoulli(self.edge_prob).bool()
            sampled_edge_idx = self.graph_edge_idx_list[1][:, sampling_mask]
            src_idx = torch.unique(self.graph_edge_idx_list[0][0])
            loop_edges = torch.stack([src_idx, src_idx], dim=0)
            self.graph_edge_idx = torch.cat([sampled_edge_idx, loop_edges], dim=1)  # (2, sampleN+selfloop)

            sampled_edge_weight = self.edge_prob[sampling_mask]
            loop_weight = torch.ones(len(src_idx), device=self.device_id)
            graph_edge_weight = torch.cat([sampled_edge_weight, loop_weight], dim=0).view(-1, 1)
            noise = torch.distributions.Beta(self.G_beta, self.G_beta).sample(
                (graph_edge_weight.shape[0],)).view(-1, 1).to(self.device_id)
            self.graph_edge_weight = graph_edge_weight * noise
        #reset stuPredlist
        self.stuPredlist = torch.zeros(len(self.full_dataset), self.num_classes).to(self.device_id)


    def training_step(self, batch, batch_idx):

        labels = batch.y
        labels = labels.view(-1, )
        teacher_logit = batch.teacherPred
        current_kd_type = None

        total_loss = 0.0
        l_size = len(labels)
        batch_size = len(labels)
        subset_size = max(1, l_size // self.ModelCount)
        loss_label = 0.0
        loss_teacher = 0.0
        discrepancy_combine = 0

        graph_weights = self.graph_weights.to(self.device_id)

        rand_idx = torch.randperm(l_size)

        for k in range(0,self.ModelCount):
                start_idx = k * subset_size
                end_idx = (k + 1) * subset_size if k < self.ModelCount - 1 else l_size
                #end_idx = start_idx + subset_size
                sub_idx = rand_idx[start_idx:end_idx]
                #sub_idx = rand_idx[rand_idx[start_idx:end_idx]]

                y_pred, node_emb, graph_emb = self.model(batch,k,output_emb =True)

                #classification loss
                out_l = y_pred.log_softmax(dim=1)
                loss_label += self.nll(out_l[sub_idx],labels[sub_idx])

                current_kd_type = self.kd_type[k]
                if current_kd_type == "G":
                    if self.useGraphKD:
                        graphHGloss = 0.0
                        self.stuPredlist[batch.GlobalGraphIdx] = y_pred
                        val = self.linear_proj(graph_emb)
                        graphHGloss = cal_HG_graph_embloss(
                            self.graph_edge_idx, batch.GlobalGraphIdx, val,
                            self.teacherEmblist, self.graph_edge_weight, len(self.dataset))
                        discrepancy_combine = (graphHGloss * self.graphKDReg)

                    if self.useGraphPooling:
                        graphAlignmentLoss = 0.0
                        teacherGraphEmb = batch.graphEmb
                        val = self.linear_proj(graph_emb)
                        graphAlignmentLoss = nodeFeatureAlignment(val, teacherGraphEmb).sum(1)  # nodeFeatureAlignment can also be used for graph embedding matching
                        discrepancy_combine = (graphAlignmentLoss * self.graphPoolingReg)

                if current_kd_type == "S":
                    if self.useSubgraphKD and k == 1:  # from MuGSI
                        num_graphs = batch_size
                        batch_node_idx = batch.batch
                        loss_cluster_each_graph = []
                        clusterMatchingloss_per_K = 0.0
                        h = []
                        cluster_num = []

                        for i in range(num_graphs):
                            if self.communityAlgo == 'louvain':
                                cluster_id = batch.louvain_cluster_id.view(-1, )
                            mask_graph = batch_node_idx == i
                            cluster_id_per_graph = cluster_id[mask_graph]
                            node_emb_per_graph = node_emb[mask_graph]
                            clusterpool = self.model.sub_mlps[k].pool(node_emb_per_graph, cluster_id_per_graph)

                            h.append(clusterpool)
                            cluster_num.append(cluster_id_per_graph.max() + 1)

                        teacherClusterInfo = batch.teacherClusterInfo

                        splitted_tensor = torch.split(teacherClusterInfo, cluster_num)

                        for i, t in enumerate(splitted_tensor):
                            if t.shape[0] <= 2:
                                val = torch.tensor(0.0).to(t.device)
                            else:
                                cos_sim_teacher = fast_cosine_sim_matrix(t, t)

                                cos_sim_stu = fast_cosine_sim_matrix(h[i], h[i])

                                val = (cos_sim_stu - cos_sim_teacher).norm(p='fro') ** 2 / 2.0

                            if torch.isnan(val) or torch.isinf(val):
                                val = torch.tensor(0.0).to(val.device)

                            loss_cluster_each_graph.append(val)
                            clusterMatchingloss_per_K += val.item()

                        loss_cluster_each_graph = torch.stack(loss_cluster_each_graph)
                        discrepancy_combine = (loss_cluster_each_graph * self.SubgraphKDReg)

                if current_kd_type == "N":
                    if self.useNodeKD:  # from MuGSI
                        num_graphs = batch_size
                        if num_graphs == 0:
                            raise RuntimeError("num_graphs is 0!")
                        batch_node_idx = batch.batch

                        teacherNodeEmb = batch.nodeEmb

                        random_walk_paths = batch.random_walk_paths

                        loss_RandomWalkCons_per_graph = []
                        loss_RandomWalkCons_per_k = 0.0
                        # ! Need to consider per-graph cond probabilities. use for loop and batch.batch to get each graph
                        for idx, i in enumerate(range(num_graphs)):
                            mask_graph = batch_node_idx == i
                            if torch.all(random_walk_paths[20 * idx:20 * (idx + 1)] == 1):
                                loss_RandomWalkCons_per_graph.append(torch.tensor(0.0, device=teacherNodeEmb.device))
                                continue
                            rwp = random_walk_paths[20 * idx:20 * (idx + 1)]
                            tne = teacherNodeEmb[mask_graph]
                            sne = node_emb[mask_graph]
                            teacherCondProb = calculate_conditional_probabilities_gemd(rwp, tne)
                            stuCondProb = calculate_conditional_probabilities_gemd(rwp, sne)
                            rwWalkLoss_per_graph = calculate_kl_loss_check(stuCondProb, teacherCondProb)

                            loss_RandomWalkCons_per_k += rwWalkLoss_per_graph
                            loss_RandomWalkCons_per_graph.append(rwWalkLoss_per_graph)

                        loss_RandomWalkCons_per_graph = torch.stack(loss_RandomWalkCons_per_graph)
                        discrepancy_combine = (loss_RandomWalkCons_per_graph * self.NodeKDReg)

                #SoftLabel KD loss
                kd_logits  = y_pred
                soften_logits = kd_logits / self.tau
                soften_target = teacher_logit / self.tau
                softmax_s = (soften_logits).log_softmax(dim=1)
                softmax_t = (soften_target).log_softmax(dim=1)
                sl_loss = self.kl_div(softmax_s, softmax_t).sum(1)

                #Update weights
                graph_weights, alpha = GEMD_update_weights(discrepancy_combine,
                                                            kd_logits,
                                                            teacher_logit,
                                                            graph_weights,
                                                            batch.GlobalGraphIdx,
                                                            self.beta ,self.num_classes)
                self.alpha_per_batch[k].append(alpha)
                self.graph_weights = graph_weights

                discrepancy_combine += (sl_loss/self.ModelCount)
                loss_teacher += (discrepancy_combine * graph_weights[batch.GlobalGraphIdx]).sum()

        loss_label /= self.ModelCount

        total_loss = (loss_label) + loss_teacher
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return total_loss

    def backward(self, loss, optimizer, optimizer_idx):
        loss.backward()
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"{name} grad has nan or inf!")
                    raise RuntimeError(f"{name} grad has nan or inf!")  # 强制中止程序

    def on_before_optimizer_step(self, optimizer,optimizer_idx):
        has_nan_inf = False
        for name, param in self.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"[PL Hook] {name} grad has nan or inf!")
                    has_nan_inf = True
        if has_nan_inf:
            raise RuntimeError("Some gradients have nan or inf before optimizer step")


    def training_epoch_end(self, outputs):
        for model_idx in range(self.ModelCount):
            self.alpha_vals[model_idx] = torch.tensor(self.alpha_per_batch[model_idx]).mean().item()

        #self.stuPredlist = torch.cat(self.stuPredlist, dim=0)

    def validation_step(self, batch, batch_idx):
        self.eval()
        tot_correct = 0.
        y = batch.y
        pred_list = [] #record preds
        for k in range(0,self.ModelCount):
            y_pred = self.model(batch,k,output_emb=False)
            out_logsoftmax = y_pred.log_softmax(dim=1)

            out_softmax = out_logsoftmax.softmax(dim=1)
            #pred_list.append(out.softmax(dim=1))
            pred_list.append(out_softmax)

        pred_all = torch.stack(pred_list)
        alpha = self.alpha_vals.unsqueeze(1).unsqueeze(1).to(self.device_id)
        logits = (pred_all * alpha).sum(dim=0)
        out = torch.log(logits + 1e-16)
        pred = torch.argmax(out ,dim=1)
        tot_correct += torch.sum(y.to('cpu') == pred.to('cpu')).item()
        N = len(y)
        self.val_acc.append((N, tot_correct))

    def on_validation_epoch_end(self):
        tot_correct = sum([i[1] for i in self.val_acc])
        N = sum([i[0] for i in self.val_acc])
        val_acc = 1. * tot_correct / N
        self.log('valid_acc', val_acc, prog_bar=True, on_epoch=True)
        self.val_acc = []

        if self.current_epoch <= 1:
            self.best_valid_acc = 0.0
            return

        if val_acc > self.best_valid_acc:
            self.best_valid_epoch = self.current_epoch
            self.best_valid_acc = val_acc
            if self.current_epoch == 0:
                self.best_valid_acc = 0.0
            if self.model_saving_path is not None:
                file_name = f"BestValidAccuracy_{self.best_valid_acc:5f}_model_weights.pt"
                save_model_weights(self.model, self.model_saving_path, file_name=file_name)

            info_common = f"Dataset:{self.dataset},fold_index:{self.fold_index}, Seed:{self.seed}, expId:{self.expId},student:{self.stu}, teacher:{self.teacher}"
            if self.use_KD:
                info_KD = f"temperature:{self.tau},  use_AdditionalAttr:{self.use_AdditionalAttr}, BestValidAccuracy: {self.best_valid_acc:6f}"
                info = info_common + ", " + info_KD
                result_filename = f"{self.dataset}_student_{self.stu}_teacher_{self.teacher}_expId_{self.random_id}.txt"
            else:
                info_no_KD = f"Not using KD, model name:{self.stu}, num hops:{self.num_hops}, BestValidAccuracy: {self.best_valid_acc:6f}"
                info = info_common + ", " + info_no_KD
                result_filename = f"{self.dataset}_student_{self.stu}_expId_{self.random_id}_noKD.txt"

            write_results_to_file(f"{self.result_saving_path}", result_filename, info)

        print(colored.colored(
            f'Seed:{self.seed}, dataset:{self.dataset}, using device:{self.device_id} BestValidationAccuracy at epoch:{self.best_valid_epoch} is {self.best_valid_acc:6f}',
            'red', 'on_yellow'))

        if not self.use_KD and self.current_epoch == self.max_epochs - 1:
            info_no_KD = f"Not using KD, model name:{self.stu}, num hops:{self.num_hops}, BestValidAccuracy: {self.best_valid_acc:6f}"
            write_results_to_file(f"{self.result_saving_path}",
                                  f"{self.dataset}_student_{self.stu}_expId_{self.random_id}_noKD.txt", info_no_KD)

class GNN_plModel(pl.LightningModule):
    def __init__(self, model, lr=5e-3, weight_decay=1e-7, lr_patience=30, gamma=0.5, model_saving_path=None,test_loader=None, use_KD=False, **kargs):
        super(GNN_plModel, self).__init__()
        self.save_hyperparameters(ignore=["model","val_loader","test_loader","teacherModel","pyg_dataset"])
        self.model = model
        self.lr = lr
        self.lr_patience = lr_patience
        self.weight_decay = weight_decay
        self.num_classes = kargs['num_classes']
        self.dataset = kargs['dataset']
        self.fold_index = kargs["dataset_index"]
        self.acc = Accuracy(top_k=1, task='multiclass', num_classes=kargs['num_classes'])
        self.test_dataloader = test_loader
        self.val_acc = []
        self.record_acc = []
        self.per_epoch_loss = []
        self.loss_cache = []
        self.best_valid_acc = -1.
        self.test_acc = 0.
        self.use_KD = use_KD
        self.max_epochs = kargs['max_epochs']
        self.seed = kargs['seed']


        self.dropout = kargs['dropout']
        self.first_layer_dropout = 0.
        self.hidden_dim = kargs['hidden_dim']
        self.random_id = kargs['random_id']
        self.model_saving_path = model_saving_path
        self.result_saving_path = kargs["result_saving_path"]
        self.device_id = kargs["device_id"]
        self.best_epoch = 0

        self.train_start_time = None
        self.train_end_time = None

        self.inference_time = []
        self.infer_start = 0.
        self.infer_end = 0.
        self.dataset = kargs["dataset"]
        self.expId = f"expId_{self.random_id}"
        self.use_AdditionalAttr = kargs.get("use_AdditionalAttr", False)

    def forward(self, data):
        x = self.model(data)
        return x

    def on_train_epoch_start(self) -> None:
        self.train_start_time = time.time()

    def training_step(self, batch, batch_idx):
        # x, y,batch = batch.x,batch.y, batch.batch
        y = batch.y
        y = y.view(-1, )
        y_pred = self.model(batch)

        if self.num_classes > 1:  # multiple classes, not binary classification
            # Classification problem: use log softmax loss
            loss = F.nll_loss(F.log_softmax(y_pred, dim=1), y)
            self.loss_cache.append(loss.detach().item())

        self.log('train_loss', loss, prog_bar=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        tot_correct = 0.
        _ , y, _ = batch.x, batch.y, batch.batch
        y_pred = self.model(batch)
        y_pred = torch.argmax(y_pred, dim=1)
        tot_correct += torch.sum(y.to('cpu') == y_pred.to('cpu')).item()
        N = len(y)
        self.val_acc.append((N,tot_correct))

    def on_validation_epoch_end(self):
        tot_correct = sum([i[1] for i in self.val_acc])
        N = sum([i[0] for i in self.val_acc])
        val_acc = 1. * tot_correct / N
        self.log('valid_acc', val_acc, prog_bar=True, on_epoch=True)
        self.val_acc = []
        if self.current_epoch <= 1:
            self.best_valid_acc = 0.0
            return
        if val_acc > self.best_valid_acc:
            self.best_valid_acc = val_acc
            self.best_epoch = self.current_epoch
            if self.current_epoch == 0:
                self.best_valid_acc = 0.0
            if self.model_saving_path is not None:
                file_name = f"BestValidAccuracy_{self.best_valid_acc:5f}_model_weights.pt"
                if self.model_saving_path is not None:
                    save_model_weights(self.model, self.model_saving_path, file_name=file_name)

            info = f"Not using KD, Seed:{self.seed}, ExpId:{self.random_id},model name:GIN,BestEpoch: {self.best_epoch}, BestTestAccuracy: {self.best_valid_acc:6f}"
            write_results_to_file(f"{self.result_saving_path}",
                                      f"{self.dataset}_expId_{self.random_id}_noKD.txt", info)

        print(colored.colored(
            f'Seed:{self.seed}, dataset:{self.dataset},using device:{self.device_id} BestValidationAccuracy at epoch:{self.best_epoch} is {self.best_valid_acc:6f}',
            'red', 'on_yellow'))

        if self.current_epoch == self.max_epochs - 1:
            info = f"Not using KD, Seed:{self.seed}, ExpId:{self.random_id},model name:GIN, BestTestAccuracy: {self.best_valid_acc:6f}"
            write_results_to_file(f"{self.result_saving_path}",
                                    f"{self.dataset}_expId_{self.random_id}_noKD.txt", info)
        print(colored.colored(
            f'Dataset:{self.dataset},fold:{self.fold_index}, Not using KD,Seed:{self.seed}, ExpId:{self.random_id},model name:GIN, TestAccuracy at epoch:{self.current_epoch} is {self.best_valid_acc:6f}:',
            'red', 'on_yellow'))


    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=8e-3, weight_decay=self.weight_decay)
        # We will reduce the learning rate

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=self.lr_patience,
                                                         min_lr=5e-7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_acc'}
        # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=self.lr_patience,gamma=self.gamma)
        # return {'optimizer':optimizer,'lr_scheduler':scheduler}

    def test_step(self, batch, batch_idx):
        batch_inference_times = []
        preds = []
        self.model.eval()
        self.model.to('cpu')  # 确保模型在cpu
        with torch.no_grad():
            # 直接使用 to_data_list 拆分单个图
            data_list = batch.to_data_list()
            for data in data_list:
                data = data.to('cpu')  # 确保数据在cpu
                start_time = time.perf_counter()
                output = self.model(data)
                end_time = time.perf_counter()
                batch_inference_times.append(end_time - start_time)
                preds.append(output)

        mean_time = sum(batch_inference_times) / len(batch_inference_times) if batch_inference_times else 0.0
        self.log("mean_infer_time", mean_time)

        return {"inference_times": batch_inference_times}

class pl_Model(pl.LightningModule):
    def __init__(self, model, lr,
                 weight_decay, lr_patience ,
                 gamma , model_saving_path=None,
                 test_loader=None,use_KD= False,**kwargs):
        super(pl_Model, self).__init__()
        """tarining and validation parameters"""
        self.save_hyperparameters(ignore=["model", "val_loader", "test_loader", "teacherModel", "pyg_dataset"])
        self.model = model
        self.lr = lr
        self.lr_patience = lr_patience
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.num_classes = kwargs["num_classes"]
        self.dataset = kwargs["dataset"]
        self.fold_index = kwargs['dataset_index']
        self.teacherModel = kwargs.get("teacherModel",None)

        self.acc = Accuracy(top_k=1,task='multiclass',num_classes=kwargs['num_classes'])
        self.test_dataset = test_loader

        self.val_acc = []
        self.record_acc = []
        self.per_epoch_loss = []
        self.loss_cache = []
        self.best_valid_acc = -1.
        self.test_acc = 0.
        self.max_epochs = kwargs['max_epochs']
        self.seed = kwargs['seed']

        self.num_hops = kwargs["num_hops"]

        self.dropout = kwargs['dropout']
        self.first_layer_dropout = 0.
        self.hidden_dim = kwargs['hidden_dim']
        self.random_id = kwargs['random_id']
        self.model_saving_path = model_saving_path
        self.loss_saving_path = kwargs["loss_saving_path"]
        self.result_saving_path = kwargs["result_saving_path"]
        self.device_id = kwargs["device_id"]

        self.train_start_time = None
        self.train_end_time = None

        self.inference_time = []
        self.infer_start = 0.
        self.infer_end = 0.

        """KD setting"""
        self.use_KD = use_KD
        if use_KD:
            self.linear_proj = nn.Linear(2*kwargs["hidden_dim"], kwargs['teacher_hidden_dim'])
            if kwargs["useNCE"]:
                print ("useNCE",kwargs["useNCE"])
                self.nce_linear = nn.Linear(2*kwargs["hidden_dim"], kwargs['teacher_hidden_dim'])
            self.kl_div = nn.KLDivLoss(reduction='batchmean')
            """KD skills"""
            self.useDropoutEdge = kwargs.get('useDropoutEdge', False)
            self.pathLength = kwargs.get('pathLength', 0)
            self.usePE = kwargs.get('usePE', False)

            self.useSoftLabel = kwargs.get("useSoftLabel", False)
            self.softLabelReg = kwargs.get("softLabelReg", 1e-1)

            self.useNodeSim = kwargs.get('useNodeSim', False)
            self.nodeSimReg = kwargs.get('nodeSimReg', 1e-3)

            self.useNodeFeatureAlign = kwargs.get('useNodeFeatureAlign', False)
            self.NodeFeatureReg = kwargs.get('NodeFeatureReg', 0.0)

            self.useClusterMatching = kwargs.get('useClusterMatching', False)
            self.ClusterMatchingReg = kwargs.get('ClusterMatchingReg', 0.0)
            self.clusterAlgo = kwargs.get('clusterAlgo', "louvain")

            self.useRandomWalkConsistency = kwargs.get('useRandomWalkConsistency', False)
            self.RandomWalkConsistencyReg = kwargs.get('RandomWalkConsistencyReg', 0.0)

            self.useMixUp = kwargs.get('useMixUp', False)
            self.MixUpReg = kwargs.get('MixUpReg', 0.0)

            self.useGraphPooling = kwargs.get('useGraphPooling', False)
            self.graphPoolingReg = kwargs.get('graphPoolingReg', 0.0)

            self.useNCE = kwargs.get('useNCE', False)
            self.NCEReg = kwargs.get('NCEReg', 0.0)


            self.teacher = kwargs["teacherModelName"]
        else:
            self.teacher = kwargs.get("teacherModelName", 'empty')
        self.stu = kwargs["studentModelName"]
        self.dataset = kwargs["dataset"]
        self.expId = f"expId_{self.random_id}"
        self.use_AdditionalAttr = kwargs.get("use_AdditionalAttr", False)

    def forward(self, data):
        x = self.model(data)
        return x

    def on_train_epoch_start(self) -> None:
        self.train_start_time = time.time()

    def training_step(self, batch, batch_idx):
        y = batch.y
        y= y.view(-1,)
        if self.use_KD:
            y_pred,node_emb,graph_emb = self.model(batch,output_emb =True) # logits after graph readout function
        else:
            y_pred = self.model(batch)

        if self.num_classes > 1: # multiple classes, not binary classification
            # Classification problem: use log softmax loss
            loss = F.nll_loss(F.log_softmax(y_pred, dim=1), y)
            self.loss_cache.append(loss.detach().item())

        if not self.use_KD:
            self.log('train_loss', loss,prog_bar=True,on_epoch=True)
            return loss
        else:
            """init loss"""
            total_loss = loss
            nodeSimLoss= 0.0
            softLabelLoss = 0.0
            featureAlignmentLoss = 0.0
            graphPoolingLoss = 0.0
            mixUpLoss = 0.0
            clusterMatchingLoss = 0.0
            graphAlignmentLoss = 0.0

            teacherPred = batch.teacherPred

            #GLNN
            if self.useSoftLabel:
                softLabelLoss = calc_KL_divergence(y_pred, teacherPred, self.kl_div)
                total_loss = total_loss + self.softLabelReg * softLabelLoss
            #NOSMOG
            if self.useNodeSim:
                teacherNodeEmb = batch.nodeEmb  # teacher node embeddings
                nodeSimLoss = calc_node_similarity(node_emb,teacherNodeEmb)
                total_loss = total_loss + self.nodeSimReg*nodeSimLoss

            if self.useNodeFeatureAlign:
                val = self.linear_proj(node_emb)
                teacherNodeEmb = batch.nodeEmb
                featureAlignmentLoss = nodeFeatureAlignment(val,teacherNodeEmb)
                total_loss = total_loss +  self.NodeFeatureReg*featureAlignmentLoss

            #MuGSI
            if self.useRandomWalkConsistency:
                rwLoss = 0.0
                num_graphs = batch.num_graphs
                batch_idx = batch.batch
                teacherNodeEmb = batch.nodeEmb
                random_walk_paths = batch.random_walk_paths
                # ! Need to consider per-graph cond probabilities. use for loop and batch.batch to get each graph
                for idx,i in enumerate(range(num_graphs)):
                    mask_graph = batch_idx == i
                    if torch.all(random_walk_paths[20*idx:20*(idx+1)]==1):
                        continue
                    teacherCondProb = calculate_conditional_probabilities(random_walk_paths[20*idx:20*(idx+1)],teacherNodeEmb[mask_graph])  #! sample_size for random_walk per graph is 20
                    stuCondProb = calculate_conditional_probabilities(random_walk_paths[20*idx:20*(idx+1)],node_emb[mask_graph])
                    rwWalkLoss_per_graph = calculate_kl_loss(stuCondProb,teacherCondProb)
                    rwLoss = rwLoss + rwWalkLoss_per_graph
                rwLoss = rwLoss/num_graphs
                total_loss = total_loss +  self.RandomWalkConsistencyReg*rwLoss
            #MuGSI
            if self.useGraphPooling:
                teacherGraphEmb = batch.graphEmb
                val = self.linear_proj(graph_emb)
                graphAlignmentLoss = nodeFeatureAlignment(val,
                                                          teacherGraphEmb)  # nodeFeatureAlignment can also be used for graph embedding matching
                total_loss = total_loss + self.graphPoolingReg * graphAlignmentLoss
            #MuGSI
            if self.useClusterMatching:
                num_graphs = batch.num_graphs
                batch_idx = batch.batch
                h = []
                cluster_num = []
                for i in range(num_graphs):
                    if self.clusterAlgo == 'louvain':
                        cluster_id = batch.louvain_cluster_id.view(-1, )
                    if self.clusterAlgo == 'metis5':
                        cluster_id = batch.metis_clusters5.view(-1, )
                    if self.clusterAlgo == 'metis10':
                        cluster_id = batch.metis_clusters10.view(-1, )
                    mask_graph = batch_idx == i
                    cluster_id_per_graph = cluster_id[mask_graph]

                    node_emb_per_graph = node_emb[mask_graph]
                    h.append(self.model.pool(node_emb_per_graph, cluster_id_per_graph))
                    cluster_num.append(cluster_id_per_graph.max() + 1)
                teacherClusterInfo = batch.teacherClusterInfo
                splitted_tensor = torch.split(teacherClusterInfo, cluster_num)
                for i, t in enumerate(splitted_tensor):
                    if t.shape[0] <= 2: continue
                    cos_sim_teacher = fast_cosine_sim_matrix(t, t)
                    cos_sim_stu = fast_cosine_sim_matrix(h[i], h[i])
                    val = (cos_sim_stu - cos_sim_teacher).norm(p='fro') ** 2 / 2.0
                    clusterMatchingLoss += val
                    # print (f"clusterMatchingLoss:{clusterMatchingLoss}.....")
                total_loss += clusterMatchingLoss * self.ClusterMatchingReg / num_graphs

            loss_dict = {'train_loss': total_loss}
            self.log_dict(loss_dict, prog_bar=True, on_epoch=True)
            return total_loss

    def validation_step(self, batch, batch_idx):
        tot_correct = 0.
        _, y, _ = batch.x, batch.y, batch.batch
        y_pred = self.model(batch)
        y_pred = torch.argmax(y_pred, dim=1)
        tot_correct += torch.sum(y.to('cpu') == y_pred.to('cpu')).item()
        N = len(y)
        self.val_acc.append((N, tot_correct))

    def on_validation_epoch_end(self):
        tot_correct = sum([i[1] for i in self.val_acc])
        N = sum([i[0] for i in self.val_acc])
        val_acc = 1. * tot_correct / N
        self.log('valid_acc', val_acc, prog_bar=True, on_epoch=True)
        self.val_acc = []

        if self.current_epoch <= 1:
            self.best_valid_acc = 0.0
            return

        if val_acc > self.best_valid_acc:
            self.best_valid_acc = val_acc
            if self.current_epoch == 0:
                self.best_valid_acc = 0.0
            if self.model_saving_path is not None:
                file_name = f"BestValidAccuracy_{self.best_valid_acc:5f}_model_weights.pt"
                if self.model_saving_path is not None:
                    save_model_weights(self.model, self.model_saving_path, file_name=file_name)

            info_common = f"Dataset:{self.dataset},fold_index:{self.fold_index}, Seed:{self.seed}, expId:{self.expId},student:{self.stu}, teacher:{self.teacher}"
            if self.use_KD:
                info_KD = f"useSoftLabel:{self.useSoftLabel}, useNodeSim:{self.useNodeSim}, useLearnableGraphPooling:{self.useGraphPooling}, useNodeFeatureAlign:{self.useNodeFeatureAlign}, useClusterMatching: {self.useClusterMatching}, useRandomWalkConsistency:{self.useRandomWalkConsistency}, useNCE:{self.useNCE}, softLabelReg:{self.softLabelReg}, nodeSimReg:{self.nodeSimReg}, NodeFeatureReg:{self.NodeFeatureReg}, ClusterMatchingReg:{self.ClusterMatchingReg}, graphPoolingReg:{self.graphPoolingReg}, pathLength:{self.pathLength}, RandomWalkConsistencyReg:{self.RandomWalkConsistencyReg}, NCEReg:{self.NCEReg}, use_AdditionalAttr:{self.use_AdditionalAttr}, BestValidAccuracy: {self.best_valid_acc:6f}"
                info = info_common + ", " + info_KD
                result_filename = f"{self.dataset}_student_{self.stu}_teacher_{self.teacher}_expId_{self.random_id}.txt"
            else:
                info_no_KD = f"Not using KD, model name:{self.stu}, num hops:{self.num_hops}, BestValidAccuracy: {self.best_valid_acc:6f}"
                info = info_common + ", " + info_no_KD
                result_filename = f"{self.dataset}_student_{self.stu}_expId_{self.random_id}_noKD.txt"

            write_results_to_file(f"{self.result_saving_path}", result_filename, info)

        print(colored.colored(f'Seed:{self.seed}, dataset:{self.dataset}, using device:{self.device_id} BestValidationAccuracy at epoch:{self.current_epoch} is {self.best_valid_acc:6f}','red', 'on_yellow'))

        if not self.use_KD and self.current_epoch == self.max_epochs - 1:
            info_no_KD = f"Not using KD, model name:{self.stu}, num hops:{self.num_hops}, BestValidAccuracy: {self.best_valid_acc:6f}"
            write_results_to_file(f"{self.result_saving_path}",
                                  f"{self.dataset}_student_{self.stu}_expId_{self.random_id}_noKD.txt", info_no_KD)

    def test(self):
        print (colored.colored("----------------Testing Stage---------------","blue","on_white"))
        self.eval()
        tot_correct = 0.
        y_preds = []
        labels = []
        with torch.no_grad():
            for batch in tqdm.tqdm(self.test_dataloader):
                batch = batch.to(self.device)
                _, y, _ = batch.x, batch.y, batch.batch
                y_pred = self.model(batch)
                labels.append(y.view(-1,))
                y_preds.append(y_pred)
        labels = torch.cat(labels, dim=0)
        y_preds = torch.cat(y_preds, dim=0)
        y_preds = torch.argmax(y_preds, dim=1)
        tot_correct += torch.sum(labels == y_preds).item()
        test_acc = 1.*tot_correct/len(labels)
        self.log('test_acc', test_acc, prog_bar=True, on_epoch=True)
        print (colored.colored(f'expId:{self.random_id},TestAccuracy at epoch:{self.current_epoch} is: {test_acc}', 'red','on_yellow'))
        self.train()
        self.test_acc=test_acc
        return test_acc

    def configure_optimizers(self):
        # We will support Adam or SGD as optimizers.
        optimizer = optim.Adam(self.parameters(), lr=8e-3, weight_decay=self.weight_decay)
        # We will reduce the learning rate

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.6, patience=self.lr_patience,
                                                         min_lr=5e-7)
        return {'optimizer': optimizer, 'lr_scheduler': scheduler, 'monitor': 'valid_acc'}
        # scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=self.lr_patience,gamma=self.gamma)
        # return {'optimizer':optimizer,'lr_scheduler':scheduler}

    def test_step(self, batch, batch_idx):
        batch_inference_times = []
        preds = []
        self.model.eval()
        self.model.to('cpu')  # 确保模型在cpu
        with torch.no_grad():
            # 直接使用 to_data_list 拆分单个图
            data_list = batch.to_data_list()
            for data in data_list:
                data = data.to('cpu')  # 确保数据在cpu
                start_time = time.perf_counter()
                output = self.model(data)
                end_time = time.perf_counter()
                batch_inference_times.append(end_time - start_time)
                preds.append(output)

        mean_time = sum(batch_inference_times) / len(batch_inference_times) if batch_inference_times else 0.0
        self.log("mean_infer_time", mean_time)

        return {"inference_times": batch_inference_times}



def IniAdaparas(K,dataset):
    alpha_vals = torch.ones(K)
    graph_weights = torch.ones(len(dataset))
    graph_weights /=  graph_weights.sum()
    return alpha_vals, graph_weights

def adagmlp_update_weights(
        logits_s, logits_t, graphs_weights,idx , beta
):
    criterion = torch.nn.KLDivLoss(reduction="none", log_target=True)
    with torch.no_grad():
        out_s = logits_s.log_softmax(dim=1)
        out_t = logits_t.log_softmax(dim=-1)
        loss = criterion(out_s, out_t).sum(1)
        errors = 1 - torch.exp(-beta * loss)  # torch.sigmoid(loss)
        #errors [batch_size] to 1
        error = torch.sum(graphs_weights[idx] * errors) / torch.sum(graphs_weights[idx])
        error = error + 1e-16
        alpha = max(torch.log((1 - error) / error + 1e-16), 1e-16)
        graphs_weights[idx] = graphs_weights[idx] * torch.exp(alpha * errors)
        graphs_weights[idx] /= graphs_weights[idx].sum()

    return graphs_weights, alpha

class AdaGMLP_plModel(pl.LightningModule):
    def __init__(self, model, lr,
                 weight_decay, lr_patience ,
                 gamma , model_saving_path=None,
                 test_loader=None,use_KD= False,**kwargs):
        super(AdaGMLP_plModel, self).__init__()
        """tarining and validation parameters"""
        self.save_hyperparameters(ignore=["model", "val_loader", "test_loader", "teacherModel", "pyg_dataset"])
        self.model = model
        self.lr = lr
        self.lr_patience = lr_patience
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.num_classes = kwargs["num_classes"]
        self.dataset = kwargs["dataset"]
        self.fold_index = kwargs['dataset_index']
        self.batch_size = kwargs['batch_size']
        self.teacherModel = kwargs.get("teacherModel", None)

        self.acc = Accuracy(top_k=1, task='multiclass', num_classes=kwargs['num_classes'])
        self.test_dataset = test_loader

        self.dropout = kwargs['dropout']
        self.first_layer_dropout = 0.
        self.hidden_dim = kwargs['hidden_dim']
        self.random_id = kwargs['random_id']
        self.model_saving_path = model_saving_path
        self.loss_saving_path = kwargs["loss_saving_path"]
        self.result_saving_path = kwargs["result_saving_path"]
        self.device_id = kwargs["device_id"]

        self.kl_div = nn.KLDivLoss(reduction="none", log_target=True)
        self.nll = nn.NLLLoss()
        self.train_acc = []
        self.val_acc = []
        self.record_acc = []
        self.per_epoch_loss = []
        self.loss_cache = []
        self.best_valid_acc = -1.
        self.test_acc = 0.
        self.max_epochs = kwargs['max_epochs']
        self.seed = kwargs['seed']
        self.best_valid_epoch = 0

        self.train_start_time = None
        self.train_end_time = None
        self.inference_time = []
        self.infer_start = 0.
        self.infer_end = 0.

        self.teacher = kwargs["teacherModelName"]
        self.stu = kwargs["studentModelName"]
        self.dataset_name = kwargs["dataset"]
        self.dataset = kwargs["pyg_dataset"]
        self.expId = f"expId_{self.random_id}"

        self.use_KD = use_KD
        self.useNodeSim = kwargs["useNodeSim"]
        self.nodeSimReg = kwargs["nodeSimReg"]

        self.useGraphPooling = kwargs.get('useGraphPooling', False)
        self.graphPoolingReg = kwargs.get('graphPoolingReg', 0.0)
        self.useRandomWalkConsistency = kwargs.get('useRandomWalkConsistency',False)
        self.RandomWalkConsistencyReg = kwargs.get('RandomWalkConsistencyReg', 0.0)
        self.useClusterMatching = kwargs.get('useClusterMatching',False)
        self.ClusterMatchingReg = kwargs.get('ClusterMatchingReg', 0.0)
        self.clusterAlgo = kwargs.get('clusterAlgo', "louvain")


        self.use_AdditionalAttr = kwargs.get("use_AdditionalAttr", False)
        self.linear_proj = nn.Linear(2 * kwargs["hidden_dim"], kwargs['teacher_hidden_dim'])

        """AdaGMLP """
        self.lamb = kwargs["lamb"]
        self.tau = kwargs["tau"]
        self.beta = kwargs["beta"]
        self.K = kwargs["K"]
        self.selective = kwargs["selective"]
        self.alpha_per_batch = {model_idx: [] for model_idx in range(self.K)} #use store every batch alpha for per sub_model
        self.alpha_vals, self.graph_weights = IniAdaparas(self.K, self.dataset)