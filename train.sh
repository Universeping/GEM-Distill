#Data preprocess
python DataPreTransform.py --dataset $dataset_name$

#GNN
python train_GNN.py --dataset PROTEINS --dataset_index 0 --hidden_dim 64 --num_layers 3 --dropout 0.0

#MLP
python train_MLP.py --useLaPE --max_epochs 350 --dataset PROTEINS --dataset_index 0 --hidden_dim 64 --out_dim 64   --lr_patience 30  --batch_size 32 --numWorkers 2

#GLNN
python train_Baseline.py  --useLaPE --max_epochs 350 --dataset PROTEINS --dataset_index 0 --use_KD --hidden_dim 64 --out_dim 64  --studentModelName MLP --teacherModelName GIN --lr_patience 30  --batch_size 32 --num_hops 1 --numWorkers 2  --useSoftLabel --softLabelReg 1.0 --KD_name GLNN

#NOSMOG
python train_Baseline.py --useLaPE  --max_epochs 350 --dataset PROTEINS --dataset_index 0  --use_KD --hidden_dim 64 --out_dim 64  D --studentModelName MLP --teacherModelName GIN --lr_patience 30  --batch_size 32 --num_hops 1 --numWorkers 2  --useSoftLabel --softLabelReg 1.0 --useNodeSim --nodeSimReg 0.1  --KD_name useNOSMOG

#MuGSI
python train_Baseline.py --useLaPE  --max_epochs 350 --dataset PROTEINS   --dataset_index 0 --use_KD --hidden_dim 64 --out_dim 64  --studentModelName MLP --teacherModelName GIN --lr_patience 30 --batch_size 32 --num_hops 1 --numWorkers 2  --useSoftLabel --softLabelReg 1.0 --useRandomWalkConsistency --RandomWalkConsistencyReg 0.0001 --useClusterMatching --ClusterMatchingReg 0.01 --useGraphPooling --graphPoolingReg 0.01 --KD_name MuGSI

#AdaGMLP
python run_Model_Ada.py --useLaPE  --tau 0.4 --K 3 --max_epochs 350 --dataset PROTEINS --dataset_index 0 --use_KD --hidden_dim 64 --out_dim 64  --studentModelName AdaGMLP   --teacherModelName GIN --lr_patience 30  --batch_size 32 --num_hops 1 --numWorkers 2

#GEMDistill
python train_GEMDistill.py --useLaPE --dataset PROTEINS --dataset_index 0 --use_KD --tau 0.4 --max_epochs 350  --K_neighbors 20 --hidden_dim 64 --out_dim 64  --studentModelName GEMDistill  --teacherModelName GIN --lr_patience 30  --batch_size 32 --numWorkers 2  --useNodeKD --NodeKDReg 0.2 --useSubgraphKD --SubgraphKDReg 0.1 --useGraphKD --graphKDReg 10 --kd_order G,S,N --KD_name GEMDistill
