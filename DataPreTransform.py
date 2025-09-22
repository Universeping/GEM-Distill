import warnings
import argparse
from torch_geometric.datasets import TUDataset
from torch_geometric.transforms import Compose
from Transforms import DegreeTransform, CustomLaplacianEigenvectorPE, PerformLouvainClustering
from termcolor import colored


warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="PROTEINS")
    args = parser.parse_args()
    args = vars(args)
    dataset_name = args['dataset']
    print(colored(f"loading dataset {dataset_name}", 'blue', 'on_white'))
    if "IMDB-BINARY" in dataset_name or "REDDIT-BINARY" in dataset_name or "COLLAB" or "IMDB-MULTI" in dataset_name:
        dataset = TUDataset(root=f'data/raw/',name=dataset_name,pre_transform=DegreeTransform())
        print ("raw:",dataset[0])
        dataset = TUDataset(root=f'data/withAdditionalAttr/',name=dataset_name,pre_transform=Compose([DegreeTransform(),
                                                                                                CustomLaplacianEigenvectorPE(),
                                                                                                PerformLouvainClustering()]))
    else:
        dataset = TUDataset(root=f'data/withAdditionalAttr/',name=dataset_name,pre_transform=Compose([
                                                                                                CustomLaplacianEigenvectorPE(),
                                                                                                PerformLouvainClustering()]))
        dataset = TUDataset(root=f'data/raw/',name=dataset_name)
        print ("raw:",dataset[0])