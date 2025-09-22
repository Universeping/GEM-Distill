from torch_geometric import transforms as T
from torch_geometric.utils import (degree,
                                   is_undirected,
                                   to_undirected,
                                   to_networkx)
import numpy as nx
import numpy as np
from torch_geometric.utils import (degree,
                                   to_networkx,
                                   to_undirected)
from torch_scatter import scatter_add
from torch_geometric.transforms import (AddLaplacianEigenvectorPE,
                                        BaseTransform)
import community
import torch


class DegreeTransform(object):
    # combine position and intensity feature, ignore edge value
    def __init__(self) -> None:
        self.deg_func = T.OneHotDegree(max_degree=10000)

    def __call__(self, data):
        data = self.deg_func(data)
        N = data.x.shape[0]
        degrees = degree(data.edge_index[0],num_nodes=N).view(-1,1).float()
        max_degree = degrees.max().item()
        degrees = degrees/max_degree
        val = torch.cat([degrees,data.x],dim=1)
        val = val[:,:65]
        data.x = val
        return data

class CustomLaplacianEigenvectorPE:
    def __init__(self, attr_name=None,start=20):
        self.attr_name = attr_name
        self.start = start
    def __call__(self, data):
        num_feats = data.num_features
        for k in [self.start,10,5]:
            num_nodes = data.num_nodes
            if k>=num_nodes-1:
                continue
            try:
                transform = AddLaplacianEigenvectorPE(k=k, attr_name=self.attr_name)
                data = transform(data)
                # Add zeros if fewer than 20 dimensions
                num_nodes = data.num_nodes
                remaining_dim = self.start - k
                if remaining_dim > 0:
                    zero_padding = torch.zeros((num_nodes, remaining_dim))
                    data.x = torch.cat((data.x, zero_padding), dim=-1)
                return data
            except:
                continue
                # print(f"Failed with k={k}, trying a smaller k. Error: {e}")


        # If all attempts failed, assign 20-dimensional all-zero vector
        # print("All attempts to calculate Laplacian Eigenvector failed. Assigning 20-d all-zero vector.")
        num_nodes = data.num_nodes
        data.x = torch.zeros((num_nodes, self.start+num_feats))
        return data

    def __repr__(self):
        return f'{self.__class__.__name__}()'

class PerformLouvainClustering(BaseTransform):
    def __call__(self, data):
        G = to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True)

        # Perform Louvain clustering
        partition = community.best_partition(G)
        # Convert partition (a dict) to a list of cluster labels
        labels = list(partition.values())
        data.louvain_cluster_id = torch.tensor(labels).view(-1, 1)
        return data


class DropEdge(T.BaseTransform):
    def __init__(self, p=0.08):
        """
        Initialize the transform.
        p: The probability of an edge being dropped.
        """
        self.p = p

    def __call__(self, data):
        """
        Apply the transform.
        data: A Data object, which includes edge_index attribute.
        """
        edge_index = data.edge_index
        num_edges = edge_index.size(1) // 2  # Each edge appears twice

        if not is_undirected(edge_index):
            edge_index = to_undirected(edge_index)

        # Create a mask of edges to drop
        mask = torch.rand(num_edges) < self.p
        drop_edges = edge_index[:, mask.repeat(2)].t().tolist()

        # Remove both directions of each dropped edge
        for i in range(0, len(drop_edges), 2):
            edge = drop_edges[i]
            reverse_edge = edge[::-1]
            edge_index = edge_index[:, ~((edge_index[0] == edge[0]) & (edge_index[1] == edge[1]))]
            edge_index = edge_index[:, ~((edge_index[0] == reverse_edge[0]) & (edge_index[1] == reverse_edge[1]))]

        data.edge_index = edge_index
        return data

    def __repr__(self):
        return '{}(p={})'.format(self.__class__.__name__, self.p)


class TeacherModelTransform(T.BaseTransform):
    def __init__(self, model,use_clustering=False,cluster_algo = "louvain"):
        self.model = model
        self.use_clustering = use_clustering
        self.cluster_algo = cluster_algo
        self.model.eval()

    def __call__(self, data):
        self.model.eval()
        with torch.no_grad():
            pred,node_emb,graphEmb = self.model(data,output_emb=True)
            data.teacherPred = pred
            data.nodeEmb = node_emb
            data.graphEmb = graphEmb
            if self.use_clustering:
                if self.cluster_algo == "louvain":
                    cluster_id = data.louvain_cluster_id
                if self.cluster_algo == "metis5":
                    cluster_id = data.metis_clusters5
                if self.cluster_algo == "metis10":
                    cluster_id = data.metis_clusters10
                # if cluster_id.max()<=2:
                #     print (colored(f"cluster num is: {cluster_id}",'red'))
                h = self.model.pool(node_emb,cluster_id.view(-1,))
                data.teacherClusterInfo = h
        return data



# generate random walk paths for path consistency regularization for KD.
class RandomPathTransform(T.BaseTransform):
    def __init__(self, sample_size=20, path_length=15):
        super(RandomPathTransform, self).__init__()
        self.sample_size = sample_size
        self.path_length = path_length

    def __call__(self, data):
        G = to_networkx(data, node_attrs=None, edge_attrs=None)
        try:
            random_paths = nx.generate_random_paths(G, self.sample_size, self.path_length)
            # print ("have isolated nodes:",contains_isolated_nodes(data.edge_index))
            data.random_walk_paths = torch.tensor(list(random_paths)).long()
            # print (1,data.random_walk_paths.shape)
            return data
        except:
            data.random_walk_paths = torch.ones((self.sample_size, self.path_length+1), dtype=torch.long)
            # print (2,data.random_walk_paths.shape)
            return data


    def __repr__(self):
        return '{}(sample_size={}, path_length={})'.format(self.__class__.__name__, self.sample_size, self.path_length)