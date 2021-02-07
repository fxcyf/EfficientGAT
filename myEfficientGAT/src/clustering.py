import metis
import torch
import random
import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
import scipy.sparse as sp


class ClusteringMachine(object):
    """
    Clustering the graph, feature set and target.
    """
    def __init__(self, args, graph, features, target, type_map=None):
        """
        :param args: Arguments object with parameters.
        :param graph: Networkx Graph.
        :param features: Feature matrix (ndarray).
        :param target: Target vector (ndarray).
        """
        self.args = args
        self.graph = graph          # nx.Graph
        self.features = features    # coo_matrix
        self.target = target        # np.array
        self.type_map = type_map    # dict
    #    self._set_sizes()
    #
    # def _set_sizes(self):
    #     """
    #     Setting the feature and class count.
    #     """
        self.feature_count = self.features.shape[1]     # num of col
        self.class_count = np.max(self.target)+1        # 1,2,... + 0

    def decompose(self):
        """
        Decomposing the graph, partitioning the features and target, creating Torch arrays.
        """
        if self.args.clustering_method == "metis":
            print("\nMetis graph clustering started.\n")
            self.metis_clustering()
        else:
            print("\nRandom graph clustering started.\n")
            self.random_clustering()
        self.general_data_partitioning()
        self.transfer_edges_and_nodes()

    def random_clustering(self):
        """
        Random clustering the nodes.
        """
        self.clusters = list(range(self.args.cluster_number)) # [cluster for cluster in range(self.args.cluster_number)]
        self.cluster_membership = {node: random.choice(self.clusters) for node in self.graph.nodes()}

    def metis_clustering(self):
        """
        Clustering the graph with Metis. For details see:
        """
        (st, parts) = metis.part_graph(self.graph, self.args.cluster_number)
        self.clusters = list(set(parts))
        self.cluster_membership = {node: membership for node, membership in enumerate(parts)}

    def general_data_partitioning(self):
        """
        Creating data partitions and train-test splits.
        """
        self.sg_nodes = {}
        self.sg_edges = {}
        self.sg_train_nodes = {}
        self.sg_valid_nodes = {}
        self.sg_test_nodes = {}
        self.sg_features = {}
        self.sg_targets = {}
        for cluster in self.clusters:
            subgraph = self.graph.subgraph([node for node in sorted(self.graph.nodes()) if self.cluster_membership[node] == cluster])
            self.sg_nodes[cluster] = [node for node in sorted(subgraph.nodes())]
            mapper = {node: i for i, node in enumerate(sorted(self.sg_nodes[cluster]))}
            self.sg_train_nodes[cluster] = sorted([mapper[node] for node in self.sg_nodes[cluster] if self.type_map[node]=='train'])
            self.sg_valid_nodes[cluster] = sorted([mapper[node] for node in self.sg_nodes[cluster] if self.type_map[node]=='valid'])
            self.sg_test_nodes[cluster] = sorted([mapper[node] for node in self.sg_nodes[cluster] if self.type_map[node]=='test'])

            # mind in sg_edges[cluster], nodes id has been mapped to 0,1,2,3,...
            # graph.edges() is unsymmetrical
            self.sg_edges[cluster] = [[mapper[edge[0]], mapper[edge[1]]] for edge in subgraph.edges()] +  [[mapper[edge[1]], mapper[edge[0]]] for edge in subgraph.edges()]
            # self.sg_train_nodes[cluster], self.sg_test_nodes[cluster] = train_test_split(list(mapper.values()), test_size = self.args.test_ratio)
            # self.sg_train_nodes[cluster] = sorted([mapper[x] for x in self.sg_train_nodes[cluster]])
            # self.sg_valid_nodes[cluster] = sorted([mapper[x] for x in self.sg_valid_nodes[cluster]])
            # self.sg_test_nodes[cluster] = sorted([mapper[x] for x in self.sg_test_nodes[cluster]])

            self.sg_features[cluster] = self.features[self.sg_nodes[cluster],:] # must sort and correspond
            self.sg_targets[cluster] = self.target[self.sg_nodes[cluster],:]

    def transfer_edges_and_nodes(self):
        """
        Transfering the data to PyTorch format.
        """
        for cluster in self.clusters:
            self.sg_nodes[cluster] = torch.LongTensor(self.sg_nodes[cluster])       # nNode * 1
            self.sg_edges[cluster] = torch.LongTensor(self.sg_edges[cluster]).t()   # t(): transpose --> 2*nEdges
            self.sg_train_nodes[cluster] = torch.LongTensor(self.sg_train_nodes[cluster])
            self.sg_valid_nodes[cluster] = torch.LongTensor(self.sg_valid_nodes[cluster])
            self.sg_test_nodes[cluster] = torch.LongTensor(self.sg_test_nodes[cluster])
            self.sg_features[cluster] = torch.FloatTensor(self.sg_features[cluster]) # nNode * nFeature
            self.sg_targets[cluster] = torch.LongTensor(self.sg_targets[cluster])   # nNode * 1
