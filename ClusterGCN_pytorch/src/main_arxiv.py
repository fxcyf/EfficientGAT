import torch
from parser import parameter_parser
from clustering_arxiv import ClusteringMachine
from clustergcn import ClusterGCNTrainer
import argparse
from utils import tab_printer, graph_reader, feature_reader, target_reader, type_reader
from ogb.nodeproppred import DglNodePropPredDataset
import networkx as nx
import pandas as pd
import numpy as np


def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    parser = argparse.ArgumentParser(description="Run .")

    parser.add_argument("--data-name", nargs="?", default="ogbn-arxiv", help="ogb dataset name")
    parser.add_argument("--data-path", nargs="?", default="../input", help="ogb dataset path")
    parser.add_argument("--edge-path", nargs="?", default="../input/ogbn_arxiv/raw/edge.csv", help="Edge list csv.")
    parser.add_argument("--type-map-path", nargs="?", default="../input/ogbn_arxiv/split/time/",
                        help="path to test.csv, train.csv,valid.csv.")
    parser.add_argument("--result-path", nargs="?", default="../results/ogbn_arxiv_results/")
    parser.add_argument("--clustering-method", nargs="?", default="metis",
                        help="Clustering method for graph decomposition. Default is the metis procedure.")
    parser.add_argument("--epochs", type=int, default=200, help="Number of training epochs. Default is 200.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout parameter. Default is 0.5.")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Learning rate. Default is 0.01.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test data ratio. Default is 0.1.")
    parser.add_argument("--cluster-number", type=int, default=10, help="Number of clusters extracted. Default is 10.")
    parser.add_argument("--cluster-batch",type=int,default=1,help="number of clusters to form a batch. Default is one")
    parser.set_defaults(layers=[16, 16, 16])
    args = parser.parse_args()

    torch.manual_seed(args.seed)    # random seed
    tab_printer(args)

    dataset = DglNodePropPredDataset(name=args.data_name, root=args.data_path)
    # (Node Property Prediction
    g, target = dataset[0]

    graph = nx.from_edgelist(pd.read_csv(args.edge_path).values.tolist())  # graph = nx.Graph(...edges)
    features = g.ndata['feat']      # coo_matrix(np.array(g.ndata['feat']))  # features = coo_matrix(...)
    target = np.array(target)   # target = np.array(size=(-1,1))
    type_map = type_reader(args.type_map_path)    # target = np.array(size=(-1,1))
    clustering_machine = ClusteringMachine(args, graph, features, target, type_map)
    clustering_machine.decompose()
    gcn_trainer = ClusterGCNTrainer(args, clustering_machine)
    gcn_trainer.train()
    gcn_trainer.test()


if __name__ == "__main__":
    main()
