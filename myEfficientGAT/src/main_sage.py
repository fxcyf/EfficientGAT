import torch
import argparse
from clustering import ClusteringMachine
from clustergat import ClusterGATTrainer
from utils import tab_printer, graph_reader, feature_reader, target_reader, type_reader, graphsage_data_reader
import os

def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    # parameters setting
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument("dataset_path", nargs="?", default="../input", help="dataset path")
    parser.add_argument("dataset_str", nargs="?", default="ppi", help="dataset name")
    # parser.add_argument("--edge-path", nargs="?", default="../input/reddit_processed/edges.csv", help="Edge list
    # csv.") parser.add_argument("--features-path", nargs="?", default="../input/reddit_processed/features.csv",
    # help="Features json.") parser.add_argument("--target-path", nargs="?",
    # default="../input/reddit_processed/target.csv", help="Target classes csv.")
    parser.add_argument("--type-map-path", nargs="?", default="../input/ppi/",
                        help="path to test.csv, train.csv,valid.csv.")
    parser.add_argument("--multilabel", action='store_true', default=True, help="multi-label classification")
    parser.add_argument("--clustering-path", nargs="?", default="../input/ppi/", help="Clustering result txt")
    parser.add_argument("--results-path", nargs="?", default="../results/ppi_results/",
                        help="path to store results. ")

    parser.add_argument("--clustering-method", nargs="?", default="metis",
                        help="Clustering method for graph decomposition. Default is the metis procedure.")
    parser.add_argument("--seed", type=int, default=5876, help="Random seed for train-test split. Default is 42.")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Use CUDA training.')  # 无需带参数
    parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs. Default is 200.")
    parser.add_argument("--dropout", type=float, default=0, help="Dropout parameter. Default is 0.5.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate. Default is 0.01.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test data ratio. Default is 0.1.")
    parser.add_argument("--cluster-number", type=int, default=50, help="Number of clusters extracted. Default is 10.")
    parser.add_argument("--cluster-batch", type=int, default=2,
                        help="number of clusters to form a batch. Default is one")

    parser.add_argument('--hidden', type=int, default=512, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=4, help='Number of head attentions.')
    parser.add_argument('--patience', type=int, default=50, help='Patience')

    parser.add_argument('--feature_type', nargs="?", default='sqr',
                        help='Nonlinearity function for feature. Can be relu, elu+1, sqr, favor+, or favor+{int}.')
    parser.add_argument('--compute_type', nargs="?", default='iter',
                        help="Which type of method to compute: "
                             "iter = iterative algorithm from Appendix B, "
                             "ps = implementation using torch.cumsum, "
                             "parallel_ps = implementation using custom log prefix sum implementation.")

    args = parser.parse_args()      # args = parser.parse_args()

    torch.manual_seed(args.seed)    # random seed
    tab_printer(args)
    # load data and metis
    type_map = type_reader(args.type_map_path)
    graph, target, features = graphsage_data_reader(args.dataset_path, args.dataset_str)

    clustering_machine = ClusteringMachine(args, graph, graph, features, target, type_map)
    clustering_machine.decompose()

    myGAT = ClusterGATTrainer(args, clustering_machine)
    myGAT.train()
    myGAT.test()


if __name__ == "__main__":
    main()
