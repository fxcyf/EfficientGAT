import torch
import argparse
from clustering import ClusteringMachine
from clustergat import ClusterGATTrainer
from utils import tab_printer, graph_reader, feature_reader, target_reader


def main():
    """
    Parsing command line parameters, reading data, graph decomposition, fitting a ClusterGCN and scoring the model.
    """
    # parameters setting
    parser = argparse.ArgumentParser(description="Run .")
    parser.add_argument("--edge-path", nargs="?", default="../input/edges.csv", help="Edge list csv.")
    parser.add_argument("--features-path", nargs="?", default="../input/features.csv", help="Features json.")
    parser.add_argument("--target-path", nargs="?", default="../input/target.csv", help="Target classes csv.")
    parser.add_argument("--clustering-method", nargs="?", default="metis", help="Clustering method for graph "
                                                                                "decomposition. Default is the metis "
                                                                                "procedure.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for train-test split. Default is 42.")
    parser.add_argument('--no_cuda', action='store_true', default=False, help='Use CUDA training.')  # 无需带参数
    parser.add_argument("--epochs", type=int, default=150, help="Number of training epochs. Default is 200.")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout parameter. Default is 0.5.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate. Default is 0.01.")
    parser.add_argument("--test-ratio", type=float, default=0.1, help="Test data ratio. Default is 0.1.")
    parser.add_argument("--cluster-number", type=int, default=20, help="Number of clusters extracted. Default is 10.")
    parser.add_argument("--cluster-batch",type=int,default=1,help="number of clusters to form a batch. Default is one")

    parser.add_argument('--hidden', type=int, default=8, help='Number of hidden units.')
    parser.add_argument('--nb_heads', type=int, default=8, help='Number of head attentions.')
    parser.add_argument('--patience', type=int, default=100, help='Patience')
    parser.add_argument('--feature_type', nargs="?", default='sqr',
                        help='Nonlinearity function for feature. Can be relu, elu+1, sqr, favor+, or favor+{int}.')
    parser.add_argument('--compute_type', nargs="?", default='iter',
                        help="Which type of method to compute: "
                             "iter = iterative algorithm from Appendix B, "
                             "ps = implementation using torch.cumsum, "
                             "parallel_ps = implementation using custom log prefix sum implementation.")

    args = parser.parse_args()  # args = parser.parse_args()

    torch.manual_seed(args.seed)  # random seed
    tab_printer(args)
    # load data and metis
    graph = graph_reader(args.edge_path)  # graph = nx.Graph(...edges)
    features = feature_reader(args.features_path)  # features = coo_matrix(...)
    target = target_reader(args.target_path)  # target = np.array(size=(-1,1))

    clustering_machine = ClusteringMachine(args, graph, features, target)
    clustering_machine.decompose()

    myGAT = ClusterGATTrainer(args, clustering_machine)
    myGAT.train()
    myGAT.test()


if __name__ == "__main__":
    main()
