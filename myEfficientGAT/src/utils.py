import torch
import numpy as np
import pandas as pd
import networkx as nx
from texttable import Texttable
from scipy.sparse import coo_matrix


def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)  # namespace to dictionary
    keys = sorted(args.keys())  # list of attribute name
    t = Texttable()
    t.add_rows([["Parameter", "Value"]] + [[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

#
def graph_reader(path):
    """
    Function to read the graph from the path.\nThe node_id must form a continual linear space without skipping a number.
    :param path: Path to the edge list.
    :return graph: NetworkX object returned.
    """
    graph = nx.from_edgelist(pd.read_csv(path).values.tolist())

    return graph


def feature_reader(path):
    """
    Reading the sparse feature matrix stored as csv from the disk.\nThe node_id must form a continual linear space without skipping a number.
    :param path: Path to the csv file.
    :return features: Dense matrix of features.
    """
    features = pd.read_csv(path)  # index and feature_id begins with 0
    node_index = features["node_id"].values.tolist()
    feature_index = features["feature_id"].values.tolist()
    feature_values = features["value"].values.tolist()
    node_count = max(node_index) + 1  # 1,2,3... + 0
    feature_count = max(feature_index) + 1  # 1,2,3... + 0
    features = coo_matrix((feature_values, (node_index, feature_index)),
                          shape=(node_count, feature_count)).toarray()  # feature in [N,C]
    return features


def target_reader(path):
    """
    Reading the target vector from disk. The node_id must form a continual linear space without skipping a number.
    :param path: Path to the target.
    :return target: Target vector.
    """
    target = np.array(pd.read_csv(path)["target"]).reshape(-1, 1)
    # reshape to be one colomn, #row will be cal automatically
    return target
