import numpy as np
import pandas as pd
import networkx as nx
from texttable import Texttable
from scipy.sparse import coo_matrix
import json
from networkx.readwrite import json_graph


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


def type_reader(path):
    test_nodes = np.loadtxt(path + "test.csv")
    train_nodes = np.loadtxt(path + "train.csv")
    valid_nodes = np.loadtxt(path + "valid.csv")
    type_map = dict().fromkeys(test_nodes, "test")
    type_map2 = dict().fromkeys(train_nodes, "train")
    type_map3 = dict().fromkeys(valid_nodes, "valid")
    type_map.update(type_map2)
    type_map.update(type_map3)
    return type_map


def graphsage_data_reader(dataset_path, dataset_str):
    with open('{}/{}/{}-G.json'.format(dataset_path, dataset_str, dataset_str)) as fp:
        graph_json = json.load(fp)
    graph_nx = json_graph.node_link_graph(graph_json)
    with open('{}/{}/{}-id_map.json'.format(dataset_path, dataset_str, dataset_str)) as fp:
        id_map = json.load(fp)
    is_digit = list(id_map.keys())[0].isdigit()
    id_map = {(int(k) if is_digit else k): int(v) for k, v in id_map.items()}
    with open('{}/{}/{}-class_map.json'.format(dataset_path, dataset_str, dataset_str)) as fp:
        class_map = json.load(fp)
    is_instance = isinstance(list(class_map.values())[0], list)
    class_map = {(int(k) if is_digit else k): (v if is_instance else int(v))
                 for k, v in class_map.items()}
    broken_count = 0
    to_remove = []
    for node in graph_nx.nodes():
        if node not in id_map:
            to_remove.append(node)
            broken_count += 1
    for node in to_remove:
        graph_nx.remove_node(node)
    with open('{}/{}/{}-feats.npy'.format(dataset_path, dataset_str, dataset_str), 'rb') as fp:
        feats = np.load(fp).astype(np.float32)
    num_data = len(id_map)
    if isinstance(list(class_map.values())[0], list):
        num_classes = len(list(class_map.values())[0])
        labels = np.zeros((num_data, num_classes), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], :] = np.array(class_map[k])
    else:
        num_classes = len(set(class_map.values()))
        labels = np.zeros((num_data, 1), dtype=np.float32)
        for k in class_map.keys():
            labels[id_map[k], 1] = class_map[k]
    return graph_nx, labels, feats
