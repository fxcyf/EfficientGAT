from datetime import datetime
import torch
import random
import numpy as np
from tqdm import trange, tqdm  # 进度条
from layers import StackedGCN
from torch.autograd import Variable
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from GATmodels import GAT
import json
from pytorchtools import EarlyStopping


class ClusterGATTrainer(object):
    """
    Training a ClusterGCN.
    """

    def __init__(self, args, clustering_machine):
        """
        :param ags: Arguments object.
        :param clustering_machine:
        """
        self.args = args
        self.clustering_machine = clustering_machine
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #     self.create_model()
        #
        # def create_model(self):
        #     """
        #     Creating a StackedGCN and transferring to CPU/GPU.
        #     """
        self.model = GAT(self.clustering_machine.feature_count,
                         self.args.hidden,
                         self.clustering_machine.class_count,
                         self.args.dropout,
                         self.args.nb_heads)
        self.model = self.model.to(self.device)  # self.model.layers <-- torch.nn.Module + self define modules

    def do_forward_pass(self, epoch, clusters):
        """
        Making a forward pass with data from a given partition.
        :param cluster: Cluster index.
        :return average_loss: Average loss on the cluster.
        :return node_count: Number of nodes.
        """
        edges = torch.LongTensor()
        macro_nodes = torch.LongTensor()
        train_nodes = torch.LongTensor()
        features = torch.LongTensor()
        target = torch.LongTensor()
        for cluster in clusters:
            edges = torch.cat((edges, self.clustering_machine.sg_edges[cluster]), 1)
            macro_nodes = torch.cat((macro_nodes, self.clustering_machine.sg_nodes[cluster]), 0)
            train_nodes = torch.cat((train_nodes, self.clustering_machine.sg_train_nodes[cluster]), 0)
            features = torch.cat((features, self.clustering_machine.sg_features[cluster]), 0)
            # print(features.shape)
            target = torch.cat((target, self.clustering_machine.sg_targets[cluster]), 0)
        edges = edges.to(self.device)
        macro_nodes = macro_nodes.to(self.device)
        train_nodes = train_nodes.to(self.device)
        features = features.to(self.device)
        target = target.to(self.device).squeeze()
        # print(train_nodes.shape,'\n',features.shape,'\n',target.shape)
        predictions = self.model(features)  # predictions in [N,out_channels]
        # np.savetxt('../input/ogbn_arxiv/results/targets-{}-{}.txt'.format(epoch, clusters[0]),
        #            target.cpu().detach().numpy())
        # np.savetxt('../input/ogbn_arxiv/results/predictions-{}-{}.txt'.format(epoch, clusters[0]),
        #            predictions.cpu().detach().numpy())
        # print('target: ',target,'\n','pred: ',predictions)
        average_loss = torch.nn.functional.nll_loss(predictions[train_nodes], target[
            train_nodes])  # The negative log likelihood loss: nll_loss(input--(N,C),targer--(N))
        node_count = train_nodes.shape[0]
        return average_loss, node_count

    def update_average_loss(self, batch_average_loss, node_count, torv="train", lossoracc="loss"):
        """
        Updating the average loss in the epoch.
        :param batch_average_loss: Loss of the cluster. 
        :param node_count: Number of nodes in currently processed cluster.
        :return average_loss: Average loss in the epoch.
        """
        if lossoracc == 'loss':
            if torv == 'train':
                self.accumulated_training_loss = self.accumulated_training_loss + batch_average_loss.item() * node_count
                self.train_node_count_seen = self.train_node_count_seen + node_count
                average_loss = self.accumulated_training_loss / self.train_node_count_seen
            elif torv == 'valid':
                self.accumulated_valid_loss = self.accumulated_valid_loss + batch_average_loss.item() + node_count
                self.valid_node_count_seen = self.valid_node_count_seen + node_count
                average_loss = self.accumulated_valid_loss / self.valid_node_count_seen
        elif lossoracc == 'acc':
            if torv == 'valid':
                self.accumulated_valid_acc = self.accumulated_valid_acc + batch_average_loss.item() + node_count
                # self.valid_node_count_seen = self.valid_node_count_seen + node_count
                average_loss = self.accumulated_valid_acc / self.valid_node_count_seen
        return average_loss

    def do_prediction(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        test_nodes = self.clustering_machine.sg_test_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        target = target[test_nodes]
        prediction = self.model(features)
        prediction = prediction[test_nodes, :]
        return prediction, target

    def do_validation(self, cluster):
        """
        Scoring a cluster.
        :param cluster: Cluster index.
        :return prediction: Prediction matrix with probabilities.
        :return target: Target vector.
        """
        edges = self.clustering_machine.sg_edges[cluster].to(self.device)
        macro_nodes = self.clustering_machine.sg_nodes[cluster].to(self.device)
        valid_nodes = self.clustering_machine.sg_valid_nodes[cluster].to(self.device)
        features = self.clustering_machine.sg_features[cluster].to(self.device)
        target = self.clustering_machine.sg_targets[cluster].to(self.device).squeeze()
        prediction = self.model(features)
        average_loss = torch.nn.functional.nll_loss(prediction[valid_nodes], target[
            valid_nodes])  # The negative log likelihood loss: nll_loss(input--(N,C),targer--(N))
        # acc = accuracy_score(target[valid_nodes].cpu().detach().numpy(),prediction[valid_nodes].cpu().detach().numpy().argmax(1))
        node_count = valid_nodes.shape[0]
        return average_loss, node_count, prediction[valid_nodes], target[valid_nodes]

    def train(self):
        """
        Training a model.
        """
        print("Training started.\n")
        epochs = trange(self.args.epochs, desc="Train Loss")  # 进度条
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        earlystopping = EarlyStopping(patience=self.args.patience)
        train_loss = []
        valid_loss = []
        for epoch in epochs:  # go through every cluster per epoch
            self.model.train()
            self.train_node_count_seen = 0
            self.accumulated_training_loss = 0
            # self.accumulated_valid_acc = 0
            random.shuffle(self.clustering_machine.clusters)
            cluster_batch = self.args.cluster_batch
            batch_num = int(self.args.cluster_number / cluster_batch)
            for i in range(batch_num):
                # for cluster in self.clustering_machine.clusters:
                clusters = self.clustering_machine.clusters[cluster_batch * i:cluster_batch * (i + 1)]
                # print(clusters)
                self.optimizer.zero_grad()  # set all parameters to zeros
                batch_average_loss, node_count = self.do_forward_pass(epoch,
                                                                      clusters)  # mind this, we may use more than one cluster
                batch_average_loss.backward()
                self.optimizer.step()
                average_train_loss = self.update_average_loss(batch_average_loss, node_count)
            train_loss.append(average_train_loss)
            # validation for early stopping
            self.model.eval()
            predictions = []
            targets = []
            self.valid_node_count_seen = 0
            self.accumulated_valid_loss = 0
            for cluster in self.clustering_machine.clusters:
                batch_average_loss, node_count, prediction, target = self.do_validation(cluster)
                average_valid_loss = self.update_average_loss(batch_average_loss, node_count, torv='valid')
                predictions.append(prediction.cpu().detach().numpy())
                targets.append(target.cpu().detach().numpy())
            valid_loss.append(average_valid_loss)
            targets = np.concatenate(targets)
            predictions = np.concatenate(predictions).argmax(1)
            valid_acc = accuracy_score(targets, predictions)
            # average_valid_acc = self.update_average_loss(valid_acc,node_count, torv = 'valid',lossoracc='acc')
            # update valid acc but dont update valid node seen

            epochs.set_description("Train Loss: {}, Valid Loss: {}, Valid ACC: {}"
                                   .format(round(average_train_loss, 4), round(average_valid_loss, 4),
                                           round(valid_acc, 4)))
            earlystopping(average_valid_loss, self.model)
            if earlystopping.early_stop:
                print("Early Stopping in epoch{}\n".format(epoch))

                break;
        self.model.load_state_dict(torch.load('checkpoint.pt'))

    def test(self):
        """
        Scoring the test and printing the F-1 score.
        """
        self.model.eval()
        self.predictions = []
        self.targets = []
        for cluster in self.clustering_machine.clusters:
            prediction, target = self.do_prediction(cluster)
            self.predictions.append(prediction.cpu().detach().numpy())
            self.targets.append(target.cpu().detach().numpy())
        self.targets = np.concatenate(self.targets)
        self.predictions = np.concatenate(self.predictions).argmax(1)
        score = f1_score(self.targets, self.predictions, average="micro")
        acc = accuracy_score(self.targets, self.predictions)
        iso_date = datetime.now().isoformat().replace(':', '-')[:-7]
        np.savetxt('../input/ogbn_arxiv/results/targets-{}.txt'.format(iso_date), self.targets)
        np.savetxt('../input/ogbn_arxiv/results/predictions-{}.txt'.format(iso_date), self.predictions)
        param = {'acc': acc,
                 'F-1 score': score,
                 'epochs': self.args.epochs,
                 'dropout': self.args.dropout,
                 'lr': self.args.learning_rate,
                 'cluster_number': self.args.cluster_number,
                 'cluster_batch': self.args.cluster_batch,
                 'hidden': self.args.hidden,
                 'nb_heads': self.args.nb_heads,
                 'patience': self.args.patience}
        with open('../input/ogbn_arxiv/results/param-{}.json'.format(iso_date), 'w') as f:
            json.dump(param, f)
        # ../input/ogbn_arxiv/results/targets.txt; ../input/results/targets.txt
        # ../input/ogbn_arxiv/results/predictions.txt; ../input/results/predictions.txt
        print("\nF-1 score: {:.4f}".format(score))
        print("\nAccuracy: {:.4f}".format(acc))
