import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, using linear attention
    """

    def __init__(self, in_features, out_features, dropout, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.dropout = dropout
        self.in_features = in_features
        self.out_features = out_features
        self.concat = concat
        self.W_Q = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.W_K = nn.Parameter(torch.empty(size=(in_features, out_features)))
        self.W_V = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W_Q.data, gain=1.414)  # Initialize Weight metrix
        nn.init.xavier_uniform_(self.W_K.data, gain=1.414)  # Initialize Weight metrix
        nn.init.xavier_uniform_(self.W_V.data, gain=1.414)  # Initialize Weight metrix

        # self.a = nn.Parameter(torch.empty(size=(2 * out_features, 1)))
        # nn.init.xavier_uniform_(self.a.data, gain=1.414)

    def forward(self, h):
        WQ = torch.mm(h, self.W_Q)  # h.shape: (N, in_features), Wh.shape: (N, out_features)
        Q = F.elu(WQ) + 1
        WK = torch.mm(h, self.W_K)
        K = F.elu(WK) + 1
        V = torch.mm(h, self.W_V)
        KV = K.t().mm(V)
        # Compute the normalizer
        Z = 1 / (Q.matmul(K.t().sum(dim=1, keepdim=True)))  # avoiding the fenmu==0
        # # Finally compute and return the new values
        # print(Q.shape,KV.shape,Z.shape)
        V = torch.mul(Q.mm(KV), Z)  # V.shape: (N,out_features)
        # print(V.shape)
        if self.concat:
            return F.elu(V)  # middle layer, multihead self-attention, concat=True
        else:
            return V  # final prediction layer, delay elu, and don't concat
