import torch
import torch.nn as nn
import torch.nn.functional as F
from LinearAttentionLyr import LinearAttention
from slim_performer import MultiHeadAttention

class GAT(nn.Module):
    def __init__(self, feature_type, compute_type, nfeat, nhid, nclass, nheads):
        """Dense version of GAT."""
        super(GAT, self).__init__()
        self.nheads = nheads
        self.nclass = nclass
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dropout = dropout

        # self.attentions = [LinearAttention(nfeat, nhid, dropout=dropout, concat=True) for _ in range(nheads)]
        # self.attentions: [(N,nhid) * nheads]
        # for i, attention in enumerate(self.attentions):
        #     self.add_module('attention_{}'.format(i), attention)
        self.attention = MultiHeadAttention(feature_type, nheads, nfeat, nhid*nheads, compute_type)    #  head_dim = hidden_dim // nheads
        self.add_model('attention',self.attention)

        # self.out_att = [LinearAttention(nhid * nheads, nclass, dropout=dropout, concat=False) for _ in range(nheads)]
        self.out_att = [MultiHeadAttention(feature_type, 1, nhid*nheads, nclass, compute_type) for _ in range(nheads)]      # nhid = nheads * head_dim
        for i, out_att in enumerate(self.out_att):
            self.add_module('out_att_{}'.format(i), out_att)

    def forward(self, x):
        # x in nNodes * nfeat
        # x = F.dropout(x, self.dropout, training=self.training)
        # x = torch.cat([att(x) for att in self.attentions], dim=1)  # concat multihead, x.shape: (N,nhid*nheads)
        x = self.attention.full_forward(x,self.attention.sample_rfs(self.device))
        x = F.dropout(x, self.dropout, training=self.training)
        # temp = torch.zeros(x.shape[0],self.nclass).to(self.device)

        x = torch.cat([out_att(x).unsqueeze(-1) for out_att in self.out_att],dim=2).sum(dim=2)
        x = x / self.nheads
        x = F.elu(x)    #?
        return F.log_softmax(x, dim=1)




