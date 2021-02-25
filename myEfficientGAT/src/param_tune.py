import numpy as np
from main_arxiv import main
import random
from main_cora import main
import os

dropout = np.linspace(0.9,0.1,17)
learning_rate = np.linspace(0.0005,0.01, 20)    # np.array([0.001,0.005,0.01,0.015,0.02,0.03,0.05])
cluster_number = np.linspace(1,10,10)
# cluster_batch = np.array([2,4,6,8,10])
hidden = np.array([5,8,10,12,15,18,20])
nb_heads = np.array([3,6,9,12])

for a in dropout:
    for b in learning_rate:
        for c in cluster_number:
            for d in hidden:
                for e in nb_heads:
                    os.system("python main_cora.py --dropout %f --learning-rate %f --cluster-number %d --hidden %d "
                              "--nb_heads %d" % (a,b,c,d,e))


    # dp = dropout[int(random()*len(dropout))]
    # lr = learning_rate[int(random()*len(learning_rate))]
    # cn = cluster_number[int(random()*len(cluster_number))]
    # cb = cluster_batch[int(random()*len(cluster_batch))]
    # hd = hidden[int(random()*len(hidden))]
    # nbh = nb_heads[int(random()*len(nb_heads))]

