import numpy as np
from main_ogbn import main
import random

dropout = np.linspace(0.1,0.8,21)
learning_rate = np.array([0.001,0.005,0.01,0.015,0.02,0.03,0.05])
cluster_number = np.linspace(100,500,9)
cluster_batch = np.array([2,4,6,8,10])
hidden = np.array([5,8,10,12,15,18,20])
nb_heads = np.array([3,6,9,12])

for i in range(20):
    dp = dropout[int(random()*len(dropout))]
    lr = learning_rate[int(random()*len(learning_rate))]
    cn = cluster_number[int(random()*len(cluster_number))]
    cb = cluster_batch[int(random()*len(cluster_batch))]
    hd = hidden[int(random()*len(hidden))]
    nbh = nb_heads[int(random()*len(nb_heads))]

