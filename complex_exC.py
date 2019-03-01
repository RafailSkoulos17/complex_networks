import pandas as pd
import networkx as nx
import numpy as np
import pickle
from random import shuffle, sample



def generate_g2(file):
    df = pd.read_excel(file)
    stamps = []
    for index, row in df.iterrows():
        stamps.append(row['timestamp'])
    shuffle(stamps)
    return stamps


def generate_g3(file, L):
    df = pd.read_excel(file)
    stamps = []
    for index, row in df.iterrows():
        stamps.append(row['timestamp'])
    count_dict = dict((x, stamps.count(x)) for x in set(stamps))
    g3_lst = []
    search_list = [i for i in range(L)]
    for key, value in count_dict.items():
        g3_lst += sample(search_list, value)
    return g3_lst


with open('graph.pickle', 'rb') as handle:
    G = pickle.load(handle)

# question 14
g2Timestamps = generate_g2("manufacturing_emails_temporal_network.xlsx")
g3Timestamps = generate_g3("manufacturing_emails_temporal_network.xlsx", G.number_of_edges())
# print(g3Timestamps)
# print(len(g3Timestamps))
# print(len(set(g3Timestamps)))
