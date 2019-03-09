import pandas as pd
import networkx as nx
import numpy as np
import pickle
from random import shuffle, sample


def createStructure(df):
    G = nx.Graph()
    for index, row in df.iterrows():
        if not G.has_node(int(row['node1'])):
            G.add_node(int(row['node1']))
        if not G.has_node(int(row['node2'])):
            G.add_node(int(row['node2']))
        if not G.has_edge(int(row['node1']), int(row['node2'])):
            G.add_edge(int(row['node1']), int(row['node2']))
    return G


def generate_g2(file):
    df = pd.read_excel(file)
    df['timestamp'] = np.random.permutation(df['timestamp'].values)
    df = df.sort_values(by=['timestamp'])
    return df


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


# with open('graph.pickle', 'rb') as handle:
#     G = pickle.load(handle)

# question 14

# use this to generate and save the G2 graph. then you just load it

# g2_df = generate_g2("manufacturing_emails_temporal_network.xlsx")
# G2 = createStructure(g2_df)
# with open('graph2.pickle', 'wb') as handle:
#     pickle.dump(G2, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('graph2.pickle', 'rb') as handle:
    G2 = pickle.load(handle)

# g3Timestamps = generate_g3("manufacturing_emails_temporal_network.xlsx", G.number_of_edges())
# print(g3Timestamps)
# print(len(g3Timestamps))
# print(len(set(g3Timestamps)))
