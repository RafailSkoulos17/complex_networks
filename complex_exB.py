import pandas as pd
import networkx as nx
import numpy as np
import pickle


def init_infection(G):
    for node in list(G.nodes):
        G.nodes[node]['infected'] = 0
    return G


def simulate_infection(file, G):
    df = pd.read_excel(file)
    I = {}
    for node in list(G.nodes):
        print(node)
        G = init_infection(G)
        G.nodes[node]['infected'] = 1
        temp = []
        t = 0
        infection_set = [node]
        for index, row in df.iterrows():
            if row['timestamp'] != t:
                if temp != []:
                    temp.append(temp[-1]+len(set(infection_set)))
                else:
                    temp.append(len(set(infection_set)))
                for inf_node in set(infection_set):
                    G.nodes[inf_node]['infected'] = 1
                infection_set = []
                t = row['timestamp']
            if G.nodes[row['node1']]['infected'] == 1 and G.nodes[row['node2']]['infected'] == 0:
                infection_set.append(row['node2'])
            if G.nodes[row['node1']]['infected'] == 0 and G.nodes[row['node2']]['infected'] == 1:
                infection_set.append(row['node1'])
        if temp != []:
            temp.append(temp[-1] + len(set(infection_set)))
        else:
            temp.append(len(set(infection_set)))
        for inf_node in set(infection_set):
            G.nodes[inf_node]['infected'] = 1
        I[node] = temp
    return I


def rank_influence(I, N):
    rList = []
    for key, value in I.items():
        temp = [i for i, n in enumerate(value) if n >= 0.8*N]
        if temp != []:
            rList.append([key, temp[0]])
        else:
            rList.append([key, len(value)+1])
    return list(sorted(rList, key=lambda x: int(x[1])))


with open('graph.pickle', 'rb') as handle:
    G = pickle.load(handle)
# G = init_infection(G)
# I = simulate_infection("manufacturing_emails_temporal_network.xlsx", G)
# with open('infection_sim.pickle', 'wb') as handle:
#     pickle.dump(I, handle, protocol=pickle.HIGHEST_PROTOCOL)
# for key, value in I.items():
#     print("with node "+str(key)+" as seed")
#     print(value)
with open('infection_sim.pickle', 'rb') as handle:
    I = pickle.load(handle)
R = rank_influence(I, G.number_of_nodes())
print(R)
