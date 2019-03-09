import math
import os
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import pandas as pd
import networkx as nx
import numpy as np
import pickle



def plot_recognition_rate(f_values, R_values, metric, ranking):

    f_values = ['{0:.2f}'.format(f) for f in f_values]
    fig, ax = plt.subplots()
    ax.plot(f_values, R_values, color='r', marker='o')
    ax.set(xlabel='f', ylabel='Recognition Rate',
           title='Influence prediction of ' + metric + ' for ' + ranking)
    plt.xticks(f_values, f_values)
    ax.grid()
    # plt.show()
    plt.savefig('figures/influence_prediction_' + '_'.join(metric.split()) + '_for_' + ranking + '.png')



def compute_Rr(f, R, L):
    """
     L is either the degree ranking vector or the clustering coefficient ranking vector
    """
    top_r = math.floor(f*len(R))
    Rf = R[:top_r]
    top_l = math.floor(f*len(L))
    Lf = L[:top_l]

    # keep only the id of the top f nodes
    Rf = [x[0] for x in Rf]
    Lf = [x[0] for x in Lf]

    common_elements = []
    for element in Rf:
        if element in Lf:
            common_elements += [element]

    return len(common_elements)/len(Rf)


def init_infection(G):
    for node in list(G.nodes):
        G.nodes[node]['infected'] = 0
    return G


def plot_infected_nodes(I):
    infected = []
    for key, value in I.items():
        infected.append(value)

    infected = np.array(infected)

    avg_infected_nodes = np.mean(infected, axis=0)
    std_infected_nodes = np.std(infected, axis=0)

    x = range(57792)
    fig, ax = plt.subplots()
    ax.errorbar(x, avg_infected_nodes, yerr=std_infected_nodes, ecolor='g', linestyle='None', marker='o', linewidth=2,
                markersize=12)
    ax.locator_params(axis='x', nbins=4)

    loc = plticker.MultipleLocator(base=10000)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)

    ax.set(xlabel='Timestep', ylabel='Infected Nodes',
           title='Mean and std number of affected nodes per timestep')
    ax.grid()
    plt.savefig('figures/mean_and_std_of_affected_nodes_per_timestep.png')
    # plt.show()


def plot_infected_nodes_sampled(I):
    infected = []
    for key, value in I.items():
        infected.append(value[0:len(value):1000])

    infected = np.array(infected)

    avg_infected_nodes = np.mean(infected, axis=0)
    std_infected_nodes = np.std(infected, axis=0)

    x = range(0, 57791, 1000)
    fig, ax = plt.subplots()
    ax.errorbar(x, avg_infected_nodes, yerr=std_infected_nodes, ecolor='g', linestyle='None', marker='o')
    ax.locator_params(axis='x', nbins=4)

    loc = plticker.MultipleLocator(base=10000)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)

    ax.set(xlabel='Timestep', ylabel='Infected Nodes',
           title='Mean and std number of affected nodes per timestep')
    ax.grid()
    # plt.show()
    plt.savefig('figures/sampled_mean_and_std_of_affected_nodes_per_timestep.png')


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


def rank_by_avg_influence(I, N):
    rList = []
    for key, value in I.items():
        temp1 = [i for i, n in enumerate(value) if n >= 0.8 * N]
        if temp1!=[]:
            temp = value[:temp1[0]+1]
            diff_lst = [(i+1)*n for i, n in enumerate([j - i for i, j in zip(temp[:-1], temp[1:])])]
            rList.append([key, sum(diff_lst)/value[temp1[0]]])
        else:
            rList.append([key, len(value) + 1])
    return list(sorted(rList, key=lambda x: int(x[1])))


def create_graph_for_specific_timestamp(edgeList):
    G = nx.Graph()
    for edge in edgeList:
        G.add_edge(edge[0], edge[1])
    return G


def calculate_temporal_closeness_centrality(timeD):
    clossD = defaultdict(list)
    for t in timeD:
        Gt = create_graph_for_specific_timestamp(timeD[t])
        closs = nx.closeness_centrality(Gt)  # returns dictionary {'node1':closs1, 'node2': closs2...}
        for key, value in closs.items():
            clossD[key].append(value)
    for node in clossD:
        clossD[node] = np.mean(clossD[node])
    return clossD

# def create_edgelist_per_time(file):
#     timeD = {} # create edgelist
#     df = pd.read_excel(file)
#     for index, row in df.iterrows():
#         vertex1 = int(row[0])
#         vertex2 = int(row[1])
#         t = int(row[2])
#
#         # Create edgelist per time - Adding one edge at a time
#         if t in timeD:
#             timeD[t].append((vertex1, vertex2))
#         else:
#             timeD[t] = []
#             timeD[t].append((vertex1, vertex2))
#     with open('edgelist_per_time.pickle', 'wb') as handle:
#         pickle.dump(timeD, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_temporal_degree(timeD):
    tempDegreeD = defaultdict(list)
    for t in range(1, max(timeD)+1):
        Gt = create_graph_for_specific_timestamp(timeD[t])
        # tempDegreeList = list(dict(Gt.degree(Gt.nodes())).values())
        tempDegreeDict = dict(Gt.degree(Gt.nodes()))
        for key, value in tempDegreeDict.items():
            tempDegreeD[key].append(value)
    for node in tempDegreeD:
        tempDegreeD[node] = np.mean(tempDegreeD[node])
    return tempDegreeD


figures_dir = 'figures'
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)


with open('graph.pickle', 'rb') as handle:
    G = pickle.load(handle)
# G = init_infection(G)
# I = simulate_infection("manufacturing_emails_temporal_network.xlsx", G)
# with open('infection_sim_v2.pickle', 'wb') as handle:
#     pickle.dump(I, handle, protocol=pickle.HIGHEST_PROTOCOL)

# for key, value in I.items():
#     print("with node "+str(key)+" as seed")
#     print(value)
with open('infection_sim_v2.pickle', 'rb') as handle:
    I = pickle.load(handle)

#  Question 9
plot_infected_nodes_sampled(I)
plot_infected_nodes(I)

#  Question 10
R = rank_influence(I, G.number_of_nodes())
# print(R)

# Question 11

degree = dict(G.degree)
clustering_coefficient = dict(nx.clustering(G))

degree_list = degree.items()
cc_list = clustering_coefficient.items()

cc_list = list(sorted(cc_list, key=lambda x: int(x[1]), reverse=True))
degree_list = list(sorted(degree_list, key=lambda x: int(x[1]), reverse=True))

f_values = np.arange(0.05, 0.55, 0.05)
Rrd = [compute_Rr(f, R, degree_list) for f in f_values]
Rrc = [compute_Rr(f, R, cc_list) for f in f_values]

plot_recognition_rate(f_values, Rrd, "Degree", "R")
plot_recognition_rate(f_values, Rrc, "Clustering Coefficient", "R")


# Question 12

readfile = "manufacturing_emails_temporal_network.xlsx"
# create_edgelist_per_time(readfile)
with open('edgelist_per_time.pickle', 'rb') as handle:
    edge_list_per_time = pickle.load(handle)

temporal_degree = calculate_temporal_degree(edge_list_per_time)
temporal_closeness_centrality = calculate_temporal_closeness_centrality(edge_list_per_time)

temporal_degree_list = temporal_degree.items()
temporal_closeness_centrality_list = temporal_closeness_centrality.items()

temporal_degree_list = list(sorted(temporal_degree_list, key=lambda x: int(x[1]), reverse=True))
temporal_closeness_centrality_list = list(sorted(temporal_closeness_centrality_list, key=lambda x: int(x[1]), reverse=True))

Rrtd = [compute_Rr(f, R, temporal_degree_list) for f in f_values]
Rrtc = [compute_Rr(f, R, temporal_closeness_centrality_list) for f in f_values]

plot_recognition_rate(f_values, Rrtc, "temporal closeness centrality", "R")
plot_recognition_rate(f_values, Rrtd, "temporal degree", "R")

# Question 13
R_star = rank_by_avg_influence(I, G.number_of_nodes())
# print(R_star)

R_star_rd = [compute_Rr(f, R_star, degree_list) for f in f_values]
Rstar_rc = [compute_Rr(f, R_star, cc_list) for f in f_values]
Rstar_rr = [compute_Rr(f, R_star, R) for f in f_values]


plot_recognition_rate(f_values, R_star_rd, "Degree", "R'")
plot_recognition_rate(f_values, Rstar_rc, "Clustering_Coefficient", "R'")
plot_recognition_rate(f_values, Rstar_rr, "Influence_Ranking", "R'")
