import math

import pandas as pd
import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

def plot_recognition_rate(f_values, R_values, metric):

    f_values = ['{0:.2f}'.format(f) for f in f_values]
    fig, ax = plt.subplots()
    ax.plot(f_values, R_values, color='r', marker='o')
    ax.set(xlabel='f', ylabel='Recognition Rate',
           title='Recognition rate per f for ' + metric + ' metric')
    plt.xticks(f_values, f_values)
    ax.grid()
    plt.show()


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
    plt.show()


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
    plt.show()


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

#  Question 10
R = rank_influence(I, G.number_of_nodes())
print(R)

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

plot_recognition_rate(f_values, Rrd, "Degree")
plot_recognition_rate(f_values, Rrc, "Clustering Coefficient")

# Question 12

# Question 13
R_star = rank_by_avg_influence(I, G.number_of_nodes())
print(R_star)



