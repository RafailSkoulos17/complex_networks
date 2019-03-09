import pandas as pd
import networkx as nx
import numpy as np
import pickle
from random import randint
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


def generate_g2(file):
    df = pd.read_excel(file)
    df['timestamp'] = np.random.permutation(df['timestamp'].values)
    df = df.sort_values(by=['timestamp'])
    return df


def generate_g3(file, G):
    L = G.number_of_edges()
    edge_dict = {}
    for i, e in enumerate(list(G.edges)):
        edge_dict[i] = e
    df = pd.read_excel(file)
    stamps = []
    for index, row in df.iterrows():
        stamps.append(row['timestamp'])
    g3_lst = {}
    for s in stamps:
        link = randint(0, L-1)
        if link not in g3_lst.keys():
            g3_lst[link] = [s]
        else:
            g3_lst[link].append(s)
    for key in g3_lst.keys():
        g3_lst[key] = list(set(g3_lst[key]))
    dict_df = dict()
    dict_df['node1'] = {}
    dict_df['node2'] = {}
    dict_df['timestamp'] = {}
    i = 0
    for key in g3_lst.keys():
        for t in g3_lst[key]:
            dict_df['node1'][i] = edge_dict[key][0]
            dict_df['node2'][i] = edge_dict[key][1]
            dict_df['timestamp'][i] = t
            i += 1
    return pd.DataFrame.from_dict(dict_df).sort_values(by=['timestamp'])


def init_infection(G):
    for node in list(G.nodes):
        G.nodes[node]['infected'] = 0
    return G


def simulate_infection(df, G):
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


with open('graph.pickle', 'rb') as handle:
    G = pickle.load(handle)
G = init_infection(G)
# # question 14, 15
#
# g2_df = generate_g2("manufacturing_emails_temporal_network.xlsx")
# I = simulate_infection(g2_df, G)
# with open('infection_sim_G2.pickle', 'wb') as handle:
#     pickle.dump(I, handle, protocol=pickle.HIGHEST_PROTOCOL)

# g3_df = generate_g3("manufacturing_emails_temporal_network.xlsx", G)
# I = simulate_infection(g3_df, G)
# with open('infection_sim_G3.pickle', 'wb') as handle:
#     pickle.dump(I, handle, protocol=pickle.HIGHEST_PROTOCOL)

