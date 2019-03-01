import pandas as pd
import networkx as nx
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


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
    ax.errorbar(x, avg_infected_nodes, yerr=std_infected_nodes, ecolor='g', linestyle='None', marker='o')
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
plot_infected_nodes(I)

#  Question 10
R = rank_influence(I, G.number_of_nodes())
print(R)



