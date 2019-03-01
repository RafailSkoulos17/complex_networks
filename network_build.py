import copy

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker


def plot_degree_distribution(deg):
    degrees = list(deg.values())
    deg_freq = {}
    for deg in set(degrees):
        deg_freq[str(deg)] = degrees.count(deg) / len(degrees)

    x = map(int, list(deg_freq.keys()))
    y = map(float, list(deg_freq.values()))
    x, y = zip(*sorted(zip(x, y)))

    fig, ax = plt.subplots()
    ax.plot(x, y, color='r', marker='o')
    ax.set(xlabel='Degrees', ylabel='Frequency',
           title='Degree distribution')

    loc = plticker.MultipleLocator(base=10)  # this locator puts ticks at regular intervals
    ax.xaxis.set_major_locator(loc)

    # plt.xscale('log')
    # plt.yscale('log')

    ax.grid()
    plt.show()


def make_network(file):
    df = pd.read_excel(file)
    net = {}
    for index, row in df.iterrows():
        if str(row['node1']) not in net:
            net[str(row['node1'])] = [str(row['node2'])]
        else:
            if str(row['node2']) not in net[str(row['node1'])]:
                net[str(row['node1'])] += [str(row['node2'])]
        if str(row['node2']) not in net:
            net[str(row['node2'])] = [str(row['node1'])]
        else:
            if str(row['node1']) not in net[str(row['node2'])]:
                net[str(row['node2'])] += [str(row['node1'])]
    return net


def calculateAssortativity(net, deg, L):
    d2 = sum(map(lambda x: x ** 2, list(deg.values())))
    d3 = sum(map(lambda x: x ** 3, list(deg.values())))
    dLink = 0
    for i in deg.keys():
        for j in net.keys():
            if j in net[i]:
                dLink += (deg[i]-deg[j])**2
    dLink = dLink/2
    return 1-dLink/(d3-(d2**2)/(2*L))


def calculateClusteringCoeff(net, deg):
    c = {}
    for key, value in net.items():
        li = 0
        for v in net[key]:
            for v_check in net[key]:
                if v in net[v_check]:
                    li += 1
        if deg[key] == 1:
            c[key] = 0
        else:
            c[key] = li/(deg[key]*(deg[key]-1))
    #print(c)
    return sum(list(c.values()))/len(net)


# net = make_network("manufacturing_emails_temporal_network.xlsx")
# with open('network.json', 'w') as outfile:
#     json.dump(net, outfile)
with open('network.json') as json_file:
    net = json.load(json_file)

deg = {}
for key, value in net.items():
    if len(set(value)) != len(value):
        print("malakia was played")
    deg[key] = len(value)

numNodes = len(net)
numLinks = int(sum([len(value) for key, value in net.items()])/2)
graphDensity = 2*numLinks/(numNodes*(numNodes-1))
avgDegree = 2*numLinks/numNodes
varDegree = np.var(list(deg.values()))
print("numNodes: "+str(numNodes))
print("numLinks: "+str(numLinks))
print("graphDensity: "+str(graphDensity))
print("avgDegree: "+str(avgDegree))
print("varDegree: "+str(varDegree))


clustCoeff = calculateClusteringCoeff(net, deg)
assortativity = calculateAssortativity(net, deg, numLinks)
print("clustCoeff: "+str(clustCoeff))
print("assortativity: "+str(assortativity))


plot_degree_distribution(deg)
