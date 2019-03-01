import pandas as pd
import json
import numpy as np


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


# net = make_network("manufacturing_emails_temporal_network.xlsx")
# with open('network.json', 'w') as outfile:
#     json.dump(net, outfile)
with open('network.json') as json_file:
    net = json.load(json_file)

deg = {}
for key, value in net.items():
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
