import pandas as pd
import networkx as nx
import numpy as np
import pickle


def createStructure(file):
    df = pd.read_excel(file)
    G = nx.Graph()
    for index, row in df.iterrows():
        if not G.has_node(int(row['node1'])):
            G.add_node(int(row['node1']))
        if not G.has_node(int(row['node2'])):
            G.add_node(int(row['node2']))
        if not G.has_edge(int(row['node1']), int(row['node2'])):
            G.add_edge(int(row['node1']), int(row['node2']))
    return G


# G = createStructure("manufacturing_emails_temporal_network.xlsx")
# with open('graph.pickle', 'wb') as handle:
#     pickle.dump(G, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('graph.pickle', 'rb') as handle:
    G = pickle.load(handle)

# question A1
numNodes = G.number_of_nodes()
numLinks = G.number_of_edges()
graphDensity = 2*numLinks/(numNodes*(numNodes-1))
avgDegree = 2*numLinks/numNodes
deg = dict(G.degree(G.nodes))
varDegree = np.var(list(deg.values()))
print("numNodes: "+str(numNodes))
print("numLinks: "+str(numLinks))
print("graphDensity: "+str(graphDensity))
print("avgDegree: "+str(avgDegree))
print("varDegree: "+str(varDegree))

# question A3
assortativity = nx.degree_assortativity_coefficient(G)
print("assortativity: "+str(assortativity))

# question A4
clustCoeff = nx.average_clustering(G)
print("clustCoeff: "+str(clustCoeff))

# question A5
avg_hopcount = nx.average_shortest_path_length(G)
diameter = -1
for value in dict(nx.all_pairs_shortest_path_length(G)).values():
    if max(value.values())>diameter:
        diameter = max(value.values())
print("avg_hopcount: "+str(avg_hopcount))
print("diameter: "+str(diameter))

# question A6
isSmallWorld = nx.omega(G)
print("omega coefficient: "+str(isSmallWorld))