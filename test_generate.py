from graph_tool.generation import random_graph
from graph_tool.generation import random_rewire
from graph_tool.generation import generate_sbm
import numpy as np
import random
import math
import os

#outputs the graph collection. Only for undirected graphs!
def write_to_files(collection_name, graphs, labels):
    path = os.path.join('datasets', collection_name)
    if not os.path.exists(path):
        os.mkdir(path)

    with open(os.path.join(path, collection_name + '_A.txt'), 'w+') as f:
        node_id = 1
        for i, graph in enumerate(graphs, start=1):
            for edge in graph.edges():
                f.write(str(graph.vertex_index[edge.source()]+node_id))
                f.write(', ')
                f.write(str(graph.vertex_index[edge.target()]+node_id))
                f.write('\n')
                f.write(str(graph.vertex_index[edge.target()]+node_id))
                f.write(', ')
                f.write(str(graph.vertex_index[edge.source()]+node_id))
                f.write('\n')
            node_id += graph.num_vertices()

    with open(os.path.join(path, collection_name + '_edge_labels.txt'), 'w+') as f:
        for i, graph in enumerate(graphs, start=1):
            for edge in graph.edges():
                f.write('1')
                f.write('\n')
                f.write('1')
                f.write('\n')

    with open(os.path.join(path, collection_name + '_graph_indicator.txt'), 'w+') as f:
        for i, graph in enumerate(graphs, start=1):
            for node in graph.vertices():
                f.write(str(i))
                f.write('\n')

    with open(os.path.join(path, collection_name + '_node_labels.txt'), 'w+') as f:
        for i, graph in enumerate(graphs, start=1):
            for node in graph.vertices():
                node_degree = len(graph.get_out_edges(node))
                f.write(str(node_degree))
                f.write('\n')

    with open(os.path.join(path, collection_name + '_graph_labels.txt'), 'w+') as f:
        for label in labels:
            f.write(str(label))
            f.write('\n')



def generate_GNE(n, m):
    ak = math.floor(2 * m / n)
    dm = 2 * m - ak * n
    g = random_graph(n, lambda i: ak + 1 if i < dm else ak, directed=False, random=False)
    random_rewire(g, model='erdos', n_iter=10, edge_sweep=True)
    return g

def generate_GNP(n, p):
    m = 0
    for i in range(round(n*(n-1)/2)):
        if random.random() < p:
            m += 1
    return generate_GNE(n, m)

def generate_block_model(nodes, groups, in_group_p, between_group_p):
    group_memberships = []
    group_sizes = [0] * groups
    for i in range(nodes):
        group_memberships.append((i % groups))
        group_sizes[i % groups] += 1
    probabilities = np.ndarray([groups, groups])
    for i in range(groups):
        for j in range(groups):
            if i == j:
                probabilities[i][j] = in_group_p * group_sizes[i] * group_sizes[j]
            else:
                probabilities[i][j] = between_group_p * group_sizes[i] * group_sizes[j] / 2
    return generate_sbm(group_memberships, probabilities)

labels = []
graphs = []
for i in range(50):
    #graphs.append(generate_GNP(70, 0.08))
    graphs.append(generate_block_model(100, 2, 0.2, 0.01))
    labels.append(1)

for i in range(50):
    #graphs.append(generate_GNP(72, 0.08))
    temp_graph = generate_block_model(100, 2, 0.2, 0.01)
    random_rewire(temp_graph, model='erdos', n_iter=10, edge_sweep=True)
    graphs.append(temp_graph)
    labels.append(2)
#graphs.append(generate_GNP(100, 0.2))
#graphs.append(generate_block_model(100, 2, 0.8, 0.2))
write_to_files("EXP3", graphs, labels)