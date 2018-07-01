from graph_tool.generation import random_graph
from graph_tool.generation import random_rewire
from graph_tool.generation import generate_sbm
import numpy as np
import random
import math
import os
from distutils.dir_util import copy_tree
import graph_tool

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

def write_modified_dataset(previous_name, new_name, graphs, prop_maps):
    path = os.path.join('datasets', new_name)
    orig_path = os.path.join('datasets', previous_name)
    #THIS ASSUMES THAT TARGET FOLDER DOES NOT EXIST
    copy_tree(orig_path, path)

    with open(os.path.join(path, new_name + '_A.txt'), 'w+') as f:
        node_id = 0
        current_line = 1
        for i, graph in enumerate(graphs, start=0):
            for edge in graph.edges():
                #find correct edge to write
                right_edge = 0
                for e in graph.edges():
                    if prop_maps[i][e] == current_line:
                        right_edge = e
                        break
                f.write(str(graph.vertex_index[right_edge.source()]+node_id))
                f.write(', ')
                f.write(str(graph.vertex_index[right_edge.target()]+node_id))
                f.write('\n')
                current_line += 1


def read_dataset(collection_name):
    path = os.path.join('datasets', collection_name)

    node_to_graph_ids = [-1]
    graph_number = 0
    with open(os.path.join(path, collection_name + '_graph_indicator.txt'), 'r') as f:
        li = f.readline().rstrip('\n')
        while li != '':
            x = int(li)
            node_to_graph_ids.append(x-1)
            graph_number = max(graph_number, x)
            li = f.readline().rstrip('\n')
    graphs = []
    for i in range(graph_number):
        graphs.append(graph_tool.Graph())
    prop_maps = []
    for gra in graphs:
        x = gra.new_edge_property('int')
        prop_maps.append(x)
    orig_line = 1
    with open(os.path.join(path, collection_name + '_A.txt'), 'r') as f:
        li = f.readline().rstrip('\n')
        while li != '':
            x = li.split(', ')
            x1 = int(x[0])
            x2 = int(x[1])
            graph_no = node_to_graph_ids[x1]
            e = graphs[graph_no].add_edge(x1, x2)
            prop_maps[graph_no][e] = orig_line
            orig_line += 1
            li = f.readline().rstrip('\n')
    return graphs, prop_maps

def rewire_dataset_with_preserved_degree(orig_dataset_name, new_dataset_name):
    graphs, prop_maps = read_dataset(orig_dataset_name)
    for g in graphs:
        random_rewire(g, model='configuration', n_iter=10, edge_sweep=True)
    write_modified_dataset(orig_dataset_name, new_dataset_name, graphs, prop_maps)


def rewire_dataset_partially(orig_dataset_name, new_dataset_name, p=0.1):
    graphs, prop_maps = read_dataset(orig_dataset_name)
    for g in graphs:
        pins = g.new_edge_property('int')
        for e in g.edges():
            if random.random() < p:
                pins[e] = 0
            else:
                pins[e] = 1
        random_rewire(g, model='erdos', n_iter=10, edge_sweep=True, pin=pins)
    write_modified_dataset(orig_dataset_name, new_dataset_name, graphs, prop_maps)



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



"""
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
"""

rewire_dataset_with_preserved_degree('MUTAG', 'MUTAGSHUFFLED1')
rewire_dataset_partially('MUTAG', 'MUTAGSHUFFLED2', 0.1)