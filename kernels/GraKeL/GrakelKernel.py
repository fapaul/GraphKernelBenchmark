import subprocess
import os
import numpy as np
from .. import kernel
from scipy import sparse as sps
import grakel
from grakel.graph import Graph
from sklearn.utils import Bunch
from collections import Counter

class GrakelKernel(kernel.Kernel):

    def __init__(self, dataset_name, output_path, dataset_path, parameters, kernel_name):
        super().__init__(dataset_name, output_path, dataset_path)
        self.parameters = parameters
        self.kernel_name = kernel_name

    def compile(self):
        pass

    def load_data(self):
        #is being done once kernel computes
        pass

    def compute_kernel_matrices(self):
        ds = self.read_data(self.datasetname)
        G, labels = ds.data, ds.target
        kernel = 0
        #TODO configure other kernels and parameters
        if self.kernel_name == "Propagation":
            kernel = grakel.Propagation(n_jobs=None, verbose=False, normalize=True, random_seed=42, M='TV', t_max=4, w=0.0001)
        elif self.kernel_name == "RandomWalk":
            kernel = grakel.RandomWalk()
        elif self.kernel_name == "PyramidMatch":
            kernel = grakel.PyramidMatch(with_labels=True, L=4, d=6)
        elif self.kernel_name == "VertexHistogram":
            kernel = grakel.VertexHistogram()
        elif self.kernel_name == "WeisfeilerLehman":
            kernel = grakel.GraphKernel([{"name": "weisfeiler_lehman"},{"name": self.parameters[0]}])
        elif self.kernel_name == "ShortestPath":
            kernel = grakel.ShortestPath()
        elif self.kernel_name == "GraphletSampling":
            kernel = grakel.GraphletSampling()
        elif self.kernel_name == "LovaszTheta":
            kernel = grakel.LovaszTheta()
        elif self.kernel_name == "SVMTheta":
            kernel = grakel.SvmTheta()
        elif self.kernel_name == "MultiscaleLaplacian":
            kernel = grakel.MultiscaleLaplacian()
        elif self.kernel_name == "NeighborhoodHash":
            kernel = grakel.NeighborhoodHash()
        elif self.kernel_name == "NeighborhoodSubgraphPairwiseDistance":
            kernel = grakel.NeighborhoodSubgraphPairwiseDistance()
        elif self.kernel_name == "GraphHopper":
            kernel = grakel.GraphHopper()
        elif self.kernel_name == "SubgraphMatching":
            kernel = grakel.SubgraphMatching()
        elif self.kernel_name == "EdgeHistogram":
            kernel = grakel.EdgeHistogram()
        elif self.kernel_name == "OddSth":
            kernel = grakel.OddSth()
        else:
            kernel = grakel.GraphKernel([{"name": self.parameters[0]}])
        kernelmatrix = kernel.fit_transform(G)
        np.savetxt(os.path.join(self.output_path,self.kernel_name), kernelmatrix, delimiter=' ', newline='\n')
        return [os.path.join(self.output_path,self.kernel_name)]

    def read_data(
        self,
        name,
        nl = True,
        na = False,
        el = True,
        ea = False,
        with_classes=True,
        prefer_attr_nodes=False,
        prefer_attr_edges=False,
        produce_labels_nodes=False,
        as_graphs=False,
        is_symmetric=False):
        '''
        Create a dataset iterable for GraphKernel.

        Parameters
        ----------
        name : str
            The dataset name.

        with_classes : bool, default=True
            Return an iterable of class labels based on the enumeration.

        produce_labels_nodes : bool, default=False
            Produce labels for nodes if not found.
            Currently this means labeling its node by its degree inside the Graph.
            This operation is applied only if node labels are non existent.

        prefer_attr_nodes : bool, default=False
            If a dataset has both *node* labels and *node* attributes
            set as labels for the graph object for *nodes* the attributes.

        prefer_attr_edges : bool, default=False
            If a dataset has both *edge* labels and *edge* attributes
            set as labels for the graph object for *edge* the attributes.

        as_graphs : bool, default=False
            Return data as a list of Graph Objects.

        is_symmetric : bool, default=False
            Defines if the graph data describe a symmetric graph.

        Returns
        -------
        Gs : iterable
            An iterable of graphs consisting of a dictionary, node
            labels and edge labels for each graph.

        classes : np.array, case_of_appearance=with_classes==True
            An one dimensional array of graph classes aligned with the lines
            of the `Gs` iterable. Useful for classification.

        '''
        folder_path = "datasets/"
        indicator_path = "./"+folder_path+str(name)+"/"+str(name)+"_graph_indicator.txt"
        edges_path = "./"+folder_path + str(name) + "/" + str(name) + "_A.txt"
        node_labels_path = "./"+folder_path + str(name) + "/" + str(name) + "_node_labels.txt"
        node_attributes_path = "./"+folder_path+str(name)+"/"+str(name)+"_node_attributes.txt"
        edge_labels_path = "./"+folder_path + str(name) + "/" + str(name) + "_edge_labels.txt"
        edge_attributes_path = \
            "./"+folder_path + str(name) + "/" + str(name) + "_edge_attributes.txt"
        graph_classes_path = \
            "./"+folder_path + str(name) + "/" + str(name) + "_graph_labels.txt"

        # node graph correspondence
        ngc = dict()
        # edge line correspondence
        elc = dict()
        # dictionary that keeps sets of edges
        Graphs = dict()
        # dictionary of labels for nodes
        node_labels = dict()
        # dictionary of labels for edges
        edge_labels = dict()

        # Associate graphs nodes with indexes
        with open(indicator_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                ngc[i] = int(line[:-1])
                if int(line[:-1]) not in Graphs:
                    Graphs[int(line[:-1])] = set()
                if int(line[:-1]) not in node_labels:
                    node_labels[int(line[:-1])] = dict()
                if int(line[:-1]) not in edge_labels:
                    edge_labels[int(line[:-1])] = dict()

        # Extract graph edges
        with open(edges_path, "r") as f:
            for (i, line) in enumerate(f, 1):
                edge = line[:-1].replace(' ', '').split(",")
                elc[i] = (int(edge[0]), int(edge[1]))
                Graphs[ngc[int(edge[0])]].add((int(edge[0]), int(edge[1])))
                if is_symmetric:
                    Graphs[ngc[int(edge[1])]].add((int(edge[1]), int(edge[0])))

        # Extract node attributes
        if prefer_attr_nodes and na and os.path.exists(node_attributes_path):
            with open(node_attributes_path, "r") as f:
                for (i, line) in enumerate(f, 1):
                    node_labels[ngc[i]][i] = \
                        [float(num) for num in
                         line[:-1].replace(' ', '').split(",")]
        # Extract node labels
        elif nl and os.path.exists(node_labels_path):
            with open(node_labels_path, "r") as f:
                for (i, line) in enumerate(f, 1):
                    node_labels[ngc[i]][i] = int(line[:-1])
        elif produce_labels_nodes:
            for i in range(1, len(Graphs)+1):
                node_labels[i] = dict(Counter(s for (s, d) in Graphs[i] if s != d))

        # Extract edge attributes
        if prefer_attr_edges and ea and os.path.exists(edge_attributes_path):
            with open(edge_attributes_path, "r") as f:
                for (i, line) in enumerate(f, 1):
                    attrs = [float(num)
                             for num in line[:-1].replace(' ', '').split(",")]
                    edge_labels[ngc[elc[i][0]]][elc[i]] = attrs
                    if is_symmetric:
                        edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = attrs

        # Extract edge labels
        elif el and os.path.exists(edge_labels_path):
            with open(edge_labels_path, "r") as f:
                for (i, line) in enumerate(f, 1):
                    edge_labels[ngc[elc[i][0]]][elc[i]] = int(line[:-1])
                    if is_symmetric:
                        edge_labels[ngc[elc[i][1]]][(elc[i][1], elc[i][0])] = \
                            int(line[:-1])

        Gs = list()
        if as_graphs:
            for i in range(1, len(Graphs)+1):
                Gs.append(Graph(Graphs[i], node_labels[i], edge_labels[i]))
        else:
            for i in range(1, len(Graphs)+1):
                Gs.append([Graphs[i], node_labels[i], edge_labels[i]])

        if with_classes:
            classes = []
            with open(graph_classes_path, "r") as f:
                for line in f:
                    classes.append(int(line[:-1]))

            classes = np.array(classes, dtype=np.int)
            return Bunch(data=Gs, target=classes)
        else:
            return Bunch(data=Gs)

