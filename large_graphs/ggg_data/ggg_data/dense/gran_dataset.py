import os

import networkx as nx
import numpy as np
import torch

from ggg_data.dense.utils.helpers import _data_helper, graph_dump, graph_load


class Enzymes:
    def __init__(
        self,
        data_dir=os.path.expanduser("~/.datasets"),
        filename=None,
        min_nodes=20,
        max_nodes=1000,
        size=5000,
        create_rand=False,
        rand_name=None,
        name=None,
        node_attributes=None,
        graph_labels=None,
    ):
        self.size = size
        self.gran_data_dir = 'gran_data'
        self.min_num_nodes = min_nodes
        self.max_num_nodes = max_nodes
        self.name = name
        self.node_attributes = node_attributes
        self.graph_labels = graph_labels

        self.Py_data = []
        self.data_dir = data_dir
        if filename is None and not create_rand:
            filename = (
                f"{self.name}_{int((size * 2)/ 1000)}k.pt"
            )
        elif filename is None and create_rand:
            filename = f"{self.name}_{int((size * 2)/ 1000)}k_{rand_name}.pt"
        self.filename = filename
        os.makedirs(data_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fpath = os.path.join(self.data_dir, self.filename)

        if not (os.path.isfile(fpath) and self.filename.endswith(".pt")):
            self.graph_load_batch()
            self.save(self.data_dir, self.filename)
        else:
            self.load(self.data_dir, self.filename)

    def __getitem__(self, item):
        return self.Py_data[item]

    def save(self, data_dir, filename):
        graph_dump(Py_data=self.Py_data, dir_=os.path.join(data_dir, filename))

    def load(self, data_dir, filename):
        self.Py_data = graph_load(dir_=os.path.join(data_dir, filename))

    def graph_load_batch(self):
        '''
          load many graphs, e.g. enzymes
          :return: a list of graphs
          '''
        print('Loading graph dataset: ' + str(self.name))
        G = nx.Graph()
        # load data
        path = os.path.join(self.gran_data_dir, self.name)
        data_adj = np.loadtxt(
            os.path.join(path, '{}_A.txt'.format(self.name)), delimiter=',').astype(int)
        if self.node_attributes:
            data_node_att = np.loadtxt(
                os.path.join(path, '{}_node_attributes.txt'.format(self.name)),
                delimiter=',')
        data_node_label = np.loadtxt(
            os.path.join(path, '{}_node_labels.txt'.format(self.name)),
            delimiter=',').astype(int)
        data_graph_indicator = np.loadtxt(
            os.path.join(path, '{}_graph_indicator.txt'.format(self.name)),
            delimiter=',').astype(int)
        if self.graph_labels:
            data_graph_labels = np.loadtxt(
                os.path.join(path, '{}_graph_labels.txt'.format(self.name)),
                delimiter=',').astype(int)

        data_tuple = list(map(tuple, data_adj))
        # print(len(data_tuple))
        # print(data_tuple[0])

        # add edges
        G.add_edges_from(data_tuple)
        # add node attributes
        for i in range(data_node_label.shape[0]):
            if self.node_attributes:
                G.add_node(i + 1, feature=self.data_node_att[i])
            G.add_node(i + 1, label=data_node_label[i])
        G.remove_nodes_from(list(nx.isolates(G)))

        # remove self-loop
        G.remove_edges_from(nx.selfloop_edges(G))

        # print(G.number_of_nodes())
        # print(G.number_of_edges())

        # split into graphs
        graph_num = data_graph_indicator.max()
        node_list = np.arange(data_graph_indicator.shape[0]) + 1
        graphs = []
        max_nodes = 0
        for i in range(graph_num):
            graph = _data_helper()
            # find the nodes for each graph
            nodes = node_list[data_graph_indicator == i + 1]
            G_sub = G.subgraph(nodes)
            if self.graph_labels:
                G_sub.graph['label'] = data_graph_labels[i]
            # print('nodes', G_sub.number_of_nodes())
            # print('edges', G_sub.number_of_edges())
            # print('label', G_sub.graph)
            if G_sub.number_of_nodes() >= self.min_num_nodes and G_sub.number_of_nodes(
            ) <= self.max_num_nodes:
                graphs.append(G_sub)
                if G_sub.number_of_nodes() > max_nodes:
                    max_nodes = G_sub.number_of_nodes()
                # print(G_sub.number_of_nodes(), 'i', i)
                # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
                # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))

                # Get adjacency matrix
                A = torch.tensor(nx.to_numpy_matrix(G_sub))
                # Get attributes X as degree of nodes in A
                X = A.sum(dim=1)

                graph.x = torch.tensor(X)
                graph.A = torch.tensor(A)
                self.Py_data.append(graph)

            if len(self.Py_data) > self.size:
                print(f"Done! Loaded dataset of size {len(self.Py_data)}")

        return self.Py_data
