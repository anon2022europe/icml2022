import os
import pickle

import networkx as nx
import numpy as np
import torch

from ggg.data.dense.utils.helpers import _data_helper


class CommSmall:
    def __init__(
        self,
        data_dir=os.path.expanduser("~/.datasets"),
        filename=None,
        num_communities=2,
        min_nodes=20,
        max_nodes=20,
        size=5000,
        p_inter=0.05,
        create_rand=False,
        name=None,
    ):
        self.size = size
        self.p_inter = p_inter
        self.min_nodes = min_nodes
        self.max_nodes = max_nodes
        self.num_communities = num_communities

        self.Py_data = []
        self.data_dir = data_dir
        if filename is None and not create_rand:
            filename = (
                f"community_N_nodes{size}_maxN{max_nodes}_minN{min_nodes}.sparsedataset"
            )
        elif filename is None and create_rand:
            filename = f"community_N_nodes{size}_maxN{max_nodes}_minN{min_nodes}_{name}.sparsedataset"
        self.filename = filename
        os.makedirs(data_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fpath = os.path.join(self.data_dir, self.filename)

        if not (os.path.isfile(fpath) and self.filename.endswith(".sparsedataset")):
            self.generate()
            self.save(self.data_dir, self.filename)
        else:
            self.load(self.data_dir, self.filename)

    def __getitem__(self, item):
        return self.Py_data[item]

    def save(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), "wb") as f:
            pickle.dump(self.Py_data, f)

    def load(self, data_dir, filename):
        with open(os.path.join(data_dir, filename), "rb") as f:
            self.Py_data = pickle.load(f)

    def generate(self):

        data_A = []
        data_X = []
        # transform graphs to matrices : A | get attributes : X
        for _ in range(self.size):
            self.chosen_max = np.random.random_integers(
                self.min_nodes, self.max_nodes, size=1
            )[0]
            g = self.n_community()

            # Get adjacency matrix
            A = torch.tensor(nx.to_numpy_matrix(g))
            # Get attributes X as degree of nodes in A
            X = A.sum(dim=1)

            data_X.append(X)
            data_A.append(A)

        self.data_A = data_A
        self.data_X = data_X

        for idx in range(self.size):
            graph = _data_helper()
            graph.x = torch.tensor(self.data_X[idx])
            graph.A = torch.tensor(self.data_A[idx])
            self.Py_data.append(graph)

        return self.Py_data

    def n_community(self):
        """
        Adapted from: https://github.com/ermongroup/GraphScoreMatching/blob/master/utils/data_generators.py
        Args:
            num_communities: number of communities
            max_nodes: maximum nodes of the graph
            p_inter: connection between communities

        Returns:
            Small community graph
        """

        c_sizes = [self.chosen_max // self.num_communities] * self.num_communities
        max_nodes = self.chosen_max // self.num_communities * self.num_communities
        p_inter = (self.p_inter * max_nodes) / (
            self.num_communities
            * (self.num_communities - 1)
            // 2
            * (max_nodes // self.num_communities) ** 2
        )

        graphs = [
            nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))
        ]

        G = nx.disjoint_union_all(graphs)
        communities = list(G.subgraph(c) for c in nx.connected_components(G))
        add_edge = 0
        for i in range(len(communities)):
            subG1 = communities[i]
            nodes1 = list(subG1.nodes())
            for j in range(i + 1, len(communities)):
                subG2 = communities[j]
                nodes2 = list(subG2.nodes())
                has_inter_edge = False
                for n1 in nodes1:
                    for n2 in nodes2:
                        if np.random.rand() < p_inter:
                            G.add_edge(n1, n2)
                            has_inter_edge = True
                            add_edge += 1
                if not has_inter_edge:
                    G.add_edge(nodes1[0], nodes2[0])
                    add_edge += 1
        print(
            "connected comp: ",
            len(list(G.subgraph(c) for c in nx.connected_components(G))),
            "add edges: ",
            add_edge,
        )
        print(G.number_of_edges())
        return G
