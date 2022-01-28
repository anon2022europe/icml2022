import os

import torch

import numpy as np
import networkx as nx
from networkx.generators.trees import random_tree

from ggg_data.dense.utils.helpers import _data_helper, graph_dump, graph_load

class RandTrees:
    def __init__(
        self,
        data_dir=os.path.expanduser("~/.datasets"),
        filename=None,
        sizes_of_trees=40,
        size=5000,
        create_rand=False,
        curriculum=False,
        name=None,
    ):
        self.size = size
        self.size_of_trees = sizes_of_trees

        self.Py_data = []
        data_dir=os.path.expanduser(data_dir)
        self.data_dir = data_dir
        if filename is None and not create_rand and curriculum:
            filename = f"Trees_n_{sizes_of_trees}_{int((size * 2)/ 1000)}k_curriculum.pt"
        elif filename is None and not create_rand:
            filename = f"Trees_n_{sizes_of_trees}_{int(size/1000)}k.pt"
        elif filename is None and create_rand:
            filename = (
                f"Trees_n_{sizes_of_trees}_{int(size/1000)}k_{name}.pt"
            )
        self.filename = filename
        os.makedirs(data_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fpath = os.path.join(self.data_dir, self.filename)

        if not (os.path.isfile(fpath) and self.filename.endswith(".pt")):
            self.generate()
            if curriculum:
                self.curriculum_trees()
            self.save(self.data_dir, self.filename)
        else:
            self.load(self.data_dir, self.filename)

    def __getitem__(self, item):
        return self.Py_data[item]

    def save(self, data_dir, filename):
        graph_dump(Py_data=self.Py_data, dir_=os.path.join(data_dir, filename))

    def load(self, data_dir, filename):
        self.Py_data = graph_load(dir_=os.path.join(data_dir, filename))

    def generate(self):

        data_A = []
        data_X = []
        # transform graphs to matrices : A | get attributes : X
        for _ in range(self.size):
            g = random_tree(self.size_of_trees)

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

    def curriculum_trees(self):
        temp_list = []
        for G in self.Py_data:
            # get indices of single edge nodes
            idx_single_e = np.where(G.x == 1)

            # Option1: disconnect nodes
            new_X = G.x.clone().detach().numpy()
            np.put(new_X, idx_single_e, 0)

            G.A[idx_single_e, :] = 0
            G.A[:, idx_single_e] = 0

            graph = _data_helper()
            graph.x = torch.tensor(new_X)
            graph.A = G.A
            temp_list.append(graph)

            # # Option2: remove nodes with only 1 edge
            # new_A = np.delete(G.A.clone().detach(), idx_single_e, axis=1)
            # new_A = np.delete(new_A, idx_single_e, axis=0)
            # new_A = F.pad(torch.tensor(new_A), (1, self.size_of_trees), mode='constant', value=0.0)
            # new_X = new_A.sum(dim=1)
            #
            # graph = _data_helper()
            # graph.x = new_X
            # graph.A = new_A
            # temp_list.append(graph)

        for g in temp_list:
            self.Py_data.append(g)