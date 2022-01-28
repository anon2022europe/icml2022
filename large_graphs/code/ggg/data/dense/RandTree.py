import os
import pickle

import torch
import networkx as nx
from networkx.generators.trees import random_tree

from ggg.data.dense.utils.helpers import _data_helper


class RandTrees:
    def __init__(
        self,
        data_dir="Trees",
        filename=None,
        sizes_of_trees=40,
        size=5000,
        create_rand=False,
        name=None
    ):
        self.size = size
        self.size_of_trees = sizes_of_trees

        self.Py_data = []
        self.data_dir = data_dir
        if filename is None and not create_rand:
            filename=f"Trees_n_{sizes_of_trees}_{int(size/1000)}k.sparsedataset"
        elif filename is None and create_rand:
            filename=f"Trees_n_{sizes_of_trees}_{int(size/1000)}k_{name}.sparsedataset"
        self.filename = filename
        os.makedirs(data_dir,exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fpath=os.path.join(self.data_dir, self.filename)

        if not (os.path.isfile(fpath) and self.filename.endswith(
                ".sparsedataset"
        )):
            self.generate()
            self.save(self.data_dir,self.filename)
        else:
            self.load(self.data_dir,self.filename)


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