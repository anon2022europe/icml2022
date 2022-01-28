import os
import pickle

import torch
import numpy as np
import networkx as nx
import torchvision.transforms as transforms

from pygraphviz import *
from ggg_data.dense.utils.helpers import _data_helper, graph_dump, graph_load
from ggg_data.dense.house_floorplans.house_dst_utils import FloorplanGraphDataset

# Needed dictionaries for drawing house graphs
ID_COLOR = {1: 'brown', 2: 'magenta', 3: 'orange', 4: 'gray', 5: 'red', 6: 'blue', 7: 'cyan', 8: 'green', 9: 'salmon', 10: 'yellow'}
ROOM_CLASS = {"living_room": 1, "kitchen": 2, "bedroom": 3, "bathroom": 4, "missing": 5, "closet": 6, "balcony": 7, "corridor": 8,
              "dining_room": 9, "laundry_room": 10}

CLASS_ROM = {}
for x, y in ROOM_CLASS.items():
    CLASS_ROM[y] = x

class HousePlans:
    def __init__(
        self,
        data_dir=os.path.expanduser("~/.datasets"),
        filename=None,
        size=5000,
        create_rand=False,
        name=None,
        target_set='A'
    ):
        """ Get dataset from house floorplans

        target_set : portion of graphs to be used for cross calidation (ex. 'A':[1, 3] --> graphs with 1 to 3 nodes)
        """
        self.size = size
        self.name = name
        self.create_rand = create_rand

        self.Py_data = []
        self.data_dir = data_dir

        if filename is None and not create_rand:
            filename = f"HouseFloorplan_{int(size/1000)}k.pt"
        elif filename is None and create_rand:
            filename = f"HouseFloorplan_{int(size/1000)}k_{name}.pt"

        self.filename = filename
        os.makedirs(data_dir, exist_ok=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        fpath = os.path.join(self.data_dir, self.filename)

        if not (os.path.isfile(fpath) and self.filename.endswith(".pt")):
            # HACK: should not usually do relative paths but
            house_dst_dir = os.path.join(
                os.path.dirname(__file__), "train_data.npy"
            )

            if not os.path.isfile(house_dst_dir):
                raise ValueError("You do not have the house dataset in your directory. "
                                 "Download it at: https://github.com/ennauata/housegan ")

            else:
                self.floorplan_dst = FloorplanGraphDataset(house_dst_dir, transforms.Normalize(mean=[0.5], std=[0.5]),
                                                           target_set=target_set)

                self.Py_data = self.generate()
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
        size_floorplan_dataset = len(self.floorplan_dst)
        if self.create_rand:
            # name should be something as "rand1", "rand2", ...
            idx = int(self.name.split("rand")[1])
            # displace of training dataset size
            list_idxs = np.arange(5001*idx, 10001*idx)
            print(len(list_idxs), min(list_idxs), max(list_idxs))
        else:
            list_idxs = np.arange(0, self.size)

        data_A = []
        data_X = []
        # transform graphs to matrices : A | get attributes : X
        for iter_ in list_idxs:
            nds = self.floorplan_dst[iter_][1]
            edgs = self.floorplan_dst[iter_][2]
            g = self.get_graph_flrplan(nodes=nds, edges=edgs)

            # Get adjacency matrix
            A = torch.tensor(nx.to_numpy_matrix(g))
            # Get attributes X
            x_attr = []
            for line in g.nodes(data="room_id"):
                res = int(line[1])
                x_attr.append(res)
            X = torch.tensor(np.array(x_attr))

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

    def get_graph_flrplan(self, nodes, edges):
        # Create graph
        graph = AGraph(strict=False, directed=False)

        # Create nodes
        for k in range(nodes.shape[0]):
            nd = np.where(nodes[k] == 1)[0]
            if len(nd) > 0:
                r_id = nd[0] + 1
                color = ID_COLOR[nd[0] + 1]
                name = CLASS_ROM[nd[0] + 1]
                graph.add_node(k, room_id=r_id, label=name, color=color)

        # Create edges
        for i, p, j in edges:
            if p > 0:
                graph.add_edge(i.item(), j.item(), color='black', penwidth='4')

        nx_graph = nx.nx_agraph.from_agraph(graph)
        return nx_graph
