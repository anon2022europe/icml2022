from warnings import warn

from torch.utils.data import Dataset
import os
import sklearn
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import networkx as nx


def keep_topk_conns(adj, k=3):
    g = nx.from_numpy_array(adj)
    to_removes = [cp for cp in sorted(nx.connected_components(g), key=len)][:-k]
    for cp in to_removes:
        g.remove_nodes_from(cp)
    adj = nx.to_numpy_array(g)
    return adj


def get_spectral_embedding(adj, d, solver="arpack"):  # , ‘lobpcg’, or ‘amg’"):
    """
    Given adj is N*N, return its feature mat N*D, D is fixed in model
    :param adj:
    :return:
    """

    adj_ = np.array(adj)
    emb = sklearn.manifold.SpectralEmbedding(
        n_components=d, eigen_solver=solver
    )  # use lopcg since otherwis ethis errors now?
    res = emb.fit_transform(adj_)
    x = res.astype(np.float32)
    return x


class CondgenTCGA(Dataset):
    def __init__(self, DATA_DIR, download=True, d=10):
        self.DATA_DIR = DATA_DIR
        self.download = True  # TODO: implement
        self.d = d
        NODE_FILE = "node.dat"
        LINK_FILE = "link.dat"
        LABEL_FILE = "label.dat"
        ATTR_FILE = "attribute.dat"
        mat_names = []  # e.g. GSE_2304
        adj_mats = []  # essential data, type: list(np.ndarray)
        attr_vecs = []  # essential data, type: list(np.ndarray)
        id_maps = []  # map index to gene name if you need

        ATTR_LEN = 8

        for folder in tqdm(os.listdir(DATA_DIR)):
            if "." in folder or folder.startswith("_"):
                continue
            mat_names.append(folder)
            with open(os.path.join(DATA_DIR, folder, ATTR_FILE), "r") as f:
                attr_vec = np.loadtxt(f)
                attr_vecs.append(attr_vec[:ATTR_LEN])

            id_to_item = {}
            with open(os.path.join(DATA_DIR, folder, NODE_FILE), "r") as f:
                for i, line in enumerate(f):
                    cells = line.split("\t")
                    id_to_item[i] = cells[0]

            all_items = set(id_to_item.values())
            all_ids = set(id_to_item.keys())

            links = defaultdict(set)
            with open(os.path.join(DATA_DIR, folder, LINK_FILE), "r") as f:
                for line in f:
                    cells = line.rstrip("\n").split("\t")
                    from_id = int(cells[0])
                    to_id = int(cells[1])
                    if from_id in all_ids and to_id in all_ids:
                        links[from_id].add(to_id)

            N = len(all_ids)
            adj = np.zeros((N, N))
            for from_id in range(N):
                for to_id in links[from_id]:
                    adj[from_id, to_id] = 1
                    adj[to_id, from_id] = 1
            adj -= np.diag(np.diag(adj))
            id_map = [id_to_item[i] for i in range(N)]

            # Remove small component
            # adj = remove_small_conns(adj, keep_min_conn=4)

            # Keep large component
            adj = keep_topk_conns(adj, k=1)

            adj_mats.append(adj)
            id_maps.append(id_map)

            if int(np.sum(adj)) == 0:
                adj_mats.pop(-1)
                id_map.pop(-1)
                mat_names.pop(-1)
                attr_vecs.pop(-1)
        self.adj_mats = adj_mats
        self.id_maps = id_maps
        self.mat_names = mat_names
        self.attr_vecs = attr_vecs

    def __len__(self):
        return len(self.adj_mats)

    def __getitem__(self, idx):
        attr_vec = self.attr_vecs[idx]
        A = self.adj_mats[idx]
        x = get_spectral_embedding(A, self.d)
        return (x, A), attr_vec


class CondgenDBLP(Dataset):
    def __init__(self, DATA_DIR, download=True, d=5):
        self.DATA_DIR = DATA_DIR
        self.download = True  # TODO: implement
        self.d = d
        # script for loading NWE dblp
        # folder structure
        # - this.ipynb
        # - $DATA_DIR - *.txt

        mat_names = []  # e.g. GSE_2304
        adj_mats = []  # essential data, type: list(np.ndarray)
        attr_vecs = []  # essential data, type: list(np.ndarray)
        id_maps = []  # map index to gene name if you need

        for f in os.listdir(DATA_DIR):
            if not f.startswith(("nodes", "links", "attrs")):
                continue
            else:
                mat_names.append("_".join(f.split(".")[0].split("_")[1:]))
        mat_names = sorted([it for it in set(mat_names)])
        print("Test length", len(mat_names))
        for mat_name in mat_names:
            node_file = "nodes_" + mat_name + ".txt"
            link_file = "links_" + mat_name + ".txt"
            attr_file = "attrs_" + mat_name + ".txt"
            node_file_path = os.path.join(DATA_DIR, node_file)
            link_file_path = os.path.join(DATA_DIR, link_file)
            attr_file_path = os.path.join(DATA_DIR, attr_file)

            id_to_item = {}
            with open(node_file_path, "r") as f:
                for i, line in enumerate(f):
                    author = line.rstrip("\n")
                    id_to_item[i] = author
            all_ids = set(id_to_item.keys())

            with open(attr_file_path, "r") as f:
                attr_vec = np.loadtxt(f).T.flatten()
                attr_vecs.append(attr_vec)

            links = defaultdict(set)
            with open(link_file_path, "r") as f:
                for line in f:
                    cells = line.rstrip("\n").split(",")
                    from_id = int(cells[0])
                    to_id = int(cells[1])
                    if from_id in all_ids and to_id in all_ids:
                        links[from_id].add(to_id)

            N = len(all_ids)
            adj = np.zeros((N, N))
            for from_id in range(N):
                for to_id in links[from_id]:
                    adj[from_id, to_id] = 1
                    adj[to_id, from_id] = 1

            adj -= np.diag(np.diag(adj))
            id_map = [id_to_item[i] for i in range(N)]

            # Remove small component
            # adj = remove_small_conns(adj, keep_min_conn=4)

            # Keep large component
            adj = keep_topk_conns(adj, k=1)
            adj_mats.append(adj)
            id_maps.append(id_map)

            if int(np.sum(adj)) == 0:
                adj_mats.pop(-1)
                id_maps.pop(-1)
                mat_names.pop(-1)
                attr_vecs.pop(-1)
        self.adj_mats = adj_mats
        self.id_maps = id_maps
        self.mat_names = mat_names
        self.attr_vecs = attr_vecs

    def __len__(self):
        return len(self.adj_mats)

    def __getitem__(self, idx):
        attr_vec = self.attr_vecs[idx]
        A = self.adj_mats[idx]
        for solver in ["arpack", "lobpcg"]:
            try:
                x = get_spectral_embedding(A, self.d, solver)
                break
            except BaseException as e:
                if solver == "arpack":
                    warn(f"Got {e} for matrix {A}, retrying with lobpcg")
                else:
                    warn(f"{solver} failed for {A}, exiting")
                    raise e
        return (x, A), attr_vec
