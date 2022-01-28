from collections import defaultdict
from warnings import warn

import torch
from ipdb import set_trace
from torch.utils.data import Dataset
from tqdm import tqdm
import subprocess as sp
import os
import pandas as pd
import numpy as np
import networkx as nx

from ggg.utils.utils import pad_to_max


def download_egonet_files(root_path, parallel=True):
    """

    :param file_dict:
    :param root_path:
    :return:
    """
    old_path = os.path.abspath(os.getcwd())
    root_abs = os.path.abspath(root_path)
    FILES = ["twitter.tar.gz"]
    EGO_URL = {"twitter.tar.gz": "https://snap.stanford.edu/data/twitter.tar.gz"}
    os.chdir(root_abs)
    for f in FILES:
        dl_procs = []
        if not (os.path.exists(f) or os.path.exists(f.replace(".tar.gz", ""))):
            p = sp.Popen(["wget", "-O", f, EGO_URL[f]])
            if parallel:
                dl_procs.append(p)
            else:
                p.wait()
        for p in dl_procs:
            p.wait()
        if not os.path.exists(f.replace(".tar.gz", "")):
            p = sp.Popen(["tar", "xf", f])
            p.wait()
        os.chdir(root_abs)
    os.chdir(old_path)


class EgonetSnap(Dataset):
    PRESETS = {
        1: set(range(20)),
        2: set(range(500, 520)),
        3: set(range(927, 947)),
        4: set(range(400, 420)),
        5: set(range(600, 620)),
        6: set(range(700, 720)),
    }

    def __init__(
        self,
        dir="EgonetSNAP",
        undirected=True,
        include_ego=False,
        num_graphs=None,
        force_connected=True,
        skip_loops=True,
        clean_loops=False,
        preset20_num=None,
        select_inds=None,
        skip_features=False,
    ) -> None:
        super().__init__()
        if preset20_num is not None:
            assert preset20_num in EgonetSnap.PRESETS
            select_inds = EgonetSnap.PRESETS[preset20_num]
        self.skip_features = skip_features
        self.select_inds = select_inds
        self.clean_loops = clean_loops
        self.skip_loops = skip_loops
        self.undirected = undirected
        self.include_ego = include_ego
        self.dir = dir
        self.num_graphs = num_graphs
        self.force_connected = force_connected
        os.makedirs(self.dir, exist_ok=True)
        self.download()
        cache_file = f"/tmp/egonet{self.num_graphs}.pt"
        if os.path.exists(cache_file):
            with open(cache_file, "rb") as f:
                d = torch.load(f)
                self.Xs = d["Xs"]
                self.As = d["As"]
        else:
            self.Xs, self.As = self.parse()

    def download(self):
        download_egonet_files(self.dir)

    def parse(self):
        data_dir = os.path.join(self.dir, "twitter")
        node_edge_files = [f for f in os.listdir(data_dir) if ".edges" in f]
        node_ids = [x.split(".")[0] for x in node_edge_files]
        Xs = []
        As = []
        missing = []
        loops = []
        selected_mccs = []
        for i, ego in tqdm(
            enumerate(node_ids),
            desc="Parsing net",
            total=(len(node_ids) if self.num_graphs is None else self.num_graphs),
        ):
            if self.select_inds is not None and i not in self.select_inds:
                continue
            try:
                edge_file = f"{ego}.edges"
                feat_file = f"{ego}.feat"
                ego_feat_file = f"{ego}.egofeat"
                features = pd.read_csv(
                    os.path.join(data_dir, feat_file),
                    delimiter=" ",
                    header=None,
                    index_col=0,
                )
                ids = features.index
                feats = np.array(features).transpose()
                features = {i: f for i, f in zip(ids, feats)}
                with open(os.path.join(data_dir, edge_file), "r") as f:
                    # TODO: might be able to do this without for loop by parsing directly in numpy array and then indexing...
                    el = f.readlines()
                    g = nx.parse_edgelist(el)
                if self.force_connected:
                    mcc = max(nx.connected_components(g), key=len)
                    if len(mcc) < len(g.nodes):
                        selected_mccs.append(ego)

                    g = g.subgraph(mcc).copy()

                A = nx.to_numpy_array(g)
                X = [features[int(n)] for n in g.nodes]

                assert (A.sum(0) != 0).all()
                assert (A.sum(1) != 0).all()
                if np.diag(A).sum() != 0 and self.skip_loops:
                    loops.append(ego)
                    continue
                if np.diag(A).sum() != 0 and self.clean_loops:
                    loops.append(ego)
                    np.fill_diagonal(A, 0.0)
                assert len(X) == len(A), f"{len(X)}!={len(A)},{A.shape}"
                X = np.stack(X, 0)
                Xs.append(X)
                As.append(A)
            except KeyError as e:
                e: KeyError
                missing.append(ego)
                pass

            if self.num_graphs is not None and len(Xs) >= self.num_graphs:
                break

        if self.skip_loops and len(loops) > 0:
            l = "\n".join([str(x) for x in loops])
            msg = f"Skipped the following ego-ids since they contained self loops (nonzero diagonal){l}"
            warn(msg)
        elif self.clean_loops and len(loops) > 0:
            l = "\n".join([str(x) for x in loops])
            msg = f"Zero-filled diagonal of the following ego-ids since they contained self loops (nonzero diagonal){l}"
            warn(msg)
        if len(missing) > 0:
            m = "\n".join([str(x) for x in missing])
            msg = f"Skipped the following ego-ids since the node features had missing entries {m}"
            warn(msg)
        if len(selected_mccs) > 0:
            m = "\n".join([str(x) for x in selected_mccs])
            msg = f"Selected largest connected components on the following graphs since they had isolated nodes:{m}"
            warn(msg)
        """
        TODO. check this here
        """
        # As=np.stack(pad_to_max(As,N_dim=0,pad_dims=(0,1)),0)
        # Xs=pad_to_max(Xs,N_dim=,pad_dims=(0,))
        # Xs=np.stack(X_pad,0)
        """
        I don't get the feature dims here...seems to be dependent on the node size?
        https://snap.stanford.edu/data/readme-Ego.txt
        
        Okay, I think now it makes sense? But not 100% sur
        """
        warn(
            "Node features might not make sense here, only use with structural features"
        )
        cache_file = f"/tmp/egonet{self.num_graphs}.pt"
        if not os.path.exists(cache_file):
            with open(cache_file, "wb") as f:
                d = dict(Xs=Xs, As=As)
                torch.save(d, f)
        return Xs, As

    def __len__(self) -> int:
        return len(self.Xs)

    def __getitem__(self, index: int):
        if self.skip_features:
            return None, self.As[index]
        else:
            return self.Xs[index], self.As[index]


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("TkAgg")
    import networkx as nx
    import matplotlib.pyplot as plt

    # ds=EgonetSnap()
    N = 20
    n = int(np.ceil(np.sqrt(N)))
    ds = EgonetSnap(preset20_num=4)
    fig, axs = plt.subplots(n, n)
    for i, ax in enumerate(axs.flatten()):
        if i == N:
            break
        x, a = ds[i]
        g = nx.from_numpy_matrix(a)
        # mcc=max(nx.connected_components(g), key=len)
        # print(mcc)
        # g=g.subgraph(mcc).copy()
        pos = nx.spring_layout(g)
        nx.draw_networkx_nodes(g, pos=pos, ax=ax, node_size=10)
        nx.draw_networkx_edges(g, pos=pos, ax=ax, edge_color="#000000")
    plt.show()
