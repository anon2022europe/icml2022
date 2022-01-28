from pdb import set_trace

import torch as pt
from ogb.nodeproppred import PygNodePropPredDataset
import os
import networkx as nx
from tqdm import tqdm
import numpy as np
import functools
import hashlib
import re
import multiprocessing
import torch_geometric as tg


@functools.lru_cache(maxsize=10000)
def node_edges(n: int, e: pt.Tensor):
    """

    :param n:  node index
    :param e: [2,N] array of edges
    :return: [2,|neighbourood|]
    """
    assert type(n) is int
    inds = (e == n).any(0).nonzero().flatten()
    return e[:, inds]


def k_hop_community(edge: pt.Tensor, edges: pt.Tensor, k=1, verbose=False):
    """
    Constructs the k-hop neighbourhood of each node in the given edge, then combines the two to form a commnuity graph

    :param edge:[2] tensor
    :param edges: [2,N]
    :return:
    """
    cand_nodes = set(edge.tolist())
    nodes_done = set()
    selected_edges = []
    for K in tqdm(range(k), desc="K-hop", disable=not verbose, leave=False):
        neigh_k = []
        # for each node in the frontier, add the neighbourhood
        for n in tqdm(
            cand_nodes,
            desc=f"Visiting {K+1}-hop frontier",
            disable=not verbose,
            leave=False,
        ):
            neigh_k.append(node_edges(n, edges))
            nodes_done.add(n)
        neigh_k = pt.cat(neigh_k, -1)
        selected_edges.append(neigh_k)

        # all nodes which were not yet visited are added to frontier
        cand_nodes = set(neigh_k.flatten().unique().tolist())
        cand_nodes = cand_nodes.difference(nodes_done)

    selected_edges = pt.cat(selected_edges, -1)
    return selected_edges


def node_k_communities(node_list, edges: pt.Tensor, k=1, verbose=False, max_len=None):
    """
    Constructs the k-hop communities of which the given nodes are a part of
    :param node_list:
    :param edges:
    :param k:
    :return:
    """
    community_edges = pt.cat(
        [
            node_edges(n, edges)
            for n in tqdm(
                node_list,
                desc="Collecting community edges",
                disable=not verbose,
                leave=False,
            )
        ],
        -1,
    ).permute(1, 0)
    community_edges = set([tuple(sorted(e.tolist())) for e in community_edges])
    edges_done = set()

    communities = []
    bar = tqdm(community_edges, disable=not verbose)
    for et in bar:
        of = f"/{max_len}" if max_len is not None else ""
        desc = f"Have {len(communities)}{of} Checking cand edge"
        bar.set_description(desc=desc)
        if et not in edges_done:
            edges_done.add(et)
            e = pt.tensor(et)
            communities.append(k_hop_community(e, edges, k, verbose=verbose))
        if max_len is not None and len(communities) >= max_len:
            break
    return communities


def check_max_len(p, max_len, suffix="_nl", skip=None):
    b = os.path.basename(p)
    mlb = re.findall(f"max_len=(.+){suffix}", b)[0]
    if skip is not None:
        slb = re.findall(f"skip=([0-9]+)", b)
        if len(slb) > 0:
            slb = slb[0]
            slbi = int(slb) if slb.isnumeric() else None
            slb_ok = slbi is not None and skip == slbi
        else:
            slb_ok = False
    else:
        slb_ok = True

    return (
        max_len is None
        and mlb == "None"
        or max_len is not None
        and mlb.isnumeric()
        and int(mlb) >= max_len
        and slb_ok
    )


class Product_Categories(pt.utils.data.Dataset):
    def __init__(
        self,
        mode="different",
        k=1,
        dense=True,
        root=os.path.expanduser("~/.ogbn"),
        verbose=False,
        max_len=None,
        egonets=False,
        process_batch_size=None,
        with_labels=False,
        seed_edge_offset=0,
        size_limit=None,
    ):
        super().__init__()
        self.with_labels = with_labels
        os.makedirs(root, exist_ok=True)
        os.makedirs(os.path.join(root, "caches"), exist_ok=True)
        cache_file_name = f"product_k-hop_cat_k={k}_mode={mode}_skip={seed_edge_offset}_maxN={size_limit}_max_len={max_len}.pt"
        cache_dir = os.path.join(root, "caches")
        cache_path = os.path.join(cache_dir, cache_file_name)
        # select all edges which connects different categories, remove duplicate unidirectional edges
        # check if maybe we can subset a cache file base on the node list digest and the k in case cache_path doesn't exist
        possible_cache_files_found = (
            [
                os.path.join(cache_dir, x)
                for x in os.listdir(cache_dir)
                if f"k={k}" in x
                and f"maxN={size_limit}" in x
                and check_max_len(x, max_len, skip=seed_edge_offset, suffix=".pt")
            ]
            if not os.path.exists(cache_path)
            else None
        )
        if os.path.exists(cache_path):
            # exact match
            cf = cache_path
            if verbose:
                print(f"Using cache file \n{cf}\n")
            with open(cache_path, "rb") as f:
                d = pt.load(f)
                self.communities = d["communities"]
        elif len(possible_cache_files_found) > 0:
            cf = possible_cache_files_found[0]
            if verbose:
                print(f"Using cache file \n{cf}\n")
            with open(cf, "rb") as f:
                d = pt.load(f)
                self.communities = d["communities"][:max_len]
        else:
            if verbose:
                print(f"Didn't find cache file \n{cache_path}\n so creating it")
            self.products = PygNodePropPredDataset("ogbn-products", root=root)[0]
            edges = self.products.edges
            self.all_node_feats = self.products.x
            self.all_node_labels = self.products.y
            y = self.all_node_labels.flatten()
            if mode == "different":
                different_cats = y[edges[0]] != y[edges[1]]
                different_cats: pt.Tensor
                seed_edges = edges[
                    :, pt.nonzero(different_cats, as_tuple=False).flatten()
                ]
            elif mode == "same":
                same_cats = y[edges[0]] == y[edges[1]]
                same_cats: pt.Tensor
                seed_edges = edges[:, pt.nonzero(same_cats, as_tuple=False).flatten()]
            elif mode == "all":
                # all
                seed_edges = edges.sort(dim=0)
            else:
                raise ValueError(f"mode must be in different/same/all")
            comm = []
            bar = tqdm(
                range(seed_edges.shape[-1]),
                desc="Creating community",
                disable=not verbose,
                total=max_len,
            )
            edges_done = set()
            if max_len is None:
                max_len = seed_edges.shape[-1]
            if process_batch_size is None:
                process_batch_size = min(max_len, 100)
            batches = int(np.ceil(max_len / process_batch_size))
            i = 0
            # skip the first seed_edge_offset unique edges
            for b in tqdm(range(batches), desc="Process batch", disable=not verbose):
                unique_edges = []
                while True and i < seed_edges.shape[-1]:
                    e = seed_edges[:, i]
                    i += 1
                    egos = tuple(sorted(e.tolist()))
                    if egos not in edges_done:
                        edges_done.add(egos)
                        unique_edges.append(e)
                    if (
                        len(unique_edges) >= process_batch_size
                        or i >= seed_edges.shape[-1]
                    ):
                        break
                involved_nodes = pt.stack(unique_edges).unique()
                # pre-filter to the subgraph of all communities we want to sample to speed up loop
                # this might want to be batched to generate very large samples...
                sub_nodes, sub_edges, _inv, _sub_edge_mask = tg.utils.k_hop_subgraph(
                    involved_nodes, k, edges, relabel_nodes=True
                )
                subX = self.all_node_feats[sub_nodes]
                subY = y[sub_nodes]
                node_map = {
                    old: new for old, new in zip(involved_nodes.tolist(), _inv.tolist())
                }
                for e in unique_edges:
                    egos = tuple([node_map[o] for o in e.tolist()])
                    e_new = pt.tensor(egos)
                    # select the edges found in the joint subgraph of  of k-hop neighbourhoos
                    comm_nodes, sel_edges, _ci, comm_mask = tg.utils.k_hop_subgraph(
                        e_new, k, sub_edges, relabel_nodes=True
                    )
                    if size_limit is not None and (
                        len(comm_nodes) - (int(egonets) * 2) > size_limit
                    ):
                        continue
                    commX = subX[comm_nodes]
                    commY = subY[comm_nodes]
                    if egonets:
                        _ciset = _ci.flatten.unique().tolist()
                        egonodes = pt.tensor([x for x in _ciset if x not in egos])
                        sel_edges, _edge_attrs = tg.utils.subgraph(egonodes, sel_edges)
                        commX = commX[egonodes]
                        commY = commY[egonodes]
                    while len(edges_done) < seed_edge_offset:
                        continue
                    comm.append((sel_edges, commX, commY))
                    bar.update()
                    bar.set_description(f"Creating community {len(comm)}/{max_len}")
                    if max_len is not None and len(comm) >= max_len:
                        break
                if max_len is not None and len(comm) >= max_len:
                    break
            self.communities = comm
            with open(cache_path, "wb") as f:
                pt.save(dict(communities=self.communities), f)
            bar.close()

        self.dense = dense

    def __len__(self):
        return len(self.communities)

    def __getitem__(self, item):
        edge_list, X, Y = self.communities[item]
        if self.dense:
            G = nx.from_edgelist(edge_list.permute(1, 0).tolist())
            A = nx.to_numpy_array(G)
        else:
            A = edge_list
        if self.with_labels:
            return X, A, Y
        else:
            return X, A


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from matplotlib.colors import CSS4_COLORS

    d = Product_Categories(
        mode="different",
        verbose=True,
        with_labels=True,
        process_batch_size=100,
        seed_edge_offset=0,
        max_len=10000,
        size_limit=1000,
    )
    print(len(d))
    j = 0
    while j < 64:
        break
        fig, ax = plt.subplots(2, 2, figsize=[30, 30])
        for i, a in zip(range(len(d)), ax.flatten()):
            X, A, labels = d[j + i]
            G = nx.from_numpy_matrix(A)
            cats = labels.unique()
            CSS = list(CSS4_COLORS.keys())
            cols = {int(c): CSS[i] for i, c in enumerate(cats)}
            print(cats)
            nx.draw(G, ax=a, node_color=[cols[int(x)] for x in labels])
        fig.savefig("/tmp/products.pdf")
        plt.show(fig)
        j += 4
    if False:
        d = Product([0, 1], verbose=True, max_len=9, k=1)
        print(node_edges.cache_info())
        print(len(d))
        X, A, Y = d[0]
        G = nx.from_numpy_matrix(A)
        nx.draw(G)
        plt.show()
