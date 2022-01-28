import torch
import os
import torch.nn.functional as F
from sklearn.manifold import SpectralEmbedding
import warnings
from graph_stat import *

warnings.filterwarnings("ignore")
from pprint import pprint
import numpy as np

import matplotlib.pyplot as plt
import networkx as nx

import copy


def sacred_copy(o):
    """Perform a deep copy on nested dictionaries and lists.
    If `d` is an instance of dict or list, copies `d` to a dict or list
    where the values are recursively copied using `sacred_copy`. Otherwise, `d`
    is copied using `copy.deepcopy`. Note this intentionally loses subclasses.
    This is useful if e.g. `d` is a Sacred read-only dict. However, it can be
    undesirable if e.g. `d` is an OrderedDict.
    :param o: (object) if dict, copy recursively; otherwise, use `copy.deepcopy`.
    :return A deep copy of d."""
    if isinstance(o, dict):
        return {k: sacred_copy(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [sacred_copy(v) for v in o]
    else:
        return copy.deepcopy(o)


def show_graph(
    adj,
    base_adj=None,
    remove_isolated=True,
    epoch=None,
    sample=None,
    dataset="",
    opt=None,
    suffix="",
):
    fig, ax = plt.subplots(ncols=2 if base_adj is not None else 1)
    gen_ax = ax[0] if base_adj is not None else ax
    if not isinstance(adj, np.ndarray):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = copy.deepcopy(adj)

    adj_ -= np.diag(np.diag(adj_))

    gr = nx.from_numpy_array(adj_)
    assert (adj_ == adj_.T).all()
    if remove_isolated:
        gr.remove_nodes_from(list(nx.isolates(gr)))
    nx.draw(gr, node_size=10, ax=gen_ax)
    gen_ax.set_title("gen")

    d = compute_graph_statistics(adj_)
    pprint(d)

    if base_adj is not None:
        base_ax = ax[1]
        base_gr = nx.from_numpy_array(base_adj)
        nx.draw(base_gr, node_size=10, ax=base_ax)
        base_ax.set_title("base")
        bd = compute_graph_statistics(base_adj)
        diff_d = {}
        for k in list(d.keys()):
            diff_d[k] = round(abs(d[k] - bd[k]), 4)
        log = f"{diff_d.keys()}\n{diff_d.values()}\n"
        logpath = os.path.join(opt.output_dir, f"condgenlog_{dataset!a}.log")
        with open(logpath, "a") as f:
            f.write(log)
        print(log)

    save_path = os.path.join(
        opt.output_dir, f"condgen_{dataset!a}_{epoch!a}_{sample!a}{suffix}.pdf"
    )
    fig.savefig(save_path)


def make_symmetric(m):
    m_ = torch.transpose(m)
    w = torch.max(m_, m_.T)
    return w


def make_adj(x, n):
    res = torch.zeros(n, n)
    i = 0
    for r in range(1, n):
        for c in range(r, n):
            res[r, c] = x[i]
            res[c, r] = res[r, c]
            i += 1
    return res


def cat_attr(x, attr_vec):
    if attr_vec is None:
        return x
    attr_mat = attr_vec.repeat(x.size()[0], 1)
    x = torch.cat([x, attr_mat], dim=1)
    return x


def get_spectral_embedding(adj, d):
    """
    Given adj is N*N, return its feature mat N*D, D is fixed in model
    :param adj:
    :return:
    """

    if torch.is_tensor(adj):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = adj
    emb = SpectralEmbedding(n_components=d)
    res = emb.fit_transform(adj_)
    x = torch.from_numpy(res).float()
    return x


def normalize(adj):
    dev=adj.device
    adj = adj.data.cpu().numpy()
    adj_ = adj + np.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = np.diag(np.power(rowsum, -0.5).flatten())
    degree_mat_sqrt = np.diag(np.power(rowsum, 0.5).flatten())
    adj_normalized = degree_mat_inv_sqrt.dot(adj_).dot(degree_mat_sqrt)
    return torch.from_numpy(adj_normalized).float().to(dev)


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = (
        adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    )
    return sparse_to_tuple(adj_normalized)


def keep_topk_conns(adj, k=3):
    g = nx.from_numpy_array(adj)
    to_removes = [cp for cp in sorted(nx.connected_components(g), key=len)][:-k]
    for cp in to_removes:
        g.remove_nodes_from(cp)
    adj = nx.to_numpy_array(g)
    return adj


def remove_small_conns(adj, keep_min_conn=4):
    g = nx.from_numpy_array(adj)
    for cp in list(nx.algorithms.components.connected_components(g)):
        if len(cp) < keep_min_conn:
            g.remove_nodes_from(cp)
    adj = nx.to_numpy_array(g)
    return adj


def top_n_indexes(arr, n):
    idx = np.argpartition(arr, min(arr.size - 1, arr.size - n), axis=None)[-n:]
    width = arr.shape[1]
    return [divmod(i, width) for i in idx]


def topk_adj(adj, k):
    adj_ = adj.data.cpu().numpy()
    # assert ((adj_ == adj_.T).all())
    adj_ = (adj_ - np.min(adj_)) / np.ptp(adj_)
    adj_ -= np.diag(np.diag(adj_))
    tri_adj = np.triu(adj_)
    inds = top_n_indexes(tri_adj, k // 2)
    res = torch.zeros(adj.shape)
    for ind in inds:
        i = ind[0]
        j = ind[1]
        res[i, j] = 1.0
        res[j, i] = 1.0
    return res


def test_gen(model, n, attr_vec, z_size, twice_edge_num, bd=None):
    fixed_noise = torch.randn((n, z_size), requires_grad=True)
    if attr_vec is not None:
        fixed_noise = cat_attr(fixed_noise, attr_vec)
    a_ = model.decoder(fixed_noise)
    # print(F.sigmoid(a_))
    a_ = topk_adj(F.sigmoid(a_), twice_edge_num)
    # print(a_)
    if bd:
        show_graph(a_, bd)
    else:
        show_graph(a_)


def gen_adj(model, n, e, attr_vec, z_size, return_Z=False, device="cpu"):
    fixed_noise = torch.randn((n, z_size), requires_grad=True).to(device)
    if attr_vec is not None:
        fixed_noise = cat_attr(fixed_noise, attr_vec)
    rec_adj = model.decoder(fixed_noise)
    A = topk_adj(rec_adj, e * 2)
    if return_Z:
        return A, fixed_noise
    else:
        return A


def eval(adj, base_adj=None):
    if not isinstance(adj, np.ndarray):
        adj_ = adj.data.cpu().numpy()
    else:
        adj_ = copy.deepcopy(adj)

    adj_ -= np.diag(np.diag(adj_))
    gr = nx.from_numpy_array(adj_)
    assert (adj_ == adj_.T).all()

    d = compute_graph_statistics(adj_)
    pprint(d)

    if base_adj is not None:
        # base_adj = base_adj.numpy()
        base_gr = nx.from_numpy_array(base_adj)
        bd = compute_graph_statistics(base_adj)
        diff_d = {}

        for k in list(d.keys()):
            diff_d[k] = round(abs(d[k] - bd[k]), 4)
    return diff_d
