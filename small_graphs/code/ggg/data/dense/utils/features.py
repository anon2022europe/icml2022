from warnings import warn

import torch as pt

from ggg.utils.hooks import tensor_backward_clean_hook
from ggg.utils.utils import ensure_tensor
from DominantSparseEigenAD.symeig import DominantSymeig

def stable_sym_eigen(adj, eigenvectors=False, retries=5, noise_scale=1e-5):

    adj = adj.clone()

    adj=adj+pt.diag(pt.randn(adj.shape[-1],device=adj.device)*1e-5)

    for i in range(retries):
        try:
            if noise_scale:
                noise = pt.diag(pt.randn(adj.shape[-1], device=adj.device))
                ret = pt.linalg.eigh( #symeig(
                    adj + noise,
                    eigenvectors=eigenvectors,
                )
            else:
                ret = pt.linalg.eigh(adj, eigenvectors=eigenvectors) #symeig
            return ret
        except RuntimeError as e:
            print(
                f"Cought {e}, ignoring, try {i}/{retries}, trying to smooth matrix before"
            )
            a2 = adj.pow(2.0)
            a2sum = a2.sum()
            adj = adj * (a2sum / 1e-10 <= a2).float()

    ret = pt.linalg.eigh(adj,eigenvectors=eigenvectors) #symeig
    if isinstance(ret,pt.Tensor):
        ret.register_hook(lambda g:tensor_backward_clean_hook(g,"eigenval"))
    else:
        for r,n in zip(ret,["eigenval","eigenvec"]):
            r.register_hook(lambda g:tensor_backward_clean_hook(g,n))

    return ret


def our_dom_symeig(x, k, end_offset=1, noise_scale=0.01):
    """
    Sice the pytorch builtin gets ALL, we need to slice
    :param x:
    :param k:
    :return:
    """
    x = ensure_tensor(x)
    vals, vecs = stable_sym_eigen(x, eigenvectors=True, noise_scale=noise_scale)

    """
    NxN
    k-1x N 
    
    Laplacian 2nd to the kth eigenvector
    
    On a graph with 2 communities: 1st vector constant, 2nd should have positive and negative entries
    
    """
    #  sym_eig goes from smallest to largest)=> get from end
    N = vals.shape[-1]
    idx = pt.arange(N - k, N - end_offset)
    with pt.no_grad():
        _, counts = pt.unique(vals, return_counts=True, dim=-1)
        if (counts > 1).any().item():
            warn(f"Found non-unique eigen-values, this might cause NaNs in backwards")

    return vals[..., idx], vecs[..., idx]


def our_small_symeig(x, k, offset=1, noise_scale=0.01):
    """
    Sice the pytorch builtin gets ALL, we need to slice
    :param x:
    :param k:
    :return:
    """
    x = ensure_tensor(x)
    vals, vecs = stable_sym_eigen(x, eigenvectors=True, noise_scale=noise_scale)

    """
    NxN
    k-1x N 

    Laplacian 2nd to the kth eigenvector

    On a graph with 2 communities: 1st vector constant, 2nd should have positive and negative entries

    """
    N = vals.shape[-1]
    idx = pt.arange(offset, k)
    with pt.no_grad():
        _, counts = pt.unique(vals, return_counts=True, dim=-1)
        if (counts > 1).any().item():
            warn(f"Found non-unique eigen-values, this might cause NaNs in backwards")

    return vals[..., idx], vecs[..., idx]


def approx_small_symeig(x, k):
    raise NotImplementedError()
    """
    Since DominantSymeig.apply seems to return only a single val,vec pair, we need to loop
    :param x:
    :param k:
    :return:
    """
    vals, vecs = [], []
    for val, vec in (DominantSymeig.apply(x, i) for i in range(1, k + 1)):
        vals.append(val)
        vecs.append(vec)
    vals = pt.stack(vals, -1)
    vecs = pt.stack(vecs, -1)
    return vals, vecs


def approx_dom_symeig(x, k):
    """
    Since DominantSymeig.apply seems to return only a single val,vec pair, we need to loop
    :param x:
    :param k:
    :return:
    """
    vals, vecs = [], []
    for val, vec in (DominantSymeig.apply(x, i) for i in range(1, k + 1)):
        vals.append(val)
        vecs.append(vec)
    vals = pt.stack(vals, -1)
    vecs = pt.stack(vecs, -1)
    return vals, vecs


def largest_k_eigenval(adj, k=4):
    return stable_sym_eigen(adj, eigenvectors=True)[0][:, -k:]


def smallest_k_eigenval(adj, k=4):
    return stable_sym_eigen(adj, eigenvectors=True)[0][:, 0:k]


def k_largest_eigenvec(adj, k=2, top_k=False):
    """

    Parameters
    ----------
    adj
    k
    top_k if True, return the vec correspodning to kth largest eigenval *and larger*

    Returns
    -------

    """

    if top_k:
        return stable_sym_eigen(adj, eigenvectors=True)[1][:, :, -k:]
    else:
        return stable_sym_eigen(adj, eigenvectors=True)[1][:, :, -k]


def k_smallest_eigenvec(adj, k=2, top_k=False):
    """

    Parameters
    ----------
    adj
    k
    top_k if True, return the vec correspodning to kth largest eigenval *and larger*

    Returns
    -------

    """

    if top_k:
        return stable_sym_eigen(adj, eigenvectors=True)[1][:, :, :k]
    else:
        return stable_sym_eigen(adj, eigenvectors=True)[1][:, :, :k]


# use K *SMALLEST* eigenvalues/corresponding vectors *everywhere*

if __name__ == "__main__":
    """
    Verify that the 2nd highed eigenval of the laplacian is indeed informative of communities
    """
    from ggg.data.dense.CommunitySmall import CommSmall
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    ds = CommSmall()
    ps = ds[0]
    a = pt.stack([p.A for p in [ds[i] for i in range(3)]])
    deg = a.sum(-1, keepdim=False)
    D = pt.diag_embed(deg)
    # print(D.shape, a.shape)
    L = a - D
    _, feat = stable_sym_eigen(L, True, noise_scale=0.1)
    # print(feat.shape)
    # print(feat[:, :, :3])
    g = nx.from_numpy_array(a[0].numpy())
    print(feat.shape)
    eig_k2_direct = feat[0, :, -2].flatten()
    feat0 = stable_sym_eigen(L[0], True, noise_scale=0.1)[1][:, -2]
    eig_k2 = our_dom_symeig(L[0], 2, noise_scale=0.1)[1].flatten()
    print(eig_k2_direct.shape, eig_k2.shape)
    print(len(eig_k2_direct), len(g.nodes))
    c1 = (eig_k2_direct > 0).nonzero().flatten().tolist()
    c2 = (eig_k2_direct <= 0).nonzero().flatten().tolist()
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_color="red", nodelist=c1)
    nx.draw_networkx_nodes(g, pos, node_color="blue", nodelist=c2)
    nx.draw_networkx_edges(g, pos)
    plt.show()
    c1o = (eig_k2 > 0).nonzero().flatten().tolist()
    c2o = (eig_k2 <= 0).nonzero().flatten().tolist()
    plt.figure()
    nx.draw_networkx_nodes(g, pos, node_color="pink", nodelist=c1)
    nx.draw_networkx_nodes(g, pos, node_color="cyan", nodelist=c2)
    nx.draw_networkx_edges(g, pos)
    plt.show()
    print(eig_k2_direct.shape, feat0.shape, feat.shape, eig_k2.shape)
    print(eig_k2_direct.flatten() - feat0.flatten())
    print(eig_k2_direct.flatten() - eig_k2.flatten())
