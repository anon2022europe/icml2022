from _warnings import warn

import torch as pt
from DominantSparseEigenAD.symeig import DominantSymeig

from ggg_utils.utils.hooks import tensor_backward_clean_hook
from ggg_utils.utils.utils import ensure_tensor


def stable_sym_eigen(adj, eigenvectors=False, retries=5, noise_scale=1e-5,force_clean=False):

    adj = adj.clone()

    adj = adj + pt.diag(pt.randn(adj.shape[-1], device=adj.device) * 1e-5)

    for i in range(retries):
        try:
            if noise_scale:
                noise = pt.diag(pt.randn(adj.shape[-1], device=adj.device))*noise_scale
                if eigenvectors:
                    ret = pt.linalg.eigh(adj + noise, UPLO="U")
                else:
                    ret = pt.linalg.eigvalsh(adj + noise, UPLO="U")
            else:
                if eigenvectors:
                    ret = pt.linalg.eigh(adj, UPLO="U")
                else:
                    ret = pt.linalg.eigvalsh(adj, UPLO="U")
            # ensure we clean this in
            if isinstance(ret, pt.Tensor) and ret.requires_grad or force_clean:
                ret.register_hook(lambda g: tensor_backward_clean_hook(g, "eigenval"))
            else:
                for r, n in zip(ret, ["eigenval", "eigenvec"]):
                    if r.requires_grad or force_clean:
                        r.register_hook(lambda g: tensor_backward_clean_hook(g, n))
            return ret
        except RuntimeError as e:
            warn(
                f"Cought {e}, ignoring, try {i}/{retries}, trying to smooth matrix before"
            )
            a2 = adj.pow(2.0)
            a2sum = a2.sum()
            adj = adj * (a2sum / 1e-10 <= a2).float()

    if eigenvectors:
        ret = pt.linalg.eigh(adj, UPLO="U")
    else:
        ret = pt.linalg.eigvalsh(adj,UPLO="U")
    if isinstance(ret, pt.Tensor) and ret.requires_grad or force_clean:
        ret.register_hook(lambda g: tensor_backward_clean_hook(g, "eigenval"))
    else:
        for r, n in zip(ret, ["eigenval", "eigenvec"]):
            if r.requires_grad or force_clean:
                r.register_hook(lambda g: tensor_backward_clean_hook(g, n))

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
    return stable_sym_eigen(adj, eigenvectors=False)[0][:, -k:]


def smallest_k_eigenval(adj, offset=0,k=4,force_clean=False):
    return stable_sym_eigen(adj, eigenvectors=False,force_clean=force_clean)[:, offset:offset+k]


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

if __name__=="__main__":
    import networkx as nx
    g=nx.erdos_renyi_graph(n=20,p=0.4)
    a=pt.from_numpy(nx.to_numpy_array(g)).unsqueeze(0)
    l=smallest_k_eigenval(a,0,5)
