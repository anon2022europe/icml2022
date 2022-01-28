import torch_geometric
import networkx as nx
import numpy as np
import scipy as sp
import torch
import torch as pt

from ggg_utils.utils.eigen import stable_sym_eigen
from ggg_utils.utils.hooks import tensor_backward_clean_hook


def anon_get_eigfeatures(graph):
    """
    Adds permutation equivariant eigenfeatures to the pytorch graph object.
    Let U, l be the eigenvectors and eigenvalues of the Laplacian associated with the input graph.
    The code adds the U diag(l) matrix as node features.    Inputs:
        graph: a pytorch geometric graph object
    Returns:
        X: an torch object    Remarks:
        Procedural aspects
        * the entries have been rounded to 4 decimal digits to avoid precision errors of the solver
        * the first eigenvector (with eigenvalue 0) is not returned since it's not informative
        * in case of eigenvalue multiplicities, the eigenvectors are substituted by the constant vector (since the eigenvector choice is arbitrary)
        * the eigenvector sign ambiguity problem is dealt with by ensuring that the positive entries of each eigenvector are always larger in magnitude.        It follows from the above that:
        * The node features are always the same (up to permutation) for isomorphic graphs
        * However, there might exist different graphs with the same features
        * Each node is given n-1 features.
        * There is no need to also pass the eigenvalues as separate features. They are encoded in the magnitute of the eigenvectors.    Warning:
        If you are using this function with graphs of different sizes, make sure to zeropad the feature matrix
        (such that all nodes in all graphs have the same number of features).     Example:
        To imbue a pytorch geometric dataset (consisting of graphs with the same number of nodes) with eigenfeatures, use        for graph in dataset_torch:
            setattr(graph, 'x', get_eigfeatures(graph))    anon anon
    7 Feb 2021
    """
    graph = torch_geometric.utils.convert.to_networkx(graph)
    A = nx.to_numpy_array(graph)
    D = np.diag(sum(A))
    L = D - A
    l, U = sp.linalg.eigh(L)  # round it up -- the solver has limited precision
    l = np.round(l, decimals=5)
    U = np.round(U, decimals=5)
    n = A.shape[0]  # kill the trivial eigenspace
    U, l = U[:, 1:], l[1:]  # remove eigenspaces with multiplicity
    if len(set(l)) < n:
        for v in np.unique(l):
            idx = np.where(l == v)[0]
            if len(idx) > 1:
                U[:, idx] = 1  # eigenvalue normalization and resolve sign ambiguity
        for k in range(U.shape[1]):
            u = U[:, k]
            u *= l[k] / np.linalg.norm(u)  # encode eigenvalue in eigenvector length
            if np.max(u) <= 0:  # all negative entries
                U[:, k] = -u
            elif np.min(u) >= 0:  # all positive entries
                U[:, k] = u
            else:
                p = np.sort(u)[::-1]
                n = np.sort(-u)[::-1]
                diff = p - n
                if (
                    sum(np.abs(diff)) == 0
                ):  # cannot break ambiguity due to balanced positive and negative entries
                    U[:, k] = np.abs(u)
                else:
                    p_idx = min(np.where(diff > 0)[0])
                    n_idx = min(np.where(diff < 0)[0])
                    if p_idx < n_idx:  # the positive entries are larger in magnitude
                        U[:, k] = u
                    elif p_idx > n_idx:  # the negative entries are larger in magnitude
                        U[:, k] = -u
                    else:  # this should never happen
                        U[:, k] = np.abs(u)  # kill off any remaining precision errors
    X = np.round(U, decimals=4)  # Convert to a python tensor
    X = torch.tensor(X, dtype=torch.float)
    return X


def round(x, DECIMALS=5,with_grad=False):
    grad=x-x.detach() if with_grad else 0.0
    DEC = 10 ** DECIMALS
    return ((x * DEC).round() / DEC).detach()+grad


def anon_get_eigfeatures_torch(A: pt.Tensor, noise_scale=1e-5, k=4):
    # TODO: check if this runs, then ask anon for correctness
    D = pt.diag_embed(A.sum(-1))
    target = D - A
    return extract_canonical_k_eigenfeat(target, k, offset=1, noise_scale=noise_scale)


def extract_canonical_k_eigenfeat(target, offset=1, k=4, noise_scale=1e-5):
    need_grad = target.requires_grad
    #egival, eigvec
    l, U = stable_sym_eigen(target, eigenvectors=True, noise_scale=noise_scale)
    # round it up -- the solver has limited precision
    l = round(l,with_grad=True)
    U = round(U,with_grad=True)
    if target.dim() == 2:
        U, l = U.unsqueeze(0), l.unsqueeze()
    assert target.dim() == 3
    assert l.dim() == 2
    # kill the trivial eigenspace by skipping the lowest one
    U, l = U[:, :, offset:], l[:, offset:]
    # remove eigenspaces with multiplicity
    # have to loop because torch unique is funky
    for b in range(l.shape[0]):
        lt = l[b]
        l_uniqs, inv_idx, l_counts = pt.unique(
            lt.detach(), return_counts=True, return_inverse=True
        )
        if (l_counts == 1).all():
            continue
        else:
            # mask multiplicities in eigenvec entries with 0 (constant can be done by adding a inv-masked array)
            multiplicity_mask = (l_counts[inv_idx] == 1).float()
            U[b, :, :] = U[b, :, :] * multiplicity_mask.detach()
    # eigenvalue normalization and resolve sign ambiguity
    U = (
        U / (U + 1e-6).norm(keepdim=True) * l.unsqueeze(-1)
    )  # encode eigenvalue in eigenvector length
    # flip all eigenvector signs where the negative parts outnumber the positive parts
    less_0_mag_sum = U.clamp_max(0).abs().sum(-1, keepdim=True)
    geq_0_mag_sum = U.clamp_min(0).abs().sum(-1, keepdim=True)
    flip_sign = (-1.0) ** (less_0_mag_sum > geq_0_mag_sum).float()
    U = U * flip_sign.detach()
    # if we have exact equality, need to abs
    abs_mask = (less_0_mag_sum == geq_0_mag_sum).detach()
    U = U.abs() * abs_mask + (U * pt.logical_not(abs_mask))
    # round again to kill remaining ambiguitities
    U = round(U,with_grad=True)
    # select 1st to kth eigenvec
    feat = U[:, :, :k]
    if feat.requires_grad:
        feat.register_hook(
            lambda g: tensor_backward_clean_hook(g, "canonical-eigenfeat")
        )
    assert feat.requires_grad == need_grad
    return feat
