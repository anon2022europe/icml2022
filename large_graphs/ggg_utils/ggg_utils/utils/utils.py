import copy
import logging
from inspect import signature
from logging import info, warning
from typing import Dict, List, Tuple, Union, Optional
from warnings import warn

import attr
import networkx as nx
import torch
import torch as pt
import torch_geometric as pg
from attr.validators import in_
from pdb import set_trace
import numpy as np
from sparsemax import Sparsemax
from torch_sparse import SparseTensor
from tqdm import tqdm
from torch import nn

from ggg_utils.utils.spectral_norm import sn_wrap
from ggg_utils.utils.adam import Adam

def torch_sparse_coo(s:SparseTensor)->Tuple[pt.Tensor,pt.Tensor]:
    # following https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html
    row, col, edge_attr = s.t().coo()
    edge_index = torch.stack([row, col], dim=0)
    return edge_index,edge_attr


def get_laplacian(A):
    # undirected graphs, in and out degree shttps://github.com/buwantaiji/DominantSparseEigenAD/blob/master/examples/TFIM_vumps/symmetric.pyame
    D = A.sum(-1)
    D = pt.diag_embed(D)
    L = D - A
    return D


def ensure_tensor(x):
    if pt.is_tensor(x):
        return x
    elif isinstance(x, np.ndarray):
        return pt.from_numpy(x)
    else:
        return pt.tensor(x)

# An ordinary implementation of Swish function
# from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
class Swish(nn.Module):
    def forward(self, x):
        return x * pt.sigmoid(x)


# A memory-efficient implementation of Swish function
# from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
class SwishImplementation(pt.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * pt.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = pt.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


# MemoryEfficientSwish
# from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/utils.py
class MESwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


InEfficientSwish = Swish
Swish = MESwish
def get_act(swish):
    if swish is True or swish == "swish":
        act = Swish
    elif swish == "leaky":
        act = lambda: torch.nn.LeakyReLU(0.1)
    elif swish == "celu":
        act = torch.nn.CELU
    elif swish == "sigmoid":
        act=pt.nn.Sigmoid
    elif swish=="relu":
        act = torch.nn.ReLU
    else:
        raise NotImplementedError(f"get_act doesn't support {swish}")
    return act

def batch_trace(X):
    """
    Expect a matrix of shape B N N, returns the trace in shape B
    :param X:
    :return:
    """
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    trace = diag.sum(dim=-1)
    return trace


def batch_diagonal(X):
    """
    Extracts the diagonal from the last two dims of a tensor
    :param X:
    :return:
    """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class kcycles:
    """
    Following
    careful, 5 cycle form is wrong
    https://mathworld.wolfram.com/GraphCycle.html
    and
    https://theory.stanford.edu/~virgi/cs267/papers/cycles-ayz.pdf
    """

    def __init__(self):
        super(kcycles, self).__init__()

    def calculate_kpowers(self):
        self.k1_matrix = self.adj_matrix.float()
        self.k2_matrix = self.k1_matrix @ self.adj_matrix.float()
        self.k3_matrix = self.k2_matrix @ self.adj_matrix.float()
        self.k4_matrix = self.k3_matrix @ self.adj_matrix.float()
        self.k5_matrix = self.k4_matrix @ self.adj_matrix.float()
        self.k6_matrix = self.k5_matrix @ self.adj_matrix.float()

    def k3_cycle(self):
        """
        K3: take A^3, take the trace of that, divide by 6
        :return:
        """
        c3 = batch_trace(self.k3_matrix)
        return c3 / 6

    def k4_cycle(self):
        """
        4 cycles: calculate A^2 A^4, take the traces of both, take the sum d of the squared elements of the diagonal of A^2
        i.e., trace A^2 elementwise squared, sum the traces and subtract 2*d, then divide by 8
        :return:
        """
        tr_a4 = batch_trace(self.k4_matrix)
        tr_a2 = batch_trace(self.k2_matrix)

        d = batch_trace(self.k2_matrix ** 2)

        c4 = tr_a4 + tr_a2 - 2 * d
        return c4 / 8

    def k5_cycle(self):
        """
        A^5,A^3,A^2 get the traces,
        :return:
        """
        tr_a5 = batch_trace(self.k5_matrix)
        tr_a3 = batch_trace(self.k3_matrix)

        ng_h5 = 0
        a3t = torch.diagonal(self.k3_matrix, dim1=-2, dim2=-1)
        d = torch.diagonal(self.k2_matrix, dim1=-2, dim2=-1) - 2
        ng_h5_t = (d * a3t).sum(dim=-1)

        ng_c3 = tr_a3

        c5_t = tr_a5 - 5 * ng_h5_t - 5 * ng_c3
        return c5_t / 10

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6_matrix)

        term_2_t = batch_trace(self.k3_matrix ** 2)

        term3_t = torch.sum(self.adj_matrix * self.k2_matrix.pow(2), dim=[-2, -1])

        d_t4 = batch_diagonal(self.k2_matrix)
        a_4_t = batch_diagonal(self.k4_matrix)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)

        term_5_t = batch_trace(self.k4_matrix)

        term_6_t = batch_trace(self.k3_matrix)

        term_7_t = batch_diagonal(self.k2_matrix).pow(3).sum(-1)

        term8_t = torch.sum(self.k3_matrix, dim=[-2, -1])

        term9_t = batch_diagonal(self.k2_matrix).pow(2).sum(-1)

        term10_t = batch_trace(self.k2_matrix)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return c6_t / 12

    def k_cycles(self, adj_matrix, verbose=False):
        self.adj_matrix = adj_matrix

        self.calculate_kpowers()

        if verbose:
            print("3 Cycles")
        k3 = self.k3_cycle().unsqueeze(-1).float()
        if verbose:
            print("4 Cycles")
        k4 = self.k4_cycle().unsqueeze(-1).float()
        if verbose:
            print("5 Cycles")
        k5 = self.k5_cycle().unsqueeze(-1).float()
        if verbose:
            print("6 Cycles")
        k6 = self.k6_cycle().unsqueeze(-1).float()

        kcycles = torch.cat([k3, k4, k5, k6], dim=-1)

        return kcycles

    def triangles_(self, adj_matrix, k_, prev_k=None):
        if prev_k is None:
            k_matrix = torch.matrix_power(adj_matrix.float(), k_)
        else:
            k_matrix = prev_k @ adj_matrix.float()
        egd_l = torch.diagonal(k_matrix, dim1=-2, dim2=-1)
        return egd_l, k_matrix


def label2onehot(labels, dim):
    """Convert label indices to dense one-hot vectors."""

    labels = ensure_tensor(labels)
    device = labels.device
    labels = labels.long()

    out = torch.zeros(list(labels.size()) + [dim]).to(device)
    out.scatter_(len(out.size()) - 1, labels.unsqueeze(-1), 1.0)
    return out


def dense_graph_batch_oh(item_list, k_paths=0):
    one_hot = 5
    max_num = 0
    item_list = [(ensure_tensor(x), ensure_tensor(a)) for x, a in item_list]
    for x, a in item_list:
        max_num = max(max_num, a.shape[-1])

    padded_items = [expand_item(x, a, max_num, one_hot) for x, a in item_list]
    X_stack = pt.stack([p[0] for p in padded_items])
    A_b = pt.stack([p[1] for p in padded_items])

    X_b = node_featues_batch(A_b, X_stack, k_paths)

    return X_b, A_b


def dense_graph_batch(item_list, k_paths=0):
    one_hot = 5
    max_num = 0
    for x, a in item_list:
        max_num = max(max_num, a.shape[-1])

    if len(x.shape) == 1:
        x_shape = one_hot + 2 + (k_paths - 2)
    else:
        x_shape = x[0].shape[-1] + 2 + (k_paths - 2)

    X_stack = pt.stack([x for x, _ in item_list])
    A_b = pt.stack([a for _, a in item_list])

    X_b = node_featues_batch(A_b, X_stack, k_paths)

    return X_b, A_b


def expand_item(x, a, max_num, one_hot):
    """
    Expand a labeled node to a one hote encoded label + the additional node features
    Parameters
    ----------
    x
    a
    max_num
    one_hot

    Returns
    -------

    """
    X = pt.zeros(max_num, one_hot)
    A = pt.zeros(max_num, max_num)
    num = a.shape[-1]
    x_b = label2onehot(x, one_hot)
    X[:num, :] = x_b
    A[:num, :num] = a

    return X, A


def single_node_featues(A, X, k_paths):
    # # Append degrees
    # app_ = A.sum(-1, keepdims=True)

    # Append graph size
    raise NotImplementedError("TODO: refactor")
    num = A.max(dim=1)[0].sum(dim=-1)
    node_mask = (pt.ones(A.shape[-1], 1, device=A.device).cumsum(dim=1)).float()
    app_ = node_mask * num.reshape(-1, 1, 1)

    X = torch.cat([X, app_], dim=-1)
    return X


def node_featues_batch(A, X, k_paths):
    kc = kcycles()
    triang_adj = A.clone()
    torch.diagonal(triang_adj, dim1=-2, dim2=-1).fill_(0)
    k_matrix = None
    B = A.shape[0]
    num = A.max(dim=1)[0].sum(
        dim=-1
    )  # rowise maximum=> only rows with some connections, i.e. present nodes, sum=> number of nodes
    node_mask = (
        pt.ones(B, A.shape[-1], 1, device=A.device).cumsum(dim=1)
        <= num.reshape(B, 1, 1)
    ).float()
    # add paths of len 1, len 2 to nodes
    for k_ in range(2, k_paths + 1):
        paths, k_matrix = kc.triangles_(triang_adj, k_, prev_k=k_matrix)
        X = torch.cat((X, node_mask * paths.reshape(B, -1, 1)), -1)

    # in_degrees = a.sum(0)
    # out_degrees = a.sum(1)
    # X_batch[i, :num, -3] = in_degrees
    # X_batch[i, :num, -2] = out_degrees
    # assign the number of nodes in the graph to all nodes
    num = node_mask * num.reshape(B, 1, 1)

    X = torch.cat([X, num], dim=-1)
    return X


def node_features_single(
    A_batch, X_batch, item_list, k_paths, kcycles_, one_hot, x_shape
):
    """
    Loops through item list, calculates node features and places them in the batch matrix
    :param A_batch:
    :param X_batch:
    :param item_list:
    :param k_paths:
    :param kcycles_:
    :param one_hot:
    :param x_shape:
    :return:
    """

    for i, (x, a) in enumerate(item_list):

        triang_adj = a.clone()
        torch.diagonal(triang_adj).fill_(0)

        num = a.shape[-1]
        x_b = label2onehot(x, one_hot)
        k_matrix = None

        for k_ in range(2, k_paths + 1):
            paths, k_matrix = triangles_(triang_adj, k_, prev_k=k_matrix)
            x_b = torch.cat((x_b, paths.reshape(num, 1)), 1)

        # in_degrees = a.sum(0)
        # out_degrees = a.sum(1)
        A_batch[i, :num, :num] = a
        X_batch[i, :num, :-1] = x_b
        # X_batch[i, :num, -3] = in_degrees
        # X_batch[i, :num, -2] = out_degrees
        X_batch[i, :num, -1] = num
    return X_batch, A_batch


def pad_to_max(xlist, N_dim=1, pad_dims=(0, 1)):
    max_N = np.max([x.shape[N_dim] for x in xlist])
    padded_list = []
    for x in xlist:
        x_padded = pad_to(x, max_N, pad_dims)
        padded_list.append(x_padded)
    return padded_list


def pad_to(x, max_N, pad_dims,pad_front=False):
    if all(x.shape[i]==max_N for i in pad_dims):
        return x
    pad_len = [
        (0, 0) if i not in pad_dims else (0, max_N - x.shape[i])
        for i in range(len(x.shape))
    ]
    if pad_front:
        pad_len=[tuple(reversed(x)) for x in pad_len]
    if torch.is_tensor(x):
        pad_len = [p for y in reversed(pad_len) for p in y]
        x_padded = torch.nn.functional.pad(x, pad_len)
    else:
        x_padded = np.pad(x, pad_len, mode="constant")
    return x_padded


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


if __name__ == "__main__":
    xs = [torch.rand(np.random.randint(1, 10, 2).tolist()) for _ in range(10)]
    print([x.shape for x in xs])
    xs = pad_to_max(xs)
    print([x.shape for x in xs])


def zero_diag(A):
    """
    In-place diagonal zeroing, careful will break any sigmoid etc after it
    :param A:
    :return:
    """
    N = A.shape[1]
    idx = torch.arange(N)
    d = torch.diagonal(A, dim1=-2, dim2=-1)
    D_zero = torch.ones_like(A) - torch.diag_embed(d)
    Az = A * D_zero
    Az[..., idx, idx] = 0.0
    return Az


def get_kernel(scorfun:str)->pt.nn.Module:
    if scorfun=="sigmoid":
        return pt.nn.Sigmoid()
    elif scorfun=="softmax":
        return pt.nn.Softmax()
    elif scorfun=="sparsemax":
        return Sparsemax()
    elif scorfun is None:
        return None
    else:
        raise NotImplementedError(f"Unknown kernel{scorfun}")
def add_to_non_edges(A, x, N=None):
    """
    :param A:
    fill: x
    :return:
    """
    d = torch.diagonal(A, dim1=-2, dim2=-1)
    Az = A + torch.diag_embed(pt.ones_like(d) * x)
    if N is not None:
        with pt.no_grad():
            non_edge_mask = pt.logical_not(adj_mask(A, N))
            non_edge_mask = non_edge_mask * x
            # nan filter
            non_edge_mask[pt.isnan(non_edge_mask)] = 0.0
        Az = Az + non_edge_mask
    return Az


def closest_multiple(i, N):
    return type(i)(np.ceil(i / N) * N)


def triangles_(adj_matrix, k_, prev_k=None):
    if prev_k is None:
        k_matrix = torch.matrix_power(adj_matrix.float(), k_)
    else:
        k_matrix = prev_k @ adj_matrix.float()
    egd_l = torch.diagonal(k_matrix, dim1=-2, dim2=-1)
    return egd_l, k_matrix


def pg_mask_from_N(max_N: int, N: pt.Tensor):
    B = N.shape[0]
    mask = pt.ones([B, max_N], device=N.device).cumsum(1) <= N.view(-1, 1)
    return mask


def zero_mask_nodes(node_feats: pt.Tensor, N: pt.Tensor):
    mask = node_mask(node_feats, N)
    return node_feats * mask


def zero_and_symmetrize(A,force_prezero=False):
    A = zero_diag(A)
    if zero_and_symmetrize.pre_zero or force_prezero:
        A = torch.triu(A) # zeros out half of the scores, but ensures we don't get 0.5 values in discrete case
        A = A + A.permute(0, 2, 1)
    else:
        A=0.5*(A+A.permute(0,2,1)) # might have 0.5 value=> needs to be applied pre-discretize?
    return A
zero_and_symmetrize.pre_zero=True
def set_zas_prezero(b:bool):
    zero_and_symmetrize.pre_zero=b


def place_mat(canvas_size, offset_2, offset_1, mat):
    # places a matrix in another matrix
    # offset_2: top down for a 1 N N tensor
    # offset_1 left right
    c = pt.zeros(canvas_size, device=mat.device)
    end_2 = offset_2 + mat.shape[-2]
    end_1 = offset_1 + mat.shape[-1]
    c[:, offset_2:end_2, offset_1:end_1] = mat
    return c


def zero_and_symmetrize_crp(oldA, new_new, new_old):
    # TODO anon: document
    # oldA: # B N N
    # new_new: B L L
    # new_old: B L Nc  => need to flip
    oldA = torch.triu(oldA)  # B N N
    L = new_new.shape[-1]
    Nold = oldA.shape[-1]
    assert new_old.shape[-2]==L
    assert new_old.shape[-1]==Nold
    new_new = pt.triu(new_new)  # L L upper triangular
    new_size = [oldA.shape[0], L + Nold, L + Nold]
    new_new = place_mat(new_size, 0, 0, new_new)

    new_old = place_mat(new_size, 0, L, new_old)
    A = place_mat(new_size, L, L, oldA)
    A = new_new + A + new_old  # B N+L N+L

    A = torch.triu(A)  # B N N
    A = zero_diag(A)
    A = A + A.permute(0, 2, 1)
    return A


def node_mask(node_feats, N):
    mask = pt.ones_like(node_feats).cumsum(1) <= N.view(-1, 1, 1)
    return mask


def zero_mask_adjmat(adjmats: pt.Tensor, N: pt.Tensor):
    mask = adj_mask(adjmats, N)
    return adjmats * mask


def adj_mask(adjmats, N):
    os = pt.ones_like(adjmats)
    row_mask = os.cumsum(-1) <= N.view(-1, 1, 1)
    col_mask = os.cumsum(-2) <= N.view(-1, 1, 1)
    mask = col_mask.logical_and(row_mask)
    return mask


def adj_mask_2(adjmats, N_row, N_col):
    os = pt.ones_like(adjmats)
    row_mask = os.cumsum(-1) <= N_col.view(-1, 1, 1)
    col_mask = os.cumsum(-2) <= N_row.view(-1, 1, 1)
    mask = col_mask.logical_and(row_mask)
    return mask


def filter_kwargs(cls, kwargs):
    kwargs = {k: kwargs[k] for k in signature(cls).parameters.keys() if k in kwargs}
    missing = [k for k in signature(cls).parameters.keys() if k not in kwargs]
    if len(missing) > 0:
        warning(f"Didn't find {missing} when creating {cls.__name__}")
    return kwargs


def kwarg_create(cls, kwargs: Dict):
    # creates the class given the kwargs, filtering out any which aren't in class definition
    kwargs = filter_kwargs(cls, kwargs)
    return cls(**kwargs)


def pac_reshape(X, pac, to_packed=False, mode="nodes"):
    if mode == "nodes" or mode == "adj":
        if to_packed:
            B = X.shape[0]
            return X.reshape(B // pac, pac, -1, X.shape[-1])
        else:
            B = X.shape[0] * pac
            return X.reshape(B, -1, X.shape[-1])
    elif mode == "N":
        if to_packed:
            B = X.shape[0]
            return X.reshape(B // pac, pac)
        else:
            B = X.shape[0] * pac
            return X.reshape(B)
    else:
        raise ValueError(f"Don't know mode {mode}")


def enable_asserts(state):
    enable_asserts.state = state


enable_asserts.state = None


def asserts_enabled():
    return enable_asserts.state == True


def maybe_assert(x=None, func=None):
    if asserts_enabled():
        if x is not None:
            assert x
        if func is not None:
            assert func()


def pdf(weights):
    sum = weights.sum()
    assert sum != 0
    return weights / sum


class MLP(pt.nn.Module):
    def __init__(self, widths: List[int], act=None, norm=None, spectral_norm=False):
        super(MLP, self).__init__()
        layers = [sn_wrap(pt.nn.Linear(widths[0], widths[1]), spectral_norm)]
        if norm is not None:
            layers.append(norm(widths[1]))
        if act is not None:
            layers.append(act() if not isinstance(act, pt.nn.Module) else act)
        for i in range(1, len(widths) - 1):
            layers.append(
                sn_wrap(pt.nn.Linear(widths[i], widths[i + 1]), spectral_norm)
            )
            if norm is not None:
                layers.append(norm(widths[i + 1]))
            if act is not None:
                layers.append(act() if not isinstance(act, pt.nn.Module) else act)

        self.mlp = pt.nn.Sequential(*layers)
    def forward(self, X):
        return  self.mlp(X)


class Rezero(pt.nn.Module):
    def __init__(self,m,indim=None,outdim=None):
        super(Rezero, self).__init__()
        self.gate = torch.nn.Parameter(pt.zeros([]))
        self.proj = (
            pt.nn.Identity()
            if indim is None or indim == outdim
            else pt.nn.Linear(indim, outdim)
        )
        self.m=m
    def forward(self,x):
        act=self.m(x)
        return act*self.gate+self.proj(x)

class RezeroMLP(pt.nn.Module):
    def __init__(self, widths: List[int], act=None, norm=None, spectral_norm=False):
        super(RezeroMLP, self).__init__()
        self.mlp=Rezero(MLP(widths,act,norm,spectral_norm),widths[0],widths[-1])

    def forward(self, X):
        return self.mlp(X)


class EMA(pt.nn.Module):
    def __init__(self, shape: Tuple[int], smoothing=0.9, trainable=False):
        super(EMA, self).__init__()
        self.smoothing = smoothing
        self.shape = shape
        self.trainable = trainable
        self.ema = pt.nn.Parameter(pt.zeros(shape), requires_grad=trainable)

    def forward(self, X):
        grad = self.trainable and self.training
        self.ema.requires_grad = grad
        with pt.set_grad_enabled(grad):
            self.ema.data = self.ema * (1 - self.smoothing) + self.smoothing * X
            return self.ema.clone()


def squash_nonfinite(g, name=None, squash_all=False):
    if name is None:
        name = ""
    nans = pt.isfinite(g) == False
    if nans.any():
        n_nans = len(nans.nonzero())
        N = len(nans.reshape(-1))
        perc = n_nans / N
        logging.warning(f"Squashing {n_nans}/{N} ({perc:0.2e}) non-finite - {name}")
        if squash_all:
            g = pt.zeros_like(g)
        else:
            g[nans] = 0.0
    return g
def squash_nan(g, name=None, squash_all=False):
    if name is None:
        name = ""
    nans = pt.isnan(g)
    if nans.any():
        n_nans = len(nans.nonzero())
        N = len(nans.reshape(-1))
        perc = n_nans / N
        logging.warning(f"Squashing {n_nans}/{N} ({perc:0.2e}) nans - {name}")
        if squash_all:
            g = pt.zeros_like(g)
        else:
            g[nans] = 0.0
    return g

def squash_infs(g, name=None, squash_all=False):
    if name is None:
        name = ""
    nans = pt.logical_or(pt.isposinf(g),pt.isneginf(g))
    if nans.any():
        n_nans = len(nans.nonzero())
        N = len(nans.reshape(-1))
        perc = n_nans / N
        logging.warning(f"Squashing {n_nans}/{N} ({perc:0.2e}) infs - {name}")
        if squash_all:
            g = pt.zeros_like(g)
        else:
            g[nans] = 0.0
    return g

def torch_int(instance, attribute, value):
    return any(value.dtype==t for t in [pt.int,pt.long,pt.short,pt.int8,pt.int16,pt.int32,pt.int64])
def torch_float(instance, attribute, value):
    return any(value.dtype==t for t in [pt.float,pt.double,pt.float16,pt.float32,pt.float64])
@attr.s(eq=False)
class Graph:
    x=attr.ib(type=pt.Tensor,default=None)
    edges=attr.ib(type=Union[SparseTensor,pt.Tensor], default=None)
    edge_weight=attr.ib(default=None,type=Optional[pt.Tensor]) # None if SparseTensor, otherwise hold the values of the Adjmat
    num_nodes=attr.ib(default=None,type=Union[List[int],int,pt.Tensor])
    num_edges=attr.ib(default=None,type=Union[List[int],int,pt.Tensor])
    num_chunks=attr.ib(default=None,type=pt.Tensor)
    chunk_sizes=attr.ib(default=None,type=pt.Tensor)
    overlap_region_sizes=attr.ib(default=None,type=pt.Tensor)
    membership_matrix=attr.ib(default=None,type=pt.Tensor)
    def __attrs_post_init__(self):
        if self.x is not None:
            assert self.x.dim() in {2,3}, f"Node features should be [\sum_b n_b, f] or [B,n_max,f]"
        if pt.is_tensor(self.edges):
            assert self.edges.dim() in {2,3}, f"Edges should be given as adjmat [B,n_max,n_max]/[n,n], or adj list [2,\sum_b n_b]"
        elif isinstance(self.edges,SparseTensor):
            assert len(self.edges.sparse_sizes())==2, f"Sparse adjmat should be \sum_b n_b squared"
            assert self.edge_weight is None
        elif self.edges is None:
            pass
        else:
            raise ValueError(f"Bad type for edges {type(self.edges)}")
        if self.edge_weight is not None:
            assert pt.is_tensor(self.edge_weight)
            assert pt.is_tensor(self.edges)
            assert self.edge_weight.dim() in {1}, f"Edge values should be given by a 1D tensor of shape [\sum_b m_b]"
        if not pt.is_tensor(self.num_edges) and not self.num_edges is None:
            if type(self.num_edges)==np.ndarray:
                self.num_edges=pt.from_numpy(self.num_edges)
            else:
                self.num_edges=pt.tensor(self.num_edges)
        if not pt.is_tensor(self.num_nodes) and not self.num_nodes is None:
            if type(self.num_nodes) == np.ndarray:
                self.num_nodes = pt.from_numpy(self.num_nodes)
            else:
                self.num_nodes = pt.tensor(self.num_nodes)

    def __eq__(self, other):
        keys=attr.asdict(self).keys()
        selfshape=self.shapes()
        othershape=other.shapes()
        if self.typ==other.typ and selfshape==othershape:
            ma=lambda x:x.all() if hasattr(x,"all") else x
            return all((ma(getattr(self,k)==getattr(other,k)) for k in keys))
        else:
            return False
    def mask(self,N:pt.Tensor):
        """
        Mask the nodes and adjacency matrix to the given N (masking applicable for single shot and fixed-block_size CRP generation)
        """
        assert self.dense

        sN=self.edges.shape[-2]
        N=pt.minimum(N,pt.tensor(sN)).to(N.device) #  if we have less nodes than request to mask, we only mask to that size
        if (N==sN).all():
            pass # and if we were to retain everything, we do a no-op
        else:
            if self.x is not None:
                self.x=self.x*node_mask(self.x,N)
            self.edges=self.edges*adj_mask(self.edges,N)
        return self
    def mask_crp(self,N_chunk:pt.Tensor,L:int):
        """
        Mask the nodes and adjacency matrix to the given N_chunk (masking applicable for general CRP generation)
        """
        assert self.dense
        num_rounds = N_chunk.size(1)

        if self.x is not None:
            x_block = self.x[:, :L, :]
            self.x[:, :L, :] = x_block * node_mask(x_block, N_chunk[:, 0])

        A_diag_block = self.edges[:, :L, :L]
        self.edges[:, :L, :L] = A_diag_block * adj_mask(A_diag_block, N_chunk[:, 0])
        for r in range(1, num_rounds):
            block_range = slice(r*L, (r+1)*L)
            A_row_block = self.edges[:, :L, block_range]
            self.edges[:, :L, block_range] = A_row_block*adj_mask_2(A_row_block, N_chunk[:, 0], N_chunk[:, r])
            A_col_block = self.edges[:, block_range, :L]
            self.edges[:, block_range, :L] = A_col_block*adj_mask_2(A_col_block, N_chunk[:, r], N_chunk[:, 0])
        return self

    @property
    def node_batch(self):
        """
        Gives a tensor of shape self.x[:-1] (i.e. without feature dim) which indicates which graph in the batch
        a node belongs to,returns None if not batched, empty nodes are marked with -1
        """
        if not self.batch:
            raise ValueError("Trying to get node_batch from non-batch graph")
        else:
            nid = self.x.new_ones(self.x.shape[:-1]) * -1
            if self.dense:
                for b, n  in enumerate(self.num_nodes):
                    nid[b,:n]=b
                return nid
            elif self.torch_sparse or self.coo_sparse:
                cn=0
                for b,n in enumerate(self.num_nodes):
                    nid[cn:cn+n]=b
                    cn+=n
                return nid
            else:
                raise NotImplementedError(f"Unknown repr {self.typ}")


    @property
    def typ(self):
        if type(self.edges)==SparseTensor:
            return "torch_sparse"
        elif pt.is_tensor(self.edges) and self.edges.dtype in {pt.int,pt.long}:
            return "COO"
        elif pt.is_tensor(self.edges) and self.edges.dtype in {pt.float,pt.double}:
            return "dense"
        else:
            raise ValueError("Couldn't determine torch_sparse,COO or dense format")

    @property
    def dense(self):
        return self.typ=="dense"
    @property
    def torch_sparse(self):
        return self.typ=="torch_sparse"
    @property
    def coo_sparse(self):
        return self.typ=="COO"
    def to_ppgn_features(self)->pt.Tensor:
        """
        Converts the graph into a batch of B F+1 N N features suitable for PPGN which uses Conv2D for feature mods.
        To construct, first we create dense adjacency matrices B N N and unsqueeze to B 1 N N.
        Then we create a zero initialized feature vector B F N N and assign each node it's node features, i.e. we set
        xppgn[b,:,n,n]=X[b,n,:].
        """
        target=self.to_dense() if not self.dense else self
        adj=target.edges.unsqueeze(1) # B N N to B 1 N N
        B,_,N,_=adj.shape
        F=target.x.shape[-1]
        feats=adj.new_zeros(B,F,N,N)
        for b in range(B):
            for n in range(N):
                feats[b,:,n,n]=target.x[b,n,:]
        return pt.cat([adj,feats],1)




    @property
    def batch(self):
        """
        Checks whether the object represents a single large graph or a batch of graphs,indicated by checking
        whether num_nodes is a scalar or a rank 1, dim>1 tensor
        """
        n= self.num_nodes.flatten()
        return n.dim()==0 or len(n)>1

    def detach(self)->'Graph':
        detached=copy.copy(self)
        for k in attr.asdict(self):
            v=getattr(self,k,None)
            if pt.is_tensor(v):
                v=v.detach()
            setattr(detached,k,v)
        return detached
    def clone(self)->'Graph':
        cloned=copy.copy(self)
        for k in attr.asdict(self):
            v=getattr(self,k,None)
            if pt.is_tensor(v):
                v=v.clone()
            setattr(cloned,k,v)
        return cloned
    def to(self,device)->'Graph':
        cloned=copy.copy(self)
        for k in attr.asdict(self):
            v=getattr(self,k,None)
            if pt.is_tensor(v):
                v=v.to(device)
            setattr(cloned,k,v)
        return cloned

    def fuse(self):
        self.num_nodes=sum(self.num_nodes)
        self.num_edges = sum(self.num_edges)
        return self
    @property
    def sum_nodes(self):
        return self.num_nodes if isinstance(self.num_nodes,int) or self.num_nodes.dim()==0 else sum(self.num_nodes)
    @property
    def sum_edges(self):
        return self.num_edges if isinstance(self.num_edges,int) or self.num_edges.dim()==0  else sum(self.num_edges)

    def avg_degree(self):
        if type(self.edges)==SparseTensor:
            ei:SparseTensor=self.edges
            idx=ei.to_torch_sparse_coo_tensor().coalesce()[0]
        else:
            idx=self.edges[0]
        return pt.mean(pg.utils.degree(idx,self.sum_nodes))

    @classmethod
    def multi_rewire(cls, graphs:List['Graph'], intra_graph_density=0.1, diffusion_strength=0.0, diff_avg_degree=64)-> 'Graph':
        """
        Uses SBM edges to connect the given graphs in a single super-graph and then use PPR edge diffusion to rewire the resulting graph.
        When setting :intra_graph_density=0.0 and diffusion_strength=0.0 (default) simply calls combine and returns the combinated graph.
        """
        sd=cls.combine(graphs)
        ends=sd.num_nodes
        nB=len(sd.num_nodes)
        starts=[0]+ends[:-1]
        # we connect every graph to every other graph with this probability
        is_sparse= type(sd.edges) == SparseTensor
        ei=sd.edges
        ew = sd.edge_weight
        n = sum(sd.num_nodes)
        if intra_graph_density>0.0:
            intra_block_density = [[0.0 if i == j else intra_graph_density for i in range(nB)] for j in range(nB)]
            intra_graph_edges = pg.utils.stochastic_blockmodel_graph(sd.num_nodes, edge_probs=intra_block_density)
            if is_sparse:
                ei=ei+SparseTensor(row=intra_graph_edges[0],col=intra_graph_edges[1],sparse_sizes=(n,n))
            else:
                ei=pt.cat([ei,intra_graph_edges],-1)
                ew = pt.cat([ei,sd.edge_weight.new_ones(intra_graph_edges.shape[-1])],-1)
        d=pg.data.Data(x=sd.x,edge_index=ei,edge_attr=ew)
        d.num_nodes=n
        if diffusion_strength>0.0:
            # NOTE: no gradient flow here
            diffusion=pg.transforms.GDC(self_loop_weight=1,
                                        normalization_in="sym",normalization_out="col",
                                        diffusion_kwargs=dict(method="ppr",alpha=diffusion_strength),
                                        sparsification_kwargs=dict(method="threshold",avg_degree=diff_avg_degree))
            d=diffusion(d)
        return Graph.from_data(d) if (diffusion_strength > 0.0 or intra_graph_density > 0.0) else sd

    @classmethod
    def from_dense_tensor(cls, x:pt.Tensor, A:pt.Tensor,N=None,K=None,N_chunk=None,OR=None,M=None,to_sparse=False, torch_sparse=False,strict=True)->'Graph':
        if N is None:
            if x is not None:
                N = (x!=0).any(-1).sum(-1).long() # zero padded nodes, assume 0=no node
                if strict:
                    warning(f"Did not receive number of nodes, inferring N from non-all-zero node features (shape:{N.shape},min:{N.min()} med:{N.median()} max:{N.max()})")
            else:
                warning(
                    f"Did not receive number of nodes or node features, cannot set n")
        elif not pt.is_tensor(N):
            N=pt.tensor(N)
        num_nodes=N
        num_chunks=K
        chunk_sizes=N_chunk
        overlap_region_sizes=OR
        membership_matrix=M
        num_edges=pt.sum(A,(-1,-2)).long() # dito zero padded edges
        if x is not None:
            if x.dim()==3:
                assert num_edges.dim()==1
                assert num_nodes.dim() == 1
            elif x.dim()==2:
                assert num_edges.dim() == 0
                assert num_nodes.dim() == 0
            else:
                raise ValueError("Malformed inputs")
        edges=A
        edge_weight = None
        ret= Graph(x,edges,edge_weight=edge_weight,num_nodes=num_nodes,num_edges=num_edges,num_chunks=num_chunks,chunk_sizes=chunk_sizes,overlap_region_sizes=overlap_region_sizes,membership_matrix=membership_matrix)
        if to_sparse:
            if torch_sparse:
                ret=ret.to_torch_sparse()
            else:
                ret=ret.to_coo_sparse()
        return ret

    @classmethod
    def combine(cls, graphs:List['Graph'])-> 'Graph':
        #TODO: handle various representations and batch requirements here..
        raise  NotImplementedError("Need to revisit combine and fuse")
        cx=pt.cat([g.x[...,:g.num_nodes,:] for g in graphs],-2)
        cn=[g.num_nodes for g in graphs]
        csn=sum(cn)
        cm=[g.num_edges for g in graphs]
        cum_sum=pt.cumsum(pt.tensor([0]+cn[:-1]),-1)
        is_sparse= type(graphs[0].edges) == SparseTensor
        if is_sparse:
            As=[g.edges.to_torch_sparse_coo_tensor().coalesce() for g in graphs]
            assert len(cum_sum)==len(As)
            # create the offset stitching
            eis=pt.cat([t.indices()+cs for t,cs  in zip(As,cum_sum)],-1)
            vs=pt.cat([t.values() for t in As],-1)
        else:
            eis=pt.cat([g.edges + cs for g, cs in zip(graphs, cum_sum)], -1)
            vs=pt.cat([g.edge_weight for g in graphs],-1)
        ret=Graph()
        ret.x=cx
        ret.edges=SparseTensor(row=eis[0], col=eis[1], value=vs, sparse_sizes=(csn, csn)) if is_sparse else eis
        ret.edge_weight=None if is_sparse else vs
        ret.num_nodes=cn
        ret.num_edges=cm
        return ret

    def batch_list(self)->List['Graph']:
        if self.batch:
            if self.dense:
                return self.to_dense(True)
            elif self.coo_sparse:
                return self.to_coo_sparse(True)
            elif self.torch_sparse:
                return self.to_torch_sparse(True)
            else:
                raise NotImplementedError(f"Unkown repr {self.typ}")
        else:
            raise RuntimeError("Tried to call batch_list on non-batch graph repr")
    def to_data(self)->pg.data.Data:
        if self.batch:
            return pg.data.Batch.from_data_list([g.to_data() for g in self.batch_list()])
        else:
            target=self if self.coo_sparse else self.to_coo_sparse()
            ei=target.edges
            ew=target.edge_weight
            d = pg.data.Data(x=target.x, edge_index=ei,edge_attr=ew,num_nodes=target.num_nodes)
            return d

    @classmethod
    def from_data(self,d:Union[pg.data.Data,pg.data.Batch])-> Union[List['Graph'],'Graph']:
        if type(d)==pg.data.Batch:
            num_nodes=pt.from_numpy(np.diff(d.__slices__["x"]))
            num_edges=pt.from_numpy(np.diff(d.__slices__["edge_index"]))
            g=Graph(d.x,d.edge_index,d.edge_attr,num_nodes,num_edges)
        else:
            ne=d.num_edges
            g=Graph(d.x,d.edge_index,d.edge_attr,num_nodes=d.num_nodes,num_edges=ne)
        return g

    def split(self)->List['Graph']:
        """De-combine into component graphs
        """
        rets=[]
        is_sparse= type(self.edges) == SparseTensor
        csn=pt.cumsum(pt.tensor([0]+self.num_nodes[:-1])-1)
        csm=pt.cumsum(pt.tensor([0]+self.num_edges[:-1]),-1)
        assert len(csn)==len(self.num_nodes)==len(self.num_edges)
        for n,m,cn,cm in zip(self.num_nodes,self.num_edges,csn,csm):
            start_n=cn
            stop_n=cn+n
            start_m=cm
            stop_m=cm+m
            x = self.x[start_n:stop_n, ...]  # slice nodes
            if is_sparse:
                ei= self.edges[start_n:stop_n, start_n:stop_n] # slice sparse as if it was dense, yay rusty
                ret=Graph()
                ret.x=x
                ret.edges=ei
                ret.num_edges=m
                ret.num_nodes=n
                rets.append(ret)
            else:
                ei= self.edges[:, start_m:stop_m] # slice edge index list
                vs=self.edge_weight[start_m:stop_m]
                ret=Graph()
                ret.x=x
                ret.edges=ei
                ret.num_edges=m
                ret.num_nodes=n
                ret.edge_weight=vs
                rets.append(ret)
        return rets

    def unpac(self, pac, in_place=False):
        if self.dense:
            target=self if in_place else self.clone()
            maybe_reshape=lambda x,m:x if x is None else pac_reshape(x,to_packed=False,pac=pac,mode=m)
            target.x,target.edges,target.num_nodes,target.num_edges=[maybe_reshape(x,m)
                                                                     for x,m in zip(
                    [target.x,target.edges,target.num_nodes,target.num_edges],
                    ["nodes","adj","N","N"]
                )]
            return target
        else:
            raise NotImplementedError("pac/unpac not yet implemented for anything but dense")
    def pac(self, pac, in_place=False):
        if self.dense:
            target=self if in_place else self.clone()
            maybe_reshape=lambda x,m:x if x is None else pac_reshape(x,to_packed=True,pac=pac,mode=m)
            target.x,target.edges,target.num_nodes,target.num_edges=[maybe_reshape(x,m)
                                                                     for x,m in zip(
                    [target.x,target.edges,target.num_nodes,target.num_edges],
                    ["nodes","adj","N","N"]
                )]
            return target
        else:
            raise NotImplementedError("pac/unpac not yet implemented for anything but dense")
    @classmethod
    def convex_combination(cls,real:'Graph',fake:'Graph')->'Graph':
        # expects pairs to be (fakeA realA) (fakeB realB) etc
        if not real.dense and fake.dense:
            raise NotImplementedError("Convex combination only supported for dense graphs")
        interpolated = []
        for (fake, real) in [(real.x,fake.x),real.edges,fake.edges]:
            if fake is not None and real is not None:
                alpha = pt.rand_like(real).type_as(real)
                interpolate = (real * alpha + ((1.0 - alpha) * fake)).requires_grad_(True)
                interpolated.append(interpolate)
            else:
                interpolated.append(None)
        return Graph.from_dense_tensor(*interpolated)
    @classmethod
    def recombine_rows(cls,real:'Graph',fake:'Graph',p)->'Graph':
        """

        Recombine two graphs in a *discrete* way, by recombining nodes and their connectivities
        p: probability of taking a row from a1
        """
        if not real.dense and fake.dense:
            raise NotImplementedError("recombine rows only supported for dense graphs")
        a1, a2, x1 , x2 = real.edges,fake.edges,real.x,fake.x
        if a1.dim() == 2:
            from_1 = pt.rand([a1.shape[-2]], device=a1.device, dtype=a1.dtype) >= p
            idx_r = from_1.nonzero(as_tuple=True)
            (r,) = idx_r
            idx = (r, r)
        elif a1.dim() == 3:
            from_1 = (
                    pt.rand([a1.shape[0], a1.shape[-2]], device=a1.device, dtype=a1.dtype) >= p
            )
            idx_r = from_1.nonzero(as_tuple=True)
            (b, r,) = idx_r
            idx = (b, r, r)
        elif a1.dim() == 4:
            B, pac, N, _ = a1.shape
            if p.dim() == 2:
                p = p.unsqueeze(1).repeat([1, pac, 1])
            from_1 = pt.rand([B, pac, N], device=a1.device, dtype=a1.dtype) >= p
            idx_r = from_1.nonzero(as_tuple=True)
            (b, p, r,) = idx_r
            idx = (b, p, r, r)
        else:
            raise NotImplementedError(
                f"Don't know how to deal with A shape {a1.shape} {a2.shape}"
            )
        # in place shit still gives errors with gradients I think
        # new_a=a2.clone()
        # new_a[idx]=a1[idx]
        a1_sel = pt.zeros_like(a2)
        a1_sel[idx] = 1.0
        a2_sel = pt.ones_like(a1_sel) - a1_sel
        new_a = a1 * a1_sel + a2_sel * a2
        if x1 is not None:
            # in place shit still gives errors with gradients I think
            # new_x=x2.clone()
            # new_x[idx]=x1[idx]
            x1_sel = pt.zeros_like(x2)
            x1_sel[idx_r] = 1.0
            x2_sel = pt.ones_like(x1_sel) - x1_sel
            new_x = x1 * x1_sel + x2_sel * x2
            return Graph(new_x,new_a,None,real.num_nodes)
        else:
            return Graph(None,new_a,None,real.num_nodes)

    def requires_grad_(self,grad_enabled=True):
        for k in attr.asdict(self).keys():
            v=getattr(self,k,None)
            if pt.is_tensor(v) and v.dtype in {pt.float,pt.double}:
                v.requires_grad_(grad_enabled)

    def clear_grad(self):
        for k in attr.asdict(self).keys():
            v=getattr(self,k,None)
            if pt.is_tensor(v) and v.dtype in {pt.float,pt.double}:
                v.grad=None

    def effective_inputs(self):
        ei=[self.x] if self.x is not None else  []
        if self.dense or self.torch_sparse:
            return ei+[self.edges]
        elif self.coo_sparse:
            return ei+[self.edge_weight]
        else:
            raise NotImplementedError("Unknown graph representation")

    @classmethod
    def to_batch(cls, targets:List['Graph'])->'Graph':
        """
        Takes in a list of individual graphs and performs batching on dense/coo_sparse/torch_sparse according to the representation
        """
        dense=all(g.dense for g in targets)
        torch_sparse=all(g.torch_sparse for g in targets)
        coo=all(g.coo_sparse for g in targets)
        if dense:
            num_nodes=[]
            num_edges=[]
            edges=[]
            x=[]
            for g in targets:
                num_nodes.append(g.num_nodes)
                edges.append(g.edges)
                x.append(g.x)
                num_edges.append(g.num_edges)
            num_nodes=pt.cat(num_nodes,-1)
            num_edges=pt.cat(num_edges,-1)
            x=pt.stack(pad_to_max(x,1,(1,)),0)
            edges=pt.stack(pad_to_max(x,-1,(-1,-2)),0)
            return Graph(x,edges,None,num_nodes,num_edges)
        elif torch_sparse:
            # shunt via coo for simplicities sake....
            targets=[g.to_coo_sparse() for g in targets]
            return Graph.to_batch(targets).to_torch_sparse()
        elif coo:
            num_nodes=[]
            num_edges=[]
            edges=[]
            edge_weights=[]
            x=[]
            cum_sum=0
            for g in targets:
                num_nodes.append(g.num_nodes)
                # offset edges to index into node list
                edges.append(g.edges+cum_sum)
                edge_weights.append(g.edge_weight)
                x.append(g.x)
                num_edges.append(g.num_edges)
                cum_sum+=g.num_nodes
            x=pt.cat(x,0)
            edges=pt.cat(edges,1)
            edge_weights=pt.cat(edge_weights,0)
            num_nodes=pt.stack(num_nodes,0)
            num_edges=pt.stack(num_edges,0)
            return Graph(x,edges,edge_weights,num_nodes,num_edges)
        else:
            raise NotImplementedError(f"Unkown repr or heterogeneous type of repr for batching")

    def to_torch_sparse(self, batch_list=False)->Union[List['Graph'], 'Graph']:
        """
        Converts the graph representation from dense or coo_sparse to torch sparse, returning either a list of graphs or a
        large disjointed graph in case this was a batch of dense graphs previously
        """
        edge_weight = None
        if self.dense:
            if self.batch:
                batch=[]
                for b,n in enumerate(self.num_nodes):
                    a=self.edges[b,:n,:n]
                    x=self.x[b,:n,:]
                    batch.append(Graph(x,SparseTensor.from_dense(a),None,n,self.num_edges[b]))
                if batch_list:
                    return batch
                else:
                    batch= Graph.to_batch([g.to_coo_sparse() for g in batch]).to_torch_sparse()

                    return batch
            else:
                return Graph(self.x,SparseTensor.from_dense(self.edges),edge_weight,self.num_nodes,self.num_edges)

        elif self.coo_sparse:
            combo_n = self.sum_nodes
            sparse = SparseTensor(row=self.edges[0], col=self.edges[1], value=self.edge_weight, sparse_sizes=(combo_n,combo_n))
            if self.batch:
                if batch_list:
                    batch=[]
                    cn=0
                    for b,n,in enumerate(self.num_nodes):
                        x=self.x[cn:cn+n,:]
                        s=sparse[cn:cn+n,cn:cn+n].coalesce()
                        batch.append(Graph(x,s,edge_weight, n, self.num_edges[b]))
                        cn += n
                    return batch
                else:
                    return Graph(self.x,sparse,edge_weight,self.num_nodes,self.num_edges)
            else:
                return Graph(self.x, sparse, edge_weight, self.num_nodes, self.num_edges)
        elif self.torch_sparse:
            if self.batch:
                if batch_list:
                    batch=[]
                    sparse=self.edges
                    cn=0
                    for b,n,in enumerate(self.num_nodes):
                        x=self.x[cn:cn+n,:]
                        s=sparse[cn:cn+n,cn:cn+n].coalesce()
                        cn+=n
                        batch.append(Graph(x,s,edge_weight, n, self.num_edges[b]))
                    return batch
            else:
                return self
        else:
            raise NotImplementedError(f"Unkown conversion {self.typ}->torch_sparse")

    def to_coo_sparse(self, batch_list=False)->Union[List['Graph'], 'Graph']:
        """
        Converts the graph representation from dense or torch_sparse to coo sparse, returning either a list of graphs or a
        large disjointed graph in case this was a batch of dense graphs previously
        """
        if self.dense:
            if self.batch:
                batch = []
                for b, n in enumerate(self.num_nodes):
                    a = self.edges[b, :n, :n]
                    x = self.x[b, :n, :]
                    ei,ew=pg.utils.sparse.dense_to_sparse(a)
                    batch.append(Graph(x, ei, ew, n, self.num_edges[b]))
                if batch_list:
                    return batch
                else:
                    return Graph.to_batch(batch)
            else:
                ei, ew = pg.utils.sparse.dense_to_sparse(self.edges)
                return Graph(self.x, ei,ew, self.num_nodes, self.num_edges)
        elif self.coo_sparse:
            if self.batch and batch_list:
                #shunt to torch_sparse for simplicity of API
                return [g.to_coo_sparse() for g in self.to_torch_sparse(batch_list)]
            else:
                return self
        elif self.torch_sparse:
            if self.batch and batch_list:
                batch=[]
                for b,n in enumerate(self.num_nodes):
                    ss=self.edges[b,:n,:n].to_torch_sparse_coo_tensor().coalesce()
                    x=self.x[b,:n,:]
                    ei=ss.indices()
                    ew=ss.values()
                    batch.append(Graph(x, ei,ew, n, self.num_edges[b]))
                return batch
            else:
                ss=self.edges.to_torch_sparse_coo_tensor().coalesce()
                ei=ss.indices()
                ew=ss.values()
                return Graph(self.x, ei,ew, self.num_nodes, self.num_edges)
        else:
            raise NotImplementedError(f"Unkown conversion {self.typ}->coo")

    def to_dense(self, batch_list=False)->Union[List['Graph'], 'Graph']:
        """
        Converts the graph representation from coo_sparse or torch_sparse to dense matrices, returning either a list of graphs or a
        padded batch single big tensor in case this was a batch of sparse graphs previously
        """
        if self.dense:
            target= self
            if target.batch and batch_list:
                target=[Graph(self.x[b,:n,:],self.edges[b,:n,:n],None,self.num_nodes[b],self.num_edges[b]) for b,n in enumerate(self.num_nodes)]
            return target
        elif self.coo_sparse:
            # shunt via sparse_torse for simplicity of API
            if batch_list:
                return [g.to_dense() for g in self.to_torch_sparse(batch_list)]
            else:
                return self.to_torch_sparse().to_dense()
        elif self.torch_sparse:
            n_max=max(self.num_nodes) if not (isinstance(self.num_nodes,int) or self.num_nodes.dim()==0) else self.num_nodes
            if self.batch:
                cno = pt.cumsum(pt.tensor([0]+self.num_nodes.tolist()[:-1]), -1)
                if batch_list:
                    return [Graph(pad_to(self.x[cn:cn+n, :],n_max,pad_dims=(1,)), pad_to(self.edges[b, :n, :n].coalesce().to_dense(),n_max,(-1,-2)),
                                  None, self.num_nodes[b], self.num_edges[b]) for
                              b, (n,cn) in enumerate(zip(self.num_nodes,cno))]
                else:
                    xs=pt.stack([pad_to(self.x[cn:cn+n,:],n_max,(0,)) for cn,n in zip(cno,self.num_nodes)],0)
                    edges=[pad_to(self.edges[cn:cn+n, cn:cn+n].coalesce().to_dense(),n_max,(0,1)) for cn,n in zip(cno,self.num_nodes)]
                    edges=pt.stack(edges,0)
                    return Graph(xs,edges,None,self.num_nodes,self.num_edges)
            else:
                return Graph(self.x, pad_to(self.edges.coalesce().to_dense(),n_max,(-1,-2)), None, self.num_nodes, self.num_edges)
        else:
            raise NotImplementedError(f"Unkown conversion {self.typ}->dense")

    def perturb_edges(self, perturbation_percentage):
        if self.dense:
            g=self.clone()
            A_triu = pt.triu(g.edges)
            # roll an erasure dice on every node
            erase_chance = pt.rand_like(A_triu) * A_triu.round()
            erased :pt.Tensor= pt.less_equal(erase_chance,perturbation_percentage)
            # mask out erased nodes
            erased_neg = pt.logical_not(erased)
            A_triu = A_triu * erased_neg
            # roll a dice for adding an edge
            # roll a dice on all empty edges which aren't empty because we erased them
            add_chance = pt.rand_like(erase_chance)
            added = (perturbation_percentage > add_chance) * (1 - A_triu).abs() * erased_neg
            # add the new connections
            A_triu = A_triu + added
            # resymmetriue
            perm=A_triu.permute(0, 1, -1, -2) if A_triu.dim()==4 else A_triu.permute(0,-1,-2) # pac and non-pac handle
            g.edges = (A_triu + perm).clamp(0,1)
            return g
        else:
            raise NotImplementedError("Permute edges only implemented for dense so far")

    def shapes(self):
        s=[]
        if self.x is not None:
            s.append(self.x.shape)
        else:
            s.append(None)
        if self.dense:
            s.append(self.edges.shape)
        elif self.torch_sparse:
            s.append(self.edges.sparse_sizes())
        else:
            s.extend(self.edges.shape,self.edge_weight.shape)
        return s,self.typ

    def num_graphs(self):
        if not self.batch:
            return 1
        else:
            return len(self.num_nodes)

    @classmethod
    def cat(cls, graphs:List['Graph'],dim=0,till=None)->'Graph':
        assert all(g.dense for g in graphs)
        take=lambda x: x[:till] if till is not None else x
        x=pt.cat([take(g.x) for g in graphs],dim)
        edges=pt.cat([take(g.edges) for g in graphs],dim)
        num_nodes=pt.cat([take(g.num_nodes) for g in graphs],dim)
        num_chunks=pt.cat([take(g.num_chunks) for g in graphs],dim)
        chunk_sizes=pt.cat([take(g.chunk_sizes) for g in graphs],dim)
        overlap_region_sizes=pt.cat([take(g.overlap_region_sizes) for g in graphs],dim)
        membership_matrix=pt.cat([take(g.membership_matrix) for g in graphs],dim)
        return Graph.from_dense_tensor(x,edges,num_nodes,num_chunks,chunk_sizes,overlap_region_sizes,membership_matrix)

    @classmethod
    def half_batch(cls, gr: 'Graph', gf: 'Graph'):
        assert gr.dense and gf.dense, "No other implemented"
        if gr.shapes()==gf.shapes():
            B=gr.edges.shape[0]
            assert B>=2
            half=int(B//2)
            if gr.x is not None:
                x1=pt.cat([gr.x[:half],gf.x[:half]],0)
                x2=pt.cat([gr.x[half:],gf.x[half:]],0)
            else:
                x1,x2=None,None
            a1=pt.cat([gr.edges[:half],gf.edges[:half]],0)
            a2=pt.cat([gr.edges[half:],gf.edges[half:]],0)
            if gr.num_nodes is not None and gf.num_nodes is not None:
                n1=pt.cat([gr.num_nodes[:half],gf.num_nodes[:half]],0)
                n2=pt.cat([gr.num_nodes[half:],gf.num_nodes[half:]],0)
            else:
                n1,n2=None,None
            return Graph(x=x1,edges=a1,num_nodes=n1),Graph(x=x2,edges=a2,num_nodes=n2)
        else:
            raise NotImplementedError("TODO: need to handle padding for merging...")


def perturb_adj(A, percentage):
    A_triu = pt.triu(A)
    # roll an erasure dice on every node
    erase_chance = pt.rand_like(A_triu) * A_triu
    erased: pt.Tensor = pt.less_equal(erase_chance, percentage)
    # mask out erased nodes
    erased_neg = pt.logical_not(erased)
    A_triu = A_triu * erased_neg
    # roll a dice for adding an edge
    # roll a dice on all empty edges which aren't empty because we erased them
    add_chance = pt.rand_like(erase_chance)
    added = (percentage > add_chance) * (1 - A_triu).abs() * erased_neg
    # add the new connections
    A_triu = A_triu + added
    # resymmetriue
    A = A_triu + A_triu.permute(0, 1, -1, -2)
    return A


class PINNWrapper(pt.nn.Module):
    def __init__(self, presum, postsum, sum_dim=-2, act=None):
        super(PINNWrapper, self).__init__()
        self.act=act
        self.presum = presum
        self.postsum = postsum
        self.sum_dim = sum_dim

    def forward(self, X, keepdim=False):
        X = self.presum(X)
        if self.act is not None:
            X = self.act(X)
        X = X.sum(self.sum_dim, keepdim=keepdim)
        X = self.postsum(X)
        return X


from attr.validators import instance_of,optional
@attr.s
class GenData:
    N=attr.ib(type=pt.LongTensor,validator=optional(torch_int)) # number of
    root=attr.ib(default=None,type=Optional[pt.FloatTensor],validator=optional(torch_float))
    Z=attr.ib(default=None, type=Optional[pt.FloatTensor], validator=optional(torch_float)) # view into root
    phi=attr.ib(default=None,type=Optional[pt.FloatTensor],validator=optional(torch_float)) # view into root
    embeddings = attr.ib(default=None, type=Optional[pt.FloatTensor], validator=optional(torch_float))  # latent embedding returned by generator
    graph=attr.ib(default=None,type=Optional[Graph])

    def maybe_mask(self)->'GenData':
        """
        This function is intended to mask the adjacency matrices and root to the available nodes if given, but I need to revisit it to ensure the graph can do this
        """
        if self.root is not None and self.root.dim()<=2:
            if not GenData.noneq_warned:
                warn(f"Skipping masking because root has dim<=2{self.root.dim()}, assuming noneq. Single warning only")
            return self
        if self.N is not None:
            if self.root is not None:
                N = self.N.to(self.root.device)
                self.root = self.root * node_mask(self.root, N).detach()
            if self.embeddings is not None:
                N = self.N.to(self.embeddings.device)
                self.embeddings= self.embeddings * node_mask(self.embeddings, N).detach()
            if self.graph is not None:
                self.graph.mask(self.N)
        return self
GenData.noneq_warned=False

@attr.s
class CRPGenData(GenData):
    K=attr.ib(type=pt.LongTensor, validator=optional(torch_int), default=None)
    N_chunk=attr.ib(type=pt.LongTensor, validator=optional(torch_int), default=None)
    OR=attr.ib(type=pt.LongTensor, validator=optional(torch_int), default=None)
    M=attr.ib(type=pt.LongTensor, validator=optional(torch_int), default=None)
    prior_embeddings=attr.ib(default=None)
    current_N=attr.ib(default=0)
    current_K=attr.ib(default=0)
    current_N_chunk=attr.ib(default=0)
    current_OR=attr.ib(default=0)
    current_M=attr.ib(default=0)
    chunks=attr.ib(factory=list)
CRPGenData.noneq_warned=False


def sample_partition_batch(max_bin_num: int, totals: pt.Tensor, bin_nums: pt.Tensor, size_weights: pt.Tensor):
    batch_size = len(totals)
    if size_weights.dim() == 2:
        size_weights = size_weights.unsqueeze(1).repeat(1, max_bin_num, 1)

    partitions = pt.zeros(batch_size, max_bin_num).long()
    converged = pt.zeros(batch_size)
    while not pt.all(converged):
        converged = pt.sum(partitions, dim=1) == totals
        for i in range(batch_size):
            if not converged[i]:
                partitions[i, :bin_nums[i]] = pt.distributions.Categorical(size_weights[i, :bin_nums[i]]).sample() + 1

    return partitions


def sample_bipartite_graph(a: List[int], b: List[int], method: str):
    if method == "havel_hakimi":
        graph = nx.bipartite.havel_hakimi_graph(a, b, nx.Graph())
    else:
        graph = nx.bipartite.configuration_model(a, b, nx.Graph())
    return graph
try:
    import memcnn
except:
    warn("Couldn't import memcnn, can't use invertible loop layer")
class InvLoopLayer(pt.nn.Module):
    """
    Utility class which allows weight sharing stacks of layers to increase expressivity
    """
    def __init__(self,l1,l2,num_loops,split_dim=-1):
        super(InvLoopLayer, self).__init__()
        coupling=memcnn.AdditiveCoupling(l1,l2,split_dim=split_dim)
        self.num_loops=num_loops
        self.layer=memcnn.InvertibleModuleWrapper(coupling)
    def forward(self,x):
        """
        Warning: x will get erased by the layer, so add it with x.clone() if you plan to use it later
        :param x:
        :return:
        """
        out=self.layer(x)
        for _ in range(self.num_loops):
            out=self.layer(out)
        return out
class InvLoopLayerDistinct(pt.nn.Module):
    """
    Utility class which allows stacks of layers to increase expressivity, with constant memory gradients.
    Much more memory heavy than the weight sharing version, but possibly more expressive?.
    """
    def __init__(self,l1s,l2s,split_dim=-1):
        super(InvLoopLayerDistinct, self).__init__()
        couplings=[memcnn.AdditiveCoupling(l1,l2,split_dim=split_dim) for l1,l2 in zip(l1s,l2s)]
        self.layers=[memcnn.InvertibleModuleWrapper(coupling) for coupling in couplings]
    def forward(self,x):
        """
        Warning: x will get erased by the layer, so add it with x.clone() if you plan to use it later
        :param x:
        :return:
        """
        out=x
        for l in self.layers:
            out=l(out)
        return out


def ind1(i,n):
    """
    The first with everyone, the 2nd with everyone but the 1st..
    :param i:
    :param n:
    :return:
    """
    return pt.arange(n-i)+i
def ind0(i,n):
    """
    The first with everyone so n times, the 2nd with everyone but the 1st..
    :param i:
    :param n:
    :return:
    """
    return i*pt.ones(n-i).int()
def pair_feats(x):
        n=x.shape[-2]
        inds0=pt.cat([ind0(i,n) for i in range(n)])
        inds1=pt.cat([ind1(i,n) for i in range(n)])
        pairs=pt.cat([x[:,inds0,:],x[:,inds1,:]],-1) # B ? 2*F
        return pairs

