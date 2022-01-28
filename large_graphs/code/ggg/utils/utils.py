import copy
from inspect import signature
from logging import info, warning
from typing import Dict, List

import torch
import torch as pt
from pdb import set_trace
import numpy as np

from ggg.utils.adam import Adam
from ggg.models.components.spectral_norm import sn_wrap

def ensure_tensor(x):
    if pt.is_tensor(x):
        return x
    elif isinstance(x, np.ndarray):
        return pt.from_numpy(x)
    else:
        return pt.tensor(x)


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
        pad_len = [
            (0, 0) if i not in pad_dims else (0, max_N - x.shape[i])
            for i in range(len(x.shape))
        ]
        if torch.is_tensor(x):
            pad_len = [p for y in reversed(pad_len) for p in y]
            x_padded = torch.nn.functional.pad(x, pad_len)
        else:
            x_padded = np.pad(x, pad_len, mode="constant")
        padded_list.append(x_padded)
    return padded_list


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
def add_to_non_edges(A, x, N=None):
    """
    :param A:
    fill: x
    :return:
    """
    d = torch.diagonal(A, dim1=-2, dim2=-1)
    Az = A + torch.diag_embed(pt.ones_like(d)*x)
    if N is not None:
        with pt.no_grad():
            non_edge_mask=pt.logical_not(adj_mask(A,N))
            non_edge_mask=non_edge_mask*x
            # nan filter
            non_edge_mask[pt.isnan(non_edge_mask)]=0.0
        Az=Az+non_edge_mask
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


def zero_and_symmetrize(A):
    A = zero_diag(A)
    A = torch.triu(A)
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


def kwarg_create(cls, kwargs: Dict):
    # creates the class given the kwargs, filtering out any which aren't in class definition
    kwargs = {k: kwargs[k] for k in signature(cls).parameters.keys() if k in kwargs}
    missing = [k for k in signature(cls).parameters.keys() if k not in kwargs]
    if len(missing) > 0:
        warning(f"Didn't find {missing} when creating {cls.__name__}")
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

def enable_asserts(state):
    enable_asserts.state=state

enable_asserts.state=None
def asserts_enabled():
    return enable_asserts.state==True
def maybe_assert(x=None,func=None):
    if asserts_enabled():
        if x is not None:
            assert x
        if func is not None:
            assert func()


def pdf(weights):
    sum = weights.sum()
    assert sum != 0
    return weights / sum
