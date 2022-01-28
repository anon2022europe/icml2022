from typing import Union

import torch as pt
from torch import nn
import torch_geometric as tg


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


class SkipBlock(pt.nn.Module):
    def __init__(self, inner, proj=None) -> None:
        super().__init__()
        self.inner = inner
        if proj is not None:
            self.proj = nn.Linear(*proj)
        else:
            self.proj = None

    def forward(self, x):
        xi = self.inner(x)
        if self.proj:
            xadd = self.proj(x)
        else:
            xadd = x
        return xadd + xi


class PermuteBatchnorm1d(pt.nn.BatchNorm1d):
    """
    Applies Batchnorm to the feature dimensions.
    Tensors arrive in B N F shape and Batchnorm1D applies to 1 dimension=> permute and unpermute
    pytorch_geometric actually has an implementation of this...should double check
    """

    def forward(self, input: pt.Tensor) -> pt.Tensor:
        return super().forward(input.permute(0, 2, 1)).permute(0, 2, 1)


class PairNorm(pt.nn.Module):
    # following https://pytowrch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/norm/pair_norm.html#PairNorm, but for dense Batches
    def __init__(
        self, scale_individually=False, eps=1e-5, s=1.0, node_dim=1, center="pre"
    ):
        super().__init__()
        self.s = s
        self.scale_individually = scale_individually
        self.eps = eps
        self.s = s
        self.node_dim = node_dim
        self.center = center

    def forward(self, X):
        col_mean = X.mean(dim=self.node_dim, keepdim=True)
        if self.center == "pre":
            X = X - col_mean
        if not self.scale_individually:
            div = (
                self.eps
                + X.pow(2).sum(-1, keepdim=True).mean(self.node_dim, keepdim=True)
            ).sqrt()
        else:
            div = self.eps + X.norm(2, -1, keepdim=True)
        X = self.s * X / div
        if (
            self.center == "post"
        ):  # scs style frmo https://github.com/LingxiaoShawn/PairNorm/blob/master/layers.py
            X = X - col_mean
        return X


class NodeNumberNorm(pt.nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C

    def forward(self, X):
        return X / self.C


class GraphSizeNormDeg(pt.nn.Module):
    def forward(self, X, A):
        deg = A.sum(-1, keepdims=True).clamp_min(
            1.0
        )  # calculate node degrees, lower bound by 1 since we don't want to explode anything
        return X / deg


class GraphSizeNormMaxDeg(pt.nn.Module):
    def forward(self, X, A):
        deg = A.sum(-1, keepdims=True).clamp_min(
            1.0
        )  # calculate node degrees, lower bound by 1 since we don't want to explode anything
        return X / deg.max(-1, keepdims=True).values


class GraphSizeNormV(pt.nn.Module):
    def forward(self, X, A):
        # sum along rows and check nonzero, then sum up along rows to  get number of nodes, clamp the minimum to 1 to not make things explode
        num_nodes = (
            (A.sum(-1, keepdims=True) != 0.0).sum(-1, keepdims=True).clamp_min(1.0)
        )
        return X / num_nodes


class NodeFeatNorm(pt.nn.Module):
    SUPPORTED = {
        "instance",
        "layer",
        "instance-affine",
        "layer-affine",
        "identity",
        "pair",
        "pair-si",
        "pair-scs",
        "graph-size-c",
        "graph-size-v",
        "graph-size-deg",
        "graph-size-maxdeg",
    }

    def __init__(self, feat_dim, mode="instance", max_nodes=None, affine=False):
        super().__init__()
        self.mode = mode
        if mode == "graph-size-c" and max_nodes is None:
            raise ValueError(f"When using {mode} need to provide max node number")
        self.feat_dim = feat_dim
        if mode == "instance" or mode=="instance-affine":
            self.norm = pt.nn.GroupNorm(
                num_groups=feat_dim, num_channels=feat_dim, affine=affine or "affine" in mode
            )  # GroupNorm with Group=Channel== Instance Norm, but agnostic over training dimenions as long as N C exists
        elif mode == "layer" or mode=="layer-affine":
            #self.norm = pt.nn.LayerNorm(feat_dim)
            self.norm=pt.nn.GroupNorm(num_groups=1,num_channels=feat_dim,affine=affine)
        elif mode == "layer-list":
            self.norm = pt.nn.LayerNorm(*feat_dim)
        elif mode == "pair-si":
            self.norm = PairNorm(scale_individually=True)
        elif mode == "pair-scs":
            self.norm = PairNorm(scale_individually=True)
        elif mode == "pair":
            # less restrictive, just require the inter-node distance to be constant
            self.norm = PairNorm(scale_individually=False)
        elif mode == "identity":
            self.norm = pt.nn.Identity()
        elif mode == "graph-size-c":
            self.norm = NodeNumberNorm(max_nodes)
        elif mode == "graph-size-v":
            self.norm = GraphSizeNormV()
        elif mode == "graph-size-deg":
            self.norm = GraphSizeNormDeg()
        elif mode == "graph-size-maxdeg":
            self.norm = GraphSizeNormMaxDeg()
        else:
            raise NotImplementedError(mode)

    def forward(self, X, A=None):
        # X: B N F
        if self.mode in {"instance","layer","instance-affine","layer-affine"}:
            orig_dim = X.dim()
            if orig_dim == 2:
                ret = self.norm(X)
            else:
                ret = self.norm(X.permute(0, 2, 1)).permute(0, 2, 1)

            return ret
        elif self.mode in {"graph-size-deg", "graph-size-v", "graph-size-maxdeg"}:
            return self.norm(X, A)
        else:
            return self.norm(X)


class FeedForward(nn.Module):
    def __init__(self, dimensions: list(), n_layers: int(), dropout=0.0):
        super().__init__()

        self.n_layers = n_layers - 1
        assert len(dimensions) == n_layers

        self.layers = pt.nn.ModuleList()
        for l_ in range(self.n_layers):
            self.layers.append(pt.nn.Linear(dimensions[l_], dimensions[l_ + 1]))
        self.dropout = pt.nn.Dropout(dropout)

    def forward(self, x, activation=pt.nn.Identity()):
        for l_ in range(self.n_layers):
            x = self.dropout(activation(self.layers[l_](x)))

        return x


class ConcatAggregate(nn.Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return pt.cat(x, dim=self.dim)


class DenseSequential(nn.Module):
    """"""

    def __init__(self, layers: Union[nn.ModuleList, list, tuple], aggregate=None):
        super().__init__()
        if type(layers) is not nn.ModuleList:
            layers = nn.ModuleList(*layers)
        self.layers = layers
        if aggregate is None:
            aggregate = ConcatAggregate()
        self.agg = aggregate

    def forward(self, x):
        outs = [x]
        for l in self.layers:
            agg = self.agg(outs)
            o = l(agg)
            outs.append(o)
        return outs[-1]
