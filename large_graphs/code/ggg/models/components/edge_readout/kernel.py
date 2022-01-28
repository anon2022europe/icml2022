import torch
import torch as pt
from torch import nn as nn
from torch.nn import Sequential

from ggg.models.components.discriminators.kCycleGIN import get_act
from ggg.models.components.pointnet_st import PointNetBlock
from ggg.models.components.spectral_norm import sn_wrap
from ggg.utils.logging import summarywriter, global_step, tensor_imshow, log_hists
from ggg.utils.utils import zero_diag, zero_and_symmetrize, add_to_non_edges


class KernelEdges(nn.Module):
    def __init__(self, p=2):
        super().__init__()
        self.p = p
        self.log_sigma = torch.nn.Parameter(torch.ones(()), requires_grad=True)

    def forward(self, X):
        dists = torch.zeros(X.shape[0], X.shape[1], X.shape[1], device=X.device)
        sigma = self.log_sigma.exp()
        for i in range(X.shape[1]):
            for j in range(i):
                if i == j:
                    continue
                x1 = X[:, i, :]
                x2 = X[:, j, :]
                d = torch.norm(x1 - x2, p=self.p)
                dists[:, i, j] = d
                # TODO/Note: can't do directed graphs like this
                dists[:, j, i] = d
        # assert not torch.isnan(sigma).any()
        S = -((dists / sigma) ** 2)
        ## can't zero_grad here buuuuuut, we can still make sure the diagonal isn't zero
        _A = (-S).exp()
        A = zero_diag(_A)

        return A


class RescaledSoftmax(nn.Module):
    def __init__(
        self,
        p=2,
        bias_mode=False,
        spectral_norm=None,
        feat_dim=None,
        bias_hidden=128,
        inner_activation=None,
    ):
        super().__init__()
        if inner_activation is None:
            inner_activation = nn.ReLU
        self.p = p
        self.sigma = 1.0
        if bias_mode:
            # node wise temperature
            self.bias = nn.Sequential(
                PointNetBlock(feat_dim, bias_hidden, spectral_norm=spectral_norm),
                inner_activation(),
                PointNetBlock(bias_hidden, 1, spectral_norm=spectral_norm),
            )
        else:
            self.bias = None

    def forward(self, X):
        # X: B N F
        # TODO: QK/QQ attention
        # inner product => B N N
        prod = X @ X.permute(0, -1, -2)
        if self.bias is True or self.bias == "add":
            prod = prod + self.bias(X)
        elif self.bias == "mult":
            prod = prod * self.bias(X)
        zeroed = zero_diag(prod)
        sm = zeroed.softmax(-1)
        # zero diag again
        sm_zero = zero_diag(sm)
        # renormalizec
        ma = sm_zero.max(-1)[0].unsqueeze(-1)
        mi = sm_zero.min(-1)[0].unsqueeze(-1)
        e = torch.finfo(ma.dtype).eps
        A_nonsym = (sm_zero - mi) / (ma - mi + e)
        # symmetrize
        Atriu = torch.triu(A_nonsym)
        A = Atriu + Atriu.permute(0, -1, -2)
        return A


class BiasedSigmoid(nn.Module):
    def __init__(
        self,
        feat_dim,
        hidden_dim=128,
        spectral_norm=None,
        bias_mode="scalar",
        act=None,
        scalar_bias_dual_exp=False,
        sin_sq=False,
        max_communities=None# none=> no community score, otherwise we use a community score MLP with hidden dim
    ):
        super().__init__()
        act=get_act(act)
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.mode = bias_mode
        self.sin_sq = sin_sq
        self.gate = torch.nn.Parameter(torch.zeros([]))
        self.dual_exp = scalar_bias_dual_exp
        if bias_mode == "scalar":
            self.trunk = Sequential(
                sn_wrap(nn.Linear(feat_dim, hidden_dim), spectral_norm),
                act(),
                sn_wrap(nn.Linear(hidden_dim, 1), spectral_norm),
            )
        elif bias_mode == "scalar-indep":
            self.b = torch.nn.Parameter(torch.zeros([]))
            self.bp = torch.nn.Parameter(torch.zeros([]))
            self.bn = torch.nn.Parameter(torch.zeros([]))
        elif bias_mode == "nodes":
            # node wise temperature
            self.trunk = nn.Sequential(
                PointNetBlock(feat_dim, hidden_dim, spectral_norm=spectral_norm),
                act(),
                PointNetBlock(hidden_dim, 1, spectral_norm=spectral_norm),
            )
        self.K=torch.nn.Parameter(torch.ones([])*5)
        self.max_communities=max_communities
        if max_communities is not None:
            self.community_score_net=pt.nn.Sequential(pt.nn.Linear(feat_dim,hidden_dim),act(),pt.nn.Linear(hidden_dim,max_communities))
            self.community_gate=pt.nn.Parameter(pt.zeros([]))
        else:
            self.community_score_net=None
            self.community_gate = None

    def community_score(self,X):
        Q=self.community_score_net(X)
        score=Q@Q.permute(0,2,1)
        return score

    def forward(self, X, A=None, N=None):
        # X: B N F
        # aggregate along node_dim
        # B 1 1
        # B N N
        if self.mode == "scalar-indep":
            if self.dual_exp:
                b = self.bp.exp() - self.bn.exp()
            else:
                b = self.b
            b=0.0
        else:
            x = X.mean(1)
            b = self.trunk(x).unsqueeze(-1) * self.gate
        # B, N, F
        prod = X @ X.permute(0, 2, 1) / X.shape[-1]
        # add community score if we use it
        if self.community_score_net is not None:
            comm_score=self.community_score(X)
            prod=prod*(1+self.community_gate*comm_score)
        # zerodiag and normalize
        prod=zero_diag(prod)
        assert N is not None
        max_entry=add_to_non_edges(prod, float("-inf"), N=N).reshape(prod.shape[0], -1).max(-1).values.reshape(-1, 1, 1).detach()
        min_entry=add_to_non_edges(prod, float("inf"), N=N).reshape(prod.shape[0], -1).min(-1).values.reshape(-1, 1, 1).detach()
        assert torch.isfinite(max_entry).all()
        assert torch.isfinite(min_entry).all()

        tensor_imshow("prenorm-prod", prod[0])
        K=5.0
        # divide by either the maximum absolute entry or K (whichever is larger), then multiply by K again
        # no change if the max_entry is smaller than K, otherwise normalizes to K
        prod=((prod-min_entry)/(max_entry-min_entry+1e-5)-0.5)*2 *K
        # B N N
        biased_prod = prod + b
        try:
            if log_hists():
                summarywriter().add_histogram(
                    "presigmoid", biased_prod, global_step=global_step()
                )
                summarywriter().add_histogram("presigmoid-X", X, global_step=global_step())
        except:
            pass
        Ximg = X[0]


        if log_hists():
            tensor_imshow("presigmoid-Ximg", Ximg)
            tensor_imshow("presigmoid-XXt-img", biased_prod[0])
        if self.sin_sq:
            _A = torch.sin(biased_prod).pow(2)
        else:
            _A = torch.sigmoid(biased_prod)
        # symmetrize
        A = zero_and_symmetrize(_A)
        return A


if __name__ == "__main__":
    k = KernelEdges()
    b = BiasedSigmoid(feat_dim=10, spectral_norm="diff", bias_mode="nodes")
    r = RescaledSoftmax(feat_dim=10, spectral_norm="diff", bias_mode=True)
    X = torch.randn(3, 5, 10)
    s = k(X)
    l = s.sum()
    l.backward()
    X = torch.randn(3, 5, 10)
    s = b(X)
    l = s.sum()
    l.backward()
    X = torch.randn(3, 5, 10)
    s = r(X)
    l = s.sum()
    l.backward()
