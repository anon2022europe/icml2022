import math

import torch
import torch as pt
from torch import nn as nn

from ggg.models.utils import get_act
from ggg.models.components.pointnet_st import PointNetBlock
from ggg.utils.utils import zero_diag, zero_and_symmetrize, asserts_enabled, RezeroMLP, PINNWrapper
import torch.nn.functional as F

def split_outer(X,split_across=None,out_gpu=None):
    if split_across is None:
        prod = X @ X.permute(0, 1, 3, 2)
        return prod
    else:
        to=math.ceil(X.shape[2]/len(split_across))
        chunks=[X[(i-1)*to:i*to].to(split_across[i-1]) for i in range(1,len(split_across)+1)]
        prod_chunks=[c@X.to(dev).permute(0,1,3,2) for c,dev in zip(chunks,split_across)]
        prod=pt.cat([c.to(out_gpu) for c in prod_chunks],2)
        return prod


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



class DoubleSoftmax(nn.Module):
    # TODO: need to very delicately remove the garbage I added
    def __init__(
        self,
        feat_dim,
        hidden_dim=128,
        spectral_norm=None,
        act=None,
        temperature=0.66,
        k=2,  # how many implicit edge states there are: 0,1,...K
        k_in=None,
        edge_scoring="prod",  #  prod, l2
        with_reinforce=False,
        on_gpu=None,# parameter GPU
        outer_gpus=None,#list of gpus to split the outer product across evenly
        discretize=True,
        joined_feat=False,
        root_readd_dim=None,
        explore_change=None# tuple float,float or none

    ):
        super().__init__()
        rrd=0 if not root_readd_dim else root_readd_dim
        self.root_readd_dim=root_readd_dim
        act = get_act(act)
        assert k >= 2, f"Need at least edge/noedge states, got k={k}"
        if k_in is None:
            k_in=k
        assert k_in >= 2, f"Need at least edge/noedge states, got k_in={k_in}"
        assert hidden_dim%k_in==0, f"Need {hidden_dim}%{k_in}=0, got {hidden_dim%k_in}"
        self.edge_scoring = edge_scoring
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        if root_readd_dim is not None and False:
            self.root_proj=pt.nn.Linear(feat_dim+root_readd_dim,hidden_dim)
        else:
            self.root_proj=None
        if joined_feat:
            self.join_nn=PINNWrapper(presum=RezeroMLP([feat_dim+rrd,hidden_dim,hidden_dim],act),postsum=RezeroMLP([hidden_dim,hidden_dim],act),act=act())
        else:
            self.join_nn=None
        self.edge_feat =RezeroMLP(widths=[feat_dim+rrd +(0 if not joined_feat else hidden_dim), hidden_dim, hidden_dim*(2 if joined_feat else 1)], act=act)
        if hidden_dim!=(feat_dim+rrd):
            self.proj=pt.nn.Linear(feat_dim+rrd+(hidden_dim if joined_feat else 0),hidden_dim)
        else:
            self.proj=None
        self.gate = torch.nn.Parameter(torch.zeros([]))
        self.temperature = temperature
        self.score_mlp = RezeroMLP([k_in, hidden_dim, k], act=pt.nn.ReLU)
        self.k = k
        self.k_in=k_in
        self.zo = pt.nn.Parameter(
            pt.ones([k, 1]), requires_grad=False
        )  # 0,1....,1 vector to aggregate edges afterwards
        self.zo[0, 0] = 0.0
        self.with_reinforce = with_reinforce
        self.outer_gpus=outer_gpus
        self.param_gpu=on_gpu
        self.discretize=discretize
        self.explore_change=explore_change

    def forward(self, X, A=None, N=None,root_to_readd=None,k_edges=None):
        # X: B N F
        # aggregate along node_dim
        # B, N, F => B,N,F,2
        if root_to_readd is not None:
            X=pt.cat([X,root_to_readd],-1)
        if self.root_proj is not None:
            X=self.root_proj(X)
        if self.param_gpu is not None:
            X=X.to(self.param_gpu)
        if self.join_nn is not None:
            j=self.join_nn(X,keepdim=True).repeat([1,X.shape[1],1])
            X=pt.cat([X,j],-1)
        feats=self.edge_feat(X)
        if self.proj is not None:
            X=self.proj(X)
        Xact = feats.reshape(
            feats.shape[0], feats.shape[1], feats.shape[2] // self.k_in, self.k_in
        )
        Xskip = X.reshape(X.shape[0], X.shape[1], X.shape[2] // self.k_in, self.k_in)
        if asserts_enabled():
            assert Xskip.shape == Xact.shape

        # again, rezero for stability
        X = Xskip + self.gate * Xact
        # do the outer product preserving the 2 final dim
        X = X.permute(0, 3, 1, 2)
        if self.edge_scoring == "l2":
            X = X.contiguous()
            # prod=pt.cdist(X,X) # throws invalid gradient errors
            R = X.norm(dim=-1, keepdim=True).repeat(1, 1, 1, X.shape[-2])
            prod = R + R.permute(0, 1, 3, 2) - 2 * split_outer(X,self.outer_gpus,self.param_gpu)
            prod = prod * -1  # larger distance => less likely to have edge
            # prod=pt.cdist(X,X) # throws invalid gradient errors
        else:
            prod=split_outer(X,self.outer_gpus,self.param_gpu)
            #prod = X @ X.permute(0, 1, 3, 2)  # B 2 N N

        prod = prod.permute(0, 2, 3, 1).contiguous()  # B N N K
        prod = self.score_mlp(prod.reshape(prod.shape[0], -1, self.k_in)).reshape(
            *prod.shape[:-1],self.k
        )
        if self.explore_change is not None and self.training:
            B=prod.shape[0]
            Bclean=int((1-self.explore_change[0])*B)
            Bexplore=B-Bclean
            assert Bexplore+Bclean==B and Bexplore>=2 and Bclean>=2 # proper split 1 not breaking norms
            inds=pt.arange(B)
            clean=prod[inds[:Bclean]]
            explore=prod[inds[Bclean:]]
            exp_range=explore.max()-explore.min()
            explore=explore+pt.randn_like(explore)*self.explore_change[1]*max(exp_range,1.0)
            prod=pt.cat([clean,explore],0)
        if k_edges is None: # normal sampling
            prod = (
                    prod - prod.max(dim=-1, keepdim=True).values
            )  # stabilize prod against overflow
            # GM+straight through estimator
            A2 = F.gumbel_softmax(prod, self.temperature, hard=self.discretize)
            A = (A2 @ self.zo.detach()).squeeze(-1)
        else:
            raise NotImplementedError("Edge forcing removed in cleanup")
        # symmetrize
        A = zero_and_symmetrize(A)
        if self.with_reinforce:
            log_probs = (
                pt.log((prod.softmax(-1) * A2).sum(-1)).reshape(A2.shape[0], -1).sum(-1)
            )  # log probability of each adjacency matrix
            return A, log_probs
        else:
            return A


if __name__ == "__main__":
    r = RescaledSoftmax(feat_dim=10, spectral_norm="diff", bias_mode=True)
    X = torch.randn(3, 5, 10)
    X = torch.randn(3, 5, 10)
    s = b(X)
    l = s.sum()
    l.backward()
    X = torch.randn(3, 5, 10)
    s = r(X)
    l = s.sum()
    l.backward()
