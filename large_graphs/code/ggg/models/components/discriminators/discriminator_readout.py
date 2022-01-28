from _warnings import warn

import torch
import torch as pt

from ggg.data.dense.utils.features import smallest_k_eigenval
from ggg.models.components.spectral_norm import SpectralNorm, SpectralNormNonDiff
from ggg.models.components.utilities_classes import NodeFeatNorm, Swish
from ggg.utils.utils import kcycles, maybe_assert


class DiscriminatorReadout(torch.nn.Module):
    _MODES = {"score", "graph", "nodes"}

    def __init__(
        self,
        in_dim,
        hidden_dim,
        out_dim,
        kc_flag=True,
        swish=False,
        spectral_norm=None,
        dropout=None,
        eigenfeat4=False,
        pac=1,
        agg="sum",
        norm_in_type="instance",
        norm_hidden_type="layer",
        max_nodes=None,
    ):
        super().__init__()
        self.kc_flag = kc_flag
        self.pac = pac
        self.eigenfeat4 = eigenfeat4
        self.agg = agg
        self.norm_in_type = norm_in_type
        self.norm_hidden_type = norm_hidden_type
        add_feats = 0
        if self.kc_flag:
            self.kcycles = kcycles()
            add_feats += 4
        if self.eigenfeat4:
            add_feats += 4
        effective_node_feats = in_dim + add_feats
        self.inp_norm = NodeFeatNorm(
            in_dim * self.pac, mode=norm_in_type, max_nodes=max_nodes
        )
        self.hidden_norm = NodeFeatNorm(hidden_dim, mode=norm_hidden_type)
        self.hidden_lin = torch.nn.Linear(effective_node_feats * self.pac, hidden_dim)

        if swish is True or swish == "swish":
            self.act = Swish()
        elif swish == "leaky":
            self.act = torch.nn.LeakyReLU(0.1)
        else:
            self.act = torch.nn.ReLU()
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.out_lin = torch.nn.Linear(hidden_dim, out_dim)
        if spectral_norm == "diff":
            self.hidden_lin = SpectralNorm(self.hidden_lin)
            self.out_lin = SpectralNorm(self.out_lin)
        elif spectral_norm == "nondiff":
            self.hidden_lin = SpectralNormNonDiff(self.hidden_lin)
            self.out_lin = SpectralNormNonDiff(self.out_lin)

    def forward(self, X, adj, mode="score"):
        if mode not in DiscriminatorReadout._MODES:
            raise ValueError(
                f"Only know discriminator modes {DiscriminatorReadout._MODES}"
            )
        if self.dropout:
            X = self.dropout(X)
        if self.pac > 1:
            # B,F is [B1,B2,B3,..B_pac] F
            # move to  [B,pac*F]
            assert X.dim() == 3
            X = X.reshape(-1, self.pac, X.shape[-2], X.shape[-1]).permute(0, 2, 1, 3)
            X = X.reshape(X.shape[0], X.shape[1], -1)
        if mode == "nodes":
            # return node features
            return X
        maybe_assert(func=lambda: pt.isfinite(X).all())
        X = self.inp_norm(X, adj)
        maybe_assert(func=lambda: pt.isfinite(X).all())
        X = self.act(X)
        maybe_assert(func=lambda: pt.isfinite(X).all())
        if self.agg == "sum":
            xs = X.sum(dim=-2)
        elif self.agg == "mean":
            xs = X.mean(dim=-2)
        elif self.agg == "max":
            xs = X.max(dim=-2).values
        elif self.agg == "lse":
            xs = X.logsumexp(dim=-2)
        else:
            raise ValueError("Unkown agg method in DiscReadout")
        maybe_assert(func=lambda: pt.isfinite(xs).all())
        if self.kc_flag:
            kcycles = self.kcycles.k_cycles(adj)
            if self.pac > 1:
                kcycles = kcycles.reshape(-1, self.pac, kcycles.shape[-1]).permute(
                    0, 2, 1
                )
                kcycles = kcycles.reshape(kcycles.shape[0], -1)
            if kcycles.shape[-1]<4:
                zero_pad=pt.zeros(kcycles.shape[0],4-kcycles.shape[-1],device=kcycles.device)
                kcycles=pt.cat([kcycles,zero_pad],-1)
            xs = torch.cat([xs, kcycles], dim=-1)
        maybe_assert(func=lambda: pt.isfinite(xs).all())
        if self.eigenfeat4:
            try:
                adj: pt.Tensor
                D = adj.sum(-1, keepdim=True)
                L = D - adj
                eigen_vals = smallest_k_eigenval(L, 4)
            except:
                warn(f"Getting eigenvals failed, replacing with zeros")
                eigen_vals = pt.zeros(xs.shape[0], 4).type_as(xs)
            if eigen_vals.shape[-1]<4:
                zero_pad=pt.zeros(eigen_vals.shape[0],4-eigen_vals.shape[-1],device=eigen_vals.device)
                eigen_vals=pt.cat([eigen_vals,zero_pad],-1)

            if self.pac > 1:
                eigen_vals = eigen_vals.reshape(
                    -1, self.pac, eigen_vals.shape[-1]
                ).permute(0, 2, 1)
                eigen_vals = eigen_vals.reshape(eigen_vals.shape[0], -1)
            xs = torch.cat([xs, eigen_vals], dim=-1)

        maybe_assert(func=lambda: pt.isfinite(xs).all())
        xl = self.hidden_lin(xs)
        maybe_assert(func=lambda: pt.isfinite(xl).all())
        if mode == "graph":
            # return graph features
            return xl
        xl = self.hidden_norm(xl)
        maybe_assert(func=lambda: pt.isfinite(xl).all())
        xt = self.act(xl)
        maybe_assert(func=lambda: pt.isfinite(xt).all())
        if self.dropout:
            xt = self.dropout(xt)
        xout = self.out_lin(xt)
        maybe_assert(func=lambda: pt.isfinite(xout).all())
        return xout