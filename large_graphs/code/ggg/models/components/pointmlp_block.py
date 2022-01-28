import torch as pt
from ipdb import set_trace
from torch import nn as nn

from ggg.models.components.attention.MultiHead import MultiHeadAttention
from ggg.models.components.pointnet_st import PointNetBlock
from ggg.models.components.spectral_norm import SpectralNorm
from ggg_utils.utils.utils import RezeroMLP, MLP, get_act


def point_mlp(inp,h,act,out=None,spectral=None):
    out=out if out is not None else h
    return pt.nn.Sequential(
                    PointNetBlock(inp,h,spectral_norm=spectral,activation=act),
                    PointNetBlock(h, out, spectral_norm=spectral, activation=act),
    )

class PointMLPBlock(nn.Module):
    def __init__(
            self,
            in_feat,
            out_feat,
            num_heads=1,
            inner_activation="relu",
            attention_mode="QK",
            score_function="sigmoid",
            smyrf=None,
            norm_type="layer",
            block_skip=True,
            rezero_skip=False,
            att_rezero=False,
            spectral_norm=None,
    ):
        super().__init__()
        if isinstance(inner_activation,str):
            inner_activation=get_act(inner_activation)()
        self.att=point_mlp(in_feat,out_feat,inner_activation,out=out_feat,spectral=spectral_norm)
        if block_skip:
            self.proj = (
                pt.nn.Linear(in_feat, out_feat)
                if in_feat != out_feat
                else pt.nn.Identity()
            )
            if isinstance(self.proj, pt.nn.Linear) and spectral_norm:
                self.proj = SpectralNorm(self.proj)
        else:
            self.proj = None
        self.gate = (
            1.0
            if not (rezero_skip and block_skip)
            else pt.nn.Parameter(pt.zeros([]), requires_grad=True)
        )

    def forward(self, X, mask=None):
        out = self.att(X)
        if self.proj:
            out = out * self.gate + self.proj(X)
        # maybe add an MLP HERE
        return out
