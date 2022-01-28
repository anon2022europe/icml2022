import torch as pt
from ipdb import set_trace
from torch import nn as nn

from ggg.models.components.attention.MultiHead import MultiHeadAttention
from ggg.models.components.spectral_norm import SpectralNorm


class AttentionBlock(nn.Module):
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
        self.att = MultiHeadAttention(
            in_features=in_feat,
            out_features=out_feat,
            num_heads=num_heads,
            activation=inner_activation,
            mode=attention_mode,
            score_function=score_function,
            spectral_norm=spectral_norm,
            norm_type=norm_type,
            rezero=att_rezero,
            smyrf=smyrf,
        )
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
        out = self.att(X, mask=None)
        if self.proj:
            out = out * self.gate + self.proj(X)
        # maybe add an MLP HERE
        return out
