import abc
from inspect import signature

import attr
import torch as pt
from attr.validators import in_
from torch import nn as nn

from ggg.models.components.abstract_conf import AbstractConf
from ggg.models.components.attention.block import AttentionBlock
from ggg.models.components.mlp_block import MLPBlock
from ggg.models.components.pointmlp_block import PointMLPBlock
from ggg.models.components.spectral_norm import SPECTRAL_NORM_OPTIONS
from ggg.models.components.utilities_classes import NodeFeatNorm
from ggg.utils.utils import kwarg_create
from ggg.utils.utils import node_mask


class GenTrunk(nn.Module):
    @abc.abstractmethod
    def forward(self, X, A=None, N=None):
        pass


class AttentionTrunk(GenTrunk):
    def __init__(
        self,
        feat_dim,
        attn_feat_dim,
        n_layers=2,
        inner_activation="relu",
        attention_mode="QQ",
        score_function="sigmoid",
        smyrf=None,
        norm_type="layer",
        block_skip=True,
        rezero_skip=False,
        att_rezero=False,
        spectral_norm=None,
        num_heads=5,
    ):
        super(AttentionTrunk, self).__init__()
        attn_blocks = [
            AttentionBlock(
                feat_dim,
                attn_feat_dim,
                inner_activation=inner_activation,
                attention_mode=attention_mode,
                score_function=score_function,
                smyrf=smyrf,
                norm_type=norm_type,
                rezero_skip=rezero_skip,
                att_rezero=att_rezero,
                block_skip=block_skip,
                spectral_norm=spectral_norm,
                num_heads=num_heads,
            )
        ]
        for _ in range(n_layers - 1):
            attn_blocks.append(
                AttentionBlock(
                    attn_feat_dim,
                    attn_feat_dim,
                    inner_activation=inner_activation,
                    attention_mode=attention_mode,
                    score_function=score_function,
                    smyrf=smyrf,
                    norm_type=norm_type,
                    rezero_skip=rezero_skip,
                    att_rezero=att_rezero,
                    block_skip=block_skip,
                    spectral_norm=spectral_norm,
                    num_heads=num_heads,
                )
            )
        self.attn_blocks = pt.nn.Sequential(*attn_blocks)

    def forward(self, X, A=None, N=None):
        if N is None:
            mask = node_mask(X, N).detach()
        else:
            mask = None
        for attn in self.attn_blocks:
            X = attn(X, mask=mask)
        return X, A, N

class MLPTrunk(GenTrunk):

    def __init__(self, feat_dim, attn_feat_dim, n_layers=2, inner_activation="relu", attention_mode="QQ",
                 score_function="sigmoid", smyrf=None, norm_type="layer", block_skip=True, rezero_skip=False,
                 att_rezero=False, spectral_norm=None, num_heads=5):
        super(MLPTrunk, self).__init__()
        attn_blocks = [
            MLPBlock(
                feat_dim,
                attn_feat_dim,
                inner_activation=inner_activation,
                attention_mode=attention_mode,
                score_function=score_function,
                smyrf=smyrf,
                norm_type=norm_type,
                rezero_skip=rezero_skip,
                att_rezero=att_rezero,
                block_skip=block_skip,
                spectral_norm=spectral_norm,
                num_heads=num_heads,
            )
        ]
        for _ in range(n_layers - 1):
            attn_blocks.append(
                MLPBlock(
                    attn_feat_dim,
                    attn_feat_dim,
                    inner_activation=inner_activation,
                    attention_mode=attention_mode,
                    score_function=score_function,
                    smyrf=smyrf,
                    norm_type=norm_type,
                    rezero_skip=rezero_skip,
                    att_rezero=att_rezero,
                    block_skip=block_skip,
                    spectral_norm=spectral_norm,
                    num_heads=num_heads,
                )
            )
        self.attn_blocks = pt.nn.Sequential(*attn_blocks)

    def forward(self, X, A=None, N=None):
        if N is not None:
            mask = node_mask(X, N).detach()
        else:
            mask = None
        for attn in self.attn_blocks:
            X = attn(X, mask=mask)
        return X, A, N

class PointMLPTrunk(GenTrunk):

    def __init__(self, feat_dim, attn_feat_dim, n_layers=2, inner_activation="relu", attention_mode="QQ",
                 score_function="sigmoid", smyrf=None, norm_type="layer", block_skip=True, rezero_skip=False,
                 att_rezero=False, spectral_norm=None, num_heads=5):
        super(PointMLPTrunk, self).__init__()
        attn_blocks = [
            PointMLPBlock(
                feat_dim,
                attn_feat_dim,
                inner_activation=inner_activation,
                attention_mode=attention_mode,
                score_function=score_function,
                smyrf=smyrf,
                norm_type=norm_type,
                rezero_skip=rezero_skip,
                att_rezero=att_rezero,
                block_skip=block_skip,
                spectral_norm=spectral_norm,
                num_heads=num_heads,
            )
        ]
        for _ in range(n_layers - 1):
            attn_blocks.append(
                PointMLPBlock(
                    attn_feat_dim,
                    attn_feat_dim,
                    inner_activation=inner_activation,
                    attention_mode=attention_mode,
                    score_function=score_function,
                    smyrf=smyrf,
                    norm_type=norm_type,
                    rezero_skip=rezero_skip,
                    att_rezero=att_rezero,
                    block_skip=block_skip,
                    spectral_norm=spectral_norm,
                    num_heads=num_heads,
                )
            )
        self.attn_blocks = pt.nn.Sequential(*attn_blocks)

    def forward(self, X, A=None, N=None):
        if N is not None:
            mask = node_mask(X, N).detach()
        else:
            mask = None
        for attn in self.attn_blocks:
            X = attn(X, mask=mask)
        return X, A, N


@attr.s
class GenTrunkHpars(AbstractConf):
    OPTIONS = dict(attention=AttentionTrunk,mlp=MLPTrunk,pointmlp=PointMLPTrunk)
    name = attr.ib(default="attention", validator=in_(OPTIONS))
    feat_dim = attr.ib(default=32)
    attn_feat_dim = attr.ib(default=32)
    n_layers = attr.ib(default=2)
    inner_activation = attr.ib(default="relu")
    attention_mode = attr.ib(default="QK")
    score_function = attr.ib(default="softmax")
    smyrf = attr.ib(default=None)
    # with rezero we don't need norm anywhere...and even if we had it, we should be using prenorm
    # see https://arxiv.org/pdf/2003.04887.pdf for rezero, for prenorm: https://arxiv.org/pdf/1906.01787.pdf
    norm_type = attr.ib(default="identity", validator=in_(NodeFeatNorm.SUPPORTED))
    block_skip = attr.ib(default=True)
    rezero_skip = attr.ib(default=True)
    att_rezero = attr.ib(default=True)
    spectral_norm = attr.ib(default=None, validator=in_(SPECTRAL_NORM_OPTIONS))
    num_heads = attr.ib(default=5)

    def make(self):
        cls = GenTrunkHpars.OPTIONS[self.name]
        kwargs = attr.asdict(self)
        return kwarg_create(cls, kwargs)
