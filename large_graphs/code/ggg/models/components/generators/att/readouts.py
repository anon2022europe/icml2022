import abc

import attr
import torch as pt
from attr.validators import in_
from torch import nn as nn

from ggg.models.components.abstract_conf import AbstractConf
from ggg.models.components.attention.MultiHead import MultiHeadAttention
from ggg.models.components.edge_readout.attention import AttentionEdgeReadout
from ggg.models.components.edge_readout.kernel import (
    BiasedSigmoid,
    RescaledSoftmax,
    KernelEdges,
)
from ggg.models.components.edge_readout.softmax_variants import (
    DoubleSoftmax,
)
from ggg.models.components.utilities_classes import NodeFeatNorm
from ggg.utils.utils import kwarg_create, zero_and_symmetrize
from ggg.models.components.node_readout.attention import AttentionNodeReadout
from ggg_utils.utils.utils import RezeroMLP, get_act


class EdgeReadout(nn.Module):
    @abc.abstractmethod
    def forward(self, X, A, N):
        pass


class AttentionWeights(EdgeReadout):
    def __init__(
        self,
        feat_dim,
        num_heads=1,
        inner_activation="relu",
        attention_mode="QK",
        att_skip=True,
        spectral_norm=None,
        readout_score_function="sigmoid",
        smyrf=None,
        norm_type="identity",
    ):
        super(AttentionWeights, self).__init__()
        # might want to keep  num_heads =1 for readout, otherwise difficult to reason about
        self.attn = MultiHeadAttention(
            in_features=feat_dim,
            num_heads=num_heads,
            activation=inner_activation,
            mode=attention_mode,
            score_function=readout_score_function,
            spectral_norm=spectral_norm,
            norm_type=norm_type,
            smyrf=smyrf,
        )

    def forward(self, X, A=None, N=None):
        q, k, v = X, X, X
        _, weights, _scores = self.attn.forward(
            q, k, v, return_attention_and_scores=True
        )
        weights = zero_and_symmetrize(weights)
        return weights

class NonEqEdgeReadout(EdgeReadout):
    def __init__(
            self,
            feat_dim,
            num_heads=1,
            inner_activation="relu",
            attention_mode="QK",
            att_skip=True,
            spectral_norm=None,
            readout_score_function="sigmoid",
            smyrf=None,
            norm_type="identity",
            max_N=None
    ):
        super(NonEqEdgeReadout, self).__init__()
        # might want to keep  num_heads =1 for readout, otherwise difficult to reason about
        self.attn = pt.nn.Linear(feat_dim,max_N*max_N)
        self.max_N=max_N
        self.act=get_act(readout_score_function)()


    def forward(self, X, A=None, N=None):
        max_N=self.max_N
        logits=self.attn(X).reshape(-1,max_N,max_N)
        weights=self.act(logits)
        weights = zero_and_symmetrize(weights)
        return weights


class QQSig(EdgeReadout):
    def forward(self, X, A=None, N=None):
        kernel = X @ X.permute(0, 2, 1)
        return pt.sigmoid(kernel)


@attr.s
class EdgeReadoutHpars(AbstractConf):
    OPTIONS = dict(
        attention_weights=AttentionWeights,
        QQ_sig=QQSig,
        biased_sigmoid=BiasedSigmoid,
        rescaled_softmax=RescaledSoftmax,
        gaussian_kernel=KernelEdges,
        attention_readout=AttentionEdgeReadout,
        double_softmax=DoubleSoftmax,
        noneq=NonEqEdgeReadout
    )
    name = attr.ib(default="biased_sigmoid")
    feat_dim = attr.ib(default=32)
    spectral_norm = attr.ib(default=None)
    act = attr.ib(default="relu")
    bias_mode = attr.ib(default="scalar-indep")
    num_heads = attr.ib(default=1)
    score_function = attr.ib(default="softmax")
    readout_score_function = attr.ib(default="softmax")
    count_feat = attr.ib(default=32)
    inner_activation = attr.ib(default="relu")
    attention_mode = attr.ib(default="QK")
    smyrf = attr.ib(factory=dict)
    att_skip = attr.ib(default=True)
    hard = attr.ib(default=True)
    hidden_dim = attr.ib(default=128)
    p = attr.ib(default=None)
    bias_hidden = attr.ib(default=128)
    norm_type = attr.ib(default="layer-affine", validator=in_(NodeFeatNorm.SUPPORTED))
    max_communities=attr.ib(default=None)# either compute or an integer
    max_N=attr.ib(default=None)

    def make(self):
        cls = EdgeReadoutHpars.OPTIONS[self.name]
        kwargs = attr.asdict(self)
        return kwarg_create(cls, kwargs)


@attr.s
class NodeReadoutHpars(AbstractConf):
    OPTIONS = dict(attention=AttentionNodeReadout)
    name = attr.ib(default="attention", validator=in_(OPTIONS))
    feat_dim = attr.ib(default=32)
    num_heads = attr.ib(default=5)
    att_skip = attr.ib(default=True)
    layers = attr.ib(default=1)
    node_attrib_dim = attr.ib(default=5)
    inner_activation = attr.ib(default="relu")
    attention_mode = attr.ib(default="QK")
    spectral_norm = attr.ib(default=None)
    score_function = attr.ib(default="softmax")
    smyrf = attr.ib(factory=dict)

    def make(self):
        cls = NodeReadoutHpars.OPTIONS[self.name]
        kwargs = attr.asdict(self)
        return kwarg_create(cls, kwargs)
