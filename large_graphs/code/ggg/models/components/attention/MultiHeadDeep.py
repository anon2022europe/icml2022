from typing import List

import torch
from torch import nn as nn

from ggg.models.components.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from ggg.models.components.utilities_classes import (
    DenseSequential,
    PermuteBatchnorm1d,
    NodeFeatNorm,
)
from ggg.models.components.spectral_norm import sn_wrap


class MultiHeadDeepAttention(nn.Module):
    def __init__(
        self,
        in_features,
        out_features=None,
        head_num=1,
        activation=None,
        out_activation=None,
        inner_layers: List[int] = None,
        mode="QK",
        spectral_norm=None,
    ):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param head_num: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super().__init__()
        if in_features % head_num != 0:
            raise ValueError(
                "`in_features`({}) should be divisible by `head_num`({})".format(
                    in_features, head_num
                )
            )
        self.in_features = in_features
        self.mode = mode
        if out_features is None:
            out_features = in_features

        self.inner_feat_sizes = inner_layers
        if inner_layers is None or len(inner_layers) == 0:
            inner_layers = [in_features, out_features]
            inner_layers_out = [out_features, out_features]
        else:
            inner_layers = [in_features] + inner_layers + [out_features]
            inner_layers_out = [out_features] + inner_layers + [out_features]

        def mk_proj(feat_sizes, activation, out_act=None):
            layers = nn.ModuleList()
            for l in range(1, len(feat_sizes)):
                i = sum(feat_sizes[:l])
                o = feat_sizes[l]
                local_layer = []
                local_layer.append(NodeFeatNorm(i))
                local_layer.append(sn_wrap(nn.Linear(i, o), spectral_norm))
                if activation is not None and (
                    l < len(feat_sizes) - 1 or out_act is None
                ):
                    local_layer.append(activation())
                elif out_act is not None:
                    local_layer.append(out_act())
                layers.append(nn.Sequential(*local_layer))
            return DenseSequential(layers)

        self.linear_q = mk_proj(inner_layers, activation)
        if self.mode == "QK":
            self.linear_k = mk_proj(inner_layers, activation)
        else:
            self.linear_k = None
        self.linear_v = mk_proj(inner_layers, activation)
        self.linear_o = mk_proj(inner_layers_out, activation, out_activation)

        self.out_features = out_features
        self.head_num = head_num
        self.activation = activation() if activation is not None else None
        self.out_activation = out_activation() if out_activation is not None else None

    def forward(self, q, k, v, mask=None, return_attention_and_scores=False):
        # q,k,v: tensors of batch_size, seq_len, in_feature
        q = self.linear_q(q)
        v = self.linear_v(v)
        if self.mode == "QQ":
            k = q
        elif self.mode == "QK":
            k = self.linear_k(k)

        if self.activation is not None:
            q = self.activation(q)
            k = self.activation(k)
            v = self.activation(v)

        q = self._reshape_to_batches(q)
        k = self._reshape_to_batches(k)
        v = self._reshape_to_batches(v)
        if mask is not None:
            mask = mask.repeat(self.head_num, 1, 1)
        if return_attention_and_scores:
            y, _attn, _attn_scores = ScaledDotProductAttention()(
                q, k, v, mask, return_attention_and_scores=return_attention_and_scores
            )
            _attn = self._reshape_from_batches(_attn, return_attention_and_scores)
            _attn_scores = self._reshape_from_batches(
                _attn_scores, return_attention_and_scores
            )
        else:
            y = ScaledDotProductAttention()(
                q, k, v, mask, return_attention_and_scores=return_attention_and_scores
            )
            y = self._reshape_from_batches(y)

        y = self.linear_o(y)
        if self.out_activation is not None:
            y = self.out_activation(y)
        if return_attention_and_scores:
            return y, _attn, _attn_scores
        else:
            return y

    @staticmethod
    def gen_history_mask(x):
        """Generate the mask that only uses history data.

        :param x: Input tensor.
        :return: The mask.
        """
        batch_size, seq_len, _ = x.size()
        return (
            torch.tril(torch.ones(seq_len, seq_len))
            .view(1, seq_len, seq_len)
            .repeat(batch_size, 1, 1)
        )

    def _reshape_to_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        sub_dim = in_feature // self.head_num
        return (
            x.reshape(batch_size, seq_len, self.head_num, sub_dim)
            .permute(0, 2, 1, 3)
            .reshape(batch_size * self.head_num, seq_len, sub_dim)
        )

    def _reshape_from_batches(self, x, return_attention_and_scores=False):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        if return_attention_and_scores:
            out_dim = in_feature
        else:
            out_dim = self.out_features * self.head_num

        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def extra_repr(self):
        return "in_features={}, head_num={}, bias={}, activation={} inner_feat_sizes=[]".format(
            self.in_features,
            self.head_num,
            self.bias,
            self.activation,
            self.inner_feat_sizes,
        )
