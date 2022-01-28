import abc

import torch
from torch import nn as nn

from ggg.models.components.attention.MultiHead import MultiHeadAttention


class NodeReadout(nn.Module):
    @abc.abstractmethod
    def forward(self, X, A, N):
        pass


class AttentionNodeReadout(NodeReadout):
    def __init__(
        self,
        feat_dim,
        node_attrib_dim,
        num_heads=1,
        layers=1,
        inner_activation=None,
        out_activation=None,
        attention_mode="QQ",
        score_function="sigmoid",
        spectral_norm=None,
        smyrf=None,
    ):
        super().__init__()
        self.embed_features = node_attrib_dim
        self.node_attrib_dim = node_attrib_dim
        self.head_num = num_heads
        self._layers = nn.ModuleList()
        if self.node_attrib_dim > 0:
            for _ in range(layers - 1):
                self._layers.append(
                    MultiHeadAttention(
                        in_features=feat_dim,
                        out_features=feat_dim,
                        num_heads=self.head_num,
                        activation=inner_activation,
                        mode=attention_mode,
                        score_function=score_function,
                        spectral_norm=spectral_norm,
                        smyrf=smyrf,
                    )
                )
            self._layers.append(
                MultiHeadAttention(
                    in_features=feat_dim,
                    out_features=node_attrib_dim,
                    num_heads=1,
                    mode=attention_mode,
                    score_function=score_function,
                    spectral_norm=spectral_norm,
                    smyrf=smyrf,
                )
            )

    def forward(self, X, A=None, N=None) -> torch.Tensor:
        if self.node_attrib_dim == 0:
            return torch.Tensor(X.shape[0],X.shape[1],0)
        else:
            q, k, v = X, X, X
            for i, att in enumerate(self._layers):
                Zi, A, _ = att(q=q, k=k, v=v, return_attention_and_scores=True)
            return Zi
