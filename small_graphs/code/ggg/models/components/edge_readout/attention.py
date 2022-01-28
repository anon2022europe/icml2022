import torch
from torch import nn as nn

import torch as pt
from ggg.models.components.attention.MultiHead import MultiHeadAttention
from ggg.models.components.attention.scaled_dot_product import MemSparsemax
from ggg.models.components.utilities_classes import Swish
from ggg.models.components.edge_readout.kernel import KernelEdges
from ggg.utils.utils import zero_and_symmetrize

class AttentionEdgeReadout(nn.Module):
    def __init__(
        self,
        feat_dim,
        num_heads=1,
        score_function="softmax",
        readout_score_function="sigmod",
        count_feat=None,
        inner_activation=None,
        out_activation=None,
        attention_mode="QK",
        spectral_norm=None,
        norm_type="layer",
        smyrf=None,
    ):
        super().__init__()
        assert count_feat is not None or readout_score_function not in {
            "mlp_ratio_sig",
            "mlp_score_thresh",
        }
        self.readout = readout_score_function
        fancy = readout_score_function in {
            "mlp_ratio_sig",
            "mlp_score_thresh",
            "score_thresh",
        }
        self.fancy = fancy
        self.embed_features = feat_dim
        self.attn = MultiHeadAttention(
            in_features=feat_dim,
            out_features=1 if not fancy else feat_dim,
            num_heads=num_heads,
            activation=inner_activation,
            mode=attention_mode,
            score_function=score_function,
            spectral_norm=spectral_norm,
            norm_type=norm_type,
            smyrf=smyrf,
        )

        if fancy:
            self.score_attn = MultiHeadAttention(
                in_features=feat_dim,
                out_features=1,
                num_heads=num_heads,
                activation=inner_activation,
                mode=attention_mode,
                score_function=score_function,
                spectral_norm=spectral_norm,
                norm_type=norm_type,
                smyrf=smyrf,
            )

        if count_feat is None:
            self.count_mlp = None
        else:
            self.count_mlp = pt.nn.Sequential(
                pt.nn.Linear(feat_dim, count_feat),
                pt.nn.LeakyReLU(0.1),
                pt.nn.Linear(count_feat, 1),
            )

    def forward(self, X, A=None, N=None) -> torch.Tensor:
        Z = X
        Zi = Z
        att, _, scores = self.attn(q=Zi, k=Zi, v=Zi, return_attention_and_scores=True)
        if self.readout == "sigmoid":
            A = pt.nn.Sigmoid()(scores)
        elif self.readout == "softmax":
            A = scores.softmax(-1)
        elif self.readout == "sparsemax":
            A = MemSparsemax(dim=-1)(scores)
        elif self.readout == "mlp_ratio_sig":
            # predicts scores on each node, then predicts the partition sum on these nodes, using a sigmoid to  limit domain
            temp_attn, _, temp_scores = self.score_attn(
                att, att, att, return_attention_and_scores=True
            )
            temp = self.count_mlp(att).exp()
            A = pt.nn.Sigmoid()(scores * temp_attn)
        elif self.readout == "mlp_ratio_sparse":
            # predicts scores on each node, then predicts the partition sum on these nodes, using a sigmoid to  limit domain
            temp_attn, _, temp_scores = self.score_attn(
                att, att, att, return_attention_and_scores=True
            )
            temp = self.count_mlp(att).exp()
            A = MemSparsemax(dim=-1)(scores * temp_attn)
        elif self.readout == "mlp_score_thresh":
            # predicts scores on each node, then thresholds any below a certain score
            _, _, scores = self.score_attn(
                att, att, att, return_attention_and_scores=True
            )
            threshs = self.count_mlp(att)
            survivors = (scores - threshs).clamp_min(0.0).squeeze(-1)
            A = (survivors / (survivors.sum(-1, keepdim=True) + 1e-9)).clamp(0.0, 1.0)
        elif self.readout == "score_thresh":
            _, _, scores = self.score_attn(
                att, att, att, return_attention_and_scores=True
            )
            survivors = scores.clamp_min(0.5).squeeze(-1)
            A = (survivors / (survivors.sum(-1, keepdim=True) + 1e-9)).clamp(0.0, 1.0)
        elif self.readout == "gumbel_softmax":
            X_ = att @ att.permute(0, 2, 1)  # 2,N,N
            X_ = X_.permute(1, 2, 0)  # N,N 2

            S = torch.nn.functional.gumbel_softmax(X_, tau=1, hard=False, eps=1e-10, dim=-1)
            A = S.permute(2, 0, 1)

        else:
            raise NotImplementedError(f"Don't know{self.readout}")

        return A
