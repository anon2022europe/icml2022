# taken from https://github.com/CyberZHG/torch-multi-head-attention/blob/master/torch_multi_head_attention/multi_head_attention.py
from copy import deepcopy
from logging import warning
from warnings import warn

import torch
import torch as pt
import torch.nn as nn

from pdb import set_trace
from ggg.models.components.attention.scaled_dot_product import (
    ScaledDotProductAttention,
)
from ggg.models.components.utilities_classes import Swish, NodeFeatNorm
from ggg.models.components.pointnet_st import LinearTransmissionLayer
from ggg.models.components.spectral_norm import sn_wrap

__all__ = ["MultiHeadAttention"]

try:
    from smyrf.torch.attn import SmyrfAttention, lsh_clustering

    class OurSmyrf(SmyrfAttention):
        def forward(
            self,
            queries,
            keys,
            values,
            attn_mask=None,
            progress=False,
            norm_factor=1,
            return_attn_map=False,
            return_attention_and_scores=False,
            att_sig="softmax",
        ):
            bs, q_seqlen, dim = queries.shape
            bs, k_seqlen, dim = keys.shape
            v_dim = values.shape[-1]
            assert queries.device == keys.device, "Queries, keys in different devices"
            device = queries.device

            # prepare mask if not None
            if attn_mask is not None:
                # We expect first dimension to be batch_size and second dimension seq. length
                if len(attn_mask.shape) == 1:
                    attn_mask = attn_mask.unsqueeze(0)
                # repeat for n_hashes, heads
                attn_mask = attn_mask.unsqueeze(0).repeat(
                    self.n_hashes, queries.shape[0] // attn_mask.shape[0], 1
                )

            with torch.no_grad():
                # XBOX+ transform
                self.xbox_plus.set_norms(queries, keys)
                Queries = self.xbox_plus.Q(queries)
                Keys = self.xbox_plus.K(keys)

                num_clusters = Queries.shape[1] // self.q_attn_size
                assert num_clusters == (
                    Keys.shape[1] // self.k_attn_size
                ), "Unequal number of clusters for queries and keys."

                if self.clustering_algo == "lsh":
                    q_positions, k_positions = lsh_clustering(
                        Queries, Keys, **self.clustering_params, attn_mask=attn_mask
                    )
                else:
                    raise NotImplementedError("This algorithm is not supported")

                q_positions = q_positions.reshape(self.n_hashes, bs, -1)
                k_positions = k_positions.reshape(self.n_hashes, bs, -1)

            # free memory
            del Queries
            del Keys

            q_rev_positions = torch.argsort(q_positions, dim=-1)
            q_offset = torch.arange(bs, device=queries.device).unsqueeze(-1) * q_seqlen
            k_offset = torch.arange(bs, device=queries.device).unsqueeze(-1) * k_seqlen

            q_flat = (q_positions + q_offset).reshape(-1)
            k_flat = (k_positions + k_offset).reshape(-1)

            # sorted queries, keys, values
            s_queries = (
                queries.reshape(-1, dim)
                .index_select(0, q_flat)
                .reshape(-1, self.q_attn_size, dim)
            )
            s_keys = (
                keys.reshape(-1, dim)
                .index_select(0, k_flat)
                .reshape(-1, self.k_attn_size, dim)
            )
            s_values = (
                values.reshape(-1, v_dim)
                .index_select(0, k_flat)
                .reshape(-1, self.k_attn_size, v_dim)
            )

            inner = s_queries @ s_keys.transpose(2, 1)
            inner = inner / norm_factor

            # mask out attention to padded tokens
            if attn_mask is not None:
                inner = (
                    attn_mask.reshape(-1)[k_flat]
                    .reshape(-1, self.k_attn_size)
                    .unsqueeze(1)
                    + inner
                )

            # free memory
            if not return_attn_map:
                del q_positions, k_positions

            # softmax denominator
            dots_logsumexp = torch.logsumexp(inner, dim=-1, keepdim=True)
            # softmax
            dots = torch.exp(inner - dots_logsumexp)
            # dropout
            dots = self.dropout(dots)

            # n_hashes outs
            bo = (dots @ s_values).reshape(self.n_hashes, bs, q_seqlen, -1)

            # undo sort
            q_offset = (
                torch.arange(bs * self.n_hashes, device=queries.device).unsqueeze(-1)
                * q_seqlen
            )
            q_rev_flat = (q_rev_positions.reshape(-1, q_seqlen) + q_offset).reshape(-1)
            o = (
                bo.reshape(-1, v_dim)
                .index_select(0, q_rev_flat)
                .reshape(self.n_hashes, bs, q_seqlen, -1)
            )

            slogits = dots_logsumexp.reshape(self.n_hashes, bs, -1)
            logits = torch.gather(slogits, 2, q_rev_positions)

            # free memory
            del q_rev_positions

            if att_sig == "sigmoid":
                probs = torch.nn.Sigmoid()(logits)
                out = torch.sum(o * probs.unsqueeze(-1), dim=0)
            elif att_sig == "softmax":
                probs = torch.exp(logits - torch.logsumexp(logits, dim=0, keepdim=True))
                out = torch.sum(o * probs.unsqueeze(-1), dim=0)
            else:
                raise ValueError(f"Unkown att_sig {att_sig}")

            if return_attn_map:
                return out, (q_positions, k_positions)
            elif return_attention_and_scores:
                return out, probs, logits
            else:
                return out


except:
    warn("Smyrf not found, can't use O(logN N) attention")


class SmyrfWrapper(nn.Module):
    def __init__(self, smyrfdict):
        super().__init__()
        self.smyrfdict = smyrfdict

    def forward(
        self,
        queries,
        keys,
        values,
        attn_mask=None,
        progress=False,
        norm_factor=1,
        return_attn_map=False,
        att_sig="softmax",
        return_attention_and_scores=False,
    ):
        assert queries.dim() == 3
        smyrf_args = deepcopy(self.smyrfdict)
        cluster_size = smyrf_args.pop("cluster_size")
        for x in "qk":
            smyrf_args[f"{x}_cluster_size"] = cluster_size
        smyrf = OurSmyrf(**smyrf_args)
        return smyrf.forward(
            queries,
            keys,
            values,
            attn_mask=attn_mask,
            progress=progress,
            norm_factor=norm_factor,
            return_attn_map=return_attn_map,
            att_sig=att_sig,
            return_attention_and_scores=return_attention_and_scores,
        )


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        in_features,
        out_features=None,
        num_heads=1,
        bias=True,
        activation="relu",
        mode="QK",
        score_function="sigmoid",
        spectral_norm=None,
        name=None,
        norm_type="layer",
        rezero=False,
        smyrf=None,
    ):
        """Multi-head attention.

        :param in_features: Size of each input sample.
        :param num_heads: Number of heads.
        :param bias: Whether to use the bias term.
        :param activation: The activation after each linear transformation.
        """
        super(MultiHeadAttention, self).__init__()
        acts = {"relu": torch.nn.ReLU, "swish": Swish, "leaky": torch.nn.LeakyReLU}
        self.norm_type = norm_type
        if type(activation) is str:
            activation = acts[activation]()
        self.smyrf = smyrf
        while in_features % num_heads != 0:
            warning(f"Dropping num_heads from {num_heads} to {num_heads-1} trying to make it compatible with in_dim {in_features}")
            num_heads=num_heads-1
            #raise ValueError(
            #    "`in_features`({}) should be divisible by `head_num`({})".format(
            #        in_features, num_heads
            #    )
            #)
        self.mode = mode
        if smyrf is None or len(smyrf) == 0:
            self.attn = ScaledDotProductAttention()
            out_in_features = in_features
        else:
            self.attn = SmyrfWrapper(smyrf)
            out_in_features = smyrf["n_hashes"]
            out_in_features = in_features
        self.in_features = in_features
        if out_features is None:
            out_features = in_features
        self.out_features = out_features
        self.head_num = num_heads
        self.activation = activation
        self.bias = bias
        self.score_function = score_function
        self.name = name

        self.linear_q = sn_wrap(
            nn.Linear(in_features, in_features, bias), spectral_norm
        )
        if self.mode == "QK":
            self.linear_k = sn_wrap(
                nn.Linear(in_features, in_features, bias), spectral_norm
            )
        else:
            self.linear_k = None
        self.linear_v = sn_wrap(
            nn.Linear(in_features, in_features, bias), spectral_norm
        )
        self.mlp = nn.Sequential(
            sn_wrap(nn.Linear(out_in_features, out_features, bias), spectral_norm),
            activation,
            sn_wrap(nn.Linear(out_features, out_features, bias), spectral_norm),
        )
        self.q_prenorm = NodeFeatNorm(in_features, mode=self.norm_type)
        self.out_prenorm = NodeFeatNorm(out_in_features, mode=self.norm_type)
        if rezero:
            self.gate=pt.nn.Parameter(pt.zeros([]))
        else:
            self.gate=None

    def forward(self, q, k=None, v=None, mask=None, return_attention_and_scores=False):
        # q,k,v: tensors of batch_size, seq_len, in_feature
        q_skip = q
        q = self.q_prenorm(q)
        if k is None:
            k = q
        if v is None:
            v = q
        ql = self.linear_q(q)
        vl = self.linear_v(v)
        if self.mode == "QQ":
            kl = ql
        elif self.mode == "QK":
            kl = self.linear_k(k)

        if self.smyrf is None or len(self.smyrf) == 0:
            ql = self._reshape_to_batches(ql)
            kl = self._reshape_to_batches(kl)
            vl = self._reshape_to_batches(vl)
            if mask is not None:
                mask = mask.repeat(self.head_num, 1, 1)
            if return_attention_and_scores:
                y, _attn, _attn_scores = self.attn(
                    ql,
                    kl,
                    vl,
                    mask,
                    return_attention_and_scores=return_attention_and_scores,
                    att_sig=self.score_function,
                )
            else:
                y = self.attn(
                    ql,
                    kl,
                    vl,
                    mask,
                    return_attention_and_scores=return_attention_and_scores,
                    att_sig=self.score_function,
                )
            y_att = self._reshape_from_batches(y)
        else:
            if return_attention_and_scores:
                y_att, _attn, _attn_scores = self.attn(
                    ql,
                    kl,
                    vl,
                    mask,
                    return_attention_and_scores=return_attention_and_scores,
                    att_sig=self.score_function,
                )
            else:
                y_att = self.attn(
                    ql,
                    kl,
                    vl,
                    mask,
                    return_attention_and_scores=return_attention_and_scores,
                    att_sig=self.score_function,
                )

        if self.gate is not None:
            y_att=self.gate*y_att
        y_att = y_att + q_skip
        y_att = self.out_prenorm(y_att)
        yo = self.mlp(y_att)

        assert yo.shape[-1] == self.out_features
        if return_attention_and_scores:
            assert _attn.shape[-2] == yo.shape[-2]
            assert _attn.shape[-1] == yo.shape[-2]
            return yo, _attn, _attn_scores
        else:
            return yo

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

    def _reshape_from_batches(self, x):
        batch_size, seq_len, in_feature = x.size()
        batch_size //= self.head_num
        out_dim = in_feature * self.head_num
        return (
            x.reshape(batch_size, self.head_num, seq_len, in_feature)
            .permute(0, 2, 1, 3)
            .reshape(batch_size, seq_len, out_dim)
        )

    def extra_repr(self):
        return "in_features={}, head_num={}, bias={}, activation={}".format(
            self.in_features,
            self.head_num,
            self.bias,
            self.activation,
        )


class AttentionPool(pt.nn.Module):
    def __init__(
        self,
        node_feat,
        in_features,
        pool="sum",
        weight_score_function="softmax",
        dim=-2,
        head_num=1,
        bias=True,
        activation=None,
        out_activation=None,
        mode="QK",
        score_function="sigmoid",
        spectral_norm=None,
        name=None,
        norm_type="layer",
        smyrf=None,
    ):
        super().__init__()
        self.pool = pool
        self.weight_score_function = weight_score_function
        self.dim = dim
        self.attn = MultiHeadAttention(
            in_features,
            out_features=1,
            num_heads=head_num,
            bias=bias,
            activation=activation,
            mode=mode,
            score_function=score_function,
            spectral_norm=spectral_norm,
            name=name,
            norm_type=norm_type,
            smyrf=smyrf,
        )

    def forward(self, X):
        attn = self.attn(X, X, X)
        if self.weight_score_function == "softmax":
            attn = attn.softmax(self.dim)
        elif self.weight_score_function == "sigmoid":
            attn = pt.nn.Sigmoid()(attn)
        else:
            raise NotImplementedError(f"Don't know {self.weight_score_function}")
        Xattn = X * attn
        if self.pool == "sum":
            ret = Xattn.sum(self.dim)
        else:
            raise NotImplementedError(f"Don't know {self.pool}")
        return ret
