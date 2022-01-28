import math

import torch as pt
from torch import nn

from ggg.models.components.spectral_norm import sn_wrap


class PointNetBlock(nn.Module):
    def __init__(
        self,
        input_feat_dim,
        output_feat_dim,
        no_B=False,
        spectral_norm=None,
        activation=None,
    ):
        super().__init__()
        self.input_feat_dim = input_feat_dim
        output_feat_dim = output_feat_dim
        self.A = nn.Parameter(pt.Tensor(input_feat_dim, output_feat_dim))
        self.cT = nn.Parameter(pt.Tensor(1, output_feat_dim))
        self.activation = activation
        if no_B:
            self.register_parameter("B", None)
        else:
            self.B = nn.Parameter(pt.Tensor(input_feat_dim, output_feat_dim))
        self.reset_parameters()
        self.spectral_norm = spectral_norm
        self._sns = []
        if spectral_norm is not None:
            self._sns.extend(
                [
                    sn_wrap(self, spectral_norm, name="A"),
                    sn_wrap(self, spectral_norm, name="cT"),
                ]
            )
            if not no_B:
                self._sns.append(sn_wrap(self, spectral_norm, name="B"))

            for s in self._sns:
                for n, p in s.named_parameters():
                    self.register_parameter(n.replace("module.", ""), p)

    def reset_parameters(self):
        # following https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        pt.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        fan_in, _ = pt.nn.init._calculate_fan_in_and_fan_out(self.A)
        bound = 1 / math.sqrt(fan_in)
        pt.nn.init.uniform_(self.cT, -bound, bound)
        if self.B is not None:
            pt.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))

    def forward(self, X):
        """
        Expecting shape B N F or N F
        following eq 8 in https://openreview.net/pdf?id=HkxTwkrKDB
        :param X:
        :return:
        """
        if self.spectral_norm:
            for n in self._sns:
                n._update_u_v()
        xa = X @ self.A
        N = xa.shape[-2]
        o1 = pt.ones(N, 1, device=X.device)
        c = o1 @ self.cT
        if self.B is not None:
            xb = 1 / N * o1 @ o1.t() @ X @ self.B
            out = xa + xb + c
        else:
            out = xa + c
        if self.activation:
            out = self.activation(out)
        return out


class LinearTransmissionLayer(nn.Module):
    def __init__(
        self,
        input_feat_dim,
        output_feat_dim,
        dropout=None,
        activation=None,
        spectral_norm=None,
    ):
        super().__init__()
        self.input_feat_dim = input_feat_dim
        self.output_feat_dim = output_feat_dim
        self.B = nn.Parameter(pt.Tensor(self.input_feat_dim, self.output_feat_dim))
        self.cT = nn.Parameter(pt.Tensor(1, self.output_feat_dim))
        self.reset_parameters()
        if dropout:
            self.dropout = pt.nn.Dropout(dropout)
        else:
            self.dropout = None
        self.activation = activation
        self.spectral_norm = spectral_norm
        self._sns = []
        if spectral_norm is not None:
            self._sns.extend(
                [
                    sn_wrap(self, spectral_norm, name="B"),
                    sn_wrap(self, spectral_norm, name="cT"),
                ]
            )
            for s in self._sns:
                for n, p in s.named_parameters():
                    self.register_parameter(n.replace("module.", ""), p)

    def reset_parameters(self):
        # following https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/linear.py
        pt.nn.init.kaiming_uniform_(self.B, a=math.sqrt(5))
        fan_in, _ = pt.nn.init._calculate_fan_in_and_fan_out(self.B)
        bound = 1 / math.sqrt(fan_in)
        pt.nn.init.uniform_(self.cT, -bound, bound)

    def forward(self, X):
        """
        Expecting shape B N F or N F
        following eq 8 in https://openreview.net/pdf?id=HkxTwkrKDB
        :param X:
        :return:
        """
        if self.dropout:
            X = self.dropout(X)
        N = X.shape[-2]
        o1 = pt.ones(N, 1, device=X.device)
        c = o1 @ self.cT
        xb = 1 / N * (o1 @ o1.t()) @ X @ self.B
        out = xb + c
        if self.activation:
            out = self.activation(out)
        return out

    def extra_repr(self) -> str:
        return f"in_features={self.B.shape[0]},out_features={self.B.shape[1]}"
