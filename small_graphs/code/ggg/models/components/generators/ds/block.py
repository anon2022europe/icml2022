import torch
from torch import nn as nn


class DS_block(nn.Module):
    def __init__(self, MLP_X_dim: list(), MLP_Q_dim: list(), dropout_rate=0.0):
        super().__init__()

        self.x_depth = len(MLP_X_dim) - 1
        self.q_depth = len(MLP_Q_dim) - 1

        self.MLP_X, self.skip_x, self.bn_x = self.build_MLP(MLP_X_dim, dropout_rate)
        self.MLP_Q, self.skip_q, self.bn_q = self.build_MLP(MLP_Q_dim, dropout_rate)

    def build_MLP(self, dim, dropout_rate):
        depth = dim
        bn = torch.nn.ModuleList()
        layers = torch.nn.ModuleList()
        skip_c = torch.nn.ModuleList()
        for l_ in range(len(depth) - 1):
            skip_c.append(nn.Linear(depth[l_], depth[l_ + 1]))
            bn.append(torch.nn.InstanceNorm1d(depth[l_ + 1], affine=True))
            layers.append(
                FeedForward(
                    [depth[l_], depth[l_ + 1]], n_layers=2, dropout=dropout_rate
                )
            )

        return layers, skip_c, bn

    def forward(self, Z0):

        Q = Z0
        for l_ in range(self.x_depth):
            Q_ = self.MLP_X[l_](Q)
            SC = self.skip_x[l_](Q)

            output = SC + Q_
            Q = self.bn_x[l_](output.permute(0, 2, 1)).permute(0, 2, 1)

        x = Q

        y = torch.mean(x, dim=1)
        y = y.unsqueeze(dim=1).repeat(1, x.shape[1], 1)

        x_y = torch.cat((x, y), dim=2)

        Q = x_y
        for l_ in range(self.q_depth):
            Q_ = self.MLP_Q[l_](Q)
            SC = self.skip_q[l_](Q)

            output = SC + Q_
            Q = self.bn_q[l_](output.permute(0, 2, 1)).permute(0, 2, 1)

        X = Q

        return X


class FeedForward(nn.Module):
    def __init__(self, dimensions: list(), n_layers: int(), dropout=0.0):
        super().__init__()

        self.n_layers = n_layers - 1
        assert len(dimensions) == n_layers

        self.layers = torch.nn.ModuleList()
        for l_ in range(self.n_layers):
            self.layers.append(nn.Linear(dimensions[l_], dimensions[l_ + 1]))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, activation=torch.nn.ReLU()):
        for l_ in range(self.n_layers):
            x = self.dropout(activation(self.layers[l_](x)))

        return x
