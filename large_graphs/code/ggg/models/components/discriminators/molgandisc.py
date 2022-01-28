import torch
from torch import nn as nn


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, out_dim, dropout_rate=0.0):
        super(GraphConvolution, self).__init__()

        layers = []
        layers.append(nn.Linear(input_dim, out_dim))
        self.linear = nn.Sequential(*layers)

        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, inputs, activation=None):
        adjacency_tensor, hidden_tensor, node_tensor = inputs
        adj = adjacency_tensor

        annotations = (
            torch.cat((hidden_tensor, node_tensor), -1)
            if hidden_tensor is not None
            else node_tensor
        )

        output = self.linear(annotations)
        output = torch.matmul(adj, output)
        output = output + self.linear(annotations)
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class GraphAggregation(nn.Module):
    def __init__(self, in_features, out_features, n_dim, dropout_rate=0):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(
            nn.Linear(in_features + n_dim, out_features), nn.Sigmoid()
        )
        self.tanh_linear = nn.Sequential(
            nn.Linear(in_features + n_dim, out_features), nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, activation):
        i = self.sigmoid_linear(inputs)  # i: BxNx128
        j = self.tanh_linear(inputs)  # j: BxNx128
        output = torch.sum(torch.mul(i, j), 1)  # output: Bx128
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class MolGAN_Discriminator(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        conv_channels=None,
        readout_hidden=64,
        swish=False,
        spectral_norm=None,
    ):
        super(MolGAN_Discriminator, self).__init__()

        auxiliary_dim = 128
        self.layers_ = [[node_feature_dim, 128], [128 + node_feature_dim, 64]]

        self.bn = torch.nn.ModuleList()
        self.gcn_layers = torch.nn.ModuleList()
        for l_ in self.layers_:
            self.gcn_layers.append(GraphConvolution(l_[0], l_[1]))

        self.agg_layer = GraphAggregation(64, auxiliary_dim, node_feature_dim)

        # Multi dense layer [128x64]
        layers = []
        for c0, c1 in zip([auxiliary_dim], [64]):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
        self.linear_layer = nn.Sequential(*layers)  # L1: 256x512 | L2: 512x256

        # Linear map [128x1]
        self.output_layer = nn.Linear(64, 1)

    def forward(self, node, adj):
        h = None
        for l in range(len(self.layers_)):
            h = self.gcn_layers[l](inputs=(adj, h, node))
        annotations = torch.cat((h, node), -1)
        h = self.agg_layer(annotations, torch.nn.Tanh())
        h = self.linear_layer(h)

        output = self.output_layer(h)

        return output