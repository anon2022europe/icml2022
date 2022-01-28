r"""Functional interface"""
from __future__ import division

import attr
from torch import nn as nn

from ggg.models.components.generators.att.discretization import DiscretizationHpars
from ggg.models.components.generators.att.readouts import (
    EdgeReadoutHpars,
    NodeReadoutHpars,
)
from ggg.models.components.generators.att.roots import GenRootHpars
from ggg.models.components.generators.att.trunks import GenTrunkHpars
from ggg.models.components.utilities_classes import NodeFeatNorm
import torch as pt


class Generator(nn.Module):
    def __init__(
        self,
        num_node_weights: pt.Tensor,
        root_hpars: GenRootHpars,
        trunk_hpars: GenTrunkHpars,
        edge_readout_hpars: EdgeReadoutHpars,
        node_readout_hpars: NodeReadoutHpars,
        discretization_hpars: DiscretizationHpars,
        hist_hook=None,
    ):
        super(Generator, self).__init__()

        self.root = root_hpars.make(num_node_weights)
        self.root_hpars = root_hpars

        self.trunk = trunk_hpars.make()
        self.trunk_hpars = trunk_hpars

        self.edge_readout_hpars = edge_readout_hpars
        self.edge_readout = edge_readout_hpars.make()

        self.node_readout_hpars = node_readout_hpars
        self.node_readout = node_readout_hpars.make()

        self.discretization = discretization_hpars.make()
        self.discretization_hpars = discretization_hpars
        self.hist_hook = hist_hook

    def root_forward(self, batch_size=1, device=None, Z=None, X=None, A=None, N=None):
        if X is None:
            X, A, N, Z = self.root.forward(batch_size, device, Z=Z, N=N)
        self.maybe_histogram(A, N, X, "root")
        return X, A, N, Z

    def feature_forward(
        self, batch_size=1, device=None, Z=None, X=None, A=None, N=None
    ):
        X, A, N, Z = self.root_forward(batch_size, device, Z, X, A, N)
        X, A, N = self.trunk(X=X, N=N, A=A)
        self.maybe_histogram(A, N, X, "features")
        return X, A, N, Z

    def readout(self, X, A, N):
        nodes = self.node_readout(X=X, A=A, N=N)
        edges = self.edge_readout(X=X, A=A, N=N)
        if self.discretization:
            edges = self.discretization(edges)
        self.maybe_histogram(A, N, X, "readout")
        return nodes, edges, N

    def maybe_histogram(self, A, N, X, name):
        if self.hist_hook is not None:
            for n, t in zip(["X", "A", "N"], [X, A, N]):
                if t is not None:
                    self.hist_hook(f"{name}-{n}", t)

    def forward(self, batch_size=1, device=None, Z=None, X=None, A=None, N=None):
        X, A, N, Z = self.feature_forward(batch_size, device, Z, X, A, N)
        X, A, N = self.readout(X, A, N)
        return X, A, N, Z

    def sample(self, batch_size=1, device=None, Z=None, X=None, A=None, N=None):
        return self.forward(batch_size, device, Z, X, A, N)
