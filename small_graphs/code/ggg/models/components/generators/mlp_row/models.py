r"""Functional interface"""
from __future__ import division

import torch
import torch as pt

import torch.nn as nn
import torch.distributions as td

from ggg.models.components.utilities_classes import FeedForward
from ggg.models.components.edge_readout.kernel import (
    KernelEdges,
    BiasedSigmoid,
    RescaledSoftmax,
)
from ggg.utils.utils import zero_diag
from ggg.models.components.generators.ds.block import DS_block


class MLP_generator(nn.Module):
    """Baseline invariant MLP generator that operates with a set of points
    Inputs:
    e_dim: embedding dimension, n_dim: node feature dimension, n_node_dist: node distribution,
    layers_: layers of MLP, discretization: type of discretization in Adj matrix, edge_readout_type: final readout type,
    spectral norm: boolean, bias_mode: depending on the readout, edge_bias_hidden: dimension for previous parameter,
    inner_activation: non-linear activation, temperature: for discretization, dropout_rate: rate for dropout,
    cycle_opt: different MLP architectures (fixed Z, standard)"""

    def __init__(
        self,
        embedding_dim,
        n_dim,
        n_node_dist,
        layers_,
        finetti_dim=None,
        discretization="relaxed_bernoulli",
        edge_readout_type="biased_sigmoid",
        spectral_norm=None,
        bias_mode="scalar",
        edge_bias_hidden=128,
        inner_activation=None,
        temperature=0.1,
        dropout_rate=0.0,
        cycle_opt="standard",
        seed_batch_size=None,
        trainable_z=True,
        train_fix_context=True,
        dynamic_creation=False,
        flip_finetti=False,
        finneti_MLP=False,
        replicated_Z=False,
        batch_embedding=True,
        device=None,
    ):
        super().__init__()

        self.finetti_dim = finetti_dim
        seed_batch_shape = [seed_batch_size] if seed_batch_size is not None else [1]
        self.seed_batch_shape = seed_batch_shape
        self.seed_batch_size = seed_batch_size
        self.trainable_z = trainable_z
        self.train_fix_context = train_fix_context
        self.cycle_opt = cycle_opt
        self.flip_finetti = flip_finetti
        self.dynamic_creation = dynamic_creation
        self.replicated_Z = replicated_Z

        self.n_dim = n_dim
        self.e_dim = embedding_dim
        self.z_dim = embedding_dim + 1  # for number of nodes
        self.finneti_MLP = finneti_MLP
        self.batch_embedding = batch_embedding
        self.n_node_dist = n_node_dist
        KNOWN_CYCLE_OPTS = {"standard", "finetti_noDS", "finetti_ds"}
        assert cycle_opt in KNOWN_CYCLE_OPTS

        if "finetti" in cycle_opt and not dynamic_creation:
            assert all(x is not None for x in [seed_batch_size, seed_batch_shape])
            self.seedN = self.get_N_tensor(
                seed_batch_shape, N=torch.tensor(self.n_node_dist.probs.shape[0])
            )
            self.create_params(self.seedN, seed_batch_shape, seed_batch_size)
            print("Non-dynamically creating finetti stuff")
        else:
            self.Z0_init = None
            self.finetti_u_init = None
        if (
            "finetti" in self.cycle_opt
            and not dynamic_creation
            and not self.flip_finetti
        ):
            assert self.Z0_init is not None
        elif "finetti" in self.cycle_opt and not dynamic_creation and self.flip_finetti:
            assert self.finetti_u_init is not None
        if edge_readout_type == "biased_sigmoid":
            assert bias_mode in {"nodes", "scalar"}
        if edge_readout_type == "rescaled_softmax":
            assert bias_mode in {True, False, "mult", "add"}, bias_mode

        self.depth = [self.z_dim] + layers_

        if self.cycle_opt == "standard":
            self.add_feat = 0
            self.standard_build(dropout_rate)
            self.node_readout = FeedForward(
                [self.depth[-1] + self.add_feat, n_dim],
                n_layers=2,
                dropout=dropout_rate,
            )
        elif "finetti" in self.cycle_opt:
            self.add_feat = 0
            self.finetti_build(dropout_rate)
            if self.cycle_opt == "finetti_ds":
                self.node_readout = FeedForward(
                    [self.z_dim + self.add_feat, n_dim],
                    n_layers=2,
                    dropout=dropout_rate,
                )
            else:
                self.node_readout = FeedForward(
                    [self.depth[-1] + self.add_feat, n_dim],
                    n_layers=2,
                    dropout=dropout_rate,
                )

        self.n_node_dist = n_node_dist
        self.discretization = discretization
        self.temperature = temperature

        self.edge_readout_type = edge_readout_type
        if self.edge_readout_type == "gaussian_kernel":
            self.edge_readout = KernelEdges()
        elif self.edge_readout_type == "biased_sigmoid":
            self.edge_readout = BiasedSigmoid(
                feat_dim=self.depth[-1],
                spectral_norm=spectral_norm,
                act=inner_activation,
                bias_mode=bias_mode,
            )
        elif self.edge_readout_type == "rescaled_softmax":
            self.edge_readout = RescaledSoftmax(
                feat_dim=self.depth[-1],
                spectral_norm=spectral_norm,
                bias_mode=bias_mode,
                inner_activation=inner_activation,
            )
        elif self.edge_readout_type == "QQ_sig":
            self.edge_readout = torch.nn.Sigmoid()
        else:
            self.edge_readout = None

    def create_params(self, seedN, seed_batch_shape, seed_batch_size, device=None):
        _Ns, Z0 = self.get_Z0(seedN, seed_batch_shape, seed_batch_size)
        if self.replicated_Z:
            # get only that single Z0 , we will replicate it across the batch size later
            Z0 = Z0[0].unsqueeze(0)
        if not self.batch_embedding:
            Z0 = Z0.reshape(Z0.shape[1], -1)
        finetti_u = self.get_finetti_u(seed_batch_size)
        if device:
            finetti_u = finetti_u.to(device)
            Z0 = Z0.to(device)
        self.Z0_init = torch.nn.Parameter(Z0, requires_grad=self.trainable_z)
        self.finetti_u_init = torch.nn.Parameter(
            finetti_u, requires_grad=self.trainable_z
        )

    def forward(self, Z0, finetti_u=None):

        if self.cycle_opt == "standard":
            Q = self.standard_fw(Z0)
        elif self.cycle_opt == "gru":
            Q = self.gru_fw(Z0, finetti_u)
        elif "finetti" in self.cycle_opt:
            Q = self.finetti_fw(Z0, finetti_u)

        X = self.node_readout(Q)

        if self.edge_readout_type == "QQ_sig":
            A = torch.nn.Sigmoid()(Q @ Q.transpose(1, 2))
        else:
            A = self.edge_readout(Q)

        if self.discretization == "relaxed_bernoulli":
            A = self.discretize(A)

        return X, A, Q

    def standard_build(self, dropout_rate):
        """The standard build of the invariant MLP, input set of points -> MLP -> moved set of points,
        MLP with norm and skip connection

        :param dropout_rate:
        :return model build:
        """

        self.bn = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.skip_c = torch.nn.ModuleList()
        for l_ in range(len(self.depth) - 1):
            self.bn.append(torch.nn.InstanceNorm1d(self.depth[l_], affine=True))
            self.skip_c.append(nn.Linear(self.depth[l_], self.depth[l_ + 1]))
            self.layers.append(
                FeedForward(
                    [self.depth[l_], self.depth[l_ + 1]],
                    n_layers=2,
                    dropout=dropout_rate,
                )
            )

    def standard_fw(self, Z0):
        """Forward standard function

        :param Z0:noise
        :return Q:embeddings
        """
        Q = Z0
        for l_ in range(len(self.layers)):
            Q = self.bn[l_](Q.permute(0, 2, 1)).permute(0, 2, 1)
            Q_ = self.layers[l_](Q, activation=pt.nn.ReLU())
            SC = self.skip_c[l_](Q)
            Q = Q_ + SC

        return Q

    def finetti_build(self, dropout_rate):
        """Starting as a fixed initial matrix, moved points with an MLP

        :param dropout_rate:
        :return model for fixed noise vectors:
        """

        if self.finneti_MLP:
            layers = []
            for c0, c1 in zip([self.finetti_dim, 64, 128], [64, 128, self.finetti_dim]):
                layers.append(nn.Linear(c0, c1))
                layers.append(nn.ReLU())
            self.MLP0 = nn.Sequential(*layers)

        self.depth = [self.depth[0] + self.finetti_dim] + self.depth[1:]
        self.bn = torch.nn.ModuleList()
        self.layers = torch.nn.ModuleList()
        self.skip_c = torch.nn.ModuleList()
        for l_ in range(len(self.depth) - 1):
            self.bn.append(torch.nn.InstanceNorm1d(self.depth[l_], affine=True))
            self.skip_c.append(nn.Linear(self.depth[l_], self.depth[l_ + 1]))
            self.layers.append(
                FeedForward(
                    [self.depth[l_], self.depth[l_ + 1]],
                    n_layers=2,
                    dropout=dropout_rate,
                )
            )

        if self.cycle_opt == "finetti_ds":
            self.ds = DS_block(
                [self.depth[-1], 128], [128 * 2, self.z_dim], dropout_rate=dropout_rate
            )

    def finetti_fw(self, Z0, u):
        """Forward function for finetti build

        :param Z0:noise vector
        :param u:context vector
        :return Q:node embeddings
        """
        if self.finneti_MLP:
            u_ = self.MLP0(u)
        else:
            u_ = u

        # noise = torch.randn([Z0.size()[0], Z0.size()[1], 10])
        Q = torch.cat((Z0, u_.unsqueeze(dim=1).repeat(1, Z0.size()[1], 1)), dim=-1)

        for l_ in range(len(self.layers)):
            Q = self.bn[l_](Q.permute(0, 2, 1)).permute(0, 2, 1)
            Q_ = self.layers[l_](Q, activation=pt.nn.ReLU())
            SC = self.skip_c[l_](Q)
            Q = Q_ + SC

        if self.cycle_opt == "finetti_ds":
            Q = self.ds(Q)

        return Q

    def discretize(self, A):
        relaxedA = td.RelaxedBernoulli(self.temperature, probs=A).rsample()
        # hard relaxed bernoulli= create 0 vector with gradients attached, add to rounded values
        grads_only = relaxedA - relaxedA.detach()
        Ar = relaxedA.round() + grads_only
        # rezero and resymmetrize
        Az = zero_diag(Ar)
        Atriu = torch.triu(Az)
        A = Atriu + Atriu.permute(0, 2, 1)

        return A

    def rezero_superflous(self, A, N, X, batch_size):
        # zero out superflous node/Adj entries
        # NOTE: need to do it like this since in-place assignment causes some anomaly with autograd
        # assuming N is 1 to MAX_NODE, no + 1 needed here
        def rezero(X, nmax):
            n = nmax.reshape(-1, 1, 1)
            o = torch.ones_like(X)
            cm_row = o.cumsum(dim=-1)
            row_mask = (cm_row <= n).type_as(X)
            cm_col = o.cumsum(dim=-2)
            col_mask = (cm_col <= n).type_as(X)
            mask = row_mask * col_mask
            return X * mask

        A_out = rezero(A, N)
        X_out = rezero(X, N)
        return A_out, X_out

    def get_Z0(self, N, batch_shape, batch_size, device=None):
        nmax = len(self.n_node_dist.probs)
        shape = [nmax, self.e_dim]
        shape = batch_shape + shape
        noise = torch.randn(*shape, device=N.device if device is None else device)
        if batch_size is not None:
            Ns = N[:, None, None] * torch.ones([batch_size, nmax, 1], device=N.device)
            for b in range(batch_size):
                nm = N[b].int().item()
                noise[b, nm:, :] = 0.0
                Ns[b, nm:, :] = 0.0
        else:
            Ns = N * torch.ones(batch_shape + [nmax, 1], device=N.device)
        assert Ns.dim() == noise.dim()
        Z0 = torch.cat([noise, Ns], -1)
        return Ns, Z0

    def get_finetti_u(self, batch_size, device=None):
        shape = [self.finetti_dim]
        shape = [batch_size] + shape
        if device:
            noise = torch.randn(*shape, device=device)
        else:
            noise = torch.randn(*shape)
        return noise

    def sample(
        self,
        batch_size=None,
        N=None,
        save_G=False,
        device=None,
        realX=None,
        conditional=False,
        external_finetti_u=None,
    ):
        batch_shape = [batch_size] if batch_size is not None else [1]
        N = self.get_N_tensor(batch_shape, device=device, N=N)

        Ns, Z0 = self.get_Z0(N, batch_shape, batch_size, device=device)

        # Fixed Z0, nonfixed finetti_u=> Fixed node embedding, changing universal embedding
        if "finetti" in self.cycle_opt:
            if self.dynamic_creation and self.Z0_init is None:
                # TODO: REALLY don't want to have dynamic creation...
                self.create_params(N, batch_shape, batch_size, device)
            # fixed Z, various u
            if not self.flip_finetti:
                # clone preserve gradient but we don't actually change the parameter by hand
                if device is not None and self.Z0_init.device != device:
                    self.Z0_init = self.Z0_init.to(device)
                Z = self.Z0_init.clone()

                if self.replicated_Z:
                    Z: torch.Tensor
                    # replcate across batch dim
                    Z = Z.repeat([Ns.shape[0], 1, 1])
                if not self.batch_embedding:
                    # TODO: this is mathematically meaningless, but for consistencies sake
                    Z = Z.reshape(Ns.shape[0], Z.shape[0], -1)
                if self.train_fix_context and self.training:
                    # NOTE: TODO: important that you call .eval() before sampling when this is done
                    if device is not None and self.finetti_u_init.device != device:
                        self.finetti_u_init = self.finetti_u_init.to(device)
                    finetti_u = self.finetti_u_init.clone()
                else:
                    finetti_u = self.get_finetti_u(batch_size, device)
            else:
                # clone preserve gradient but we don't actually change the parameter by hand
                if device is not None and self.finetti_u_init.device != device:
                    self.finetti_u_init = self.finetti_u_init.to(device)
                finetti_u = self.finetti_u_init.clone()
                if self.train_fix_context and self.training:
                    # NOTE: TODO: important that you call .eval() before sampling when this is done
                    if device is not None and self.Z0_init.device != device:
                        self.Z0_init = self.Z0_init.to(device)
                    Z = self.Z0_init.clone()
                    if self.replicated_Z:
                        Z: torch.Tensor
                        # replcate across batch dim
                        Z = Z.repeat([Ns.shape[0], 1, 1])
                else:
                    Z = Z0

            if external_finetti_u is not None:
                finetti_u = external_finetti_u
            X, A, Q = self.forward(Z, finetti_u)
        else:
            if conditional:
                Z0 = torch.cat([Z0, realX], dim=-1)
            Z = Z0
            X, A, Q = self.forward(Z0)

        # re-append node_numbers so things line up for discriminator
        X = torch.cat([X, Ns.to(X.device)], -1)

        A_out, X_out = self.rezero_superflous(A, N, X, batch_size)

        # TODO: furhter disentangle this
        if save_G:
            if "finetti" in self.cycle_opt:
                return Z0, finetti_u
            else:
                return Z0
        else:
            if "finetti" in self.cycle_opt:
                return X_out, A_out, Z, finetti_u, Q
            else:
                return X_out, A_out, Z, None, Q

    def get_N_tensor(self, batch_shape, device=None, N=None):
        if N is not None:
            N = torch.ones(batch_shape, dtype=torch.int).type_as(N) * N
        else:
            N = self.n_node_dist.sample(batch_shape) + 1

        N: torch.Tensor
        assert torch.is_tensor(N)
        # a graph with 1 node is not a graph...
        N = torch.max(N, torch.ones_like(N) * 2)
        if device is not None:
            N = N.to(device)
        return N
