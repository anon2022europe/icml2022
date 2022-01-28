r"""Functional interface"""
from __future__ import division

import math
from typing import Union

import torch
import torch.nn as nn
import torch.distributions as td

from torch.distributions import Normal

from ggg.models.components.attention.MultiHead import MultiHeadAttention
from ggg.models.components.attention.MultiHeadDeep import MultiHeadDeepAttention

from ggg.models.components.pointnet_st import PointNetBlock, LinearTransmissionLayer
from ggg.models.components.node_readout.attention import AttentionNodeReadout
from ggg.models.components.edge_readout.kernel import (
    KernelEdges,
    BiasedSigmoid,
    RescaledSoftmax,
)
from ggg.utils.utils import zero_diag
from ggg.models.components.node_readout.attention import AttentionNodeReadout
import torch.distributions as td

from torch_geometric.nn import DenseGINConv
from ggg.models.components.generators.ds.block import DS_block


def triangles_(adj_matrix, k_, prev_k=None):
    if prev_k is None:
        k_matrix = torch.matrix_power(adj_matrix.float(), k_)
    else:
        k_matrix = prev_k @ adj_matrix.float()
    egd_l = torch.diagonal(k_matrix, dim1=-2, dim2=-1)
    return egd_l, k_matrix


class PointNetSTGen(nn.Module):
    def __init__(
        self,
        embedding_dim,
        node_out_dim,
        n_node_dist,
        n_set_layers=2,
        num_heads=1,
        finetti_dim=None,
        inner_activation=None,
        out_activation=None,
        edge_readout_type="biased_sigmoid",
        attention_mode="QQ",
        score_function="sigmoid",
        discretization="relaxed_bernoulli",  # alternatively gumbel softmax et
        temperature=0.1,
        spectral_norm=None,
        bias_mode="nodes",  # for rescaled softmax: True/false, for biased sigmoid:scalar/nodes
        edge_bias_hidden=128,
        cycle_opt="standard",
        seed_batch_size=None,
        trainable_z=True,
        train_fix_context=True,
        dynamic_creation=False,
        flip_finetti=False,
        finneti_MLP=False,
        replicated_Z=False,
        smyrf=None,
    ):
        """
        TODO: should make this batch-size independent
        :param seed_batch_shape:
        """
        super().__init__()
        self.e_dim = embedding_dim
        self.finetti_dim = finetti_dim
        seed_batch_shape = [seed_batch_size] if seed_batch_size is not None else [1]
        self.seed_batch_shape = seed_batch_shape
        self.seed_batch_size = seed_batch_size
        self.trainable_z = trainable_z
        self.train_fix_context = train_fix_context
        self.cycle_opt = cycle_opt
        self.replicated_Z = replicated_Z

        KNOWN_CYCLE_OPTS = {"standard", "finetti_noDS", "finetti_ds"}
        assert cycle_opt in KNOWN_CYCLE_OPTS
        self.batch_embedding = True
        self.n_node_dist = n_node_dist

        if "finetti" in cycle_opt:
            assert all(x is not None for x in [seed_batch_size, seed_batch_shape])
            self.create_params(seedN, seed_batch_shape, seed_batch_size)
            print("Non-dynamically creating finetti stuff")
        else:
            self.node_embed_Z0_init = None
            self.context_vec_init = None
        if "finetti" in self.cycle_opt:
            assert self.node_embed_Z0_init is not None

        if edge_readout_type == "biased_sigmoid":
            assert bias_mode in {"nodes", "scalar"}
        if edge_readout_type == "rescaled_softmax":
            assert bias_mode in {True, False, "mult", "add"}, bias_mode
        self.e_dim = embedding_dim
        self.node_out_dim = node_out_dim
        self.attention_e_dim = self.e_dim + 1  # +1 for number of nodes
        self.attention_mode = attention_mode
        self.discretization = discretization
        self.temperature = temperature
        self.num_heads = num_heads
        self.cycle_opt = cycle_opt
        self.finneti_MLP = finneti_MLP
        self.score_function = score_function

        if self.cycle_opt == "standard":
            self.add_feat = 0
            self.fnti_ds_noise = 0
            self.standard_build(
                n_set_layers, inner_activation, out_activation, spectral_norm
            )
        elif "finetti" in self.cycle_opt:
            self.add_feat = self.finetti_dim
            self.finetti_cbuild(
                n_set_layers, inner_activation, out_activation, spectral_norm
            )

        self.node_readout = AttentionNodeReadout(
            feat_dim=self.attention_e_dim + self.add_feat,
            node_attrib_dim=node_out_dim,
            inner_activation=inner_activation,
            out_activation=out_activation,
            attention_mode=attention_mode,
            spectral_norm=spectral_norm,
            score_function=score_function,
        )

        self.edge_readout_type = edge_readout_type
        if self.edge_readout_type == "gaussian_kernel":
            self.edge_readout = KernelEdges()
        elif self.edge_readout_type == "QQ_sig":
            self.edge_readout = torch.nn.Sigmoid()
        elif self.edge_readout_type == "biased_sigmoid":
            self.edge_readout = BiasedSigmoid(
                feat_dim=self.attention_e_dim + self.add_feat,
                spectral_norm=spectral_norm,
                act=inner_activation,
                bias_mode=bias_mode,
            )
        elif self.edge_readout_type == "rescaled_softmax":
            print(bias_mode)
            self.edge_readout = RescaledSoftmax(
                feat_dim=self.attention_e_dim + self.add_feat,
                spectral_norm=spectral_norm,
                bias_mode=bias_mode,
                inner_activation=inner_activation,
            )
        elif self.edge_readout_type == "attention_weights":
            self.edge_readout = MultiHeadAttention(
                in_features=self.attention_e_dim + self.add_feat,
                num_heads=1,
                activation=inner_activation,
                mode=self.attention_mode,
                score_function="sigmoid",
                spectral_norm=spectral_norm,
                smyrf=smyrf,
            )
        else:
            self.edge_readout = None

        self.n_node_dist = n_node_dist

    def create_params(self, seed_batch_shape, seed_batch_size, device=None):
        # get full batch with max_N node embedings
        _Ns = self.sample_Ns(
            seed_batch_size, given_N=torch.tensor(self.n_node_dist.probs.shape[0])
        )
        Z0 = self.sample_node_embed_Z0(_Ns, device=None)
        if self.replicated_Z:
            # get only that single Z0 , we will replicate it across the batch size later
            Z0 = Z0[0].unsqueeze(0)
        if not self.batch_embedding:
            Z0 = Z0.reshape(Z0.shape[1], -1)
        context_vec = self.sample_context_vec(seed_batch_size)
        if device:
            context_vec = context_vec.to(device)
            Z0 = Z0.to(device)
        self.node_embed_Z0_init = torch.nn.Parameter(Z0, requires_grad=self.trainable_z)
        self.context_vec_init = torch.nn.Parameter(
            context_vec, requires_grad=self.trainable_z
        )

    def sample_Ns(
        self, batch_size: Union[int, torch.Tensor], given_N=None, device=None
    ):
        # fooo
        batch_shape = [batch_size]
        if given_N is not None:
            # all examples will have the given n
            N = torch.ones(batch_shape, dtype=torch.int).type_as(given_N) * given_N
        else:
            # sample random node numbers
            N = self.n_node_dist.sample(batch_shape) + 1

        N: torch.Tensor
        assert torch.is_tensor(N)
        # a graph with 1 node is not a graph...
        N = torch.max(N, torch.ones_like(N) * 2)
        if device is not None:
            N = N.to(device)
        return N

    def standard_build(
        self, n_set_layers, inner_activation, out_activation, spectral_norm
    ):
        self.norm_pre = torch.nn.ModuleList()
        self.norm_post = torch.nn.ModuleList()
        self.pre = torch.nn.ModuleList()
        self.post = torch.nn.ModuleList()

        assert (
            n_set_layers >= 3 and n_set_layers % 2 == 1
        ), "Need at least 1 layer pre and post Linear Transmission layer, and an odd number overall since we stay symmetric"
        point_net_len = (n_set_layers - 1) // 2
        self.transmision = torch.nn.Sequential(
            torch.nn.InstanceNorm1d(self.attention_e_dim, affine=True),
            LinearTransmissionLayer(
                input_feat_dim=self.attention_e_dim,
                output_feat_dim=self.attention_e_dim,
                activation=inner_activation,
            ),
        )
        for l in range(point_net_len):
            self.norm_pre.append(
                torch.nn.InstanceNorm1d(self.attention_e_dim, affine=True)
            )
            self.norm_post.append(
                torch.nn.InstanceNorm1d(self.attention_e_dim, affine=True)
            )
            self.pre.append(
                PointNetBlock(
                    input_feat_dim=self.attention_e_dim,
                    output_feat_dim=self.attention_e_dim,
                    activation=inner_activation,
                    spectral_norm=spectral_norm,
                )
            )
            self.post.append(
                PointNetBlock(
                    input_feat_dim=self.attention_e_dim,
                    output_feat_dim=self.attention_e_dim,
                    activation=inner_activation
                    if l < point_net_len - 1
                    else out_activation,
                    spectral_norm=spectral_norm,
                )
            )

    def finetti_cbuild(
        self, n_set_layers, inner_activation, out_activation, spectral_norm
    ):
        if self.finneti_MLP:
            layers = []
            for c0, c1 in zip([self.finetti_dim, 64, 128], [64, 128, self.finetti_dim]):
                layers.append(nn.Linear(c0, c1))
                layers.append(nn.ReLU())
            self.MLP0 = nn.Sequential(*layers)

        self.norm_pre = torch.nn.ModuleList()
        self.norm_post = torch.nn.ModuleList()
        self.pre = torch.nn.ModuleList()
        self.post = torch.nn.ModuleList()

        assert (
            n_set_layers >= 3 and n_set_layers % 2 == 1
        ), "Need at least 1 layer pre and post Linear Transmission layer, and an odd number overall since we stay symmetric"
        point_net_len = (n_set_layers - 1) // 2
        self.transmision = torch.nn.Sequential(
            torch.nn.InstanceNorm1d(self.attention_e_dim, affine=True),
            LinearTransmissionLayer(
                input_feat_dim=self.attention_e_dim,
                output_feat_dim=self.attention_e_dim,
                activation=inner_activation,
                spectral_norm=spectral_norm,
            ),
        )
        for l in range(point_net_len):
            self.norm_pre.append(
                torch.nn.InstanceNorm1d(self.attention_e_dim, affine=True)
            )
            self.norm_post.append(
                torch.nn.InstanceNorm1d(self.attention_e_dim, affine=True)
            )
            self.pre.append(
                PointNetBlock(
                    input_feat_dim=self.attention_e_dim,
                    output_feat_dim=self.attention_e_dim,
                    activation=inner_activation,
                    spectral_norm=spectral_norm,
                )
            )
            self.post.append(
                PointNetBlock(
                    input_feat_dim=self.attention_e_dim,
                    output_feat_dim=self.attention_e_dim,
                    activation=inner_activation
                    if l < point_net_len - 1
                    else out_activation,
                    spectral_norm=spectral_norm,
                )
            )
        if self.cycle_opt == "finetti_ds":
            self.ds = DS_block(
                [self.attention_e_dim + self.finetti_dim, 128],
                [128 * 2, self.attention_e_dim + self.finetti_dim],
            )

    def forward(self, Z0, u=None):
        if self.cycle_opt == "standard":
            Zi = self.standard_ds(Z0)
        elif "finetti" in self.cycle_opt:
            Zi = self.finetti_fw(Z0, u)
        else:
            raise ValueError(f"Unkown value cycle_opt {self.cycle_opt}")

        X, _ = self.node_readout(Zi, Zi, Zi)

        if self.edge_readout_type == "attention_weights":
            _, _, A = self.edge_readout(Zi, Zi, Zi, return_attention_and_scores=True)
        elif self.edge_readout_type == "QQ_sig":
            A = self.edge_readout(Zi @ Zi.permute(0, 2, 1))
        else:
            A = self.edge_readout(Zi)

        if self.discretization == "relaxed_bernoulli":
            A = self.discretize(A)

        return X, A, Zi

    def standard_ds(self, Z0):
        Zi = Z0
        for l, (norm, point) in enumerate(zip(self.norm_pre, self.pre)):

            Zi = norm(Zi.permute(0, 2, 1)).permute(0, 2, 1)
            Zi_ = point(Zi)
            Zi = Zi + Zi_
        Zi = self.transmision(Zi)
        for l, (norm, point) in enumerate(zip(self.norm_post, self.post)):

            Zi = norm(Zi.permute(0, 2, 1)).permute(0, 2, 1)
            Zi_ = point(Zi)
            Zi = Zi + Zi_
        return Zi

    def finetti_fw(self, Z0, u):
        if self.finneti_MLP:
            u_ = self.MLP0(u)
        else:
            u_ = u
        num_nodes = Z0.size()[1]
        # replicate finetti across Nodes, separately for each batch
        # NOTE: slice to batch_size which can be < the *actual* batch size, since we don't drop dangling batches by default
        Zi = torch.cat(
            (Z0[: u_.shape[0]], u_.unsqueeze(dim=1).repeat(1, num_nodes, 1)), dim=-1
        )

        Zi = self.standard_ds(Zi)

        if self.cycle_opt == "finetti_ds":
            Zi = self.ds(Zi)

        return Zi

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

    def sample_node_embed_Z0(self, Ns, device=None):
        assert len(Ns.shape) == 0
        batch_size = Ns.shape[0]
        nmax = len(self.n_node_dist.probs)
        shape = [batch_size, nmax, self.e_dim]
        noise = torch.randn(*shape, device=Ns.device if device is None else device)
        Ns = Ns.reshape(batch_size, 1, 1).repeat(1, nmax, 1)
        assert Ns.dim() == noise.dim()
        if batch_size is not None:
            for b in range(batch_size):
                nm = Ns[b].int().item()
                noise[b, nm:, :] = 0.0
                Ns[b, nm:, :] = 0.0
        Z0 = torch.cat([noise, Ns], -1)
        return Z0

    def sample_context_vec(self, batch_size, device=None):
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
        N: Union[int, torch.Tensor] = None,
        external_context_vec=None,
        device=None,
        realX=None,
        save_G=False,
    ):
        # if realX is None:
        #     N=realX.shape[1]
        # elif realX.shape[0]:
        #     batch_shape=[realX.shape[0]]
        if torch.is_tensor(N):
            Ns = N
        elif type(N) is int:
            Ns = self.sample_Ns(batch_size, device=device, given_N=N)
        else:
            raise ValueError(
                f"N must be either a vector of Ns (to use directly)or a single N (to set the same node numbering for everything"
                f" or None (to draw it from the underlying dist"
            )

        # Fixed Z0, nonfixed finetti_u=> Fixed node embedding, changing universal embedding
        if "finetti" in self.cycle_opt:
            # GG gan
            A, Q, X, node_embed_Z, context_vec = self.geometric_sampling(
                Ns,
                device=device,
                external_context_vec=external_context_vec,
                realX=realX,
            )
        else:
            # RG GAN
            A, Q, X, node_embed_Z = self.random_graph_sampling(Ns, device, realX=realX)
            context_vec = None

        # re-append node_numbers so things line up for discriminator
        X = torch.cat([X, Ns.to(X.device)], -1)

        A_out, X_out = self.rezero_superflous(A, N, X, batch_size)

        # TODO: furhter disentangle this
        if save_G:
            if "finetti" in self.cycle_opt:
                return node_embed_Z, context_vec
            else:
                return
        else:
            return X_out, A_out, node_embed_Z, context_vec, Q

    def random_graph_sampling(self, Ns, device, realX=None):
        random_node_Z0 = self.sample_node_embed_Z0(Ns, device=device)
        if realX is not None:
            random_node_Z0 = torch.cat([random_node_Z0, realX], dim=-1)
        X, A, Q = self.forward(random_node_Z0)
        return A, Q, X, random_node_Z0

    def geometric_sampling(
        self, Ns, device=None, external_context_vec=None, realX=None
    ):
        assert len(Ns.shape) == 1
        batch_size = Ns.shape[0]
        # clone preserve gradient but we don't actually change the parameter by hand
        if device is not None and self.node_embed_Z0_init.device != device:
            self.node_embed_Z0_init = self.node_embed_Z0_init.to(device)
        Z = self.node_embed_Z0_init.clone()
        if self.replicated_Z:
            Z: torch.Tensor
            # replcate across batch dim
            Z = Z.repeat([Ns.shape[0], 1, 1])
        if not self.batch_embedding:
            # TODO: this is mathematically meaningless, but for consistencies sake
            Z = Z.reshape(Ns.shape[0], Z.shape[0], -1)
        if external_context_vec is not None:
            contex_vec = external_context_vec
        elif self.train_fix_context and self.training:
            # for seeing what happens if we fix the context vector and train it instead of sampling
            # NOTE: TODO: important that you call .eval() before sampling when this is done
            if device is not None and self.context_vec_init.device != device:
                self.context_vec_init = self.context_vec_init.to(device)
            contex_vec = self.context_vec_init.clone()
        else:
            contex_vec = self.sample_context_vec(batch_size, device)
        if realX is not None:
            # conditional generation
            Z = torch.cat([Z, realX], -1)
        X, A, Q = self.forward(Z, contex_vec)
        return A, Q, X, Z, contex_vec
