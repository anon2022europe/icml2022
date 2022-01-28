import math
import os
from typing import Any

from ipdb import set_trace
from pytorch_lightning import LightningModule
import attr
from torch.utils.data import DataLoader, Subset
import torch as pt
from torch.utils.data.dataset import Dataset

from ggg.data.dense.GGG_DenseData import GGG_DenseData
import torch.distributions as td

from ggg.models.components.edge_readout.attention import AttentionEdgeReadout
from ggg.models.components.edge_readout.kernel import (
    KernelEdges,
    BiasedSigmoid,
    RescaledSoftmax,
)
from ggg.models.components.utilities_classes import Swish, SkipBlock
from ggg.models.components.pointnet_st import PointNetBlock, LinearTransmissionLayer
from ggg.models.components.attention.MultiHead import MultiHeadAttention
from ggg.utils.utils import sacred_copy, zero_diag
import torch
import numpy as np
import torch.distributions as td
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx


def rand_blocks(N, max_steps=100000, p=None):
    if N == 2:
        p = 0.5
        q = p / np.log(N)
        return np.array([[p, q], [q, p]])
    else:
        p = np.random.rand(N, N)
        p = p @ p.T
        for i in range(max_steps):
            psum = p.sum(-1)
            p = p / psum[:, None]
            psum = p.sum(0)
            p = p / psum[None, :]
            if (p.sum(0) == 1).all() and (p.sum(1) == 1).all():
                break
        if not ((p.sum(0) == 1).all() and (p.sum(1) == 1).all()):
            raise ValueError("Stochastic iteration did not converge")
    return p


class SBM(Dataset):
    def __init__(
        self,
        n_nodes_per_block=(5, 5),  # number of nodes
        n_graphs=100,  # how many graphs we want to model
        repeat=True,
        adjacency=True,  # whether to return adjacency matrices or probabilitis directly
    ):
        super().__init__()
        self.n_blocks = len(n_nodes_per_block)
        self.n_nodes_per_block = n_nodes_per_block
        P = rand_blocks(self.n_blocks)
        if repeat:
            g = nx.adjacency_matrix(
                nx.stochastic_block_model(n_nodes_per_block, P)
            ).todense()
            self.dataset = [g] * n_graphs
        else:
            self.dataset = [
                nx.adjacency_matrix(
                    nx.stochastic_block_model(n_nodes_per_block, P).todense()
                )
                for _ in range(n_graphs)
            ]
        self.dataset = np.stack([np.asarray(x) for x in self.dataset], 0)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    @property
    def exp(self):
        return np.mean(self.dataset, 0).astype(float)


def plot_shift(Z_init, Z_learned):
    n_nodes = Z_init.shape[-2]
    n_latent = Z_init.shape[-1]
    n_plots = np.floor(n_latent / 2)
    rows = int(np.ceil(np.sqrt(n_plots)))
    cols = rows
    fig, axs = plt.subplots(
        nrows=rows, ncols=cols, figsize=(cols * 4, rows * 4), facecolor=[1, 1, 1]
    )
    for dim_idx, dim in enumerate(2 * np.arange(0, n_plots, dtype=np.int)):
        ax = axs.flatten()[dim_idx]
        ax.set_title(f"Z[{dim_idx}]")
        ax.plot(Z_learned[:, dim], Z_learned[:, dim + 1], "xr")
        ax.plot(Z_init[:, dim], Z_init[:, dim + 1], "og")
        for i in range(n_nodes):
            ax.plot(
                [Z_learned[i, dim], Z_init[i, dim]],
                [Z_learned[i, dim + 1], Z_init[i, dim + 1]],
                "k-",
            )
    return fig, ax


def plot_shift_ksi(ksi_init, ksi_learned):
    assert ksi_learned.shape[1] >= 3
    n_graphs = ksi_init.shape[0]

    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        ksi_learned[:, 0], ksi_learned[:, 1], ksi_learned[:, 2], marker="x", c="r"
    )
    ax.scatter(ksi_init[:, 0], ksi_init[:, 1], ksi_init[:, 2], marker="o", c="g")
    for i in range(n_graphs):
        ax.plot(
            [ksi_learned[i, 0], ksi_init[i, 0]],
            [ksi_learned[i, 1], ksi_init[i, 1]],
            [ksi_learned[i, 2], ksi_init[i, 2]],
            "k-",
        )
    return fig, ax


def plot_kernels(
    A,
    A_target,
):
    n_samples = A.shape[0]
    outs = []
    for i in range(n_samples):
        fig = plt.figure(figsize=(3 * 8, 6), facecolor=[1, 1, 1])

        ax = fig.add_subplot(1, 3, 1)
        ax.imshow(A[i, :, :])
        ax.set_title("A")

        ax = fig.add_subplot(1, 3, 2)
        ax.imshow(A_target[i, :, :])
        ax.set_title("A_target")

        ax = fig.add_subplot(1, 3, 3)
        ax.imshow(np.abs(A_target[i, :, :] - A[i, :, :]), vmin=0, vmax=1)
        ax.set_title(
            f'error:{np.linalg.norm(A_target[i, :, :] - A[i, :, :], ord="fro") / np.linalg.norm(A_target[i, :, :], ord="fro")}'
        )
        outs.append(fig)
    return outs


def make_hiddens(hidden_width, func, n, act=True):
    H = []
    for _ in range(n):
        H.append(func(hidden_width, hidden_width))
        if act:
            H.append(Swish())
    return H


@attr.s
class SBMCheckPars:
    F = attr.ib(default=3)

    readout = attr.ib(default="attention_weights")
    explicit_expectation_target = attr.ib(default=False)

    sbm_pars = attr.ib(
        factory=lambda: dict(
            n_nodes_per_block=(5, 5),  # number of nodes
            n_graphs=int(1e5),  # how many graphs we want to model
            repeat=True,
            adjacency=True,  # whether to return adjacency matrices or probabilitis directly
        )
    )
    readout_kwargs = attr.ib(factory=dict)
    discretization = attr.ib(default="relaxed_bernoulli")
    temperature = attr.ib(default=0.66)
    use_mlp = attr.ib(default=False)

    lr = attr.ib(default=1e-3)
    epoch_exp = attr.ib(default=10000)
    max_steps = attr.ib(default=int(0.5e6))
    batch_size = attr.ib(default=100)  # n_graphs
    plot_every = attr.ib(default=None)
    verbose = attr.ib(default=False)  # add histogram
    max_plots = attr.ib(default=5)

    @classmethod
    def from_sacred(cls, d):
        return cls(**sacred_copy(d))

    @property
    def N(self):
        return sum(self.sbm_pars["n_nodes_per_block"])


class SBMCheck(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams if isinstance(hparams, dict) else attr.asdict(hparams)
        self.hpars = SBMCheckPars.from_sacred(self.hparams)
        self.X = pt.nn.Parameter(
            pt.randn(1, self.hpars.N, self.hpars.F), requires_grad=True
        )

        self.edge_readout_type = self.hpars.readout
        if self.hpars.use_mlp:
            self.model = pt.nn.Sequential(
                pt.nn.Linear(self.hpars.F, self.hpars.F),
                pt.nn.ReLU(),
                pt.nn.Linear(self.hpars.F, self.hpars.F),
            )
        else:
            self.model = pt.nn.Identity()
        if self.edge_readout_type == "gaussian_kernel":
            self.readout = KernelEdges()
        elif self.edge_readout_type == "QQ_sig":
            self.sig = torch.nn.Sigmoid()
            self.readout = lambda x: self.sig(x @ x.permute(0, 2, 1))
        elif self.edge_readout_type == "biased_sigmoid":
            self.readout = BiasedSigmoid(
                feat_dim=self.hpars.F, **self.hpars.readout_kwargs
            )
        elif self.edge_readout_type == "rescaled_softmax":
            self.readout = RescaledSoftmax(
                feat_dim=self.hpars.F, **self.hpars.readout_kwargs
            )
        elif self.edge_readout_type == "attention_weights":
            self.att = MultiHeadAttention(
                in_features=self.hpars.F, **self.hpars.readout_kwargs
            )
            self.readout = lambda x: self.att.forward(
                x, x, x, return_attention_and_scores=True
            )[-1]
        else:
            raise NotImplementedError()

    def prepare_data(self) -> None:
        self.train_set = SBM(**self.hpars.sbm_pars)

    def forward(self, A):
        """
        A: [B,N,N]
        :param A:
        :return: A_fake
        """
        if not pt.is_tensor(A):
            A = pt.from_numpy(A)
        assert A.dim() == 3
        X = self.X
        X = self.model(X)
        A_fake = self.readout(X.repeat([A.shape[0], 1, 1]))
        if self.hpars.discretization == "relaxed_bernoulli":
            A_fake = self.discretize(A_fake)
        return A_fake

    def discretize(self, A):
        relaxedA = td.RelaxedBernoulli(self.hpars.temperature, probs=A).rsample()
        # hard relaxed bernoulli= create 0 vector with gradients attached, add to rounded values
        grads_only = relaxedA - relaxedA.detach()
        Ar = relaxedA.round() + grads_only
        # rezero and resymmetrize
        Az = zero_diag(Ar)
        Atriu = torch.triu(Az)
        A = Atriu + Atriu.permute(0, 2, 1)

        return A

    def loss_func(self, fake, real):
        if self.hpars.explicit_expectation_target:
            Fmean = fake.mean(0)
            return pt.nn.functional.mse_loss(
                Fmean, pt.from_numpy(self.train_set.exp).float()
            )
        else:
            return pt.nn.functional.mse_loss(fake, real)

    def training_step(self, batch, batch_idx):
        A = batch
        A = A.float()
        fake = self.forward(A)
        if self.hpars.verbose:
            self.logger.experiment.add_histogram(
                "fake", fake, global_step=self.global_step
            )
        loss = self.loss_func(fake, A)
        log = dict(mse=loss, mse_rel=loss / A.norm())
        ret = dict(loss=loss, log=log, progress_bar=log)
        return ret

    def configure_optimizers(self):
        opt = pt.optim.Adam(self.parameters(), self.hpars.lr, weight_decay=1e-4)
        sched = dict(
            scheduler=pt.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", patience=2
            ),
            monitor="loss",
        )
        return [opt], [sched]

    def train_dataloader(self) -> DataLoader:
        dl = DataLoader(self.train_set, batch_size=self.hpars.batch_size, shuffle=False)
        return dl

    def log_weights(self, hist=False):
        norms = {}
        nmax = 0.0
        for name, param in self.named_parameters():
            n = pt.norm(param)
            if n > nmax:
                nmax = n
            self.logger.experiment.add_scalar(
                f"{name}_fro", n, global_step=self.trainer.total_batch_idx
            )
            if hist:
                self.logger.experiment.add_histogram(f"{name}_hist", param)
        self.logger.experiment.add_scalar(
            f"W_fro_max", nmax, global_step=self.trainer.total_batch_idx
        )

    def on_epoch_end(self) -> None:
        self.log_weights(hist=self.hpars.verbose)
        self.make_plots(show=False)

    def make_plots(self, save_dir=None, show=True):
        target = self.train_set[: min(self.hpars.batch_size, self.hpars.max_plots)]
        outs = plot_kernels(self.forward(target).detach().numpy(), target)
        for i, o in enumerate(outs):
            self.logger.experiment.add_figure(
                f"{self.hpars.readout}-Kernels{i}", o, global_step=self.global_step
            )
        plt.close("all")
