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

from ggg.models.components.utilities_classes import Swish, SkipBlock
from ggg.models.components.pointnet_st import PointNetBlock, LinearTransmissionLayer
from ggg.models.components.attention.MultiHead import MultiHeadAttention
from ggg.utils.utils import sacred_copy, zero_diag
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def kernel(X, alpha=3):
    """
    The kernel function used to build a graph from a set of points.
    Setting alpha to a larger number will render the kernels closer to being binary,
    but can cause vanishing gradient issues.

        X: the points (dimension n_graphs x n_nodes x n_dim)
        alpha: a positive constant determining how close to {0,1} the kernel output is.
    """

    Xnorm = X.reshape(-1, X.shape[2])  # Xnorm: (n_graphsxn_nodes) x n_dim
    Xnorm = torch.norm(Xnorm, dim=1)  # Xnorm: (n_graphsxn_nodes)
    Xnorm = Xnorm.unsqueeze(dim=1)  # Xnorm: (n_graphsxn_nodes) x 1
    Xnorm = Xnorm.repeat(1, X.shape[2])  # Xnorm: (n_graphsxn_nodes) x n_dim
    Xnorm = Xnorm.reshape(X.shape)  # Xnorm: n_graphs x n_nodes x n_dim
    X = X / Xnorm
    K = torch.bmm(X, X.transpose(1, 2))  # K: n_graphs x n_nodes x n_nodes
    K = torch.sigmoid(alpha * K)  # K: n_graphs x n_nodes x n_nodes
    return K


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


class BaseGen(pt.nn.Module):
    def __init__(self, temperature=0.66, alpha=5) -> None:
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha

    def discretize(self, logits):
        return kernel(logits, self.alpha)

    def discretize_bernoulli(self, logits):
        logA = logits @ logits.permute(0, 2, 1)
        probs = logA.softmax(-1)
        relaxedA = td.RelaxedBernoulli(self.temperature, probs=probs).rsample()
        # hard relaxed bernoulli= create 0 vector with gradients attached, add to rounded values
        grads_only = relaxedA - relaxedA.detach()
        Ar = relaxedA.round() + grads_only
        # rezero and resymmetrize
        Az = zero_diag(Ar)
        Atriu = pt.triu(Az)
        A = Atriu + Atriu.permute(0, 2, 1)

        return A


class MLP(BaseGen):
    def __init__(
        self, feat_size, hidden_dim=128, out_dim=64, temperature=0.66, depth=1, alpha=5
    ) -> None:
        super().__init__(temperature, alpha)
        self.feat_size = feat_size
        self.trunk = pt.nn.Sequential(
            pt.nn.Linear(feat_size, hidden_dim),
            Swish(),
            *make_hiddens(hidden_dim, pt.nn.Linear, depth - 2),
            pt.nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, Z0):
        logits = self.trunk(Z0)
        return self.discretize(logits)


class DS(BaseGen):
    def __init__(
        self, feat_size, hidden_dim=128, out_dim=64, temperature=0.66, depth=1, alpha=5
    ) -> None:
        super().__init__(temperature, alpha)
        self.feat_size = feat_size
        mods = [
            SkipBlock(
                PointNetBlock(feat_size, hidden_dim), proj=(feat_size, hidden_dim)
            ),
            Swish(),
        ]
        halves = depth // 2
        for i in range(halves - 1):
            mods.extend(
                [
                    SkipBlock(PointNetBlock(hidden_dim, hidden_dim)),
                    Swish(),
                ]
            )
        mods.extend(
            [
                SkipBlock(LinearTransmissionLayer(hidden_dim, hidden_dim)),
                Swish(),
            ]
        )

        for i in range(halves):
            mods.append(
                SkipBlock(
                    PointNetBlock(
                        hidden_dim, hidden_dim if i < halves - 1 else out_dim
                    ),
                    proj=None if i < halves - 1 else (hidden_dim, out_dim),
                )
            )
            if i < halves - 1:
                mods.append(Swish())

        self.trunk = pt.nn.Sequential(*mods)

    def forward(self, Z0):
        logits = self.trunk(Z0)
        return self.discretize(logits)


class KernelDataset(Dataset):
    def __init__(
        self,
        n_nodes=10,  # number of nodes
        n_graphs=100,  # how many graphs we want to model
        n_dim=2,  # dimension where the points live
        alpha=5,
    ):
        super().__init__()
        dataset = []

        # n-cycle
        t = np.linspace(0, 2 * math.pi * (n_nodes - 1) / n_nodes, n_nodes)
        X = np.array([np.cos(t), np.sin(t)]).T
        dataset.append(X)

        # random geometric graphs
        for i in range(n_graphs - 1):
            X = (np.random.rand(n_nodes, 2) - 0.5) * 2
            dataset.append(X)
        self.X_target = torch.from_numpy(np.array(dataset))
        self.K_target = kernel(self.X_target, alpha=alpha)

    def __len__(self):
        return len(self.K_target)

    def __getitem__(self, item):
        return self.K_target[item]


class SkipMHSA(pt.nn.Module):
    """
    Skipblock MultiHEAD self Attention
    """

    def __init__(self, in_features, **kwargs):
        super().__init__()
        out_feat = kwargs.get("out_features", None)
        if out_feat is None:
            proj = None
        else:
            proj = (in_features, out_feat)
        self.inner = SkipBlock(MultiHeadAttention(in_features, **kwargs), proj=proj)

    def forward(self, X):
        return self.inner(X)


class SkipMHSA(MultiHeadAttention):
    def __init__(
        self,
        in_features,
        out_features=None,
        num_heads=1,
        bias=True,
        activation=None,
        out_activation=None,
        mode="QK",
        score_function="sigmoid",
        spectral_norm=None,
    ):
        super().__init__(
            in_features,
            out_features,
            num_heads,
            bias,
            activation,
            out_activation,
            mode,
            score_function,
        )


class Att(BaseGen):
    def __init__(
        self, feat_size, hidden_dim=128, out_dim=64, temperature=0.66, depth=1, alpha=5
    ) -> None:
        super().__init__(temperature, alpha)
        self.feat_size = feat_size
        l = [
            SkipMHSA(
                feat_size,
                out_features=hidden_dim if depth > 1 else out_dim,
                score_function="softmax",
                out_activation=None if depth < 2 else Swish(),
            ),
            *make_hiddens(
                hidden_dim,
                lambda x, y: SkipMHSA(
                    x, out_features=y, score_function="softmax", out_activation=Swish()
                ),
                depth - 2,
                act=False,
            ),
        ]
        if depth > 1:
            l.append(
                SkipMHSA(hidden_dim, out_features=out_dim, score_function="softmax")
            )
        self.trunk = pt.nn.Sequential(*l)

    def forward(self, Z0):
        logits = self.trunk(Z0)
        return self.discretize(logits)


@attr.s
class CollisionPars:
    dataset = attr.ib(default="kernel-custom")
    inner_kwargs = attr.ib(factory=dict)
    max_N = attr.ib(default=10)
    model = attr.ib(
        default="mlp-rand",
        validator=attr.validators.in_(
            {
                "mlp-rand",
                "mlp-traj",
                "ds-rand",
                "attention-rand",
                "ds-traj",
                "attention-traj",
            }
        ),
    )
    lr = attr.ib(default=1e-3)
    data_dir = attr.ib(default="data")
    graph_ind = attr.ib(default=0)
    epoch_exp = attr.ib(default=1000)
    max_steps = attr.ib(default=30000)
    batch_size = attr.ib(default=1)  # n_graphs
    rand_dim = attr.ib(default=3)
    fixed_dim = attr.ib(default=18)
    fixed_trainable = attr.ib(default=True)
    fixed_replicated = attr.ib(default=False)
    plot_every = attr.ib(default=None)
    sphere_Z = attr.ib(default=True)
    alpha = attr.ib(default=5)
    depth = attr.ib(default=4)
    hidden_width = attr.ib(default=256)
    out_dim = attr.ib(default=2)  # ==n_dim

    @classmethod
    def from_sacred(cls, d):
        return cls(**sacred_copy(d))


class CollisionDemo(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams if isinstance(hparams, dict) else attr.asdict(hparams)
        self.hpars = CollisionPars.from_sacred(self.hparams)
        self.init_fixed()
        feat_size = (
            self.hpars.rand_dim
            if not "traj" in self.hpars.model
            else self.hpars.fixed_dim + self.hpars.rand_dim
        )
        if "mlp" in self.hpars.model:
            self.generator = MLP(
                feat_size,
                hidden_dim=self.hpars.hidden_width,
                out_dim=self.hpars.out_dim,
                depth=self.hpars.depth,
                alpha=self.hpars.alpha,
            )
        elif "ds" in self.hpars.model:
            self.generator = DS(
                feat_size,
                hidden_dim=self.hpars.hidden_width,
                out_dim=self.hpars.out_dim,
                depth=self.hpars.depth,
                alpha=self.hpars.alpha,
            )
        elif "attention" in self.hpars.model:
            self.generator = Att(
                feat_size,
                hidden_dim=self.hpars.hidden_width,
                out_dim=self.hpars.out_dim,
                depth=self.hpars.depth,
                alpha=self.hpars.alpha,
            )
        else:
            raise ValueError(f"Unkown arch {self.hpars.model}")

    def init_fixed(self):
        if self.hpars.fixed_replicated:
            fixed = pt.randn(self.hpars.max_N, self.hpars.fixed_dim)
        else:
            fixed = pt.randn(
                self.hpars.batch_size, self.hpars.max_N, self.hpars.fixed_dim
            )
        if self.hpars.sphere_Z:
            fixed = fixed / fixed.norm(dim=-1, keepdim=True)
        self.Z0_fixed = pt.nn.Parameter(fixed, self.hpars.fixed_trainable)
        self.Z0_fixed_init = self.Z0_fixed.detach().clone()

    def get_fixed(self, N):
        if self.hpars.fixed_replicated:
            F = (
                self.Z0_fixed[:N, :]
                .clone()
                .unsqueeze(0)
                .repeat([self.hpars.batch_size, 1, 1])
            )
        else:
            F = self.Z0_fixed[:, :N, :].clone()
        return F

    def prepare_data(self) -> None:
        if self.hpars.dataset == "kernel-custom":
            self.train_set = KernelDataset(
                self.hpars.max_N,
                self.hpars.batch_size,
                self.hpars.out_dim,
                self.hpars.alpha,
            )
        else:
            self.train_set = GGG_DenseData(
                self.hpars.data_dir, dataset=self.hpars.dataset
            )
        self.train_set = Subset(
            self.train_set, [self.hpars.graph_ind] * self.hpars.epoch_exp
        )

    def forward(self, A):
        """
        A: [B,N,N]
        :param A:
        :return: A_fake
        """
        assert A.dim() == 3
        N = A.shape[-1]
        if "traj" in self.hpars.model:
            ksi = pt.randn(A.shape[0], 1, self.hpars.rand_dim).repeat([1, N, 1])
            F = self.get_fixed(N)
            Z0 = pt.cat([F, ksi], dim=-1)
        else:
            Z0 = pt.randn(A.shape[0], N, self.hpars.rand_dim)
        A_fake = self.generator(Z0)
        return A_fake

    def loss_func(self, fake, real):
        return pt.nn.functional.mse_loss(fake, real)

    def training_step(self, batch, batch_idx):
        if self.hpars.dataset == "kernel-custom":
            A = batch
        else:
            _, A = batch
        A = A.float()
        fake = self.forward(A)
        self.logger.experiment.add_histogram("fake", fake, global_step=self.global_step)
        loss = self.loss_func(fake, A)
        log = dict(mse=loss)
        ret = dict(loss=loss, log=log, progress_bar=log)
        return ret

    def configure_optimizers(self):
        self.generator: BaseGen
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
        self.log_weights(hist=True)
        self.make_plots(show=False)

    def make_plots(self, save_dir=None, show=True):
        if "traj" in self.hpars.model:
            for n_graph in range(self.Z0_fixed.shape[0]):
                fig, ax = plot_shift(
                    self.Z0_fixed_init[n_graph].detach().numpy(),
                    self.Z0_fixed[n_graph].detach().numpy(),
                )
                title = f"Z{n_graph}_{self.hpars.model}_{self.hpars.dataset}"
                fig.suptitle(title)
                self.logger.experiment.add_figure(
                    title, fig, global_step=self.global_step
                )
                if save_dir:
                    fig.savefig(os.path.join(save_dir, f"{title}_change.pdf"))
                if show:
                    plt.show(fig)
        target = self.train_set[: self.hpars.batch_size]
        outs = plot_kernels(self.forward(target).detach().numpy(), target.numpy())
        for i, o in enumerate(outs):
            self.logger.experiment.add_figure(
                f"{self.hpars.model}-Kernels{i}", o, global_step=self.global_step
            )
        plt.close("all")
