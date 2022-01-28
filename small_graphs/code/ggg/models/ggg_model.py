from logging import warning, info, debug
from typing import Dict, Union, List, Optional
from warnings import warn

from attr.validators import in_
from ipdb import set_trace
from pytorch_lightning.loggers import TensorBoardLogger
from torch import Tensor

from ggg.data.dense.GGG_DenseData import GGG_DenseData
from ggg.data.dense.hpars import DatasetHpars
from ggg.models.components.abstract_conf import AbstractConf
from ggg.utils.grad_penalty import GradPenHpars
from ggg.models.components.samplers.base import SamplerHpars
from ggg.optim.conf import OptSchedHpars, OptHpars
from ggg.utils.hooks import backward_trace_hook_t
from ggg.utils.logging import register_exp, register_trainer, set_log_hists
from ggg.utils.utils import (
    pac_reshape,
    ensure_tensor,
    ensure_tensor,
    pdf,
    single_node_featues, enable_asserts,
)

try:
    from ggg.warmup import GradualWarmupScheduler
except:
    warn("Need to get warmup sched again")
    pass

import attr
import matplotlib

from ggg.evaluation.plots.graph_grid import cluster_plot_molgrid

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ggg.utils.grad_flow import plot_grad_flow

import torch
import torch as pt
import torch.distributions as ptd
from pytorch_lightning import LightningModule
import networkx as nx

from torch.utils.data import DataLoader

from ggg.models.components.generators.att.models import Generator
from ggg.models.components.generators.att.roots import GenRootHpars
from ggg.models.components.generators.att.trunks import GenTrunkHpars
from ggg.models.components.generators.att.discretization import DiscretizationHpars
from ggg.models.components.generators.att.readouts import (
    NodeReadoutHpars,
    EdgeReadoutHpars,
)
from ggg.models.components.discriminators.kCycleGIN import (
    Discriminator,
    DiscriminatorHpars,
)

from torchvision.utils import make_grid
import math


@attr.s
class GGG_Hpar(AbstractConf):
    @classmethod
    def children(cls) -> Dict:
        return dict(
            discriminator_hpars=DiscriminatorHpars,
            root_hpars=GenRootHpars,
            trunk_hpars=GenTrunkHpars,
            node_readout_hpars=NodeReadoutHpars,
            edge_readout_hpars=EdgeReadoutHpars,
            discretization_hpars=DiscretizationHpars,
            disc_opt_hpars=OptHpars,
            gen_opt_hpars=OptHpars,
            penalty_hpars=GradPenHpars,
            sampler_hpars=SamplerHpars,
            dataset_hpars=DatasetHpars,
        )

    discriminator_hpars = attr.ib(default=DiscriminatorHpars())
    root_hpars = attr.ib(default=GenRootHpars())
    trunk_hpars = attr.ib(default=GenTrunkHpars())
    edge_readout_hpars = attr.ib(default=EdgeReadoutHpars())
    node_readout_hpars = attr.ib(default=NodeReadoutHpars())
    discretization_hpars = attr.ib(default=DiscretizationHpars())
    penalty_hpars = attr.ib(default=GradPenHpars())
    sampler_hpars = attr.ib(default=SamplerHpars())
    disc_opt_hpars = attr.ib(default=OptHpars.disc_default())
    gen_opt_hpars = attr.ib(default=OptHpars.gen_default())
    dataset_hpars = attr.ib(default=DatasetHpars())
    disc_every = attr.ib(default=1)
    gen_every = attr.ib(default=1)
    precision=attr.ib(default=32,validator=in_({32,16}))
    contrast_mode = attr.ib(
        default="fake-struct_fake",
        validator=attr.validators.in_(
            {"real_fake", "fake_fake", "fake-struct_fake", "fake-struct-detach_fake"}
        ),
    )  # real/fake node embeddings with fake adjacency matrix
    # old parameters
    weight_clip = attr.ib(default=False)  # enforce lipschitz via weight clip
    log_weights = attr.ib(default=False)
    # hyper parameters
    #data loaders
    batch_size = attr.ib(default=20)
    shuffle = attr.ib(default=True)
    num_workers = attr.ib(default=8)
    pin_memory = attr.ib(default=True)
    # "outside" parameters
    save_dir = attr.ib(default="GGG_save")
    exp_name = attr.ib(default="GG")
    # plotting parameters
    plot_lcc = attr.ib(default=False)  # plot only largest connected component
    grid_every = attr.ib(default=100)
    grid_max_row = attr.ib(default=5)
    pac = attr.ib(default=1)  # acts as a multiplier on batch_size for sampling
    score_penalty_lambda = attr.ib(default=0.0)
    independent_penalty_samples = attr.ib(
        default=0
    )  # sample this many additional samples from the generator to calculate gradient penalty..
    viz = attr.ib(default=True)
    grad_flow_every = attr.ib(default=100)
    plot_to_file = attr.ib(default=True)
    plot_to_tensor = attr.ib(default=True)
    log_hists = attr.ib(default=False)
    asserts=attr.ib(default=False)

    @classmethod
    def with_dims(
        cls, node_feat=20, gen_noise_dim=16, gen_joined_dim=16, pac=1, gen_feat_dim=32
    ):
        hpars = cls()
        hpars.discriminator_hpars.pac = pac
        hpars.pac = pac
        hpars.discriminator_hpars.node_feature_dim = node_feat
        hpars.root_hpars.context_dim = gen_joined_dim
        hpars.root_hpars.node_embedding_dim = gen_noise_dim
        hpars.trunk_hpars.feat_dim = gen_joined_dim + gen_noise_dim
        hpars.trunk_hpars.attn_feat_dim = gen_feat_dim
        hpars.node_readout_hpars.feat_dim = gen_feat_dim
        hpars.node_readout_hpars.node_attrib_dim = node_feat
        hpars.edge_readout_hpars.feat_dim = gen_feat_dim
        return hpars

    def __attrs_post_init__(self):
        if (
            not self.batch_size == self.root_hpars.embedding_batch_size
        ) and not self.root_hpars.allow_greater:
            warning(
                f"allow greater is False, making sure embedding batch size==training_batch_size({self.batch_size}, was {self.root_hpars.embedding_batch_size})"
            )
            self.root_hpars.embedding_batch_size = self.batch_size
        if self.root_hpars.name in {"geometric", "random_joined", "fixed_context"}:
            assert (
                self.root_hpars.node_embedding_dim + self.root_hpars.context_dim + self.root_hpars.extra_features
                == self.trunk_hpars.feat_dim
            ), f"{self.root_hpars.node_embedding_dim}+{self.root_hpars.context_dim}+{self.root_hpars.extra_features}" \
               f"!={self.trunk_hpars.feat_dim}"
        else:
            assert self.root_hpars.node_embedding_dim == self.trunk_hpars.feat_dim
        if self.pac:
            assert self.discriminator_hpars.pac

    @classmethod
    def report_base(cls,
                    n_layers=6,
                    phi_dim=25,
                    context_dim=25,
                    batch_size=16,
                    gen_every=5,
                    trunk="attention",
                    root="geometric",
                    dataset="nx_star",
                    nx_kwargs: Optional[Dict] = None,
                    num_labels=None,
                    node_feature=0,
                    edge_readout="attention_weights",
                    on=("real","fake","real-perturbed","fake-perturbed"),
                    modes=("LP","LP","LP","LP"),
                    structured_features=True, use_laplacian=True,
                    exp_name="GG", extra_features=0, k_eigenvals=4,
                    gcn_norm_type="identity", penalty_agg="sum", input_agg="sum",
                    contrast_mode="fake-struct_fake", readout_agg="sum", simple_disc=False, allow_greater=True,
                    readout_score_function="sigmoid", edge_scoref="softmax", gen_trunk_score_function="sigmoid"

    ):
        nfeatdim = context_dim+phi_dim+extra_features
        attn_feat_dim = max(120, 3 * nfeatdim)
        hpars=GGG_Hpar(exp_name=exp_name, gen_every=gen_every,
            dataset_hpars=DatasetHpars(
            dataset=dataset,
            use_laplacian=use_laplacian,
            num_labels=num_labels,
            structured_features=structured_features,
            dataset_kwargs=nx_kwargs,
            cut_train_size=False,
            # dataset=dataset,
            # k_eigenvals=k_eigenvals,
            # num_labels=num_labels,
            # structured_features=structured_features,
        ),
            contrast_mode=contrast_mode,
            batch_size=batch_size,
            root_hpars=GenRootHpars(node_embedding_dim=phi_dim,context_dim=context_dim,embedding_batch_size=batch_size if not allow_greater else 1,
                                    name=root, trainable=True, extra_features=extra_features,
                                    ),
            trunk_hpars=GenTrunkHpars(
                name=trunk,
                n_layers=n_layers,
                feat_dim=nfeatdim,
                attn_feat_dim=attn_feat_dim,
                score_function=gen_trunk_score_function,
                num_heads=5,
                norm_type="identity",
                att_rezero=True,
                block_skip=True,
                rezero_skip=True
            ),
            edge_readout_hpars=EdgeReadoutHpars(
                name=edge_readout,
                feat_dim=attn_feat_dim,
                score_function=edge_scoref,
                readout_score_function=readout_score_function,
                num_heads=1,
                norm_type="layer",
            ),
            node_readout_hpars=NodeReadoutHpars(feat_dim=attn_feat_dim, node_attrib_dim=node_feature,
                                                ),

            discriminator_hpars=DiscriminatorHpars(node_attrib_dim=node_feature,
                                                   kc_flag=True,
                                                   conv_channels=[32, 64, 64, 128],
                                                   eigenfeat4=True,
                                                   add_global_node=False,
                                                   gcn_norm_type=gcn_norm_type,
                                                   readout_agg=readout_agg,
                                                   simple_disc=simple_disc,
                                                   ),
            penalty_hpars=GradPenHpars(on=on, penalty_agg=penalty_agg, input_agg=input_agg,
                                       modes=modes),
        )

        return hpars

    @classmethod
    def pointnet_st(cls, dataset, root="geometric"):
        raise NotImplementedError()
        n_attention_layers = (7,)
        hyper = dict(
            device="cuda:2",
            n_attention_layers=7,
            cut_train_size=False,
            edge_readout="attention_weights",
            architecture="deepset",
            dataset_kwargs=dict(dir="data"),
            label_one_hot=5,
            embed_dim=25,
            finetti_dim=25,
            kc_flag=True,
            disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
            cycle_opt="finetti_noDS",
            score_function="softmax",
            finetti_trainable=True,
            finetti_train_fix_context=False,
            dynamic_finetti_creation=False,
            replicated_Z=False,
            finneti_MLP=False,
            structured_features=True,
            node_feature_dim=4,
            use_laplacian=True,
            disc_contrast="fake_fake",
        )

    @classmethod
    def mlprow(cls, dataset, root="geometric"):
        raise NotImplementedError()
        if dataset == "MolGAN_5k":
            hyper = dict(
                dataset="MolGAN_5k",
                device="cuda:0",
                n_attention_layers=12,
                disc_contrast="fake_fake",
                cut_train_size=False,
                edge_readout="QQ_sig",
                architecture="mlp_row",
                MLP_layers=[128, 256, 512],
                dataset_kwargs=dict(DATA_DIR="/ggg/data"),
                label_one_hot=5,
                kc_flag=True,
                disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
                cycle_opt="finetti_noDS",
                finetti_trainable=True,
                finetti_train_fix_context=False,
                dynamic_finetti_creation=False,
                replicated_Z=False,
                finneti_MLP=False,
                structured_features=False,
            )
        elif dataset == "anu_graphs_chordal9":
            device = ("cuda:0",)
            n_attention_layers = (12,)
            cut_train_size = (False,)
            edge_readout = ("QQ_sig",)
            architecture = ("mlp_row",)
            disc_contrast = ("fake_fake",)
            MLP_layers = ([128, 256, 512],)
            dataset_kwargs = (dict(DATA_DIR="/ggg/data"),)
            label_one_hot = (5,)
            kc_flag = (True,)
            disc_conv_channels = ([32, 64, 64, 64, 128, 128, 128],)
            cycle_opt = ("standard",)
            finetti_trainable = (True,)
            finetti_train_fix_context = (False,)
            dynamic_finetti_creation = (False,)
            replicated_Z = (False,)
            finneti_MLP = (False,)
            node_feature_dim = (4,)
            use_laplacian = (True,)
            structured_features = (True,)
        elif dataset == "CommunitySmall_20":
            hyper = dict(
                dataset="CommunitySmall_20",
                device="cuda:0",
                n_attention_layers=12,
                cut_train_size=False,
                edge_readout="QQ_sig",
                disc_contrast="fake_fake",
                architecture="mlp_row",
                MLP_layers=[128, 256, 512],
                dataset_kwargs=dict(DATA_DIR="/ggg/data"),
                label_one_hot=5,
                kc_flag=True,
                disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
                cycle_opt="finetti_noDS",
                finetti_trainable=True,
                finetti_train_fix_context=False,
                dynamic_finetti_creation=False,
                replicated_Z=False,
                finneti_MLP=False,
                node_feature_dim=4,
                use_laplacian=True,
                structured_features=True,
            )

    @classmethod
    def attention(cls, dataset,dataset_kwargs=None,data_repeat=None):
        if dataset_kwargs is None:
            dataset_kwargs={}
        hpars=GGG_Hpar.attention_nx()
        hpars.dataset_hpars.dataset=dataset
        hpars.dataset_hpars.dataset_kwargs = dataset_kwargs
        hpars.dataset_hpars.repeat=data_repeat
        return hpars

    @classmethod
    def attention_community(cls,root="geometric", allow_greater=True):
        dataset="CommunitySmall_20"
        dataset_kwargs={}
        hpars=GGG_Hpar.attention_nx()
        hpars.dataset_hpars.dataset=dataset
        hpars.dataset_hpars.dataset_kwargs = dataset_kwargs
        return hpars


    @classmethod
    def attention_nx(cls,nx_dataset="nx_star",nx_kwargs:Optional[Dict]=None, root="geometric", allow_greater=True):
        if nx_kwargs is None:
            nx_kwargs=dict()
        trunk="attention"
        contrast_mode = "fake-struct_fake"
        edge_readout = "biased_sigmoid"
        edge_score = "softmax"
        edge_readout_score = "sigmoid"
        eigenfeat4 = True
        kc_flag = True
        context_dim = 50
        phi_dim = 50
        n_attention_layers = 5
        num_labels = None
        batch_size = 16
        disc_conv_channels = [32, 64, 128, 128,128, 128, 128]
        gen_trunk_score_function = "softmax"
        node_attrib_dim = 0
        use_laplacian = True
        structured_features = True
        nfeatdim = context_dim + phi_dim
        NUM_HEADS = 7
        attn_feat_dim = 25*NUM_HEADS
        hpars = GGG_Hpar(
            penalty_hpars=GradPenHpars(),
            dataset_hpars=DatasetHpars(
                dataset=nx_dataset,
                cut_train_size=False,
                use_laplacian=use_laplacian,
                num_labels=num_labels,
                structured_features=structured_features,
                dataset_kwargs=nx_kwargs
            ),
            contrast_mode=contrast_mode,
            batch_size=batch_size,
            root_hpars=GenRootHpars(
                node_embedding_dim=phi_dim,
                context_dim=context_dim,
                embedding_batch_size=batch_size if not allow_greater else 1,
                name=root,
                trainable=True,
                allow_greater=allow_greater,
            ),
            disc_every=5,
            trunk_hpars=GenTrunkHpars(
                name=trunk,
                n_layers=n_attention_layers,
                feat_dim=nfeatdim,
                attn_feat_dim=attn_feat_dim,
                score_function=gen_trunk_score_function,
                num_heads=NUM_HEADS,
                norm_type="identity",# layer works as well
                att_rezero=True,
                block_skip=True,
                rezero_skip=True
            ),
            edge_readout_hpars=EdgeReadoutHpars(
                name=edge_readout,
                feat_dim=attn_feat_dim,
                score_function=edge_score,
                readout_score_function=edge_readout_score,
                num_heads=1,
                max_communities="compute"
            ),
            node_readout_hpars=NodeReadoutHpars(
                feat_dim=attn_feat_dim,
                node_attrib_dim=node_attrib_dim,
                num_heads=NUM_HEADS,
            ),
            discriminator_hpars=DiscriminatorHpars(
                node_attrib_dim=node_attrib_dim,
                kc_flag=kc_flag,
                conv_channels=disc_conv_channels,
                eigenfeat4=eigenfeat4,
                add_global_node=False,
            ),
        )
        return hpars


class GGG(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hpars = GGG_Hpar.from_dict(hparams)
        set_log_hists(self.hpars.log_hists)
        # self.hparams = self.hpars.to_dict()
        self.hparams.update(self.hpars.to_dict())
        # create the model
        # node weights
        self.prepare_data()
        enable_asserts(self.hpars.asserts)
        self.train_set: GGG_DenseData
        if self.hpars.edge_readout_hpars.max_communities=="compute":
            max_comm=math.ceil(math.sqrt(self.train_set.max_N))
            warning(f"Setting max_communities in readout {self.hpars.edge_readout_hpars.name} to {max_comm}")
            self.hpars.edge_readout_hpars.max_communities=max_comm
        num_node_weights = ensure_tensor(self.train_set.node_dist_weights())
        self.num_node_dist = ptd.Categorical(num_node_weights)
        #
        self.generator: Generator = Generator(
            num_node_weights,
            self.hpars.root_hpars,
            self.hpars.trunk_hpars,
            self.hpars.edge_readout_hpars,
            self.hpars.node_readout_hpars,
            self.hpars.discretization_hpars,
            hist_hook=lambda n, t: self.logger.experiment.add_histogram(
                n, t, global_step=self.trainer.total_batch_idx
            )
            if self.hpars.log_hists
            else None,
        )
        self.sampler = self.hpars.sampler_hpars.make()
        self.discriminator: Discriminator = self.hpars.discriminator_hpars.make(
            max_nodes=len(num_node_weights)
        )

        self.gradient_penalty = self.hpars.penalty_hpars.make()
        self.last_gen_grad_flow = -1
        self.last_disc_grad_flow = -1

    def forward(self, batch_size=None, Z=None, X=None, A=None, N=None, device=None):
        if batch_size is None:
            min_batch_size = min(
                [getattr(x, "shape", [float("inf")])[0] for x in [Z, X, A, N]]
            )
            if min_batch_size != float("inf"):
                debug(f"Using smallest batch size of input arguments {min_batch_size}")
                batch_size = min_batch_size
            else:
                debug(f"Using hpars batch size")
                batch_size = self.hpars.batch_size
        Xf, Af, Nf, Zf = self.sampler.forward(
            self.generator, batch_size, Z, X, A, N, device=device
        )
        if Xf is not None:
            assert (
                Xf.shape[-1]
                == self.hpars.discriminator_hpars.node_attrib_dim
                == self.hpars.node_readout_hpars.node_attrib_dim
            )
            assert (
                Xf.shape[-2]
                == self.train_set.max_N
                == Af.shape[-1]
                == Af.shape[-2]
            )
        self.logger: TensorBoardLogger
        if self.hpars.log_hists:
            self.logger.experiment.add_histogram(
                "Af", Af, global_step=self.trainer.total_batch_idx
            )
            if Xf is not None:
                self.logger.experiment.add_histogram(
                    "Xf", Xf, global_step=self.trainer.total_batch_idx
                )
            self.logger.experiment.add_histogram(
                "Nf", Nf, global_step=self.trainer.total_batch_idx
            )
            self.logger.experiment.add_histogram(
                "Zf", Zf, global_step=self.trainer.total_batch_idx
            )
        return Xf, Af, Nf, Zf

    def score(self, data, fake=False):
        X, A, N = data
        return self.discriminator.forward(X, A, N=N, mode="score")

    def forward_score(
        self, batch_size=None, Z=None, X=None, A=None, N=None, device=None
    ):
        Xf, Af, Nf, Zf = self.forward(batch_size, Z, X, A, N, device=None)
        if self.hpars.node_readout_hpars.node_attrib_dim==0:
            Xf=None
        fake_score = self.score((Xf, Af), fake=True)
        if X is not None:
            real_score = self.score((X, A), fake=False)
        else:
            real_score = None
        return fake_score, real_score

    def get_grad_penalty(self, real, fake=None):
        # assume pre-paced real/fake passed in
        X, A, N = real
        if X is not None:
            assert X.dim() == 4  # B,P,N,F
        assert A.dim() == 4  # B,P,N,N
        if fake is None:
            Xf, Af, Nf, Zf = self.forward(N=N, device=X.device)
            if self.hpars.node_readout_hpars.node_attrib_dim == 0:
                Xf = None
            Af, Nf = [
                pac_reshape(x, to_packed=True, pac=self.hpars.pac, mode=m)
                for x, m in zip([Xf, Af, Nf], ["adj", "N"])
            ]
            if Xf is not None:
                Xf = pac_reshape(X, to_packed=True, pac=self.hpars.pac, mode="nodes")
        else:
            Xf, Af, Nf = fake
        assert Af.dim() == 4  # B,P,N,N

        if Xf is not None:
            assert Xf.dim() == 4  # B,P,N,F
        # X, A, N = [pac_reshape(x, to_packed=True, pac=self.hpars.pac, mode=m) for x, m in
        #           zip([X, A, N], ["nodes", "adj", "N"])]
        if self.hpars.discriminator_hpars.simple_disc and self.hpars.discriminator_hpars.node_attrib_dim != 0:
            grad_penalty = self.gradient_penalty(
                self.discriminator, *[(fake, real) for fake, real in zip([Xf], [X])]
            )
        else:
            grad_penalty = self.gradient_penalty(
                self.discriminator, *[(fake, real) for fake, real in zip([Xf, Af], [X, A])]
            )
        return grad_penalty

    def training_epoch_end(
        self, outputs: Union[List[Dict[str, Tensor]], List[List[Dict[str, Tensor]]]]
    ) -> Dict[str, Dict[str, Tensor]]:
        return self.sanity_checks()

    def sanity_checks(self):
        """
        Log  the score of the discriminator on zero and one adj matrices
        Returns
        -------
        """
        with torch.no_grad():
            Xf, Af, Nf, Zf = self.forward()
            if self.hpars.node_readout_hpars.node_attrib_dim == 0:
                Xf = None
            Xf, Af, Nf = [
                pac_reshape(x, to_packed=True, pac=self.hpars.pac, mode=m)
                if x is not None
                else x
                for x, m in zip([Xf, Af, Nf], ["nodes", "adj", "N"])
            ]
            inps = dict(
                zero=[pt.zeros_like(x) if x is not None else x for x in [Xf, Af]]
                + [Nf],
                ones=[pt.ones_like(x) if x is not None else x for x in [Xf, Af]] + [Nf],
                rand=[pt.rand_like(x) if x is not None else x for x in [Xf, Af]] + [Nf],
                randn=[pt.randn_like(x) if x is not None else x for x in [Xf, Af]]
                + [Nf],
            )
            score_dict = {
                f"{k}_score": self.score(v, fake=True).mean() for k, v in inps.items()
            }

        def detach(l):
            return [x.detach() if x is not None else x for x in l]

        pen_dict = {
            f"{k}_pen": self.get_grad_penalty(detach(v), detach(v))
            for k, v in inps.items()
        }
        # self.log_dict(score_dict,on_step=True,logger=True,prog_bar=False)
        # self.log_dict(pen_dict,on_step=True,logger=True,prog_bar=False)
        log = dict(**pen_dict, **score_dict)
        return dict(log=log)

    def training_step(self, batch, batch_idx, optimizer_idx):
        # give the input device to sample in generator case
        X, A, N = batch
        if self.hpars.discriminator_hpars.node_attrib_dim == 0:
            X = None
        register_exp(self.logger.experiment)
        register_trainer(self.trainer)
        if optimizer_idx == 0:  # train discriminator
            Xf, Af, Nf, Zf = self.forward(N=N, device=A.device)
            if self.hpars.node_readout_hpars.node_attrib_dim == 0:
                Xf = None
            if self.hpars.log_hists:
                self.logger.experiment.add_histogram(
                    "Xf_disc", Xf, global_step=self.trainer.total_batch_idx
                )
                self.logger.experiment.add_histogram(
                    "Af_disc", Af, global_step=self.trainer.total_batch_idx
                )
                self.logger.experiment.add_histogram(
                    "Nf_disc", Nf, global_step=self.trainer.total_batch_idx
                )
            A, N = [
                pac_reshape(x, to_packed=True, pac=self.hpars.pac, mode=m)
                for x, m in zip([A, N], ["adj", "N"])
            ]
            Af, Nf = [
                pac_reshape(x, to_packed=True, pac=self.hpars.pac, mode=m)
                for x, m in zip([Af, Nf], ["adj", "N"])
            ]
            if X is not None:
                X = pac_reshape(X, to_packed=True, pac=self.hpars.pac, mode="nodes")
            if Xf is not None:
                Xf = pac_reshape(Xf, to_packed=True, pac=self.hpars.pac, mode="nodes")
            fake_score_raw = self.score((Xf, Af, Nf), fake=True)

            if Xf is not None and X is not None:
                assert Xf.shape == X.shape
            assert Af.shape == A.shape
            assert Nf.shape == N.shape
            if self.hpars.log_hists:
                self.logger.experiment.add_histogram(
                    "X_disc", X, global_step=self.trainer.total_batch_idx
                )
                self.logger.experiment.add_histogram(
                    "A_disc", A, global_step=self.trainer.total_batch_idx
                )
                self.logger.experiment.add_histogram(
                    "N_disc", N, global_step=self.trainer.total_batch_idx
                )
            real_score_raw = self.score((X, A, N), fake=False)
            if self.training:
                fake_score_raw.register_hook(backward_trace_hook_t)
            real_score = real_score_raw.mean()
            fake_score = fake_score_raw.mean()
            W1 = real_score - fake_score
            grad_penalty = self.get_grad_penalty((X, A, N), fake=(Xf, Af, Nf))
            if self.hpars.independent_penalty_samples > 0:
                grad_penalty = [grad_penalty]
                for _ in range(self.hpars.independent_penalty_samples):
                    grad_penalty.append(self.get_grad_penalty((X, A, N)))
                grad_penalty = pt.stack(grad_penalty).logsumexp(
                    -1
                )  # beloved smooth max approximatio/interpolation between mean and max...

            disc_loss = -W1 + self.hpars.penalty_hpars.penalty_lambda * grad_penalty
            if self.hpars.score_penalty_lambda > 0.0:
                score_penalty = (real_score_raw ** 2).mean()  # +fake_score.norm()**2
                disc_loss = disc_loss + self.hpars.score_penalty_lambda * score_penalty
            else:
                score_penalty = pt.zeros_like(disc_loss)
            log = dict(
                W1=W1,
                disc_loss=disc_loss,
                real_score=real_score,
                fake_score=fake_score,
                score_pen=score_penalty,
            )
            log["grad_pen"] = grad_penalty
            ret = dict(
                loss=disc_loss,
                log=log,
                progress_bar=log,
            )
            if (
                self.trainer.total_batch_idx % self.hpars.grid_every == 0
                and self.hpars.viz
            ):
                A = batch[1]
                self.log_graph_plot(A, Af)
                if self.hpars.log_weights:
                    self.log_weights()
        elif optimizer_idx == 1:  # train generator
            if self.hpars.weight_clip:
                with torch.no_grad():
                    for p in self.discriminator.parameters(recurse=True):
                        p.data = pt.clip(p.data, -1.0, 1)
            Xf, Af, Nf, Zf = self.forward(device=A.device)
            if self.hpars.node_readout_hpars.node_attrib_dim == 0:
                Xf = None
            if self.hpars.log_hists:
                if Xf is not None:
                    self.logger.experiment.add_histogram(
                        "Xf_gen", Xf, global_step=self.trainer.total_batch_idx
                    )
                self.logger.experiment.add_histogram(
                    "Af_gen", Af, global_step=self.trainer.total_batch_idx
                )
                self.logger.experiment.add_histogram(
                    "Nf_gen", Nf, global_step=self.trainer.total_batch_idx
                )
            fake_score_raw = self.score((Xf, Af, Nf), fake=True)
            # real_score_raw = self.score((X, A, N), fake=False)
            # gen_loss = real_score_raw.mean()-fake_score_raw.mean()
            gen_loss = -fake_score_raw.mean()
            log = dict(gen_loss=gen_loss)
            ret = dict(loss=gen_loss, log=log, progress_bar=log)
        else:
            raise ValueError("Should only have 2 optimizers")

        assert pt.isfinite(ret["loss"]).all()
        return ret

    def log_graph_plot(self, A, Af):
        Af = Af.reshape(-1, Af.shape[-2], Af.shape[-1])
        A = A.reshape(-1, A.shape[-2], A.shape[-1])
        grid_real = make_grid(
            A.unsqueeze(1),
            nrow=int(math.ceil(math.sqrt(Af.shape[0]))),
        )
        self.logger.experiment.add_image(
            "real_graph_mat", grid_real, self.trainer.total_batch_idx
        )
        nrows = int(math.ceil(math.sqrt(Af.shape[0])))
        ncols = int(math.floor(Af.shape[0] / nrows))
        nrows = min(nrows, self.hpars.grid_max_row)
        ncols = min(ncols, self.hpars.grid_max_row)
        grid_fake = make_grid(
            Af.unsqueeze(1),
            nrow=nrows,
        )
        self.logger.experiment.add_image(
            "generated_graph_mat", grid_fake, self.trainer.total_batch_idx
        )
        sample_plot = cluster_plot_molgrid(
            [nx.from_numpy_array(x.detach().cpu().int().numpy()) for x in Af],
            lcc=True,
            name="Samples",
            rc=(nrows, ncols),
        )
        self.logger.experiment.add_figure(
            "fake_graph_plot",
            sample_plot,
            global_step=self.trainer.total_batch_idx,
        )
        dataset_plot = cluster_plot_molgrid(
            [nx.from_numpy_array(x.detach().cpu().int().numpy()) for x in A],
            name="Dataset",
            lcc=True,
            rc=(nrows, ncols),
        )
        self.logger.experiment.add_figure(
            "real_graph_plot",
            dataset_plot,
            global_step=self.trainer.total_batch_idx,
        )
        plt.close("all")

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

    def optimizer_step(
        self,
        epoch,
        batch_idx,
        optimizer,
        optimizer_idx,
        second_order_closure=None,
        **kwargs,
    ):
        if optimizer_idx == 0:
            optimizer.step()
            if (epoch - self.last_disc_grad_flow) > self.hpars.grad_flow_every or (
                epoch == 0 and self.last_disc_grad_flow < 0
            ):
                ret = plot_grad_flow(
                    self.discriminator.named_parameters(),
                    epoch,
                    self.hpars.save_dir,
                    self.hpars.exp_name,
                    to_file=self.hpars.plot_to_file,
                    to_tensor=self.hpars.plot_to_tensor,
                )
                self.last_disc_grad_flow = epoch
                if ret is not None:
                    fig, ax = ret
                    self.logger.experiment.add_figure(
                        "disc_grad_flow", fig, global_step=self.trainer.total_batch_idx
                    )
            optimizer.zero_grad()
        elif optimizer_idx == 1:
            optimizer.step()
            if (epoch - self.last_gen_grad_flow) > self.hpars.grad_flow_every or (
                epoch == 0 and self.last_gen_grad_flow < 0
            ):
                ret = plot_grad_flow(
                    self.generator.named_parameters(),
                    epoch,
                    self.hpars.save_dir,
                    self.hpars.exp_name,
                    g_=True,
                    to_file=self.hpars.plot_to_file,
                    to_tensor=self.hpars.plot_to_tensor,
                )
                self.last_gen_grad_flow = epoch
                if ret is not None:
                    fig, ax = ret
                    self.logger.experiment.add_figure(
                        "gen_grad_flow", fig, global_step=self.trainer.total_batch_idx
                    )
            optimizer.zero_grad()
        plt.close()

    def configure_optimizers(self):
        # optimizers
        gen_opt = self.hpars.gen_opt_hpars.make(params=self.generator.parameters())
        gen_sched = (
            self.hpars.gen_opt_hpars.sched.make(gen_opt, self.generator.parameters())
            if self.hpars.gen_opt_hpars.sched
            else None
        )
        disc_opt = self.hpars.disc_opt_hpars.make(self.discriminator.parameters())
        disc_sched = (
            self.hpars.disc_opt_hpars.sched.make(
                disc_opt, self.discriminator.parameters()
            )
            if self.hpars.disc_opt_hpars.sched
            else None
        )
        disc_int = getattr(self.hpars.disc_opt_hpars.sched, "interval", "epoch")
        gen_int = getattr(self.hpars.gen_opt_hpars.sched, "interval", "epoch")

        def mk_dict(opt, freq, sched, intv):
            d = dict(
                optimizer=opt,
                frequency=freq,
                lr_scheduler=dict(scheduler=sched, interval=intv),
            )
            if sched is None:
                d.pop("lr_scheduler")
            return d

        opts = tuple(
            [
                mk_dict(opt, freq, sched, intv)
                for opt, freq, sched, intv in zip(
                    [disc_opt, gen_opt],
                    [
                        self.hpars.disc_every,
                        self.hpars.gen_every,
                    ],
                    [disc_sched, gen_sched],
                    [disc_int, gen_int],
                )
            ]
        )
        return opts

    def prepare_data(self):
        self.train_set = self.hpars.dataset_hpars.make()

    def train_dataloader(self):
        dl = DataLoader(
            self.train_set,
            batch_size=self.hpars.batch_size,
            shuffle=self.hpars.shuffle,
            num_workers=self.hpars.num_workers,
            pin_memory=self.hpars.pin_memory,
        )
        return dl


#%% Unit tests
class Test_GGG:
    def test_creation(self):
        hpars = GGG_Hpar.with_dims()
        self.g = GGG(hpars)

    def test_sampling(self):
        ret = self.g.forward()
        info(f"{[x.shape for x in ret]}")
        info(f"{ret}")

    def test_lightning_start(self):
        self.g.prepare_data()
        self.g.configure_optimizers()


class Test_Root:
    def test_hpars(self):
        self.h = GenRootHpars()

    def test_make_all(self):
        for n in GenRootHpars.OPTIONS.keys():
            self.h.name = n
            self.h.make(pt.rand(5))


class Test_Trunk:
    HCLS = GenTrunkHpars

    def test_hpars(self):
        self.h = type(self).HCLS()

    def test_make_all(self):
        for n in type(self).HCLS.OPTIONS.keys():
            self.h.name = n
            self.h.make()


class Test_NodeReadout:
    HCLS = NodeReadoutHpars

    def test_hpars(self):
        self.h = type(self).HCLS()

    def test_make_all(self):
        for n in type(self).HCLS.OPTIONS.keys():
            self.h.name = n
            self.h.make()


class Test_EdgeReadout:
    HCLS = EdgeReadoutHpars

    def test_hpars(self):
        self.h = type(self).HCLS()

    def test_make_all(self):
        for n in type(self).HCLS.OPTIONS.keys():
            self.h.name = n
            self.h.make()


class Test_Discretize:
    HCLS = DiscretizationHpars

    def test_hpars(self):
        self.h = type(self).HCLS()

    def test_make_all(self):
        for n in type(self).HCLS.OPTIONS.keys():
            self.h.name = n
            self.h.make()


class Test_Dataset:
    HCLS = DatasetHpars

    def test_hpars(self):
        self.h = type(self).HCLS()

    def test_make_all(self):
        self.h.make()


class Test_GradPen:
    HCLS = GradPenHpars

    def test_hpars(self):
        self.h = type(self).HCLS()

    def test_make_all(self):
        self.h.make()


if __name__ == "__main__":
    TGR = Test_Root()
    TGR.test_hpars()
    TGR.test_make_all()
    #
    TNR = Test_NodeReadout()
    TNR.test_hpars()
    TNR.test_make_all()
    #
    TER = Test_EdgeReadout()
    TER.test_hpars()
    TER.test_make_all()
    #
    TDR = Test_Discretize()
    TDR.test_hpars()
    TDR.test_make_all()
    #
    TTR = Test_Trunk()
    TTR.test_hpars()
    TTR.test_make_all()
    #
    TDS = Test_Dataset()
    TDS.test_hpars()
    TDS.test_make_all()
    #
    TGPS = Test_GradPen()
    TGPS.test_hpars()
    TGPS.test_make_all()
    #
    TG = Test_GGG()
    TG.test_creation()
    TG.test_sampling()
    TG.test_lightning_start()
    #
