import os
from datetime import date

from ggg.models.ggg_model import GGG_Hpar
from .base import ex
import attr

from ...data.dense.hpars import DatasetHpars
from ...models.components.discriminators.kCycleGIN import DiscriminatorHpars
from ...models.components.generators.att.readouts import (
    EdgeReadoutHpars,
    NodeReadoutHpars,
)
from ...models.components.generators.att.roots import GenRootHpars
from ...models.components.generators.att.trunks import GenTrunkHpars
from ...utils.grad_penalty import GradPenHpars

BASE_LR = 1e-4


@ex.config
def config():

    hyper = GGG_Hpar.with_dims().to_dict()
    save_dir = None

    epochs = 1001
    track_norm = 100

    log_k = 1

    overfit_pct = 0.0
    ckpt_period = 50
    detect_anomaly = False
    forward_clip = False
    backward_clean = False

    clip_grad_val = 0.5
    val_every = 1
    gpus = 0
    benchmark=None

date_ = str(date.today()).replace("-", "_")
# anon
@ex.named_config
def report_base_comm100_RF1():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_100", node_feature=0, context_dim=15, phi_dim=25,
                               structured_features=True, num_labels=None, conv_channels=[128, 128, 128, 128, 128],
                               exp_name="GGG_ICML{}".format(date_)).to_dict()

@ex.named_config
def report_base_comm100_RF2():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_100", node_feature=0, context_dim=25, phi_dim=35,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}".format(date_)).to_dict()

@ex.named_config
def report_base_comm100_RF3():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_100", node_feature=0, context_dim=25, phi_dim=35,
                               structured_features=True, num_labels=None, eigenfeat4_=True,
                               exp_name="GGG_ICML{}".format(date_)).to_dict()

@ex.named_config
def report_base_zync_RF1():
    hyper=GGG_Hpar.report_base(dataset="ZINC", node_feature=0, context_dim=15, phi_dim=25,
                               structured_features=True, num_labels=None, noise_dist_trunk="bernoulli",
                               edge_readout="double_softmax", dts_limit_train=5000,
                               conv_channels=[64, 64, 64, 128, 128],
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_zync_RF2():
    hyper=GGG_Hpar.report_base(dataset="ZINC", node_feature=0, context_dim=20, phi_dim=25,
                               structured_features=True, num_labels=None, dts_limit_train=5000,
                               conv_channels=[64, 64, 64, 128, 128],
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_zync_RF3():
    hyper=GGG_Hpar.report_base(dataset="ZINC", node_feature=0, context_dim=15, phi_dim=25,
                               structured_features=True, num_labels=None, dts_limit_train=5000,
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_zync_RF4():
    hyper=GGG_Hpar.report_base(dataset="ZINC", node_feature=0, context_dim=25, phi_dim=35, eigenfeat4_=True,
                               structured_features=True, num_labels=None, dts_limit_train=5000,
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_zync_RF5():
    hyper=GGG_Hpar.report_base(dataset="ZINC", node_feature=0, context_dim=25, phi_dim=35,
                               structured_features=True, num_labels=None, dts_limit_train=5000,
                               conv_channels=[64, 64, 64, 128, 128],
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_ogb_mol_RF1():
    hyper=GGG_Hpar.report_base(dataset="ogbg-molpcba", node_feature=0, context_dim=25, phi_dim=35,
                               structured_features=True, num_labels=None,
                               conv_channels=[32, 32, 32, 32, 64, 128, 128],
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_ogb_mol_RF2():
    hyper=GGG_Hpar.report_base(dataset="ogbg-molpcba", nx_kwargs=dict(max_size=70),
                               node_feature=0, context_dim=25, phi_dim=35,
                               structured_features=True, num_labels=None, eigenfeat4_=True,
                               conv_channels=[32, 32, 32, 64, 64, 128, 128],
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_ogb_mol_RF3():
    hyper=GGG_Hpar.report_base(dataset="ogbg-molpcba", node_feature=0, context_dim=25, phi_dim=35,
                               structured_features=True, num_labels=None, gen_every=3, input_agg='mean',
                               conv_channels=[32, 64, 64, 64, 128, 128], readout_agg='mean',
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_ogb_mol_RF4():
    hyper=GGG_Hpar.report_base(dataset="ogbg-molpcba", node_feature=0, context_dim=25, phi_dim=35,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

# nala
@ex.named_config
def report_base_nala_comm100_RF1():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_100", node_feature=0, context_dim=28, phi_dim=35,
                               structured_features=True, num_labels=None, eigenfeat4_=True,
                               exp_name="GGG_ICML{}".format(date_)).to_dict()

@ex.named_config
def report_base_nala_ogb_mol_RF1():
    hyper=GGG_Hpar.report_base(dataset="ogbg-molpcba", node_feature=0, context_dim=25, phi_dim=35,
                               conv_channels=[32, 32, 32, 32, 64, 128], eigenfeat4_=True,
                               structured_features=True, num_labels=None, readout_agg="mean",
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_nala_ogb_mol_RF2():
    hyper=GGG_Hpar.report_base(dataset="ogbg-molpcba", node_feature=0, context_dim=20, phi_dim=30,
                               conv_channels=[64, 64, 64, 64, 64, 128, 128],
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_nala_ogb_mol_RF3():
    hyper=GGG_Hpar.report_base(dataset="ogbg-molpcba", node_feature=0, context_dim=20, phi_dim=25,
                               structured_features=True, num_labels=None, eigenfeat4_=True,
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()


@ex.named_config
def report_base_nala_ogb_mol_RF4():
    hyper=GGG_Hpar.report_base(dataset="ogbg-molpcba", node_feature=0, context_dim=10, phi_dim=15,
                               structured_features=True, num_labels=None, eigenfeat4_=True,
                               conv_channels=[32, 32, 32, 32, 64, 64, 128],
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_zync_nala_RF1():
    hyper=GGG_Hpar.report_base(dataset="ZINC", node_feature=0, context_dim=15, phi_dim=25,
                               structured_features=True, num_labels=None, dts_limit_train=5000,
                               conv_channels=[64, 64, 64, 128, 128],
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_zync_nala_RF2():
    hyper=GGG_Hpar.report_base(dataset="ZINC", node_feature=0, context_dim=15, phi_dim=20,
                               structured_features=True, num_labels=None, dts_limit_train=5000,
                               conv_channels=[32, 32, 32, 64, 128, 128],
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def report_base_zync_nala_RF3():
    hyper=GGG_Hpar.report_base(dataset="ZINC", node_feature=0, context_dim=15, phi_dim=25,
                               structured_features=True, eigenfeat4_=True, num_labels=None, dts_limit_train=5000,
                               exp_name="GGG_ICML_{}".format(date_)).to_dict()

# @ex.named_config
# def report_base_gran_dataset_DD_RF1():
#     hyper=GGG_Hpar.report_base(dataset="gran_dataset_DD", node_feature=0, context_dim=15, phi_dim=50,
#                                structured_features=True, num_labels=None,
#                                conv_channels=[32, 32, 64, 64, 128],
#                                exp_name="GGG_ICML_{}".format(date_)).to_dict()

# @ex.named_config
# def report_base_comm400_RF1():
#     hyper=GGG_Hpar.report_base(dataset="CommunitySmall_200", node_feature=0, context_dim=35, phi_dim=25, gen_every=3,
#                                structured_features=True, num_labels=None, conv_channels=[128, 128, 128, 128, 128],
#                                exp_name="GGG_ICML{}_10".format(run)).to_dict()

# @ex.named_config
# def report_base_comm200_4_RF2():
#     hyper=GGG_Hpar.report_base(dataset="CommunitySmall_200_4", node_feature=0, context_dim=10, phi_dim=35,
#                                structured_features=True, num_labels=None,
#                                exp_name="GGG_ICML{}_8".format(run)).to_dict()

# @ex.named_config
# def report_base_mnist_RF1():
#     hyper=GGG_Hpar.report_base(dataset="GNNBenchmarkDataset", nx_kwargs=dict(name="MNIST"),
#                                node_feature=0, context_dim=25, phi_dim=35, dts_limit_train=5000,
#                                structured_features=True, num_labels=None,
#                                exp_name="GGG_ICML_{}".format(date_)).to_dict()

@ex.named_config
def jnf():
    hyper = dict(jnf=dict(N=10, eps=1e-2, ratio=0.7))


@ex.named_config
def smyrf_8_4():
    hyper = dict(smyrf=dict(cluster_size=8, n_hashes=4))


@ex.named_config
def smyrf_20_4():
    hyper = dict(smyrf=dict(cluster_size=20, n_hashes=4))


@ex.named_config
def report_base():
    # molgan 5k base
    hyper = GGG_Hpar.report_base().to_dict()


@ex.named_config
def pointnetst_QM9():
    hyper = GGG_Hpar.pointnet_st(
        dataset="MolGAN_5k",
    ).to_dict()


@ex.named_config
def mlprow_qm9():
    hyper = GGG_Hpar.mlprow(
        dataset="MolGAN_5k",
    ).to_dict()


@ex.named_config
def mlprow_chordal9():
    hyper = GGG_Hpar.mlprow(
        dataset="anu_graphs_chordal9",
    ).to_dict()


@ex.named_config
def mlprow_commsmall20():
    hyper = GGG_Hpar.mlprow(dataset="CommunitySmall_20")


@ex.named_config
def attention_qm9():
    hyper = GGG_Hpar.attention(dataset="MolGAN_5k").to_dict()


@ex.named_config
def attention_chordal9():
    hyper = GGG_Hpar.attention(dataset="anu_graphs_chordal9").to_dict()
@ex.named_config
def attention_egonet():
    hyper = GGG_Hpar.attention(dataset="egonet20-1",dataset_kwargs=dict(skip_features=True),data_repeat=50).to_dict()


@ex.named_config
def attention_community():
    hyper = GGG_Hpar.attention_community().to_dict()
@ex.named_config
def gradpen_perturbed():
    hyper=dict(penalty_hpars=dict(on=["real","fake","real-perturbed","fake-perturbed"],modes=["LP","LP","LP","LP"]))
@ex.named_config
def gradpen_perturbed_int():
    hyper=dict(penalty_hpars=dict(on=["real","fake","real-perturbed","fake-perturbed","int"],modes=["LP","LP","LP","LP","LP"]))
@ex.named_config
def attention_star():
    # nx_star, nx_circ_ladder only have "N_nodes" as input
    hyper = GGG_Hpar.attention_nx(nx_dataset="nx_star",nx_kwargs=dict(N_nodes=[2,3,4,5,6],repeat=200)).to_dict()
@ex.named_config
def attention_ministar():
    # nx_star, nx_circ_ladder only have "N_nodes" as input
    hyper = GGG_Hpar.attention_nx(nx_dataset="nx_star",nx_kwargs=dict(N_nodes=[2,3,5],repeat=500)).to_dict()

@ex.named_config
def point_reference():
    def mh():
        allow_greater=False
        root="geometric"
        nx_dataset = "nx_star"
        nx_kwargs = dict(N_nodes=[2,3,5],repeat=500)
        trunk = "pointmlp"
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
        disc_conv_channels = [32, 64, 128, 128, 128, 128, 128]
        gen_trunk_score_function = "softmax"
        node_attrib_dim = 0
        use_laplacian = True
        structured_features = True
        nfeatdim = context_dim + phi_dim
        NUM_HEADS = 7
        attn_feat_dim = 25 * NUM_HEADS
        hyper = GGG_Hpar(
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
            norm_type="identity",  # layer works as well
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
        return hyper
    hyper=mh().to_dict()
    del mh

@ex.named_config
def noneq_reference():
    def mh():
        allow_greater=False
        root="noneq"
        nx_dataset = "nx_star"
        nx_kwargs = dict(N_nodes=[2,3,5],repeat=500)
        trunk = "mlp"
        contrast_mode = "fake-struct_fake"
        edge_readout = "noneq"
        edge_score = "softmax"
        edge_readout_score = "sigmoid"
        eigenfeat4 = True
        kc_flag = True
        context_dim = 50
        phi_dim = 0
        n_attention_layers = 5
        num_labels = None
        batch_size = 16
        disc_conv_channels = [32, 64, 128, 128, 128, 128, 128]
        gen_trunk_score_function = "softmax"
        node_attrib_dim = 0
        use_laplacian = True
        structured_features = True
        nfeatdim = context_dim + phi_dim
        NUM_HEADS = 7
        attn_feat_dim = 25 * NUM_HEADS
        hyper = GGG_Hpar(
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
            norm_type="identity",  # layer works as well
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
            max_communities="compute",
            max_N=7
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
        return hyper
    hyper=mh().to_dict()
    del mh
@ex.named_config
def ggrs_reference():
    def mh():
        allow_greater=False
        root="random"
        nx_dataset = "nx_star"
        nx_kwargs = dict(N_nodes=[2,3,5],repeat=500)
        trunk = "attention"
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
        disc_conv_channels = [32, 64, 128, 128, 128, 128, 128]
        gen_trunk_score_function = "softmax"
        node_attrib_dim = 0
        use_laplacian = True
        structured_features = True
        nfeatdim = context_dim + phi_dim
        NUM_HEADS = 7
        attn_feat_dim = 25 * NUM_HEADS
        hyper = GGG_Hpar(
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
            norm_type="identity",  # layer works as well
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
        return hyper
    hyper=mh().to_dict()
    del mh
@ex.named_config
def attention_reference():
    def mh():
        allow_greater=False
        root="geometric"
        nx_dataset = "nx_star"
        nx_kwargs = dict(N_nodes=[2,3,5],repeat=500)
        trunk = "attention"
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
        disc_conv_channels = [32, 64, 128, 128, 128, 128, 128]
        gen_trunk_score_function = "softmax"
        node_attrib_dim = 0
        use_laplacian = True
        structured_features = True
        nfeatdim = context_dim + phi_dim
        NUM_HEADS = 7
        attn_feat_dim = 25 * NUM_HEADS
        hyper = GGG_Hpar(
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
            norm_type="identity",  # layer works as well
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
        return hyper
    hyper=mh().to_dict()
    del mh


@ex.named_config
def attention_cla():
    # nx_star, nx_circ_ladder only have "N_nodes" as input
    hyper = GGG_Hpar.attention_nx(nx_dataset="nx_circ_ladder",nx_kwargs=dict(N_nodes=[2,3,4,5,6],repeat=200)).to_dict()
@ex.named_config
def attention_lolli():
    # nx_lollipop takes N_path,N_cluster either int,int, list,int or list list, here we give a list list combo
    hyper = GGG_Hpar.attention_nx(nx_dataset="nx_lollipop",nx_kwargs=dict(N_path=list(range(5))*2,N_cluster=[3]*5+[4]*5,repeat=20)).to_dict()
@ex.named_config
def attention_roc():
    # nx_roc takes num_cliques,clique_sizes either int,int, list,int or list list, here we give a list list combo
    hyper = GGG_Hpar.attention_nx(nx_dataset="nx_roc",nx_kwargs=dict(num_cliques=list(range(2,7))*2,clique_sizes=[3]*5+[2]*5,repeat=20)).to_dict()
@ex.named_config
def attention_nx_large():
    # nx_roc takes num_cliques,clique_sizes either int,int, list,int or list list, here we give a list list combo
    hyper = GGG_Hpar.attention_nx(nx_dataset="nx_combo",nx_kwargs=dict(Ns=list(range(100,120,2)),repeat=5)).to_dict()
@ex.named_config
def attention_nx_large_diverse():
    # nx_roc takes num_cliques,clique_sizes either int,int, list,int or list list, here we give a list list combo
    hyper = GGG_Hpar.attention_nx(nx_dataset="nx_combo",nx_kwargs=dict(Ns=list(range(100,200,2)),repeat=5)).to_dict()
