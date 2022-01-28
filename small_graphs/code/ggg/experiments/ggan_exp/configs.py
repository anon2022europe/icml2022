import os

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

BASE_LR = 1e-4


@ex.config
def config():

    hyper = GGG_Hpar.with_dims().to_dict()
    save_dir = None

    epochs = 1001
    track_norm = 2

    log_k = 1

    overfit_pct = 0.0
    ckpt_period = 50
    detect_anomaly = False
    forward_clip = False
    backward_clean=False

    clip_grad_val = 0.5
    val_every = 1
    gpus = 0

run = 0
@ex.named_config
def report_base_qm9_RF1():
    hyper=GGG_Hpar.report_base(dataset="MolGAN_5k", node_feature=0, context_dim=5,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_1".format(run)).to_dict()

@ex.named_config
def report_base_qm9_RF2():
    hyper=GGG_Hpar.report_base(dataset="MolGAN_5k", node_feature=0, context_dim=10,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_2".format(run)).to_dict()

@ex.named_config
def report_base_comm20_RF1():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_20", node_feature=0, context_dim=25,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_3".format(run)).to_dict()

@ex.named_config
def report_base_comm50_RF1():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_50", node_feature=0, context_dim=25, phi_dim=40,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_4".format(run)).to_dict()

@ex.named_config
def report_base_comm50_RF2():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_50", node_feature=0, context_dim=25, phi_dim=40, gen_every=1,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_5".format(run)).to_dict()

@ex.named_config
def report_base_comm100_RF1():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_100", node_feature=0, context_dim=40, phi_dim=50, gen_every=1,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_6".format(run)).to_dict()

@ex.named_config
def report_base_comm100_RF2():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_100", node_feature=0, context_dim=40, phi_dim=50,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_7".format(run)).to_dict()

@ex.named_config
def report_base_comm200_RF1():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_100", node_feature=0, context_dim=40, phi_dim=80, gen_every=1,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_8".format(run)).to_dict()

@ex.named_config
def report_base_qm9_gen1():
    hyper=GGG_Hpar.report_base(dataset="MolGAN_5k", node_feature=0, context_dim=10, gen_every=1,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_9".format(run)).to_dict()

@ex.named_config
def report_base_comm50_smaller():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_50", node_feature=0, context_dim=25, phi_dim=25, gen_every=1,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_10".format(run)).to_dict()

@ex.named_config
def report_base_comm100_smaller():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_100", node_feature=0, context_dim=25, phi_dim=35,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_11".format(run)).to_dict()

@ex.named_config
def report_base_comm200_smaller():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_200", node_feature=0, context_dim=25, phi_dim=40, gen_every=1,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_12".format(run)).to_dict()

@ex.named_config
def report_base_comm20_smaller():
    hyper=GGG_Hpar.report_base(dataset="CommunitySmall_20", node_feature=0, context_dim=10,
                               structured_features=True, num_labels=None,
                               exp_name="GGG_ICML{}_13".format(run)).to_dict()

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
