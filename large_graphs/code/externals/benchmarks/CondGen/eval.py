import matplotlib

matplotlib.use("Agg")
from pprint import pprint
import networkx as nx
from graph_stat import *
from collections import defaultdict
import os
import numpy as np
import os
import attr
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict
from pprint import pprint
from collections import defaultdict

from graph_stat import *
from torch.utils.data import Dataset
from options import Options, AttrOptions
from GVGAN import *
from utils import *
from tqdm import tqdm
from pprint import pprint
from sacred import Experiment

ex = Experiment("CondgenEval")
import torch.nn as nn
import torch.optim as optim
import warnings
from ggg.data.dense.QM9.MolGAN_QM9 import QM9preprocess
from ggg.evaluation.plots.utils.post_experiment_plots import (
    generate_graphs,
    main_run_plot,
    main_run_MMD,
)
from ggg.evaluation.plots.mmds_isomorphism import cluster_plot_novelty
from ggg.evaluation.plots.graph_grid import cluster_plot_molgrid

from models import *
from train import load_data, load_model_parts, G_Wrap, load_our_data


def pad_shape(S, M):
    s0, s1 = S
    return tuple((0, M - s1, 0, M - s0))


def pad_full(ts):
    N_max = max([max(t.shape) for t in ts])
    ts = [nn.functional.pad(t, pad_shape(t.shape, N_max)) for t in ts]
    return ts


def evaluate(
    train_adj_mats,
    test_adj_mats,
    train_attr_vecs,
    test_attr_vecs,
    opt=None,
    _run=None,
    epoch=99,
    N_samples=50,
    rand_init=False,
):
    training_index = list(range(0, len(train_adj_mats)))

    z_out_size = opt.z_size + opt.av_size

    # TO MAKE EVERYTHING EASY, NO CLASS GVGAN() HERE...

    G = Generator(
        av_size=opt.av_size,
        d_size=opt.d_size,
        gc_size=opt.gc_size,
        z_size=opt.z_size,
        z_out_size=z_out_size,
        rep_size=opt.rep_size,
    )

    D = Discriminator(
        av_size=opt.av_size,
        d_size=opt.d_size,
        gc_size=opt.gc_size,
        rep_size=opt.rep_size,
    )

    if not rand_init:
        print("Training set")
        model_dict = dict(G=G, D=D)
        print("Loading model")
        load_model_parts(model_dict, opt.output_dir, epoch)
    G.eval()
    D.eval()
    return D, G
    indices = np.random.randint(len(train_adj_mats), size=N_samples)
    avecs = []
    base_adjs = []
    sampled_adjs = []
    for i in tqdm(indices, desc="Sampling"):
        base_adj = train_adj_mats[i]

        if base_adj.shape[0] <= opt.d_size:
            continue
        attr_vec = Variable(torch.from_numpy(train_attr_vecs[i]).float())

        # add a new line
        G.set_attr_vec(attr_vec)

        sample_adj = gen_adj(
            G,
            base_adj.shape[0],
            int(np.sum(base_adj)) // 2,
            attr_vec,
            z_size=opt.z_size,
        )
        # show_graph(sample_adj, base_adj=base_adj, remove_isolated=True,epoch=epoch,sample=i,dataset=os.path.basename(os.path.abspath(opt.DATA_DIR)),opt=opt)
        base_adjs.append(base_adj)
        sampled_adjs.append(sample_adj.cpu().detach().numpy())
        avecs.append(train_attr_vecs[i])
    base_adjs = [torch.from_numpy(x) for x in base_adjs]
    # base_adjs=torch.stack(pad_full(base_adjs))
    sample_adjs = [torch.from_numpy(x) for x in sampled_adjs]
    # sample_adjs=torch.stack(pad_full(sample_adjs))
    avecs = torch.from_numpy(np.stack(avecs, 0))
    with open(os.path.join(opt.output_dir, f"eval_{epoch!a}.pt"), "wb") as f:
        torch.save(dict(base_adjs=base_adjs, sample_adjs=sampled_adjs, avecs=avecs), f)


@ex.config
def conf():
    hpars = attr.asdict(AttrOptions())
    our_dataset = None
    batch_size = 20
    epoch = 99
    N_samples = 512


@ex.named_config
def dblp():
    hpars = attr.asdict(
        AttrOptions(
            output_dir="dblp_out", DATA_DIR="data/data_dblp/", av_size=10, d_size=5
        )
    )


@ex.named_config
def chordal_9():
    our_dataset = "anu_graphs_chordal9"
    hpars = attr.asdict(
        AttrOptions(
            output_dir="externals/benchmarks/CondGen/chrodal/17",
            DATA_DIR="externals/benchmarks/CondGen/data/",
            av_size=0,
            d_size=5,
        )
    )


@ex.named_config
def molgan5k():
    our_dataset = "MolGAN_5k"
    hpars = attr.asdict(
        AttrOptions(
            output_dir="externals/benchmarks/CondGen/qm95k_run/10",
            DATA_DIR="externals/benchmarks/CondGen/data/",
            av_size=0,
            d_size=5,
        )
    )


@ex.named_config
def community20():
    our_dataset = "CommunitySmall_20"
    hpars = attr.asdict(
        AttrOptions(
            output_dir="externals/benchmarks/CondGen/community20/4",
            DATA_DIR="externals/benchmarks/CondGen/data/",
            av_size=0,
            d_size=10,
        )
    )
@ex.named_config
def community100():
    our_dataset = "CommunitySmall_100"
    hpars = attr.asdict(
        AttrOptions(
            output_dir="community100",
            DATA_DIR="externals/benchmarks/CondGen/data/",
            av_size=0,
            d_size=50,
        )
    )
    epoch=38


@ex.named_config
def tcga():
    hpars = attr.asdict(
        AttrOptions(
            output_dir="tcga_out", DATA_DIR="data/data_tcga/", d_size=10, av_size=8
        )
    )
    epoch = 99
    N_samples = 50


def load_npz_keys(keys, file):
    """
    Small utility to directly load an npz_compressed file
    :param keys:
    :param file:
    :return:
    """
    out = []
    with np.load(file) as d:
        for k in keys:
            out.append(d[k])
    return tuple(out) if len(out) > 1 else out[0]


@ex.automain
def run(hpars, _run, _config, epoch, N_samples, our_dataset, batch_size):
    batch_size = 1
    print(_config)
    opt = AttrOptions(**sacred_copy(hpars))

    print("=========== OPTIONS ===========")
    pprint(opt)
    print(" ======== END OPTIONS ========\n\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opt.gpu}"

    if our_dataset is None:
        train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs = load_data(
            DATA_DIR=opt.DATA_DIR
        )
    else:
        assert opt.av_size == 0, "Our datasets don't have conditional vectors"
        train_adj_mats, test_adj_mats = load_our_data(our_dataset)
        train_attr_vecs, test_attr_vecs = None, None

    print("Got data")
    # output_dir = opt.output_dir
    D, G = evaluate(
        train_adj_mats,
        test_adj_mats,
        train_attr_vecs,
        test_attr_vecs,
        opt=opt,
        epoch=epoch,
        N_samples=3,
        rand_init=False,
    )
    G.eval()
    if len(_run.observers) >= 1:
        print(f"Changing output_dir from {opt.output_dir} to {_run.observers[0].dir}")
        opt.output_dir = _run.observers[0].dir
    elif os.path.exists(opt.output_dir):
        opt.output_dir = f"{opt.output_dir}_eval"
        os.makedirs(opt.output_dir, exist_ok=True)
    Gw = G_Wrap(G, train_adj_mats, opt.z_size, opt.d_size, attr_vecs=train_attr_vecs)
    dataset = os.path.basename(opt.DATA_DIR)
    plots_save_dir = os.path.join(opt.output_dir, "plots")
    data_save_dir = os.path.join(opt.output_dir, "data")
    for x in [plots_save_dir, data_save_dir]:
        os.makedirs(x, exist_ok=True)
    N_GRAPH = N_samples
    if N_GRAPH < 1024:
        print(f"Warning, using only {N_GRAPH} for debugging")
    Gw.sample(5, "cpu", True, opt)

    gen_graphs, _ = generate_graphs(
        Gw,
        current_epoch=opt.max_epochs,
        dataset=None,
        numb_graphs=N_GRAPH,
        save_dir=data_save_dir,
        device="cpu",
        batch_size=1 if batch_size is None else batch_size,
    )

    dataset_graphs = [nx.from_numpy_matrix(g) for g in train_adj_mats]
    # main_run_plot(opt.max_epochs,f"condgen_{dataset}",dataset, model_graphs=gen_graphs, dataset_graphs=dataset_graphs, loss_dir=None, plots_save_dir=plots_save_dir)
    (
        degree_metric,
        clustering_metric,
        cycle_metric,
        degree_assortativity_metric,
        algebraic_connectivity_metric,
    ) = main_run_MMD(
        opt.max_epochs,
        csv_dir=plots_save_dir,
        model_graphs=gen_graphs,
        dataset_graphs=dataset_graphs,
        numb_graphs=N_GRAPH,
    )
    print(cycle_metric)
    # sample_plot = cluster_plot_molgrid(gen_graphs, name="CondGEN_" + our_dataset, lcc=False, save_dir="", save=True)

    # chordal_ = True
    #
    # cluster_plot_novelty(models_g=[gen_graphs], dataset_g=dataset_graphs,
    #                      legends=["CondGEN"], numb_g_eval=5000, reps=1,
    #                      save=False, save_dir="", chordal=chordal_)
