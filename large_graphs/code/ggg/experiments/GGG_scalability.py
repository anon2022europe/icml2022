# -*- coding: utf-8 -*-
import os
import matplotlib

matplotlib.use("Agg")
import numpy as np

from ggg.evaluation.plots.utils.benchmark_gen import (
    benchmark_graph_gen,
    plot_benchmark,
)
from sacred.observers import FileStorageObserver

from argparse import Namespace
from uuid import uuid4

from ggg.models.ggg_model import (
    GGG,
    GGG_Hpar,
)

from sacred import Experiment
import attr
import copy
import sys

# TODO(adam): delete this once Sacred issue #498 is resolved
from sacred.run import Run


def sacred_copy(o):
    """Perform a deep copy on nested dictionaries and lists.
    If `d` is an instance of dict or list, copies `d` to a dict or list
    where the values are recursively copied using `sacred_copy`. Otherwise, `d`
    is copied using `copy.deepcopy`. Note this intentionally loses subclasses.
    This is useful if e.g. `d` is a Sacred read-only dict. However, it can be
    undesirable if e.g. `d` is an OrderedDict.
    :param o: (object) if dict, copy recursively; otherwise, use `copy.deepcopy`.
    :return A deep copy of d."""
    if isinstance(o, dict):
        return {k: sacred_copy(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [sacred_copy(v) for v in o]
    else:
        return copy.deepcopy(o)


EXPNAME = "PEAWGAN_SCALABILTIY"
ex = Experiment(EXPNAME)
if len(ex.observers) == 0:
    ex.observers.append(FileStorageObserver(EXPNAME))


@ex.config
def config():
    base_dir = os.path.abspath(os.getcwd())

    hyper = attr.asdict(
        GGG_Hpar(
            None,
            batch_size=2,
            device="cuda:0",
            save_dir="scale",
            n_attention_layers=12,
            embed_dim=50,
            finetti_dim=50,
        )
    )

    model = "GG-GAN"
    model_n = None

    deep = False
    deep_disc = False
    deep_gen = False
    max_num_nodes = 1000
    step = 50
    style = "box"
    device = "cuda:0"
    num_samples = 100
    warm_up = 10


def get_model_n_uuid(hyper):
    ds = hyper["dataset"]
    return f"{ds}_{uuid4()}"


def get_model_n(hyper):
    return (
        "Arch="
        + str(hyper["architecture"])
        + "_Z0dim="
        + str(hyper["embed_dim"])
        + "_CVdim="
        + str(hyper["finetti_dim"])
        + "_Trainable="
        + str(hyper["finetti_trainable"])
        + "FixCV="
        + str(hyper["finetti_train_fix_context"])
        + "DynamicCreation="
        + str(hyper["dynamic_finetti_creation"])
        + "FlipFinetti="
        + str(hyper["flip_finetti"])
        + "_Model="
        + str(hyper["cycle_opt"])
        + "_AttentionMode="
        # + str(hyper["attention_mode"])
        # + "_Heads="
        + str(hyper["num_heads"])
        + "_EdgeReadout="
        + str(hyper["edge_readout"])
        # + "_#Layers="
        # + str(hyper["n_attention_layers"])
        # + "_DiscLayers="
        # + str(hyper["disc_conv_channels"])
        # + "_EBMode="
        # + str(hyper["edge_bias_mode"])
        # + "_EBHidden="
        # + str(hyper["edge_bias_hidden"])
        # + "_DiscAdamLr="
        # + "_"
        # + str(hyper["disc_optim_args"]["lr"])[2:]
        # + "_betas="
        # + str(hyper["disc_optim_args"]["betas"])
        # + "_T="
        # + str(round(hyper["temperature"], 3))
        + "_Dataset="
        + str(hyper["dataset"])
    )


@ex.named_config
def attention_ComSmall20_emb5_sig_bigdisc_bigA():
    hyper = dict(
        dataset="CommunitySmall_20",
        device="cuda:0",
        n_attention_layers=24,
        cut_train_size=False,
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=5,
        embed_dim=5,
        finetti_dim=5,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 64, 64, 64],
        cycle_opt="finetti_noDS",
        score_function="sigmoid",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        structured_features=True,
    )


@ex.named_config
def attention_ComSmall20_emb5_sig_bigdiscnoKC_bigA():
    hyper = dict(
        dataset="CommunitySmall_20",
        device="cuda:0",
        n_attention_layers=24,
        cut_train_size=False,
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=5,
        embed_dim=5,
        finetti_dim=5,
        kc_flag=False,
        disc_conv_channels=[32, 64, 64, 64, 64, 64, 64],
        cycle_opt="finetti_noDS",
        score_function="sigmoid",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        structured_features=True,
    )


@ex.named_config
def attention_ComSmall20_emb2_sig_bigdisc_bigA():
    hyper = dict(
        dataset="CommunitySmall_20",
        device="cuda:0",
        n_attention_layers=24,
        cut_train_size=False,
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=5,
        embed_dim=2,
        finetti_dim=2,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 64, 64, 64],
        cycle_opt="finetti_noDS",
        score_function="sigmoid",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        structured_features=True,
    )


@ex.named_config
def attention_QM9_emb2_sigA_bigdisc_bigA():
    hyper = dict(
        dataset="MolGAN_5k",
        device="cuda:0",
        n_attention_layers=24,
        cut_train_size=False,
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=5,
        embed_dim=2,
        finetti_dim=2,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 64, 64, 64],
        cycle_opt="finetti_noDS",
        score_function="softmax",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        structured_features=False,
    )


@ex.named_config
def attention_QM9_emb20_sigA_bigdisc():
    hyper = dict(
        dataset="MolGAN_5k",
        device="cuda:0",
        n_attention_layers=12,
        cut_train_size=False,
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=5,
        embed_dim=20,
        finetti_dim=20,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 64, 64, 64],
        cycle_opt="finetti_noDS",
        score_function="softmax",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        structured_features=False,
    )


@ex.named_config
def condgen_dblp():
    hyper = dict(
        dataset="condgen_dblp",
        node_feature_dim=10,
        dataset_kwargs=dict(DATA_DIR="/home/anon/graphs/data_dblp"),
        label_one_hot=None,
    )


@ex.named_config
def condgen_tcga():
    hyper = dict(
        dataset="condgen_tcga",
        label_one_hot=None,
        node_feature_dim=10,  # TODO: check, this should not need 10 for the node_feature+1 setup?
        dataset_kwargs=dict(DATA_DIR="/home/anon/graphs/data_tcga"),
    )


def get_funcs(model, device, inner_args):
    implemented = {"GG-GAN"}
    if model not in implemented:
        raise NotImplementedError(f"Unkown model {model}, only know {implemented}")
    elif model == "GG-GAN":

        def create_model(n):
            node_count_weights = np.zeros(n)
            node_count_weights[-1] = 1.0

            hparams = inner_args.get("hyper")
            hparams["data_dir"] = "data"
            hparams["save_dir"] = "scale"
            hparams["node_count_weights"] = node_count_weights

            model = GGG(hparams)
            model.eval()
            assert model.training == False
            assert model.generator.training == False
            model = model.to(device)
            return model

        def sample_func(model):
            return model.sample()

    return create_model, sample_func


@ex.main
def run(
    hyper,
    model,
    model_n,
    base_dir,
    warm_up,
    deep,
    deep_gen,
    deep_disc,
    max_num_nodes,
    step,
    num_samples,
    device,
    _run: Run,
):
    hyper, model_n, base_dir = [
        sacred_copy(o)
        for o in [
            hyper,
            model_n,
            base_dir,
        ]
    ]
    if model_n is None:
        model_n = get_model_n_uuid(hyper)
    filename = None
    create_model, sample_func = get_funcs(
        model,
        device,
        dict(
            filename=filename,
            base_dir=base_dir,
            model_n=model_n,
            hyper=hyper,
            deep=deep,
            deep_disc=deep_disc,
            deep_gen=deep_gen,
        ),
    )

    logdir = _run.observers[0].dir
    plots_save_dir = os.path.join(logdir, "plots")
    os.makedirs(plots_save_dir, exist_ok=True)
    benchmark_data = benchmark_graph_gen(
        create_model,
        sample_func,
        num_nodes=(
            1,
            10,
            100,
            1000,
            2000,
            4000,
            6000,
            8000,
            10000,
            12000,
            14000,
            16000,
            18000,
            20000,
        ),
        num_samples=num_samples,
        name=f"{model}-{device}",
        warm_up=warm_up,
    )
    benchmark_data.save(logdir)
    plot_benchmark(benchmark_data, plots_save_dir)


if __name__ == "__main__":
    ex.run_commandline(sys.argv)
