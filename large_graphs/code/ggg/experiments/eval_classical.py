from sacred import Experiment
from sacred.observers import FileStorageObserver

import torch as pt
from ggg.data.dense.GGG_DenseData import GGG_DenseData
from ggg.evaluation.plots.utils.post_experiment_plots import (
    generate_graphs,
    main_run_MMD,
    main_run_plot,
)
from ggg.utils.classic_baselines import ClassicalBaseline

EXPNAME = "Classical"
ex = Experiment(EXPNAME)
if len(ex.observers) == 0:
    ex.observers.append(FileStorageObserver(EXPNAME))


@ex.config
def conf():
    device = "cpu"
    hpars = dict(
        model="BA",  # BA, Gnp
        lcc=True,
        batch_size=20,
        numb_graphs=1024,
        dataset="anu_graphs_chordal9",
        data_kwargs={},
    )

@ex.named_config
def community100():
    hpars = dict(
        dataset="CommunitySmall_100",
        data_kwargs=dict(DATA_DIR="/ggg/data"),
    )

@ex.named_config
def community():
    hpars = dict(
        dataset="CommunitySmall_20",
        data_kwargs=dict(DATA_DIR="/ggg/data"),
    )


@ex.named_config
def chordal9():
    hpars = dict(
        dataset="anu_graphs_chordal9",
    )


@ex.named_config
def qm9():
    hpars = dict(
        dataset="MolGAN_5k",
    )


@ex.automain
def run(hpars, device, _run):
    dataset = GGG_DenseData(
        dataset=hpars["dataset"], **hpars["data_kwargs"], zero_pad=False
    )
    dgraphs = [x[1] for x in dataset]
    cb = ClassicalBaseline(dgraphs, hpars["model"])
    save_dir = _run.observers[0].dir
    print(cb.param_dist)
    gen_graphs, dataset_graphs = generate_graphs(
        cb,
        current_epoch=0,
        device=device,
        batch_size=hpars["batch_size"],
        dataset=dataset,
        numb_graphs=hpars["numb_graphs"],
        save_dir=save_dir,
        save_files=False,
    )
    plots = main_run_plot(
        0,
        f"Model_{hpars.get('model')}",
        hpars.get("dataset"),
        model_graphs=gen_graphs,
        dataset_graphs=dataset_graphs,
        loss_dir=None,
        plots_save_dir=save_dir,
        lcc=hpars["lcc"],
    )
    main_run_MMD(
        0,
        csv_dir=save_dir,
        model_graphs=gen_graphs,
        dataset_graphs=dataset_graphs,
        numb_graphs=hpars["numb_graphs"],
        model_name=hpars.get("model"),
        dataset_name=hpars.get("dataset"),
    )
