import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx
from sacred import Experiment
from ggg.evaluation.plots.utils.plot_helpers import (
    get_epoch_graphs,
    get_dataset_epochs_graphs,
    check_dataset_file,
)
from ggg.evaluation.plots.utils.post_experiment_plots import (
    main_run_MMD,
    main_run_plot,
    write_to_csv,
)

from ggg.evaluation.statistics.utils.helpers import get_dist
from ggg.evaluation.plots.utils.plot_functions import (
    cluster_plot_histogram,
    cluster_plot_losses,
    cluster_one_img,
)
from ggg.evaluation.plots.graph_grid import cluster_plot_molgrid

ex = Experiment("PEAWGANTrain")


@ex.config
def config():
    """plot isomorphic graphs"""

    models_to_plot = ["GG-GAN_qm9"]
    # please make sure the first dataset mentioned is training dataset
    datasets_to_plot = ["QM9"]

    chkp_dir = None
    graphs_dir = []
    for model in models_to_plot:
        graphs_dir.append(model + "/lightning_log/plots/")

    save_dir = ""

    dataset_used = [os.path.join(os.path.expanduser("~/.datasets"), "QM9_5k.sparsedataset")]

    batch_idx = None
    numb = 5000
    final_epoch = 300
    number_g = []
    epoch_steps = []
    for _ in models_to_plot:
        number_g.append(numb)
        epoch_steps.append([final_epoch])

    handle = "graphs"


@ex.named_config
def GG_GAN_community20_btx1():
    models_to_plot = ["GG-GAN_Community"]
    datasets_to_plot = ["CommunitySmall_20"]
    graphs_dir = ["GG-GAN_Community/lightning_log/plots/"]
    dataset_used = ["ggg/data/community_N_nodes5000_maxN20_minN20.sparsedataset"]
    number_g = [5000]
    epoch_steps = [[1000]]
    batch_idx = 0


@ex.named_config
def GG_GAN_qm9_btx1():
    models_to_plot = ["GG-GAN_QM9"]
    datasets_to_plot = ["MolGAN_5k"]
    graphs_dir = ["GG-GAN_QM9/lightning_log/plots/"]
    dataset_used = ["ggg/data/QM9_5k.sparsedataset"]
    number_g = [5000]
    epoch_steps = [[1000]]
    batch_idx = 0


@ex.named_config
def GG_GAN_chordal9_btx1():
    models_to_plot = ["GG-GAN_chordal9"]
    datasets_to_plot = ["chordal"]
    graphs_dir = ["GG-GAN_chordal9/lightning_log/plots/"]
    dataset_used = ["ggg/data/chordal.npz"]
    number_g = [5000]
    epoch_steps = [[1000]]
    batch_idx = 0


def assert_inputs(_config):
    if _config["graphs_dir"] is not None and _config["chkp_dir"] is not None:
        assert len(_config["graphs_dir"] + _config["chkp_dir"]) == len(
            _config["models_to_plot"]
        ), "Number of directories to get models/graphs must be the same as the number of models to plot"
    elif _config["graphs_dir"] is not None:
        assert len(_config["graphs_dir"]) == len(
            _config["models_to_plot"]
        ), "Number of directories to get graphs must be the same as the number of models to plot"
    elif _config["chkp_dir"] is not None:
        assert len(_config["chkp_dir"]) == len(
            _config["models_to_plot"]
        ), "Number of directories to get models must be the same as the number of models to plot"

    if _config["datasets_to_plot"] is not None:
        assert len(_config["datasets_to_plot"]) == len(
            _config["dataset_used"]
        ), "Number of directories to get dataset graphs must be the same as the number of datasets to plot"


@ex.main
def main(_config):
    assert_inputs(_config)

    all_models_g = []
    for i, model in enumerate(_config["models_to_plot"]):
        for epoch in _config["epoch_steps"][i]:
            log_dir = _config["graphs_dir"][i]
            gg_ckpt_path = os.path.join(log_dir, f"{str(epoch).zfill(4)}/state.ckpt")
            all_models_g = get_epoch_graphs(
                epoch,
                _config["graphs_dir"][i],
                model_n=model,
                all_models_g=all_models_g,
                number_g=_config["number_g"][i],
                log_dir=_config["graphs_dir"][i],
                batch_idx=5,
                allow_greater=True,
            )

    dataset_g = []
    if _config["dataset_used"] is not None:
        for j, g_dir in enumerate(_config["dataset_used"]):
            check_dataset_file(g_dir, _config["datasets_to_plot"][j])
            dataset_g = get_dataset_epochs_graphs(
                g_dir, dataset_g=dataset_g, dataset=_config["datasets_to_plot"][0]
            )

    if _config["handle"] == "one_img":
        for i, model in enumerate(_config["models_to_plot"]):
            for epoch in _config["epoch_steps"][i]:
                main_run_plot(
                    current_epoch=epoch,
                    model_name=model,
                    dataset=_config["datasets_to_plot"][0],
                    model_graphs=all_models_g[i],
                    dataset_graphs=dataset_g[0],
                    loss_dir=_config["models_to_plot"][i],
                    plots_save_dir=_config["save_dir"],
                    lcc=False,
                    legend=str(epoch) + "_" + model,
                )

    if _config["handle"] == "graphs":
        for i, model in enumerate(_config["models_to_plot"]):
            for epoch in _config["epoch_steps"][i]:
                sample_plot = cluster_plot_molgrid(
                    all_models_g[i],
                    name=model + "_" + str(epoch),
                    lcc=False,
                    save_dir=_config["save_dir"],
                    save=True,
                    dataset_graphs=dataset_g[0]
                )

        # # plot dataset
        # sample_plot = cluster_plot_molgrid(
        #     dataset_g[0],
        #     name=_config["datasets_to_plot"][0],
        #     lcc=False,
        #     save_dir=_config["save_dir"],
        #     save=True,
        # )


if __name__ == "__main__":
    ex.run_commandline()
