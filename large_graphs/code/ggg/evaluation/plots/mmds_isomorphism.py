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
    write_to_csv,
)
from ggg.evaluation.plots.utils.plot_functions import (
    cluster_plot_isog,
    cluster_plot_novelty,
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

    models_to_plot = ["GG-GAN_qm9"] # GG-GAN_qm9 GG-GAN_community GG-GAN_chordal
    # please make sure the first dataset mentioned is training dataset
    datasets_to_plot = ["QM9", "QM9_rand", "QM9_rand1", "QM9_rand2"]

    # chordal
    # "Community", "Community_rand1", "Community_rand2", "Community_rand3"

    chkp_dir = None
    graphs_dir = []
    for model in models_to_plot:
        graphs_dir.append(model + "/lightning_log/plots/")

    save_dir = ""

    dataset_used = [    os.path.join(os.path.expanduser("~/.datasets"), "community_N_nodes5000_maxN20_minN20.sparsedataset"),
    os.path.join(os.path.expanduser("~/.datasets"), "community_N_nodes5000_maxN20_minN20_rand1.sparsedataset"),
    os.path.join(os.path.expanduser("~/.datasets"), "community_N_nodes5000_maxN20_minN20_rand2.sparsedataset"),
    os.path.join(os.path.expanduser("~/.datasets"), "community_N_nodes5000_maxN20_minN20_rand3.sparsedataset")]

    # os.path.join(os.path.expanduser("~/.datasets"), "chordal.npz")

    # os.path.join(os.path.expanduser("~/.datasets"), "QM9_5k.sparsedataset"),
    # os.path.join(os.path.expanduser("~/.datasets"), "QM9_rand.sparsedataset"),
    # os.path.join(os.path.expanduser("~/.datasets"), "QM9_rand1.sparsedataset"),
    # os.path.join(os.path.expanduser("~/.datasets"), "QM9_rand2.sparsedataset")

    os.path.join(os.path.expanduser("~/.datasets"), "community_N_nodes5000_maxN20_minN20.sparsedataset"),
    os.path.join(os.path.expanduser("~/.datasets"), "community_N_nodes5000_maxN20_minN20_rand1.sparsedataset"),
    os.path.join(os.path.expanduser("~/.datasets"), "community_N_nodes5000_maxN20_minN20_rand2.sparsedataset"),
    os.path.join(os.path.expanduser("~/.datasets"), "community_N_nodes5000_maxN20_minN20_rand3.sparsedataset")

    batch_idx = None
    numb = 5000
    number_g = [numb]
    epoch_steps = [[300]]


@ex.named_config
def GG_GAN_community20():
    models_to_plot = ["GG-GAN_Community"]
    datasets_to_plot = [
        "CommunitySmall_20",
        "CommunitySmall_20_rand1",
        "CommunitySmall_20_rand2",
        "CommunitySmall_20_rand3",
    ]
    graphs_dir = ["GG-GAN_Community/lightning_log/plots/"]
    dataset_used = [
        "ggg/data/community_N_nodes5000_maxN20_minN20.sparsedataset",
        "ggg/data/community_N_nodes5000_maxN20_minN20_rand1.sparsedataset",
        "ggg/data/community_N_nodes5000_maxN20_minN20_rand2.sparsedataset",
        "ggg/data/community_N_nodes5000_maxN20_minN20_rand3.sparsedataset",
    ]
    number_g = [5000]
    epoch_steps = [[1000]]


@ex.named_config
def GG_GAN_qm9():
    models_to_plot = ["GG-GAN_QM9"]
    datasets_to_plot = ["MolGAN_5k", "RandMolGAN_5k", "RandMolGAN_5k", "RandMolGAN_5k"]
    graphs_dir = ["GG-GAN_QM9/lightning_log/plots/"]
    dataset_used = [
        "ggg/data/QM9_5k.sparsedataset",
        "ggg/data/QM9_rand1.sparsedataset",
        "ggg/data/QM9_rand2.sparsedataset",
        "ggg/data/QM9_rand3.sparsedataset",
    ]
    number_g = [5000]
    epoch_steps = [[1000]]


@ex.named_config
def GG_GAN_chordal9():
    models_to_plot = ["GG-GAN_chordal9"]
    datasets_to_plot = ["chordal", "chordal_test"]
    graphs_dir = ["GG-GAN_chordal9/lightning_log/plots/"]
    dataset_used = ["ggg/data/chordal.npz", "ggg/data/chordal_test.npz"]
    number_g = [5000]
    epoch_steps = [[1000]]


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
            all_models_g = get_epoch_graphs(
                epoch,
                _config["graphs_dir"][i],
                model_n=model,
                all_models_g=all_models_g,
                number_g=_config["number_g"][i],
                log_dir=_config["graphs_dir"][i],
                batch_idx=_config["batch_idx"],
            )

    dataset_g = []
    original_dataset = _config["datasets_to_plot"][0]
    if _config["dataset_used"] is not None:
        for j, g_dir in enumerate(_config["dataset_used"]):
            check_dataset_file(g_dir, _config["datasets_to_plot"][j])
            dataset_g = get_dataset_epochs_graphs(
                g_dir, dataset_g=dataset_g, dataset=original_dataset
            )

    # # MMD dataset - datasets rand
    # for j, data_graphs in enumerate(dataset_g[1:]):
    #     print(
    #         "MMD between {} and {}".format(
    #             _config["datasets_to_plot"][0], _config["datasets_to_plot"][j + 1]
    #         )
    #     )
    #     main_run_MMD(
    #         current_epoch=_config["datasets_to_plot"][j + 1],
    #         csv_dir=os.path.join(_config["save_dir"], "stats_datasets"),
    #         model_graphs=dataset_g[0],
    #         dataset_graphs=data_graphs,
    #         numb_graphs=512,
    #         save=True,
    #     )

    # MMD model - datasets
    for i, model_graphs in enumerate(all_models_g):
        deg, clus, cycl, da, ac = [], [], [], [], []
        for j, data_graphs in enumerate(dataset_g):
            print("MMD between {} and {}".format(_config["models_to_plot"][i], _config["datasets_to_plot"][j]))
            degree_metric, clustering_metric, cycle_metric, degree_assortativity_metric, algebraic_connectivity_metric = \
                main_run_MMD(current_epoch=_config["models_to_plot"][i] + "_" + _config["datasets_to_plot"][j],
                             csv_dir=os.path.join(_config["save_dir"], "stats_models"), model_graphs=model_graphs,
                             dataset_graphs=data_graphs, numb_graphs=512, save=True)
            print(cycle_metric)
            if j != 0:
                deg.append(degree_metric)
                cycl.append(cycle_metric)
                clus.append(clustering_metric)
                da.append(degree_assortativity_metric)
                ac.append(algebraic_connectivity_metric)

        write_to_csv(current_epoch=_config["models_to_plot"][i] + "_" + "avg",
                     csv_dir=os.path.join(_config["save_dir"], "stats_models"),
                     degree=float(np.mean(deg)), clustering=float(np.mean(clus)), cycles=float(np.mean(cycl)),
                     algcon=float(np.mean(ac)), degassort=float(np.mean(da)), eccentricity=None)

    # TODO needs to change --> QM9 we compare against train dataset (to graphs not be there) and chordal we compare against the test dataset
    if "QM9" in _config["datasets_to_plot"][0] or "Comm" in _config["datasets_to_plot"][0]:
        dataset_to_evaluate_against = dataset_g[0]
        chordal_ = False
    elif "chordal" in _config["datasets_to_plot"][0]:
        dataset_to_evaluate_against = dataset_g[0]
        chordal_ = True

    cluster_plot_novelty(models_g=all_models_g, dataset_g=dataset_to_evaluate_against,
                         legends=_config["models_to_plot"], numb_g_eval=5000, reps=1,
                         save=True, save_dir=_config["save_dir"], chordal=chordal_)

    # cluster_plot_isog(models_datasets_g=all_models_g + [dataset_to_evaluate_against],
    #                   legends=_config["models_to_plot"] + _config["datasets_to_plot"], numb_g_eval=5000, reps=1,
    #                   save=True, save_dir=_config["save_dir"])


if __name__ == "__main__":
    ex.run_commandline()
