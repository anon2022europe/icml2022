import os
import csv
import pickle
import warnings
from typing import List

import torch as pt
import numpy as np
import networkx as nx
from operator import itemgetter
from tensorboard.backend.event_processing import event_accumulator
from tqdm import tqdm

from ggg.evaluation.statistics.utils.helpers import get_dist

from ggg.evaluation.plots.utils.plot_functions import (
    cluster_plot_histogram,
    cluster_plot_losses,
    cluster_one_img,
)
from ggg.evaluation.plots.graph_grid import cluster_plot_molgrid

from ggg.evaluation.statistics.utils.stat_tests import (
    degree_stats,
    clustering_stats,
    algebraic_connectivity_stats,
    degree_assortativity_coefficient_stats,
    cycle_stats,
    torch_degree_stats,
    torch_algebraic_connectivity_stats,
    torch_assortativity_coefficient_stats,
    torch_cycle_stats,
    torch_clustering_stats,
    torch_eccentricity_stats,
    get_dists,
    metrics_stats,
)
from ggg.utils.utils import pad_to_max
from ggg.data.dense.GGG_DenseData import GGG_DenseData
from ggg.data.dense.PEAWGANDenseStructureData import PEAWGANDenseStructureData
from ggg.evaluation.plots.utils.plot_helpers import get_epoch_graphs
from ggg.evaluation.plots.utils.plot_functions import (
    cluster_plot_isog,
    cluster_plot_novelty,
)


def main_run_MMD(
    current_epoch,
    csv_dir,
    model_graphs: [],
    dataset_graphs: [],
    numb_graphs=int(),
    save=True,
    dataset_name=None,
    model_name=None,
):
    index_ = np.random.choice(range(len(model_graphs)), numb_graphs, replace=False)
    d_index_ = np.random.choice(
        range(len(dataset_graphs)),
        numb_graphs,
        replace=len(dataset_graphs) < numb_graphs,
    )

    model_graphs_cut = list(itemgetter(*index_)(model_graphs))
    dataset_graphs_cut = list(itemgetter(*d_index_)(dataset_graphs))
    (
        degree_metric,
        clustering_metric,
        cycle_metric,
        degree_assortativity_metric,
        algebraic_connectivity_metric,
    ) = (0, 0, 0, 0, 0)
    print("Calculating MMD metrics...")
    print("Degree...")
    # degree_metric = degree_stats(dataset_graphs_cut,model_graphs_cut)
    degree_metric = torch_degree_stats(dataset_graphs_cut, model_graphs_cut)
    # assert pt.isclose(pt.tensor(degree_metric).float(),torch_degree,1e-3).all()
    # degree_metric = MMD_print(model_graphs_cut, dataset_graphs_cut, pp_name="degree", function=degree_stats)
    print("Clustering...")
    # clustering_metric = MMD_print(model_graphs_cut, dataset_graphs_cut, pp_name="clustering", function=clustering_stats)
    clustering_metric = torch_clustering_stats(dataset_graphs_cut, model_graphs_cut)
    # assert pt.isclose(pt.tensor(clustering_metric).float(),torch_cluster,1e-3).all()
    print("Cycles...")
    # cycle_metric=cycle_stats(dataset_graphs_cut,model_graphs_cut)
    cycle_metric = torch_cycle_stats(dataset_graphs_cut, model_graphs_cut)
    # assert pt.isclose(pt.tensor(cycle_metric).float(),torch_cycle,1e-3).all()
    # cycle_metric = MMD_print(model_graphs_cut, dataset_graphs_cut, pp_name="cycles", function=cycle_stats)
    print("Assortativity...")
    # degree_assortativity_metric = MMD_print(model_graphs_cut, dataset_graphs_cut, pp_name="assortativity_coefficient",
    #                                        function=degree_assortativity_coefficient_stats)
    try:
        degree_assortativity_metric = torch_assortativity_coefficient_stats(
            dataset_graphs_cut, model_graphs_cut
        )
    except Exception as e:
        print("Error ---> {}".format(e))
        degree_assortativity_metric = cycle_metric
    # assert pt.isclose(pt.tensor(degree_assortativity_metric).float(),toch_dac,1e-3)
    print("Algebraic...")
    # algebraic_connectivity_metric = MMD_print(model_graphs_cut, dataset_graphs_cut, pp_name="algebraic_connectivity",
    #                                          function=algebraic_connectivity_stats)
    algebraic_connectivity_metric = torch_algebraic_connectivity_stats(
        dataset_graphs_cut, model_graphs_cut
    )
    # assert pt.isclose(pt.tensor(algebraic_connectivity_metric).float(),torch_ac,1e-3)
    # stats=metrics_stats(dataset_graphs_cut,model_graphs_cut)
    # manual_stats=dict(cluster=clustering_metric,
    #                  degree=degree_metric,
    #                  connectivity=algebraic_connectivity_metric,
    #                  assortativity=degree_assortativity_metric,
    #                  cycles=cycle_metric)
    # for k in stats:
    #    assert pt.isclose(stats[k],manual_stats[k],1e-3).all()

    (
        degree_metric,
        clustering_metric,
        cycle_metric,
        degree_assortativity_metric,
        algebraic_connectivity_metric,
    ) = [
        x.item()
        for x in [
            degree_metric,
            clustering_metric,
            cycle_metric,
            degree_assortativity_metric,
            algebraic_connectivity_metric,
        ]
    ]
    if save:
        os.makedirs(csv_dir, exist_ok=True)
        # TODO simplify, is to address Nones and get item value, so not to write tensor in excel
        metric_list = [
            degree_metric,
            clustering_metric,
            cycle_metric,
            degree_assortativity_metric,
            algebraic_connectivity_metric,
        ]
        if degree_metric is not None and type(degree_metric) != float:
            degree_metric = degree_metric.item()
        if clustering_metric is not None and type(clustering_metric) != float:
            clustering_metric = clustering_metric.item()
        if cycle_metric is not None and type(cycle_metric) != float:
            cycle_metric = cycle_metric.item()
        if (
            degree_assortativity_metric is not None
            and type(degree_assortativity_metric) != float
        ):
            degree_assortativity_metric = degree_assortativity_metric.item()
        if (
            algebraic_connectivity_metric is not None
            and type(algebraic_connectivity_metric) != float
        ):
            algebraic_connectivity_metric = algebraic_connectivity_metric.item()
        write_to_csv(
            current_epoch=current_epoch,
            csv_dir=csv_dir,
            degree=degree_metric,
            clustering=clustering_metric,
            cycles=cycle_metric,
            algcon=algebraic_connectivity_metric,
            degassort=degree_assortativity_metric,
            eccentricity=None,
        )

    return (
        degree_metric,
        clustering_metric,
        cycle_metric,
        degree_assortativity_metric,
        algebraic_connectivity_metric,
    )


def MMD_print(list_, ori_list_, pp_name, function):
    try:
        result_ = function(ori_list_, list_)
        print("Result {:.> 60}".format(result_))
        return result_

    except Exception as error:
        print(
            "\n Did not compute MMD with function {} due to error: {} \n".format(
                pp_name, error
            )
        )
        return None


def write_to_csv(
    current_epoch,
    csv_dir,
    degree,
    clustering,
    algcon,
    degassort,
    cycles=None,
    eccentricity=None,
    dataset=None,
    model=None,
):
    if check_dir(os.path.join(csv_dir, "stats.csv")):
        pass
    else:
        with open(os.path.join(csv_dir, "stats.csv"), "w", newline="") as _file:
            writer = csv.DictWriter(
                _file,
                fieldnames=[
                    "epoch",
                    "mmd_degree",
                    "mmd_clustering",
                    "mmd_cycles",
                    "mmd_algebraic_connectivity",
                    "mmd_degree_assortativity_coefficient",
                    "mmd_eccentricity",
                ],
            )
            writer.writeheader()
    data = [
        model,
        dataset,
        current_epoch,
        degree,
        clustering,
        cycles,
        algcon,
        degassort,
        eccentricity,
    ]

    with open(os.path.join(csv_dir, "stats.csv"), "a") as text_file:
        writer = csv.writer(text_file)
        writer.writerow(data)


def check_dir(directory):
    return os.path.exists(directory)


def main_run_plot(
    current_epoch,
    model_name,
    dataset,
    model_graphs: [],
    dataset_graphs: [],
    loss_dir=None,
    plots_save_dir=None,
    lcc=True,
    legend=None,
):
    m_dists = get_dists(
        flags={"degree", "cycles"}, graphs=model_graphs, name=model_name
    )
    model_degree_dist = m_dists["degree"]
    model_cycles_dist = m_dists["cycles"]
    # model_degree_dist = get_dist(flag_="degree", graphs=model_graphs)
    # model_cycles_dist = get_dist(flag_="cycles", graphs=model_graphs)

    d_dists = get_dists(
        flags={"degree", "cycles"}, graphs=dataset_graphs, name="dataset"
    )
    dataset_degree_dist = d_dists["degree"]
    dataset_cycles_dist = d_dists["cycles"]
    # dataset_degree_dist = get_dist(flag_="degree", graphs=dataset_graphs)
    # dataset_cycles_dist = get_dist(flag_="cycles", graphs=dataset_graphs)

    file = get_loss_dir(loss_dir) if loss_dir is not None else None

    return cluster_plot_img(
        legend=current_epoch,
        model_name=model_name,
        dataset=dataset,
        model_graphs=model_graphs,
        dataset_graphs=dataset_graphs,
        losses_file=file if current_epoch > 0 else None,
        model_degree_dist=model_degree_dist,
        dataset_degree_dist=dataset_degree_dist,
        model_cycles_dist=model_cycles_dist,
        dataset_cycles_dist=dataset_cycles_dist,
        kde=False,
        plots_save_dir=plots_save_dir,
        lcc=lcc,
    )


def get_loss_dir(exp_dir):
    for root, dirs, files in os.walk(exp_dir):
        for file in files:
            if file.endswith(".0"):
                event_accumulator_file = event_accumulator.EventAccumulator(
                    root + "/" + file
                )
                return event_accumulator_file

            else:
                pass

    warnings.warn("Loss file not found, plot will be done without!")
    return None


_dataset_graph_cache = {}


def generate_graphs(
    model,
    current_epoch,
    device,
    batch_size,
    dataset=None,
    numb_graphs=1024,
    save_files=True,
    save_dir=None,
):
    """
    Function to generate graphs from model and save epoch wise files
    :param model:
    :param numb_graphs:
    :param save_files:
    :param save_dir:
    :return:
    """
    z_list = []
    X_tensors = []
    A_tensors = []
    contextvectors = []
    dataset_graphs = []
    modified_embeddings = []
    old_len = 0
    bar = tqdm(total=numb_graphs, desc="Generating graphs")

    generated_graphs = []
    while len(generated_graphs) < numb_graphs:
        # sample from model
        sout = [
            x
            for x in model.generator.sample(
                batch_size=batch_size, device=device
            )
        ]

        with pt.no_grad():
            if model.__class__.__name__ == "GGG":
                sout = [
                    x.cpu() if x is not None else x
                    for x in model.generator.sample(
                        batch_size=batch_size, device=device
                    )
                ]
            else:
                sout = [
                    x.cpu() if x is not None else x
                    for x in model.sample(batch_size=batch_size, device=device)
                ]
            if len(sout) == 5:
                X_out, A_out, Z, finetti_u, mod_emb = sout
            else:
                X_out, A_out = sout[:2]
                Z = pt.zeros_like(X_out) if X_out is not None else None
                finetti_u = None
                mod_emb = None
        for b in range(batch_size):
            adj_m = A_out[b].cpu().detach().numpy()

            np.fill_diagonal(adj_m, 0)
            g = nx.from_numpy_matrix(adj_m)

            # list appends
            if Z is not None:
                z_list.append(Z[b].detach().cpu())
            if X_out is not None:
                X_tensors.append(X_out[b].detach().cpu())
            A_tensors.append(A_out[b].detach().cpu())
            generated_graphs.append(g)
            if finetti_u is not None:
                contextvectors.append(finetti_u[b].detach().cpu())
            if mod_emb is not None:
                modified_embeddings.append(mod_emb[b].detach().cpu())

            if len(generated_graphs) >= numb_graphs:
                break

        new_len = len(generated_graphs)
        bar.update(new_len - old_len)
        old_len = new_len
        if new_len >= numb_graphs:
            bar.close()
            break

    if dataset is not None:
        key = dataset
        if key in _dataset_graph_cache:
            dataset_graphs = _dataset_graph_cache[key]
        else:
            for i in tqdm(
                range(len(dataset)), desc="Getting dataset graphs for plotting"
            ):
                _, a, _ = dataset[i]
                if pt.is_tensor(a):
                    adj_m = a.detach().numpy()

                np.fill_diagonal(adj_m, 0)
                d_g = nx.from_numpy_matrix(adj_m)
                dataset_graphs.append(d_g)
            _dataset_graph_cache[key] = dataset_graphs

    if save_files:
        epoch_dir = os.path.join(save_dir, str(current_epoch).zfill(4))
        os.makedirs(epoch_dir, exist_ok=True)
        if len(z_list)>0:
            z_list = pad_to_max(z_list, N_dim=0, pad_dims=(0,))
        else:
            z_list=[]

        if all(x is None for x in X_tensors) or len(X_tensors) == 0:
            X_tensors = []
        else:
            X_tensors = pad_to_max(X_tensors, N_dim=0, pad_dims=(0,))
            X_tensors = pad_to_max(X_tensors, N_dim=1, pad_dims=(1,))
            X_tensors = pt.stack(X_tensors)
        A_tensors = pad_to_max(A_tensors, N_dim=1, pad_dims=(0, 1))
        if len(z_list)>0:
            z_list=pt.stack(z_list)
        pt.save(z_list, os.path.join(epoch_dir, "Z.pt"))
        pt.save(X_tensors, os.path.join(epoch_dir, "X.pt"))
        pt.save(pt.stack(A_tensors), os.path.join(epoch_dir, "A.pt"))

        if hasattr(model, "trainer"):
            model.trainer.save_checkpoint(os.path.join(epoch_dir, "state.ckpt"))
        save_pkl(filename=os.path.join(epoch_dir, "graphs"), list=generated_graphs)
        save_pkl(filename=os.path.join(epoch_dir, "mod_emb"), list=modified_embeddings)
        save_pkl(
            filename=os.path.join(epoch_dir, "contextvectors"), list=contextvectors
        )

    return generated_graphs, dataset_graphs


def cluster_plot_img(
    legend,
    model_name,
    dataset,
    model_graphs: [],
    dataset_graphs: [],
    model_degree_dist: pt.Tensor,
    model_cycles_dist: pt.Tensor,
    losses_file=None,
    dataset_degree_dist: pt.Tensor = None,
    dataset_cycles_dist: pt.Tensor = None,
    kde=False,
    plots_save_dir=None,
    lcc=True,
):

    loss_plot = cluster_plot_losses(losses_file)
    sample_plot = cluster_plot_molgrid(model_graphs, lcc=lcc)
    dataset_plot = cluster_plot_molgrid(dataset_graphs, name="Dataset", lcc=lcc)
    degree_plot = cluster_plot_histogram(
        model_dist=model_degree_dist,
        dataset_dist=dataset_degree_dist,
        metric="degree",
        kde=kde,
        dataset_name=dataset,
        model_name=model_name,
    )
    cycle_plot = cluster_plot_histogram(
        model_dist=model_cycles_dist.sum(0),
        dataset_dist=dataset_cycles_dist.sum(0),
        metric="cycles",
        kde=kde,
        dataset_name=dataset,
        model_name=model_name,
        is_counts=True,
    )

    plots = [degree_plot, cycle_plot, loss_plot, sample_plot]
    cluster_one_img(plots, plots_save_dir, model_name=model_name, epoch=legend)
    plots.append(dataset_plot)
    return {
        k: v
        for k, v in zip(["degrees", "cycles", "loss", "samples", "dataset"], plots)
        if v is not None
    }


def save_pkl(filename, list):
    # Save vector of noise used
    with open(filename, "wb") as f:
        pickle.dump(list, f)


## GraphRNN plots
def externals_main_run_plot(
    model_graphs: [],
    dataset_graphs: [],
    loss_dir=None,
    plots_save_dir=None,
    dataset=None,
    baseline_name=None,
    epoch=None,
):
    model_degree_dist = get_dist(flag_="degree", graphs=model_graphs)
    model_cycles_dist = get_dist(flag_="cycles", graphs=model_graphs)

    dataset_degree_dist = get_dist(flag_="degree", graphs=dataset_graphs)
    dataset_cycles_dist = get_dist(flag_="cycles", graphs=dataset_graphs)

    file = get_loss_dir(loss_dir)
    external_plot_img(
        model_graphs=model_graphs,
        losses_file=file,
        model_degree_dist=model_degree_dist,
        dataset_degree_dist=dataset_degree_dist,
        model_cycles_dist=model_cycles_dist,
        dataset_cycles_dist=dataset_cycles_dist,
        kde=False,
        plots_save_dir=plots_save_dir,
        dataset=dataset,
        baseline_name=baseline_name,
        epoch=epoch,
    )


def external_plot_img(
    model_graphs: [],
    losses_file,
    model_degree_dist: [],
    model_cycles_dist: [],
    dataset_degree_dist=None,
    dataset_cycles_dist=None,
    kde=False,
    plots_save_dir=None,
    dataset=None,
    baseline_name=None,
    epoch=None,
):
    loss_plot = cluster_plot_losses(losses_file)
    sample_plot = cluster_plot_molgrid(model_graphs)
    degree_plot = cluster_plot_histogram(
        model_dist=model_degree_dist,
        dataset_dist=dataset_degree_dist,
        metric="degree",
        kde=kde,
        dataset_name=dataset,
    )
    cycle_plot = cluster_plot_histogram(
        model_dist=model_cycles_dist,
        dataset_dist=dataset_cycles_dist,
        metric="cycles",
        kde=kde,
        dataset_name=dataset,
    )

    cluster_one_img(
        [degree_plot, cycle_plot, loss_plot, sample_plot],
        plots_save_dir,
        model_name=baseline_name,
        epoch=epoch,
    )


def external_dataset_graphs(dataset: str):
    """"""
    dataset_graphs = []

    with open(dataset, "rb") as f:
        graph_list = pickle.load(f)

    # for (_, a) in graph_list:
    #         adj_m = a.detach().numpy()
    #
    #         np.fill_diagonal(adj_m, 0)
    #         d_g = nx.from_numpy_matrix(adj_m)
    #         dataset_graphs.append(d_g)

    for g in graph_list:
        A_np = g.A.detach().numpy()
        np.fill_diagonal(A_np, 0)
        G = nx.from_numpy_matrix(A_np)
        # for i, node in enumerate(G.nodes()):
        #     G.node[i]["feature"] = g.x[i].item()
        dataset_graphs.append(G)

    return dataset_graphs


## Wrapper for after experiments plots
def get_QM9_rand():
    rand1 = GGG_DenseData(
        data_dir="ggg/data",
        filename="QM9_rand1.sparsedataset",
        dataset="RandMolGAN_5k",
        zero_pad=False,
    )
    rand1 = get_graphs_from_dataset(rand1)

    rand2 = GGG_DenseData(
        data_dir="ggg/data",
        filename="QM9_rand2.sparsedataset",
        dataset="RandMolGAN_5k",
        zero_pad=False,
    )
    rand2 = get_graphs_from_dataset(rand2)

    rand3 = GGG_DenseData(
        data_dir="ggg/data",
        filename="QM9_rand3.sparsedataset",
        dataset="RandMolGAN_5k",
        zero_pad=False,
    )
    rand3 = get_graphs_from_dataset(rand3)

    return rand1, rand2, rand3


def get_chordal_rand(dataset_name=None):
    rand1 = PEAWGANDenseStructureData(
        data_dir="ggg/data",
        filename="chordal_test.npz",
        dataset="anu_graphs_chordal9_rand",
        k_eigenvals=4,
        use_laplacian=False,
        large_N_approx=False,
        zero_pad=False,
    )

    rand1 = get_graphs_from_dataset(rand1, dataset_name)

    return rand1


def get_graphs_from_dataset(untreated_graphs, dataset_name=None):
    graphs = []
    # X = features | A = adjacency | _ = #nodes
    for (X, A, _) in untreated_graphs:
        if pt.is_tensor(A):
            A = A.detach().numpy()
        np.fill_diagonal(A, 0)
        G = nx.from_numpy_matrix(A)
        graphs.append(G)
    return graphs


def get_egonet_rand(datasets_to_plot):
    return [
        PEAWGANDenseStructureData(data_dir="data", dataset=x) for x in datasets_to_plot
    ]


def plot_mmd_wrapper(
    hparams=None,
    current_epoch=None,
    plots_save_dir=None,
    dataset_name=None,
    dataset_used=None,
    pyl_log_dir=None,
    numb_g_eval=None,
    numb_g_mmd=None,
    allow_greater=False,
    model=None,
    model_name="GG-GAN",
):
    dataset_graphs = get_graphs_from_dataset(dataset_used, dataset_name=dataset_name)
    if "MolGAN" in dataset_name:
        datasets_to_plot = ["QM9", "rand1", "rand2", "rand3"]
        rand1, rand2, rand3 = get_QM9_rand()
        dataset_g = [dataset_graphs, rand1, rand2, rand3]
    elif "anu_graphs" in dataset_name:
        datasets_to_plot = ["chordal9", "rand1"]
        rand1 = get_chordal_rand(dataset_name)
        dataset_g = [dataset_graphs, rand1]
    elif "egonet" in dataset_name:
        datasets_to_plot = (
            [dataset_name]
            + [f"egonet20-{i}" for i in range(1, 7)]
            + ["egonet-rand-100"]
        )
        dataset_g = [dataset_graphs] + get_egonet_rand(datasets_to_plot[1:])
    else:
        datasets_to_plot = [dataset_name]
        dataset_g = [dataset_graphs]

    all_models_g = []
    all_models_g = get_epoch_graphs(
        current_epoch,
        g_dir=None,
        model_n=model_name,
        all_models_g=all_models_g,
        number_g=numb_g_eval,
        log_dir=plots_save_dir,
        model=model,
        allow_greater=allow_greater,
    )

    for i, model_graphs in enumerate(all_models_g):
        deg, clus, cycl, da, ac = [], [], [], [], []
        for j, data_graphs in enumerate(dataset_g):
            print(
                "MMD between {} and {}".format(
                    hparams["trunk_hpars"]["name"], datasets_to_plot[j]
                )
            )
            (
                degree_metric,
                clustering_metric,
                cycle_metric,
                degree_assortativity_metric,
                algebraic_connectivity_metric,
            ) = main_run_MMD(
                current_epoch=hparams["trunk_hpars"]["name"]
                + "_"
                + datasets_to_plot[j],
                csv_dir=os.path.join(plots_save_dir, "stats_models"),
                model_graphs=model_graphs,
                dataset_graphs=data_graphs,
                numb_graphs=numb_g_mmd,
                save=True,
                dataset_name=dataset_name,
                model_name=model_name,
            )
            if j != 0:
                deg.append(degree_metric)
                cycl.append(cycle_metric)
                clus.append(clustering_metric)
                da.append(degree_assortativity_metric)
                ac.append(algebraic_connectivity_metric)

        write_to_csv(
            current_epoch=hparams["trunk_hpars"]["name"] + "_" + "avg",
            csv_dir=os.path.join(plots_save_dir, "stats_models"),
            degree=float(np.mean(deg)),
            clustering=float(np.mean(clus)),
            cycles=float(np.mean(cycl)),
            algcon=float(np.mean(ac)),
            degassort=float(np.mean(da)),
            eccentricity=None,
        )

    if "MolGAN" or "CommunitySmall" in dataset_name:
        dataset_to_evaluate_against = dataset_g[0]
        chordal_ = False
    elif "anu_graphs" in dataset_name:
        dataset_to_evaluate_against = dataset_g[1]
        chordal_ = True

    # TODO improve isomorphic function
    cluster_plot_novelty(
        models_g=all_models_g,
        dataset_g=dataset_to_evaluate_against,
        legends=[hparams["trunk_hpars"]["name"]],
        numb_g_eval=numb_g_eval,
        reps=1,
        save=True,
        save_dir=pyl_log_dir,
        chordal=chordal_,
    )

    cluster_plot_isog(
        models_datasets_g=all_models_g + [dataset_graphs],
        legends=[hparams["trunk_hpars"]["name"], hparams["dataset_hpars"]["dataset"]],
        numb_g_eval=numb_g_eval,
        reps=1,
        save=True,
        save_dir=pyl_log_dir,
    )
