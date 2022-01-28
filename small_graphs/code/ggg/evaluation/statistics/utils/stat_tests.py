from itertools import chain
from warnings import warn

from .helpers import *

import concurrent.futures
import os
import subprocess as sp
from datetime import datetime

from scipy.linalg import eigvalsh
import networkx as nx
import numpy as np
import copy

from ggg.evaluation.statistics.utils.helpers import (
    get_dist,
    process_tensor,
    compute_mmd,
    gaussian_tv,
    gaussian,
)

PRINT_TIME = True
ORCA_DIR = "evaluation/orca"  # the relative path to the orca dir


def add_tensor(x, y):
    x, y = process_tensor(x, y)
    return x + y


def cycle_worker(G):
    observations = get_dist(flag_="cycles", graphs=[G])
    return np.array(observations)


def cycle_stats(graph_ref_list, graph_pred_list, new=False):
    """Compute the distance between the cycle distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if new:
        print("Ref cycles")
        sample_ref = get_dist(flag_="cycles", graphs=graph_ref_list)
        print("Pred cycles")
        sample_pred = get_dist(flag_="cycles", graphs=graph_pred_list_remove_empty)
    else:
        sample_ref = get_dist(flag_="cycles", graphs=graph_ref_list, mmd_plot=True)
        sample_pred = get_dist(
            flag_="cycles", graphs=graph_pred_list_remove_empty, mmd_plot=True
        )

    mmd_dist = compute_mmd_cycle(sample_ref, sample_pred, kernel=gaussian)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing cycles mmd: ", elapsed)
    return mmd_dist


def torch_cycle_stats(graph_ref_list, graph_pred_list):
    """Compute the distance between the cycle distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    sample_ref = get_dist(flag_="cycles", graphs=graph_ref_list, mmd_plot=True)
    sample_pred = get_dist(
        flag_="cycles", graphs=graph_pred_list_remove_empty, mmd_plot=True
    )

    mmd_dist = torch_mmd(sample_ref, sample_pred, kernel=gaussian_kernel)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing cycles mmd: ", elapsed)
    if pt.isnan(mmd_dist).any():
        warn(f"odel:{sample_pred}\ndataset{sample_ref}")
    return mmd_dist


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    print(len(sample_ref), len(sample_pred))
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


def torch_degree_stats(graph_ref_list, graph_pred_list, is_parallel=True):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
      graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)

    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)
    print("Lens", len(sample_ref), len(sample_pred))
    max_n = max(len(x) for x in chain(sample_ref, sample_pred))
    padded_ref = [pad_to(x, max_n) for x in sample_ref]
    padded_pred = [pad_to(x, max_n) for x in sample_pred]
    mmd_dist = torch_mmd(
        np.stack(padded_ref), np.stack(padded_pred), kernel=gaussian_kernel
    )
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())

    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


def torch_clustering_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)
    mmd_dist = torch_mmd(
        sample_ref, sample_pred, kernel=gaussian_kernel, sigma=1.0 / 10
    )
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


def algebraic_connectivity_worker(param):
    G, bins = param
    return np.array([nx.algebraic_connectivity(G)])


def metrics_worker(param):
    G, bins, metrics = param
    outs = {}
    for k in metrics:
        if k == "cluster":
            clustering_coeffs_list = list(nx.clustering(G).values())

            outs[k], _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
        if k == "degree":
            outs[k] = np.array(nx.degree_histogram(G))
        if k == "assortativity":
            outs[k] = np.array(nx.degree_assortativity_coefficient(G))
        if k == "connectivity":
            outs[k] = np.array([nx.algebraic_connectivity(G)])
        if k == "cycles":
            outs[k] = get_dist(flag_=k, graphs=[G])
    return outs


def metrics_stats(
    graph_ref_list,
    graph_pred_list,
    metrics=("degree", "cycles", "connectivity", "assortativity", "cluster"),
    bins=100,
    is_parallel=True,
):
    sample_ref = defaultdict(list)
    sample_pred = defaultdict(list)
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for metrics_hist in executor.map(
                metrics_worker, [(G, bins, metrics) for G in graph_ref_list]
            ):
                for k, v in metrics_hist.items():
                    sample_ref[k].append(v)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for metrics_hist in executor.map(
                metrics_worker,
                [(G, bins, metrics) for G in graph_pred_list_remove_empty],
            ):
                for k, v in metrics_hist.items():
                    sample_pred[k].append(v)

    else:
        for i in range(len(graph_ref_list)):
            for (
                k,
                v,
            ) in metrics_worker([graph_ref_list[i], bins, metrics]):
                # scalar_ = nx.algebraic_connectivity(graph_ref_list[i])
                sample_ref[k].append(v)
                # sample_ref.append(scalar_)

        for i in range(len(graph_pred_list_remove_empty)):
            for (
                k,
                v,
            ) in metrics_worker([graph_pred_list_remove_empty[i], bins, metrics]):
                # scalar_ = nx.algebraic_connectivity(graph_ref_list[i])
                sample_pred[k].append(v)

    mmd_dist = {
        k: torch_mmd(
            pt.from_numpy(np.stack(sample_ref[k])),
            pt.from_numpy(np.stack(sample_pred[k])),
            is_hist=k not in {"connectivity", "assortativity"},
            kernel=gaussian_kernel,
            sigma=1.0 / 10,
        )
        for k in sample_pred.keys()
    }
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print(f"Time {metrics} mmd: ", elapsed)
    return mmd_dist


def algebraic_connectivity_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                algebraic_connectivity_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                algebraic_connectivity_worker,
                [(G, bins) for G in graph_pred_list_remove_empty],
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            scalar_ = nx.algebraic_connectivity(graph_ref_list[i])
            sample_ref.append(scalar_)

        for i in range(len(graph_pred_list_remove_empty)):
            scalar_ = nx.algebraic_connectivity(graph_pred_list_remove_empty[i])
            sample_pred.append(scalar_)

    mmd_dist = compute_mmd(
        sample_ref, sample_pred, is_hist=False, kernel=gaussian, sigma=1.0 / 10
    )
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time algebraic_connectivity mmd: ", elapsed)
    return mmd_dist


def torch_algebraic_connectivity_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                algebraic_connectivity_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                algebraic_connectivity_worker,
                [(G, bins) for G in graph_pred_list_remove_empty],
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            scalar_ = nx.algebraic_connectivity(graph_ref_list[i])
            sample_ref.append(scalar_)

        for i in range(len(graph_pred_list_remove_empty)):
            scalar_ = nx.algebraic_connectivity(graph_pred_list_remove_empty[i])
            sample_pred.append(scalar_)

    mmd_dist = torch_mmd(
        sample_ref, sample_pred, kernel=gaussian_kernel, sigma=1.0 / 10, hist=False
    )
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time algebraic_connectivity mmd: ", elapsed)
    return mmd_dist


def eccentricity_worker(param):
    G, bins = param
    eccentricity_list = list(nx.eccentricity(G).values())
    hist, _ = np.histogram(
        eccentricity_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def eccentricity_stats(graph_ref_list, graph_pred_list, bins=100, is_parallel=True):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                eccentricity_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                eccentricity_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            eccentricity_list = list(nx.eccentricity(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                eccentricity_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            eccentricity_list = list(
                nx.eccentricity(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                eccentricity_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)
    mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian, sigma=1.0 / 10)
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time eccentricity mmd: ", elapsed)
    return mmd_dist


def torch_eccentricity_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                eccentricity_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                eccentricity_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            eccentricity_list = list(nx.eccentricity(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                eccentricity_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            eccentricity_list = list(
                nx.eccentricity(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                eccentricity_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)
    mmd_dist = torch_mmd(
        sample_ref, sample_pred, kernel=gaussian_kernel, sigma=1.0 / 10
    )
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time eccentricity mmd: ", elapsed)
    return mmd_dist


def degree_assortativity_coefficient_worker(param):
    G, bins = param
    scalar_ = nx.degree_assortativity_coefficient(G)
    return np.array([scalar_])


def degree_assortativity_coefficient_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                degree_assortativity_coefficient_worker,
                [(G, bins) for G in graph_ref_list],
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                degree_assortativity_coefficient_worker,
                [(G, bins) for G in graph_pred_list_remove_empty],
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            scalar_ = nx.degree_assortativity_coefficient(graph_ref_list[i])
            sample_ref.append(scalar_)

        for i in range(len(graph_pred_list_remove_empty)):
            scalar_ = nx.degree_assortativity_coefficient(
                graph_pred_list_remove_empty[i]
            )
            sample_pred.append(scalar_)
    sample_ref = [x for x in sample_ref if ~np.isnan(x)]
    sample_pred = [x for x in sample_pred if ~np.isnan(x)]
    mmd_dist = compute_mmd(
        sample_ref, sample_pred, is_hist=False, kernel=gaussian, sigma=1.0 / 10
    )
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time degree_assortativity_coefficient mmd: ", elapsed)
    return mmd_dist


def torch_assortativity_coefficient_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                degree_assortativity_coefficient_worker,
                [(G, bins) for G in graph_ref_list],
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                degree_assortativity_coefficient_worker,
                [(G, bins) for G in graph_pred_list_remove_empty],
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            scalar_ = nx.degree_assortativity_coefficient(graph_ref_list[i])
            sample_ref.append(scalar_)

        for i in range(len(graph_pred_list_remove_empty)):
            scalar_ = nx.degree_assortativity_coefficient(
                graph_pred_list_remove_empty[i]
            )
            sample_pred.append(scalar_)
    sample_ref = [x for x in sample_ref if ~np.isnan(x)]
    sample_pred = [x for x in sample_pred if ~np.isnan(x)]
    mmd_dist = torch_mmd(
        sample_ref, sample_pred, kernel=gaussian_kernel, sigma=1.0 / 10, hist=False
    )
    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time degree_assortativity_coefficient mmd: ", elapsed)
    return mmd_dist
