"""
Functions to plot graphs
"""

import os

import networkx as nx
import numpy as np

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt


def cluster_plot_molgrid(
    gen_graphs: [], dataset_graphs = None, name="GraphGAN", lcc=False, save_dir=None, save=False, rc=(3, 4)
):
    """Plots graphs in a grid as a single image

    :param gen_graphs:
    :param name:
    :param lcc:
    :param save_dir:
    :param save:
    :param rc:
    :return:
    """

    def g_plot(gorig, rand_choice, counter, ax, col, iso_temp_list, dataset_graphs):

        if not lcc:
            g = gorig
        else:
            mcc = max(nx.connected_components(gorig), key=len)
            g = gorig.subgraph(mcc).copy()

        # can use other possible choice of layouts (ex. kamada_kawai)
        pos = nx.spring_layout(g)
        nx.draw_networkx_edges(
            g, pos, edge_color="#000000", alpha=0.45, width=2, ax=col
        )
        nodes = nx.draw_networkx_nodes(
            g,
            pos,
            node_color="#29465b",
            alpha=0.97,
            linewidths=2,
            node_size=280,
            ax=col,
        )
        nodes.set_edgecolor("w")
        col.set_axis_off()
        iso_temp_list.append(gorig)
        counter += 1

        return counter, iso_temp_list

    pkl_list = gen_graphs

    # choose random graphs
    nrows_, ncols_ = rc
    rand_choice = np.random.choice(len(pkl_list), nrows_ * ncols_)

    fig, ax = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(24, 10))
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.35
    )

    compare_graphs = []
    counter = 0
    iso_temp_list = []
    for row in ax:
        if ncols_ != 1:
            for col in row:
                gorig = pkl_list[rand_choice[counter]]
                compare_graphs.append(gorig)

                counter, iso_temp_list = g_plot(
                    gorig, rand_choice, counter, ax, col, iso_temp_list, dataset_graphs
                )
        else:
            col = row
            gorig = pkl_list[rand_choice[counter]]
            compare_graphs.append(gorig)
            counter, iso_temp_list = g_plot(
                gorig, rand_choice, counter, ax, col, iso_temp_list, dataset_graphs
            )

    fig.suptitle(f"Sampled graphs from {name}", fontsize=30)
    if save:
        plt.savefig(os.path.join(save_dir, name + ".pdf"))
        plt.close()

    # counter=1
    # for g1 in compare_graphs:
    #     for g2 in dataset_graphs:
    #         if nx.is_isomorphic(g1, g2):
    #             print("Graphs is in the dataset")
    #             nx.draw(g2)
    #             plt.savefig(os.path.join(save_dir, str(counter) + ".pdf"))
    #             plt.close()
    #             counter+=1
    #             break

    return fig
