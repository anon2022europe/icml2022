import os
import pickle
import torch as pt
import numpy as np
import networkx as nx
from sacred import Experiment

from ggg.evaluation.plots.utils.plot_helpers import get_GGGAN_model

import matplotlib

matplotlib.use("TKAgg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

ex = Experiment("GraphsFromModels")


def proj(X, ax1, ax2):
    """From a 3D point in axes ax1,
    calculate position in 2D in ax2"""
    x, y, z = X
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax1.get_proj())
    return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))


from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


@ex.config
def config():
    """Configuration for graph generation from trained models"""
    # TODO build directory for node dist, computed from saved datasets

    # Model parameters
    z_dim = 7  # z dimension used in the run
    epoch = None  # epoch for model generation
    dataset = None  # Dataset used in exp
    batch_size = 32  # Batch size used in the run
    finetti_dim = None

    file_dir = "GG-GAN_QM9/lightning_log/data/embeddings/Z0"
    model_dir = "GG-GAN_QM9/lightning_log/plots/"


def graphs_single_node(_config, in_grey=False):
    end_epoch = 1000
    os.makedirs(_config["model_dir"] + "graphs/", exist_ok=True)
    z_start = pt.load(os.path.join(_config["file_dir"], "Z0_0000.pt"))
    z_end = pt.load(
        os.path.join(_config["file_dir"], "Z0_{}.pt".format(str(end_epoch).zfill(4)))
    )
    static_finetti = pt.load("GG-GAN_QM9/lightning_log/external_finetti.pt")
    values_to_interpolate = np.linspace(0, 1, 7)
    interpolation = []
    for i in values_to_interpolate:
        interpolation.append(pt.lerp(z_start, z_end, i))

    legend_ = []
    batch_idx = 0
    node_in_graph = 0
    x, y, z = [], [], []
    random_index = np.array(
        [0, 1, 2]
    )  # np.random.choice(range(_config["z_dim"]), 3, replace=False)
    model = get_GGGAN_model(1000, log_dir=_config["model_dir"])
    for numb, Z in enumerate(interpolation):
        legend_.append(numb)
        model.generator.Z0_init = pt.nn.Parameter(Z)

        x.append(Z[batch_idx][node_in_graph][random_index[0]].detach().numpy())
        y.append(Z[batch_idx][node_in_graph][random_index[1]].detach().numpy())
        z.append(Z[batch_idx][node_in_graph][random_index[2]].detach().numpy())

        X_out, A_out, Z, finetti_u, mod_emb = model.generator.sample(
            batch_size=20, device="cpu", external_finetti_u=static_finetti
        )
        adj_m = A_out[batch_idx].cpu().detach().numpy()

        g = nx.from_numpy_matrix(adj_m)

        if in_grey:
            fig = plt.figure()
            fig.patch.set_facecolor("#d8dcd6")
            fig.patch.set_alpha(0.5)
            ax = fig.add_subplot(111)
            edge_color = "#d8dcd6"
            node_color = "#7d7f7c"
            alpha_edges = 0.1
            alpha_nodes = 0.1
            save_appx = "grey_"
            pos = nx.kamada_kawai_layout(g)
            nx.draw_networkx_edges(
                g, pos, width=2, edge_color=edge_color, alpha=alpha_edges, ax=ax
            )
            nodes = nx.draw_networkx_nodes(
                g,
                pos,
                node_color=node_color,
                alpha=alpha_nodes,
                linewidths=4,
                node_size=400,
                ax=ax,
            )
            nodes.set_edgecolor("w")
            plt.axis("off")
            fig.savefig(
                _config["model_dir"] + "graphs/" + "g_" + save_appx + str(numb),
                facecolor=fig.get_facecolor(),
                edgecolor="k",
                transparent=False,
                bdpi=300,
            )
            # plt.show()
            # exit()

        fig = plt.figure()
        fig.patch.set_facecolor("#29465b")
        fig.patch.set_alpha(0.35)
        edge_color = "#000000"
        node_color = "#29465b"
        alpha_edges = 0.45
        alpha_nodes = 0.97
        save_appx = ""
        pos = nx.kamada_kawai_layout(g)
        nx.draw_networkx_edges(
            g, pos, width=2, edge_color=edge_color, alpha=alpha_edges
        )
        nodes = nx.draw_networkx_nodes(
            g,
            pos,
            node_color=node_color,
            alpha=alpha_nodes,
            linewidths=1,
            node_size=400,
        )
        nodes.set_edgecolor("w")
        plt.axis("off")
        fig.savefig(
            _config["model_dir"] + "graphs/" + "g_" + save_appx + str(numb),
            facecolor=fig.get_facecolor(),
            edgecolor="k",
            transparent=True,
            bdpi=300,
        )

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, color="#89a0b0", alpha=0.8, linewidth=1.5)
    ax.scatter(x, y, z, color="#1e488f", edgecolors="w", s=90, alpha=1, linewidth=3)
    for i in range(len(legend_)):
        ax.text(x[i] + 0.05, y[i], z[i] + 0.1, str(legend_[i]))
    # ax.set_axis_off()
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    plt.locator_params(nbins=5)
    fig.savefig(
        _config["model_dir"] + "graphs/" + "trajectory", transparent=True, dpi=300
    )


def mayavi_nodes_trajectory(_config):
    end_epoch = 1000
    os.makedirs(_config["model_dir"] + "trajects/", exist_ok=True)
    z_start = pt.load(os.path.join(_config["file_dir"], "Z0_0000.pt"))
    z_end = pt.load(
        os.path.join(_config["file_dir"], "Z0_{}.pt".format(str(end_epoch).zfill(4)))
    )
    values_to_interpolate = np.linspace(0, 1, 2)
    interpolation = []
    for i in values_to_interpolate:
        interpolation.append(pt.lerp(z_start, z_end, i))

    batch_idx = 0
    random_index = np.array(
        [0, 1, 2]
    )  # np.random.choice(range(_config["z_dim"]), 3, replace=False)
    x_prev, y_prev, z_prev = [], [], []
    from mayavi import mlab

    for i in range(len(interpolation)):
        x0, y0, z0 = [], [], []
        for node in range(9):
            x0.append(
                interpolation[i][batch_idx][node][random_index[0]].detach().numpy()
            )
            y0.append(
                interpolation[i][batch_idx][node][random_index[1]].detach().numpy()
            )
            z0.append(
                interpolation[i][batch_idx][node][random_index[2]].detach().numpy()
            )

        x_prev.append(x0)
        y_prev.append(y0)
        z_prev.append(z0)

        fig = mlab.figure()
        # if i > 0:
        #     counter = i
        #     for prev_x, prev_y, prev_z in zip(x_prev[:-1], y_prev[:-1], z_prev[:-1]):
        #         mlab.points3d(prev_x, prev_y, prev_z, color="#ffb07c", s=40, edgecolors='w', linewidth=0.5,
        #                    alpha=0.2, zorder=-counter)
        #         counter -= 1

        mlab.points3d(x0, y0, z0, scale_factor=0.15)
        mlab.show()


def matplotlib_nodes_trajectory(_config):
    end_epoch = 1000
    os.makedirs(_config["model_dir"] + "trajects/", exist_ok=True)
    z_start = pt.load(os.path.join(_config["file_dir"], "Z0_0000.pt"))
    z_end = pt.load(
        os.path.join(_config["file_dir"], "Z0_{}.pt".format(str(end_epoch).zfill(4)))
    )
    values_to_interpolate = np.linspace(0, 1, 7)
    interpolation = []
    for i in values_to_interpolate:
        interpolation.append(pt.lerp(z_start, z_end, i))

    pos_x, pos_y = [], []
    for p in range(len(interpolation)):
        pos_x.append(0.05 + 0.003 * p)
        pos_y.append(0.05 - 0.003 * p)

    batch_idx = 0
    random_index = np.array(
        [0, 1, 2]
    )  # np.random.choice(range(_config["z_dim"]), 3, replace=False)
    x_prev, y_prev, z_prev = [], [], []
    for i in range(len(interpolation)):
        x0, y0, z0 = [], [], []
        for node in range(9):
            x0.append(
                interpolation[i][batch_idx][node][random_index[0]].detach().numpy()
            )
            y0.append(
                interpolation[i][batch_idx][node][random_index[1]].detach().numpy()
            )
            z0.append(
                interpolation[i][batch_idx][node][random_index[2]].detach().numpy()
            )

        x_prev.append(x0)
        y_prev.append(y0)
        z_prev.append(z0)

        # x1000, y1000, z1000 = [], [], []
        # for node in range(9):
        #     x1000.append(interpolation[i+1][batch_idx][node][random_index[0]].detach().numpy())
        #     y1000.append(interpolation[i+1][batch_idx][node][random_index[1]].detach().numpy())
        #     z1000.append(interpolation[i+1][batch_idx][node][random_index[2]].detach().numpy())
        #
        # line_x, line_y, line_z = [], [], []
        # for j in range(9):
        #     line_x.append(x0[j])
        #     line_x.append(x1000[j])
        #
        #     line_y.append(y0[j])
        #     line_y.append(y1000[j])
        #
        #     line_z.append(z0[j])
        #     line_z.append(z1000[j])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if i > 0:
            counter = i
            for prev_x, prev_y, prev_z in zip(x_prev[:-1], y_prev[:-1], z_prev[:-1]):
                ax.scatter(
                    prev_x,
                    prev_y,
                    prev_z,
                    color="#ffb07c",
                    s=30,
                    edgecolors="w",
                    linewidth=0.2,
                    alpha=0.2,
                    zorder=-counter,
                )
                counter -= 1

        ax.scatter(
            x0,
            y0,
            z0,
            color="#c14a09",
            edgecolors="w",
            s=50,
            alpha=1,
            linewidth=1,
            zorder=20,
            depthshade=False,
            label="epoch " + str(int(values_to_interpolate[i] * end_epoch)),
        )
        # ax1.axes.set_xlim3d(left=-2, right=1.5)
        # ax1.axes.set_ylim3d(bottom=-1.5, top=1.5)
        # ax1.axes.set_zlim3d(bottom=-2, top=1)
        #
        # # make the panes transparent
        # ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # # make the grid lines transparent
        # ax1.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        # ax1.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        # ax1.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        #
        # ax1.grid(False)
        # ax1.axis('off')

        # ax.scatter(x1000, y1000, z1000, color="#154406", edgecolors='w', s=50, alpha=1, linewidth=1, label="Final position")
        # for l_ in range(0, len(line_x), 2):
        #     ax.plot(line_x[l_:l_ + 2], line_y[l_:l_ + 2], line_z[l_:l_ + 2], 'k--')
        # ax.plot([-2, -2], [1.5, -1.5], [-2, -2])
        # make the panes transparent
        # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # # make the grid lines transparent
        # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        # ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        #
        arrow_x = Arrow3D(
            [-2, -2],
            [1.5, -1.5],
            [-2, -2],
            mutation_scale=20,
            lw=0.5,
            arrowstyle="-|>",
            color="k",
            linestyle="dashed",
        )
        ax.add_artist(arrow_x)
        arrow_y = Arrow3D(
            [-2, 1.5],
            [1.5, 1.5],
            [-2, -2],
            mutation_scale=20,
            lw=0.5,
            arrowstyle="-|>",
            color="k",
            linestyle="dashed",
        )
        ax.add_artist(arrow_y)
        arrow_z = Arrow3D(
            [-2, -2],
            [1.5, 1.5],
            [-2, 1],
            mutation_scale=20,
            lw=0.5,
            arrowstyle="-|>",
            color="k",
            linestyle="dashed",
        )
        ax.add_artist(arrow_z)
        ax.axes.set_xlim3d(left=-2, right=1.5)
        ax.axes.set_ylim3d(bottom=-1.5, top=1.5)
        ax.axes.set_zlim3d(bottom=-2, top=1)

        if i > 0:
            for j in range(i):
                arr_lena = mpimg.imread(
                    _config["model_dir"] + "graphs/" + "g_grey_" + str(j) + ".png"
                )
                imagebox = OffsetImage(arr_lena, zoom=0.1)
                aprev = AnnotationBbox(imagebox, (-pos_x[j], pos_y[j]), frameon=False)
                ax.add_artist(aprev)

        arr_lena = mpimg.imread(
            _config["model_dir"] + "graphs/" + "g_" + str(i) + ".png"
        )
        imagebox = OffsetImage(arr_lena, zoom=0.1)
        ab = AnnotationBbox(imagebox, (-pos_x[i], pos_y[i]), frameon=False)
        ax.add_artist(ab)
        # # Hide axes ticks
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        # ax.set_zticklabels([])
        #
        # ax.xaxis._axinfo['juggled'] = (0, 0, 0)
        # ax.yaxis._axinfo['juggled'] = (1, 1, 1)
        # ax.zaxis._axinfo['juggled'] = (2, 2, 2)
        # plt.legend()
        ax.grid(False)
        plt.axis("off")

        fig.savefig(
            _config["model_dir"] + "trajects/" + "trajectory_" + str(i), dpi=300
        )

    ################## Trajectory of a single node
    #
    # random_index = np.random.choice(range(_config["z_dim"]), 3, replace=False)
    # list_ = [0, 2, 4, 6, 7, 8, 9]
    # print("Random selected points to construct trajectory are {} \n".format(random_index))
    # x, y, z, legend_ = [], [], [], []
    # batch_idx = 17
    # node_in_graph = 2
    # for root, dirs, files in os.walk(_config["exps_dir"]):
    #     dir_list = root.split("/")
    #     if dir_list[-1] == "Z0":
    #         specific_exp = dir_list[-3]
    #         # Plot save directory
    #         os.makedirs(_config["plots_save_dir"] + _config["specific_exp"], exist_ok=True)
    #         os.makedirs(_config["plots_save_dir"] + _config["specific_exp"] + "/graphs/", exist_ok=True)
    #
    #         for idx, embed in enumerate(np.array(files)[[list_]]):
    #             pkl_list = pt.load(os.path.join(root, embed))
    #
    #             x.append(pkl_list[batch_idx][node_in_graph][random_index[0]].numpy())
    #             y.append(pkl_list[batch_idx][node_in_graph][random_index[1]].numpy())
    #             z.append(pkl_list[batch_idx][node_in_graph][random_index[2]].numpy())
    #
    #             if idx != 0:
    #                 epoch_ = str(int(embed.split("_")[1].split(".")[0].lstrip("0")) + 1)
    #             else:
    #                 epoch_ = str(idx)
    #             legend_.append(epoch_)
    #             with open(os.path.join(_config["graphs_dir"], specific_exp) + "_" + epoch_ + ".pkl", "rb") as f:
    #                 graphs = pickle.load(f)
    #
    #             fig = plt.figure()
    #             pos = nx.kamada_kawai_layout(graphs[batch_idx])
    #             nx.draw_networkx_edges(graphs[batch_idx], pos, width=2, edge_color="#000000", alpha=0.45, linewidths=4.5)
    #             nodes = nx.draw_networkx_nodes(graphs[batch_idx], pos, node_color="#29465b", alpha=0.97,
    #                                            linewidths=4, node_size=400)
    #             nodes.set_edgecolor('w')
    #             plt.axis('off')
    #             # nx.draw_kamada_kawai(graphs[batch_idx], node_color="skyblue", width=2, edge_color="grey")
    #             fig.savefig(_config["plots_save_dir"] + _config["specific_exp"] + "/graphs/" +
    #                         "g_" + epoch_, bdpi=300)
    #             plt.close(fig)

    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.plot(x, y, z, color="#89a0b0", alpha=0.8, linewidth=1.5)
    # ax.scatter(x, y, z, color="#1e488f", edgecolors='w', s=90, alpha=1, linewidth=3)
    # for i in range(len(legend_)):
    #     ax.text(x[i]+0.05, y[i], z[i]+0.1, str(legend_[i]))
    # # ax.set_axis_off()
    # # make the panes transparent
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # # # make the grid lines transparent
    # ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # plt.locator_params(nbins=5)
    # plt.show()
    # # # Save figure
    # # fig.savefig(_config["plots_save_dir"] + _config["specific_exp"] + "/" + "trajectory", bdpi=300)


def fix_phi_matplotlib(_config, in_grey=False):
    os.makedirs(_config["model_dir"] + "graphs/", exist_ok=True)
    os.makedirs(_config["model_dir"] + "trajects/", exist_ok=True)
    end_epoch = 1000
    phi = pt.load(
        os.path.join(_config["file_dir"], "Z0_{}.pt".format(str(end_epoch).zfill(4)))
    )
    finetti_1 = pt.load("GG-GAN_QM9/lightning_log/finetti_1.pt")
    # finetti_2 = pt.load("GG-GAN_QM9/lightning_log/finetti_2.pt")
    # finetti_3 = pt.load("GG-GAN_QM9/lightning_log/finetti_3.pt")
    finetti_4 = pt.load("GG-GAN_QM9/lightning_log/finetti_4.pt")
    values_to_interpolate = np.linspace(0, 1, 5)
    interpolation = []
    for i in values_to_interpolate:
        interpolation.append(pt.lerp(finetti_1, finetti_4, i))

    # for i in values_to_interpolate[:-1]:
    #     interpolation.append(pt.lerp(finetti_2, finetti_3, i))
    #
    # for i in values_to_interpolate:
    #     interpolation.append(pt.lerp(finetti_3, finetti_4, i))

    legend_ = []
    batch_idx = 7
    x, y, z = [], [], []
    random_index = np.array(
        [0, 1, 2]
    )  # np.random.choice(range(_config["z_dim"]), 3, replace=False)
    model = get_GGGAN_model(1000, log_dir=_config["model_dir"])

    for numb, finet in enumerate(interpolation):
        legend_.append(numb)
        model.generator.Z0_init = pt.nn.Parameter(phi)

        x.append(finet[batch_idx][random_index[0]].detach().numpy())
        y.append(finet[batch_idx][random_index[1]].detach().numpy())
        z.append(finet[batch_idx][random_index[2]].detach().numpy())

        X_out, A_out, Z, finetti_u, mod_emb = model.generator.sample(
            batch_size=20, device="cpu", external_finetti_u=finet
        )
        adj_m = A_out[batch_idx].cpu().detach().numpy()

        g = nx.from_numpy_matrix(adj_m)

        if in_grey:
            fig = plt.figure()
            # fig.patch.set_facecolor('#d8dcd6')
            fig.patch.set_alpha(0.5)
            ax = fig.add_subplot(111)
            edge_color = "#d8dcd6"
            node_color = "#7d7f7c"
            alpha_edges = 0.1
            alpha_nodes = 0.1
            save_appx = "grey_"
            # pos = nx.kamada_kawai_layout(g)
            # nx.draw_networkx_edges(g, pos, width=2, edge_color=edge_color, alpha=alpha_edges, ax=ax)
            # nodes = nx.draw_networkx_nodes(g, pos, node_color=node_color, alpha=alpha_nodes,
            #                                linewidths=4, node_size=400, ax=ax)
            # nodes.set_edgecolor('w')
            # plt.axis('off')
            plt.imshow(adj_m, cmap="binary")
            fig.savefig(
                _config["model_dir"] + "graphs/" + "g_" + save_appx + str(numb),
                facecolor=fig.get_facecolor(),
                edgecolor="k",
                transparent=True,
                bdpi=300,
            )

        fig = plt.figure()
        # fig.patch.set_facecolor('#29465b')
        fig.patch.set_alpha(0.35)
        edge_color = "#000000"
        node_color = "#29465b"
        alpha_edges = 0.45
        alpha_nodes = 0.97
        save_appx = ""
        # pos = nx.kamada_kawai_layout(g)
        # nx.draw_networkx_edges(g, pos, width=2, edge_color=edge_color, alpha=alpha_edges)
        # nodes = nx.draw_networkx_nodes(g, pos, node_color=node_color, alpha=alpha_nodes,
        #                                linewidths=1, node_size=400)
        # nodes.set_edgecolor('w')
        plt.imshow(adj_m, cmap="Blues")
        plt.axis("off")
        fig.savefig(
            _config["model_dir"] + "graphs/" + "g_" + save_appx + str(numb),
            facecolor=fig.get_facecolor(),
            edgecolor="k",
            transparent=True,
            bdpi=300,
        )

    fig = plt.figure()
    scale = 2
    ax = fig.add_subplot(111, projection="3d")
    # Create a dummy axes to place annotations to
    ax2 = fig.add_subplot(111, frame_on=False)
    ax2.axis("off")
    ax2.axis([0, 1, 0, 1])

    ax.plot(x, y, z, color="#89a0b0", alpha=0.8, linewidth=1.5)
    ax.scatter(x, y, z, color="#1e488f", edgecolors="w", s=90, alpha=1, linewidth=3)
    for i in range(len(legend_)):
        ax.text(x[i] + 0.05, y[i], z[i] + 0.1, str(legend_[i]))
    # ax.set_axis_off()
    # make the panes transparent
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # # make the grid lines transparent
    ax.xaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.yaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    ax.zaxis._axinfo["grid"]["color"] = (1, 1, 1, 0)
    plt.locator_params(nbins=5)
    fig.savefig(
        _config["model_dir"] + "graphs/" + "trajectory", transparent=True, dpi=300
    )

    pos_x, pos_y = [], []
    for p in range(len(interpolation)):
        pos_x.append(0.05 + 0.0010 * p)
        pos_y.append(0.05 - 0.0015 * p)

    x_prev, y_prev, z_prev = [], [], []
    for i in range(len(interpolation)):
        x0, y0, z0 = [], [], []
        for node in range(9):
            x_pos = interpolation[i][batch_idx][random_index[0]].detach().numpy()
            y_pos = interpolation[i][batch_idx][random_index[1]].detach().numpy()
            z_pos = interpolation[i][batch_idx][random_index[2]].detach().numpy()
            x0.append(x_pos)
            y0.append(y_pos)
            z0.append(z_pos)

        x_prev.append(x_pos)
        y_prev.append(y_pos)
        z_prev.append(z_pos)

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        if i > 0:
            counter = 0
            for prev_x, prev_y, prev_z in zip(x_prev[:-1], y_prev[:-1], z_prev[:-1]):
                ax.scatter(
                    prev_x,
                    prev_y,
                    prev_z,
                    color="#ffb07c",
                    s=30,
                    edgecolors="w",
                    linewidth=0.2,
                    alpha=0.2,
                )

                x, y = proj((prev_x, prev_y, prev_z), ax, ax2)

                arr_lena = mpimg.imread(
                    _config["model_dir"] + "graphs/" + "g_grey_" + str(counter) + ".png"
                )
                imagebox = OffsetImage(arr_lena, zoom=0.1)

                aprev = AnnotationBbox(
                    imagebox,
                    xy=(x * 0.025, y * 0.025),
                    frameon=False,
                    xybox=(-10, 10),
                    xycoords="data",
                    boxcoords="offset points",
                    pad=0.3,
                    arrowprops=dict(arrowstyle="->"),
                )
                ax.add_artist(aprev)
                counter += 1

        ax.scatter(
            x0,
            y0,
            z0,
            color="#c14a09",
            edgecolors="w",
            s=50,
            alpha=1,
            linewidth=1,
            # label="epoch " + str(int(values_to_interpolate[i] * end_epoch))
        )

        arrow_x = Arrow3D(
            [-2 * scale, -2 * scale],
            [1.5 * scale, -1.5 * scale],
            [-2 * scale, -2 * scale],
            mutation_scale=20,
            lw=0.5,
            arrowstyle="-|>",
            color="k",
            linestyle="dashed",
        )
        ax.add_artist(arrow_x)
        arrow_y = Arrow3D(
            [-2 * scale, 1.5 * scale],
            [1.5 * scale, 1.5 * scale],
            [-2 * scale, -2 * scale],
            mutation_scale=20,
            lw=0.5,
            arrowstyle="-|>",
            color="k",
            linestyle="dashed",
        )
        ax.add_artist(arrow_y)
        arrow_z = Arrow3D(
            [-2 * scale, -2 * scale],
            [1.5 * scale, 1.5 * scale],
            [-2 * scale, 1 * scale],
            mutation_scale=20,
            lw=0.5,
            arrowstyle="-|>",
            color="k",
            linestyle="dashed",
        )
        ax.add_artist(arrow_z)
        ax.axes.set_xlim3d(left=-2 * scale, right=1.5 * scale)
        ax.axes.set_ylim3d(bottom=-1.5 * scale, top=1.5 * scale)
        ax.axes.set_zlim3d(bottom=-2 * scale, top=1 * scale)

        # if i > 0:
        #     for j in range(i):
        #         arr_lena = mpimg.imread(_config["model_dir"] + "graphs/" + "g_grey_" + str(j) + ".png")
        #         imagebox = OffsetImage(arr_lena, zoom=0.1)
        #         aprev = AnnotationBbox(imagebox, (-pos_x[j], pos_y[j]), frameon=False,
        #             xybox=(-50., 50.), xycoords='data', boxcoords="offset points",
        #             pad=0.3, arrowprops=dict(arrowstyle="-"))
        #         ax.add_artist(aprev)

        arr_lena = mpimg.imread(
            _config["model_dir"] + "graphs/" + "g_" + str(i) + ".png"
        )
        imagebox = OffsetImage(arr_lena, zoom=0.1)
        ab = AnnotationBbox(
            imagebox,
            (-pos_x[i], pos_y[i]),
            frameon=False,
            arrowprops=dict(arrowstyle="->"),
        )
        ax.add_artist(ab)

        ax.grid(False)
        plt.axis("off")

        fig.savefig(
            _config["model_dir"] + "trajects/" + "trajectory_" + str(i), dpi=300
        )


def spheres(_config, in_grey=False):
    phi = np.linspace(0, np.pi, 20)
    theta = np.linspace(0, 2 * np.pi, 40)
    x = np.outer(np.sin(theta), np.cos(phi))
    y = np.outer(np.sin(theta), np.sin(phi))
    z = np.outer(np.cos(theta), np.ones_like(phi))

    batch_idx = 7
    finetti_1 = pt.load("GG-GAN_QM9/lightning_log/finetti_1.pt")
    finetti_1[batch_idx] /= np.linalg.norm(finetti_1[batch_idx], axis=0)

    xi, yi, zi = [], [], []
    random_index = np.array([0, 1, 2])

    xi.append(finetti_1[batch_idx][random_index[0]].detach().item())
    yi.append(finetti_1[batch_idx][random_index[1]].detach().item())
    zi.append(finetti_1[batch_idx][random_index[2]].detach().item())

    # fig, ax = plt.subplots(1, 1, subplot_kw={'projection': '3d', 'aspect': 'equal'})
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_wireframe(x, y, z, color="k", rstride=1, cstride=1)
    ax.scatter(xi, yi, zi, s=100, c="r", zorder=10)
    plt.show()


@ex.main
def main(_config):
    spheres(_config, in_grey=True)


if __name__ == "__main__":
    ex.run_commandline()
