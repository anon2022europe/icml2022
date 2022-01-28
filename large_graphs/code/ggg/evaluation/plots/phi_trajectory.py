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
import matplotlib.image as mpimg
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d, Axes3D
from matplotlib.text import Annotation

ex = Experiment("GraphsFromModels")


class Annotation3D(Annotation):

    def __init__(self, text, xyz, *args, **kwargs):
        super().__init__(text, xy=(0, 0), *args, **kwargs)
        self._xyz = xyz

    def draw(self, renderer):
        x2, y2, z2 = proj3d.proj_transform(*self._xyz, self.axes.M)
        self.xy = (x2, y2)
        super().draw(renderer)


def _annotate3D(ax, text, xyz, *args, **kwargs):
    '''Add anotation `text` to an `Axes3d` instance.'''

    annotation = Annotation3D(text, xyz, *args, **kwargs)
    ax.add_artist(annotation)


setattr(Axes3D, 'annotate3D', _annotate3D)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))

        return np.min(zs)


def proj(X, ax1, ax2):
    """From a 3D point in axes ax1,
    calculate position in 2D in ax2"""
    x, y, z = X
    x2, y2, _ = proj3d.proj_transform(x, y, z, ax1.get_proj())
    return ax2.transData.inverted().transform(ax1.transData.transform((x2, y2)))


def single_graph_plot(gorig, col, lcc=False):
    if not lcc:
        g = gorig
    else:
        mcc = max(nx.connected_components(gorig), key=len)
        g = gorig.subgraph(mcc).copy()

    # can use other possible choice of layouts (ex. kamada_kawai_layout)
    # orange color = [225/255, 198/255, 58/255]
    # green color = [67/255, 160/255, 39/255]
    # pos=nx.circular_layout(g)
    # pos=nx.spring_layout(g,dim=2,pos=pos)
    pos = nx.nx_pydot.graphviz_layout(g, prog="neato")

    edge_color = np.array([0.1, 0.1, 0.1]).reshape(1, -1)
    node_color = np.array([225 / 255, 198 / 255, 58 / 255]).reshape(1, -1)
    nx.draw_networkx_edges(g, pos, edge_color=edge_color, alpha=0.55, width=6, ax=col)
    nodes = nx.draw_networkx_nodes(
        g,
        pos,
        node_color=node_color,
        alpha=1,
        linewidths=2,
        node_size=400,
        ax=col,
    )
    nodes.set_edgecolor(edge_color)
    col.set_axis_off()


@ex.config
def config():
    """Configuration for graph generation from trained models"""
    # TODO build directory for node dist, computed from saved datasets

    # Model parameters
    z_dim = 3  # z dimension used in the run
    epoch = None  # epoch for model generation
    dataset = None  # Dataset used in exp
    batch_size = 32  # Batch size used in the run
    finetti_dim = None

    save_dir = "ggg_qm9/lightning_log/plots/"
    model_dir = "ggg_qm9/lightning_log/plots/"


def create_ctx(numb_ctx=3, center_coord=None, all_contexts=None,
               sphere_x=None, sphere_y=None, sphere_z=None):
    centre = center_coord
    radius = 0.15

    u = np.random.normal(0, 2 * np.pi, numb_ctx)
    v = np.random.normal(-1, 1, numb_ctx)

    sphere1_xi = centre[0] + radius * np.cos(u) * np.sin(v)
    sphere1_yi = centre[1] + radius * np.sin(u) * np.sin(v)
    sphere1_zi = centre[2] + radius * np.cos(v)

    if sphere_x is None:
        sphere_x = []
        sphere_y = []
        sphere_z = []

    for idx in range(numb_ctx):
        sphere_x.append(sphere1_xi[idx])
        sphere_y.append(sphere1_yi[idx])
        sphere_z.append(sphere1_zi[idx])

    if all_contexts is None:
        all_contexts = []
    x = pt.empty(size=(numb_ctx, 1, 3))
    for i in range(numb_ctx):
        x[i][0][0] = sphere1_xi[i]
        x[i][0][1] = sphere1_yi[i]
        x[i][0][2] = sphere1_zi[i]
    all_contexts.append(x)

    return all_contexts, sphere_x, sphere_y, sphere_z


def latent_space(_config):
    numb_to_plot_per_sphere = 2

    btx_size = 25
    ctx_size = 3
    center_coord = [[0, 0, 0], [0, 0, 2], [1, 1, 0], [0, -1, 0], [0, -1, -1], [-2, 1, 0.5]]
    all_contexts, sphere_x, sphere_y, sphere_z = [], [], [], []
    for idx in range(ctx_size):
        all_contexts, sphere_x, sphere_y, sphere_z = create_ctx(numb_ctx=btx_size, center_coord=center_coord[idx],
                                                                all_contexts=all_contexts, sphere_x=sphere_x,
                                                                sphere_y=sphere_y, sphere_z=sphere_z)

    epochs = [50]
    for ep in epochs:
        epoch_ = ep
        new_fldr = "/trajectory_" + str(epoch_) + "/"
        os.makedirs(_config["save_dir"] + new_fldr, exist_ok=True)

        model = get_GGGAN_model(epoch_, log_dir=_config["model_dir"])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection=Axes3D.name)

        # make the panes transparent
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # make the grid lines transparent
        ax.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
        ax.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)

        ax.grid(False)
        ax.axis('off')

        # Hide axes ticks
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.xaxis._axinfo['juggled'] = (0, 0, 0)
        ax.yaxis._axinfo['juggled'] = (1, 1, 1)
        ax.zaxis._axinfo['juggled'] = (2, 2, 2)

        # the first column of the [ ] is the start. to change length is changing the end part the x, y or z.
        scale = 0.75
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
            [-2 * scale, 1.0 * scale],
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
            [-2 * scale, 1.5 * scale],
            mutation_scale=20,
            lw=0.5,
            arrowstyle="-|>",
            color="k",
            linestyle="dashed",
        )
        ax.add_artist(arrow_z)
        ax.axes.set_xlim3d(left=-2 * scale, right=2 * scale)
        ax.axes.set_ylim3d(bottom=-1.5 * scale, top=1.5 * scale)
        ax.axes.set_zlim3d(bottom=-2 * scale, top=1.5 * scale)

        # Create a dummy axes to place annotations to
        ax3 = fig.add_subplot(111, frame_on=False)
        ax3.axis("off")
        # ax3.axis([-2 * scale, 2 * scale, -1.5 * scale, 1.5 * scale])
        ax3.axis([0, 1, 0, 1])

        ax.scatter(sphere_x, sphere_y, sphere_z, s=10, c=[225 / 255, 198 / 255, 58 / 255], alpha=0.5, zorder=1)

        counter = 0
        all_graphs = []
        annotate_tuples = [[0, -20], [20, 0]]
        for idx, ctx in enumerate(all_contexts):
            X_out, A_out, N_out, Z_out = [
                x.cpu() if x is not None else x
                for x in model.generator.forward(batch_size=btx_size, device=None, joined_embed=ctx)
            ]

            to_plot = np.random.choice(A_out.shape[0], numb_to_plot_per_sphere, replace=False)
            annotated_z = False

            start_x_, start_y_ = proj([center_coord[idx][0], center_coord[idx][1], center_coord[idx][2]], ax, ax3)

            annotate_c = 0
            for b_idx in range(A_out.shape[0]):
                if b_idx in to_plot:
                    adj_m = A_out[b_idx].cpu().detach().numpy()

                    np.fill_diagonal(adj_m, 0)
                    g = nx.from_numpy_matrix(adj_m)

                    all_graphs.append(g)

                    fig2 = plt.figure()
                    ax2 = fig2.add_subplot(111)
                    single_graph_plot(g, col=ax2, lcc=False)
                    plt.savefig(os.path.join(_config["save_dir"] + new_fldr, str(counter).zfill(3) + ".png"),
                                dpi=100, transparent=True)
                    ax.scatter(sphere_x[counter], sphere_y[counter], sphere_z[counter], s=12,
                               c=[67 / 255, 160 / 255, 39 / 255]
                               , zorder=10)

                    arr_lena = mpimg.imread(os.path.join(_config["save_dir"] + new_fldr, str(counter).zfill(3) + ".png"))

                    imagebox = matplotlib.offsetbox.OffsetImage(arr_lena, zoom=0.05)  # 0.015
                    imagebox.image.axes = ax3

                    x = start_x_
                    y = start_y_
                    if center_coord[idx][1] != 0:
                        x = start_x_ - (0.04 + 0.025 * (center_coord[idx][1]-1))
                        y = start_y_
                    elif center_coord[idx][0] != 0:
                        x = start_x_ - (0.045 + 0.048 * (center_coord[idx][0]-1))
                        y = start_y_
                    ax3.scatter(x, y, c='b', s=15)

                    aprev = matplotlib.offsetbox.AnnotationBbox(imagebox, xy=(x, y),
                                                                xybox=(annotate_tuples[annotate_c][0], annotate_tuples[annotate_c][1]),
                                                                frameon=False, xycoords="data",
                                                                boxcoords="offset points")
                    annotate_c += 1
                    ax3.add_artist(aprev)

                    if not annotated_z:
                        ax.text(sphere_x[counter], sphere_y[counter], sphere_z[counter], f'$g_{{gg}}(\phi,z_{idx})$',
                                size=9, zorder=20, color='k')
                        annotated_z = True
                    plt.close(fig2)
                counter += 1

        # plt.show()
        fig.savefig(os.path.join(_config["save_dir"] + new_fldr, "space_" + ".pdf"), dpi=600)


@ex.main
def main(_config):
    latent_space(_config)


if __name__ == "__main__":
    ex.run_commandline()
