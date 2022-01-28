import io
import os
from warnings import warn

import PIL
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont

try:
    import umap
    import umap.plot
except:
    warn("Import error on umap, can't use umap")
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
from torchvision.transforms import ToTensor

from pathlib import Path

parent_dir = str(str(Path(os.path.abspath(__file__)).parents[4]))

matplotlib.use("Agg")

from tensorboard.backend.event_processing import event_accumulator

from .plot_helpers import *

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning)

from sklearn.neighbors import KernelDensity
from pathlib import Path

parent_dir = str(str(Path(os.path.abspath(__file__)).parents[4]))
import torch as pt


def plot_histogram(_config, metric, save=False, kde=False, model_name="GraphGAN"):
    """Histogram and density plots for distributions (degree and cycle)"""

    suffix_, title, xlabel = metric_to_use(metric)
    color_list = ["royalblue", "darkgreen"]

    run_list = list_from_pickle(_config["dist_dir"] + _config["specific_exp"] + suffix_)
    dataset_list = list_from_pickle(
        _config["dataset_dist_dir"] + _config["dataset"] + suffix_
    )

    plot_list = [run_list, dataset_list]
    if _config["handle"] == "Baseline":
        label_list = [_config["specific_exp"], _config["dataset"]]
    else:
        label_list = [model_name, _config["dataset"]]

    max_n = 0
    for list_ in plot_list:
        if len(list_) > 0 and max(list_) > max_n:
            max_n = max(list_)

    fig, ax = plt.subplots(figsize=(22, 12))
    bins = np.arange(max_n + 2) - 0.5

    if kde:
        # instantiate and fit the KDE model
        kde_run = KernelDensity(bandwidth=1.0, kernel="gaussian")
        x_run = np.array(run_list).reshape(-1, 1)
        x_drun = np.linspace(0, max_n, 1000)
        kde_run.fit(x_run)
        logprob = kde_run.score_samples(x_drun[:, None])
        # plt.fill_between(x_drun, np.exp(logprob), alpha=0.5, color=color_list[0])
        plt.plot(x_drun, np.exp(logprob), alpha=0.8, color=color_list[0])

        # instantiate and fit the KDE model
        kde_dataset = KernelDensity(bandwidth=1.0, kernel="gaussian")
        x_dataset = np.array(dataset_list).reshape(-1, 1)
        x_ddat = np.linspace(0, max_n, 1000)
        kde_dataset.fit(x_dataset)
        logprob = kde_dataset.score_samples(x_ddat[:, None])
        # plt.fill_between(x_ddat, np.exp(logprob), alpha=0.5, color=color_list[1])
        plt.plot(x_ddat, np.exp(logprob), alpha=0.8, color=color_list[1])

    sns.distplot(
        plot_list[0],
        bins=bins,
        hist=True,
        kde=False,
        norm_hist=True,
        color=color_list[0],
        kde_kws={"shade": True, "linewidth": 2, "alpha": 0.65},
        label=label_list[0],
    )
    sns.distplot(
        plot_list[1],
        bins=bins,
        hist=True,
        kde=False,
        norm_hist=True,
        color=color_list[1],
        kde_kws={"shade": True, "linewidth": 2, "alpha": 0.4},
        label=label_list[1],
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=21)
    ax.tick_params(axis="both", which="major", labelsize=21)
    ax.grid(True)
    plt.ylim([0, 1])
    plt.legend(framealpha=0.2, prop={"size": 25})
    plt.title(title, fontsize=30)
    plt.ylabel("Occurrences", fontsize=20)
    plt.xlabel(xlabel, fontsize=20)

    if save:
        # Save figure
        os.makedirs(_config["plots_save_dir"] + _config["specific_exp"], exist_ok=True)
        fig.savefig(
            _config["plots_save_dir"] + _config["specific_exp"] + "/" + metric, dpi=300
        )

        return fig

    else:
        return fig


def plot_losses(_config, save=False, baseline=False):
    """Quick plot for convergence analysis"""

    if baseline:
        events_dir = os.path.join(
            parent_dir, "externals/benchmarks/MolGAN/results/logs"
        )
        for root, dirs, files in os.walk(events_dir):
            for file in files:
                if file.endswith(".0"):
                    ea = event_accumulator.EventAccumulator(root + "/" + file)
                    ea.Reload()

                    Gl_df = pd.DataFrame(ea.Scalars("G/loss"))
                    Dl_df = pd.DataFrame(ea.Scalars("D/loss"))

        fig, ax = plt.subplots(figsize=(22, 12))
        ax.plot(Gl_df.step, Gl_df.value, alpha=0.6, label="Generator loss")
        ax.plot(Dl_df.step, Dl_df.value, alpha=0.6, label="Discriminator + LP loss")
        set_ax_off(ax, True)
        ax.grid(True)
        plt.legend(loc=1)

    else:
        events_dir = (
            _config["exps_dir"]
            + "/"
            + _config["specific_exp"]
            + "/"
            + _config["version_exp"]
            + "/"
        )

        for root, dirs, files in os.walk(events_dir):
            for file in files:
                if file.endswith(".0"):
                    ea = event_accumulator.EventAccumulator(root + "/" + file)
                    ea.Reload()

                    W_df = pd.DataFrame(ea.Scalars("W1"))
                    Gl_df = pd.DataFrame(ea.Scalars("gen_loss"))
                    Dl_df = pd.DataFrame(ea.Scalars("disc_loss"))
                else:
                    pass

        fig, ax = plt.subplots(figsize=(22, 12))
        ax.plot(W_df.step, W_df.value, alpha=0.3, label="Wasserstein distance")
        ax.plot(Gl_df.step, Gl_df.value, alpha=0.6, label="Generator loss")
        ax.plot(Dl_df.step, Dl_df.value, alpha=0.6, label="Discriminator + LP loss")
        set_ax_off(ax, True)
        ax.tick_params(axis="both", which="major", labelsize=21)
        ax.tick_params(axis="both", which="major", labelsize=21)
        ax.grid(True)
        plt.legend(loc=1, prop={"size": 25})
        plt.ylim([-25, 25])

    plt.title("Loss functions", fontsize=30)
    plt.ylabel("Loss values", fontsize=20)
    plt.xlabel("Iterations", fontsize=20)

    if save:
        # Save figure
        os.makedirs(_config["plots_save_dir"] + _config["specific_exp"], exist_ok=True)
        fig.savefig(
            _config["plots_save_dir"] + _config["specific_exp"] + "/" + "_losses",
            dpi=300,
        )
        return fig
    else:
        return fig


def plot_molgrid_tensor(adj, real, nrows_=3, ncols_=3, randsample=False):
    """Visualization of generated graphs"""

    r_ratio = nrows_ / 3.0
    c_ratio = ncols_ / 3.0
    fig, ax = plt.subplots(
        nrows=nrows_, ncols=ncols_, figsize=(int(22 * r_ratio), int(12 * c_ratio))
    )
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.35
    )
    if randsample:
        rand_choice = np.random.choice(adj.shape[0], nrows_ * ncols_)
    else:
        rand_choice = np.arange(min(adj.shape[0], ncols_ * nrows_))
    counter = 0
    for row in ax:
        for col in row:
            if counter >= len(rand_choice):
                continue
            G = nx.from_numpy_matrix(adj[rand_choice[counter]].numpy())
            nx.draw_kamada_kawai(G, ax=col)
            counter += 1
            del G
    if real:
        fig.suptitle("Real graphs", fontsize=30)
    else:
        fig.suptitle("Sampled graphs from AttentionGAN", fontsize=30)

    # Save to summary
    buf = io.BytesIO()
    fig.savefig(buf, format="jpeg")
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image)
    plt.close(fig)
    del fig
    buf.close()
    del buf

    return image


def plot_molgrid(_config, save=False, tensor=False):
    """Visualization of generated graphs"""

    if _config["handle"] != "dataset":
        file_dir = _config["graphs_dir"] + _config["specific_exp"] + ".pkl"
    else:
        file_dir = _config["dataset_graphs_dir"] + _config["dataset"] + ".pkl"

    with open(file_dir, "rb") as f:
        pkl_list = pickle.load(f)

    nrows_, ncols_ = 4, 4
    fig, ax = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(22, 12))
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.35
    )
    rand_choice = np.random.choice(len(pkl_list), nrows_ * ncols_)
    counter = 0
    iso_temp_list = []
    for row in ax:
        for col in row:
            print(rand_choice[counter])
            nx.draw_kamada_kawai(pkl_list[rand_choice[counter]], ax=col)
            iso_temp_list.append(pkl_list[rand_choice[counter]])
            counter += 1

    if _config["specific_exp"] == "MolGAN":
        fig.suptitle("Sampled graphs from MolGAN", fontsize=30)
    elif _config["handle"] == "dataset":
        fig.suptitle(_config["dataset"] + " Dataset", fontsize=30)
    else:
        fig.suptitle("Sampled graphs from GraphGAN", fontsize=30)

    # Quick isomorphism check, delete after
    iso_graphs = []
    for i, g1 in enumerate(iso_temp_list):
        for j, g2 in enumerate(iso_temp_list):
            if i != j:
                if nx.is_isomorphic(g1, g2):
                    if sorted((i, j)) not in iso_graphs:
                        iso_graphs.append(sorted((i, j)))
                        print(i, j)

    if save:
        # Save figure
        if _config["handle"] == "dataset":
            os.makedirs(_config["plots_save_dir"] + _config["dataset"], exist_ok=True)
            fig.savefig(
                _config["plots_save_dir"] + _config["dataset"] + "/" + "_graphs",
                dpi=300,
            )
        else:
            os.makedirs(
                _config["plots_save_dir"] + _config["specific_exp"], exist_ok=True
            )
            fig.savefig(
                _config["plots_save_dir"] + _config["specific_exp"] + "/" + "_graphs",
                dpi=300,
            )

        return fig
    if tensor:
        # Save to summary
        buf = io.BytesIO()
        plt.savefig(buf, format="jpeg")
        buf.seek(0)
        image = PIL.Image.open(buf)
        image = ToTensor()(image)
        fig.savefig(buf, dpi=300)

        return image

    else:
        return fig


def plot_isog(_config):
    """Unfinished/Temporary function to isomorphism test.
    How many isomorphic graphs are there in the generated sample"""
    # TODO Make this function more general (for now I will hardcode number of models for preliminary image)

    # AttentionGAN generated graphs
    models_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900]  # , 1000]
    # 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    # 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]

    Att_df = pd.DataFrame(columns=["model_step", "iso_g"], dtype=float)
    MolGAN_df = pd.DataFrame(columns=["model_step", "iso_g"], dtype=float)
    OriginalD_df = pd.DataFrame(columns=["model_step", "iso_g"], dtype=float)
    number_g = 100
    # run 5 times for standard devitation
    for iter_ in range(5):
        print(iter_)
        # MolGAN dataset
        # models_steps = [100, 200, 300]
        base_file_dir = _config["graphs_dir"] + "MolGAN"
        MolGAN_df = isog_plot(MolGAN_df, models_steps, number_g, iter_, base_file_dir)

        # GraphGAN
        # models_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        base_file_dir = _config["graphs_dir"] + _config["specific_exp"]
        Att_df = isog_plot(Att_df, models_steps, number_g, iter_, base_file_dir)

        # Original dataset
        file_dir = _config["dataset_graphs_dir"] + _config["dataset"] + ".pkl"
        total_iso_graphs = []
        with open(file_dir, "rb") as f:
            pkl_list = pickle.load(f)

        Ori_isog = isog_generated(pkl_list, number_g, total_iso_graphs)

        for i, iso_val in enumerate(np.ones(len(models_steps)) * Ori_isog[0]):
            row = i + (number_g * iter_)
            OriginalD_df.loc[row] = [models_steps[i], iso_val]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.lineplot(x="model_step", y="iso_g", data=Att_df, label="AttentionGAN")
    ax = sns.lineplot(x="model_step", y="iso_g", data=MolGAN_df, label="MolGAN")
    ax = sns.lineplot(
        x="model_step", y="iso_g", data=OriginalD_df, label="Original dataset"
    )
    set_ax_off(ax, True)
    ax.grid(True)
    plt.legend(loc=1)

    plt.title("Isomorphic graphs")
    plt.ylabel("Occurrences")
    plt.xlabel("Model step")

    # Save figure
    os.makedirs(_config["plots_save_dir"] + _config["specific_exp"], exist_ok=True)
    fig.savefig(
        _config["plots_save_dir"] + _config["specific_exp"] + "/" + "ISO", bdpi=300
    )


def plot_novelty2(_config):
    """Unfinished/Temporary function to isomorphism test
    How many isomorphic graphs are there between model graphs and dataset"""
    # TODO Make this function more general (for now I will hardcode number of models for preliminary image). Also the
    #  reference datasets are hardcoded

    # AttentionGAN generated graphs
    models_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900]  # , 1000]
    # 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000,
    # 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000]

    # Original dataset
    file_dir = _config["dataset_graphs_dir"] + _config["dataset"] + ".pkl"
    with open(file_dir, "rb") as f:
        Ori_pkl_list = pickle.load(f)

    Att_df = pd.DataFrame(columns=["model_step", "iso_g"], dtype=float)
    MolGAN_df = pd.DataFrame(columns=["model_step", "iso_g"], dtype=float)
    number_g = 50
    # run 5 times for standard deviation
    for iter_ in range(5):
        print(iter_)
        # MolGAN dataset
        base_file_dir = _config["graphs_dir"] + "MolGAN"
        MolGAN_df = novelty2_plot(
            MolGAN_df,
            models_steps,
            number_g,
            iter_,
            base_file_dir,
            dataset=Ori_pkl_list,
        )

        # GraphGAN
        # models_steps = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
        base_file_dir = _config["graphs_dir"] + _config["specific_exp"]
        Att_df = novelty2_plot(
            Att_df, models_steps, number_g, iter_, base_file_dir, dataset=Ori_pkl_list
        )

    fig, ax = plt.subplots(figsize=(10, 6))
    ax = sns.lineplot(x="model_step", y="iso_g", data=Att_df, label="AttentionGAN")
    ax = sns.lineplot(x="model_step", y="iso_g", data=MolGAN_df, label="MolGAN")
    set_ax_off(ax, True)
    ax.grid(True)
    plt.legend(loc=1)

    plt.title("Isomorphic graphs")
    plt.ylabel("Occurrences")
    plt.xlabel("Model step")

    # Save figure
    os.makedirs(_config["plots_save_dir"] + _config["specific_exp"], exist_ok=True)
    fig.savefig(
        _config["plots_save_dir"] + _config["specific_exp"] + "/" + "Novelty", bdpi=300
    )


def one_img(imgs: list(), _config):
    h_imgs = []

    ref_img = fig2img(imgs[0])
    for idx in range(0, len(imgs), 2):
        img1 = fig2img(imgs[idx])
        img1 = img1.resize(
            (int(img1.width * ref_img.height / img1.height), ref_img.height),
            resample=Image.BICUBIC,
        )
        img1 = img1.resize(
            (ref_img.width, int(img1.height * ref_img.width / img1.width)),
            resample=Image.BICUBIC,
        )
        img2 = fig2img(imgs[idx + 1])
        img2 = img2.resize(
            (int(img2.width * ref_img.height / img2.height), ref_img.height),
            resample=Image.BICUBIC,
        )
        img2 = img2.resize(
            (ref_img.width, int(img2.height * ref_img.width / img2.width)),
            resample=Image.BICUBIC,
        )
        h_ = Image.new("RGB", (img1.width + 1 + img2.width, img1.height))
        h_.paste(img1, (0, 0))
        h_.paste(img2, (img1.width + 1, 0))

        h_imgs.append(h_)

    final_img = Image.new("RGB", (h_imgs[0].width, h_imgs[0].height * 2))
    for idx in range(0, len(h_imgs), 2):
        final_img.paste(h_imgs[idx], (0, 0))
        final_img.paste(h_imgs[idx + 1], (0, h_imgs[idx].height))

    draw = ImageDraw.Draw(final_img)
    fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils/fonts")
    # print(os.path.join(fonts_path, 'Aaargh.ttf'))
    # exit()
    font = ImageFont.truetype(os.path.join(fonts_path, "Archivo-SemiBold.ttf"), 32)
    draw.text(
        (200, 30), "Parameters: " + _config["model_struct"][2:-9], (0, 0, 0), font=font
    )

    # Make transparent image (whites to transparent)
    # img = final_img.convert("RGBA")
    # datas = img.getdata()
    # newData = []
    # for item in datas:
    #     if item[0] == 255 and item[1] == 255 and item[2] == 255:
    #         newData.append((255, 255, 255, 0))
    #     else:
    #         newData.append(item)
    #
    # img.putdata(newData)
    # final_img = img

    # Save figure
    os.makedirs(_config["plots_save_dir"] + _config["specific_exp"], exist_ok=True)
    final_img.save(
        _config["plots_save_dir"]
        + _config["specific_exp"]
        + "/"
        + _config["model_epoch"]
        + ".png"
    )


def trajectory_plot():
    config_copy = _config.copy()
    random_index = np.random.choice(range(_config["z_dim"]), 3, replace=False)
    list_ = [0, 2, 4, 6, 7, 8, 9]
    print(
        "Random selected points to construct trajectory are {} \n".format(random_index)
    )
    x, y, z, legend_ = [], [], [], []
    batch_idx = 17
    node_in_graph = 2
    for root, dirs, files in os.walk(_config["exps_dir"]):
        dir_list = root.split("/")
        if dir_list[-1] == "Z0":
            specific_exp = dir_list[-3]
            # Plot save directory
            os.makedirs(
                _config["plots_save_dir"] + _config["specific_exp"], exist_ok=True
            )
            os.makedirs(
                _config["plots_save_dir"] + _config["specific_exp"] + "/graphs/",
                exist_ok=True,
            )

            for idx, embed in enumerate(np.array(files)[[list_]]):
                pkl_list = pt.load(os.path.join(root, embed))

                x.append(pkl_list[batch_idx][node_in_graph][random_index[0]].numpy())
                y.append(pkl_list[batch_idx][node_in_graph][random_index[1]].numpy())
                z.append(pkl_list[batch_idx][node_in_graph][random_index[2]].numpy())

                if idx != 0:
                    epoch_ = str(int(embed.split("_")[1].split(".")[0].lstrip("0")) + 1)
                else:
                    epoch_ = str(idx)
                legend_.append(epoch_)
                with open(
                    os.path.join(_config["graphs_dir"], specific_exp)
                    + "_"
                    + epoch_
                    + ".pkl",
                    "rb",
                ) as f:
                    graphs = pickle.load(f)

                fig = plt.figure()
                pos = nx.kamada_kawai_layout(graphs[batch_idx])
                nx.draw_networkx_edges(
                    graphs[batch_idx],
                    pos,
                    width=2,
                    edge_color="#000000",
                    alpha=0.45,
                    linewidths=4.5,
                )
                nodes = nx.draw_networkx_nodes(
                    graphs[batch_idx],
                    pos,
                    node_color="#29465b",
                    alpha=0.97,
                    linewidths=4,
                    node_size=400,
                )
                nodes.set_edgecolor("w")
                plt.axis("off")
                # nx.draw_kamada_kawai(graphs[batch_idx], node_color="skyblue", width=2, edge_color="grey")
                fig.savefig(
                    _config["plots_save_dir"]
                    + _config["specific_exp"]
                    + "/graphs/"
                    + "g_"
                    + epoch_,
                    bdpi=300,
                )
                plt.close(fig)

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
    plt.show()
    # Save figure
    fig.savefig(
        _config["plots_save_dir"] + _config["specific_exp"] + "/" + "trajectory",
        bdpi=300,
    )


# ------------------------------------ To plot in cluster
def cluster_plot_histogram(
    model_dist: pt.Tensor,
    dataset_dist: pt.Tensor,
    metric=None,
    kde=False,
    dataset_name=None,
    is_counts=False,
    val_start=1,
    model_name="GG-GAN",
):
    """Histogram and density plots for distributions (degree and cycle)"""

    suffix_, title, xlabel = metric_to_use(metric)
    if dataset_dist.sum() != 0:
        plot_list = [model_dist, dataset_dist]
    else:
        plot_list = [model_dist]

    label_list = [model_name, dataset_name]
    color_list = ["royalblue", "darkgreen"]

    max_n = 0
    for samples in plot_list:
        # largest element of distribution => find cumsum max of each element, then count how many are below
        s_max_n = samples.shape[-1]
        if s_max_n > max_n:
            max_n = s_max_n

    fig, ax = plt.subplots(figsize=(22, 12))
    bins = np.arange(max_n + 1) + 0.5

    if kde:
        raise NotImplementedError("Haven't adapted kde to the tensorized version yet")
        # instantiate and fit the KDE model
        kde_run = KernelDensity(bandwidth=1.0, kernel="gaussian")
        x_drun = np.linspace(0, max_n, 1000)
        kde_run.fit(model_dist.values, sample_weight=model_dist.counts)
        logprob = kde_run.score_samples(x_drun[:, None])
        # plt.fill_between(x_drun, np.exp(logprob), alpha=0.5, color=color_list[0])
        plt.plot(x_drun, np.exp(logprob), alpha=0.8, color=color_list[0])

        # instantiate and fit the KDE model
        kde_dataset = KernelDensity(bandwidth=1.0, kernel="gaussian")
        x_ddat = np.linspace(0, max_n, 1000)
        kde_dataset.fit(dataset_dist.values, sample_weight=dataset_dist.counts)
        logprob = kde_dataset.score_samples(x_ddat[:, None])
        # plt.fill_between(x_ddat, np.exp(logprob), alpha=0.5, color=color_list[1])
        plt.plot(x_ddat, np.exp(logprob), alpha=0.8, color=color_list[1])

    if is_counts:
        samples = model_dist
        vals = pt.arange(samples.shape[-1])
        counts = samples
    else:
        vals, counts = pt.unique(model_dist.flatten(), return_counts=True)
    sns.distplot(
        vals,
        bins=bins,
        hist=True,
        kde=False,
        norm_hist=True,
        color=color_list[0],
        hist_kws={"weights": counts.to(dtype=pt.float64).numpy()},
        kde_kws={
            "shade": True,
            "linewidth": 2,
            "alpha": 0.65,
            "weights": counts.to(dtype=pt.float64).numpy(),
        },
        label=label_list[0],
    )

    if is_counts:
        samples = dataset_dist
        dataset_vals = pt.arange(samples.shape[-1])
        dataset_counts = samples
    else:
        dataset_vals, dataset_counts = pt.unique(
            dataset_dist.flatten(), return_counts=True
        )
    sns.distplot(
        dataset_vals,
        bins=bins,
        hist=True,
        kde=False,
        norm_hist=True,
        color=color_list[1],
        hist_kws={"weights": dataset_counts.to(dtype=pt.float64).numpy()},
        kde_kws={
            "shade": True,
            "linewidth": 2,
            "alpha": 0.4,
            "weights": dataset_counts.to(dtype=pt.float64).numpy(),
        },
        label=label_list[1],
    )
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(axis="both", which="major", labelsize=21)
    ax.tick_params(axis="both", which="major", labelsize=21)
    ax.grid(True)
    plt.ylim([0, 1])
    plt.legend(framealpha=0.2, prop={"size": 25})
    plt.title(title, fontsize=30)
    plt.ylabel("Occurrences", fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    return fig


def cluster_plot_losses(event_accumulator_file):
    """Quick plot for convergence analysis"""
    if event_accumulator_file is not None:
        ea = event_accumulator_file.Reload()

        W_df = pd.DataFrame(ea.Scalars("W1"))
        Gl_df = pd.DataFrame(ea.Scalars("gen_loss"))
        Dl_df = pd.DataFrame(ea.Scalars("disc_loss"))

        fig, ax = plt.subplots(figsize=(22, 12))
        ax.plot(W_df.step, W_df.value, alpha=0.3, label="Wasserstein distance")
        ax.plot(Gl_df.step, Gl_df.value, alpha=0.6, label="Generator loss")
        ax.plot(Dl_df.step, Dl_df.value, alpha=0.6, label="Discriminator + LP loss")

        set_ax_off(ax, True)
        ax.tick_params(axis="both", which="major", labelsize=21)
        ax.tick_params(axis="both", which="major", labelsize=21)
        ax.grid(True)
        plt.legend(loc=1, prop={"size": 25})
        plt.ylim([-105, 105])

        plt.title("Loss functions", fontsize=30)
        plt.ylabel("Loss values", fontsize=20)
        plt.xlabel("Iterations", fontsize=20)
    else:
        fig, ax = plt.subplots(figsize=(22, 12))

    return fig


def cluster_plot_molgrid(
    gen_graphs: [], name="GraphGAN", lcc=False, save_dir=None, save=False
):
    """Visualization of generated graphs"""

    pkl_list = gen_graphs

    nrows_, ncols_ = 3, 4
    fig, ax = plt.subplots(nrows=nrows_, ncols=ncols_, figsize=(26, 12))
    plt.subplots_adjust(
        left=None, bottom=None, right=None, top=None, wspace=0.25, hspace=0.35
    )
    rand_choice = np.random.choice(len(pkl_list), nrows_ * ncols_)
    counter = 0
    iso_temp_list = []
    for row in ax:
        for col in row:
            gorig = pkl_list[rand_choice[counter]]
            if not lcc:
                g = gorig
            else:
                mcc = max(nx.connected_components(gorig), key=len)
                g = gorig.subgraph(mcc).copy()
            # nx.draw_kamada_kawai(pkl_list[rand_choice[counter]], ax=col)
            # pos = nx.spring_layout(pkl_list[rand_choice[counter]])
            if "Tree" in name:
                # Use prog= "dot" for top down trees | "twopi" for circular trees
                pos = graphviz_layout(g, prog="twopi")
            else:
                pos = nx.spring_layout(g)
            nx.draw_networkx_edges(
                g, pos, edge_color="#000000", alpha=0.45, width=2.5, ax=col
            )
            nodes = nx.draw_networkx_nodes(
                g,
                pos,
                node_color="#29465b",
                alpha=0.97,
                linewidths=2,
                node_size=350,
                ax=col,
            )
            nodes.set_edgecolor("w")
            col.set_axis_off()
            iso_temp_list.append(gorig)
            counter += 1

    # fig.suptitle(f"Sampled graphs from {name}", fontsize=30)
    if save:
        plt.savefig(os.path.join(save_dir, name + ".pdf"))

    return fig


def cluster_plot_isog(
    models_datasets_g: [[]],
    legends: [],
    numb_g_eval=100,
    reps=5,
    save=False,
    save_dir="",
    boxplot=True,
):
    """
    How many isomorphic graphs are there in the generated sample
    :param epoch_steps: steps the model ran
    :param models_g: a list containing the graphs for each epoch of each model ([[[]]])
    :param dataset_g: dataset used
    :param legends: plot legends of the names
    :param numb_g_eval: number of graphs to evaluate isomorphism
    :param reps: repetitions to calculate variance
    :return: a figure to save
    """

    assert len(models_datasets_g) == len(legends)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, graphs in enumerate(models_datasets_g):
        diver_list = []
        df = pd.DataFrame(columns=["model_step", "iso_g"], dtype=float)
        print("Calculating isomorphic graphs in {}".format(legends[i]))
        for _ in tqdm(range(reps)):
            df, diver_list = get_isog(
                df, pkl_graphs=graphs, number_g=numb_g_eval, diver_list=diver_list
            )

        if not boxplot:
            ax = sns.lineplot(x="model_step", y="iso_g", data=df, label=legends[i])
        if boxplot:
            percentages = []
            for numb_graphs in diver_list:
                unique_per = (numb_graphs / numb_g_eval) * 100
                percentages.append(unique_per)
            print(diver_list, (np.mean(diver_list) / numb_g_eval) * 100)
            ax.boxplot(percentages, labels=[legends[i]], positions=[i])

    set_ax_off(ax, True)
    ax.grid(True)
    plt.legend(loc=1)

    plt.title("Diversity (model-model)")
    plt.ylabel("Number of isomorphism classes")
    plt.xlabel("Model")

    if save:
        fig.savefig(os.path.join(save_dir, "diversity"), bdpi=300)

    return fig


def cluster_plot_novelty(
    models_g: [[]],
    dataset_g: [[]],
    legends: [],
    numb_g_eval=100,
    reps=5,
    save=False,
    save_dir="",
    boxplot=True,
    chordal=False,
):
    """

    :param epoch_steps:
    :param models_datasets_g:
    :param legends:
    :param numb_g_eval:
    :param reps:
    :param save:
    :param save_dir:
    :return:
    """
    assert len(models_g) == len(legends)

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, graphs in enumerate(models_g):
        diver_list = []
        df = pd.DataFrame(columns=["model_step", "iso_g"], dtype=float)
        print("Calculating isomorphic graphs in {}".format(legends[i]))
        for _ in range(reps):
            df, diver_list = get_novelty(
                df,
                pkl_graphs=graphs,
                number_g=numb_g_eval,
                dataset=dataset_g,
                diver_list=diver_list,
                chordal=chordal,
            )

        if not boxplot:
            ax = sns.lineplot(x="model_step", y="iso_g", data=df, label=legends[i])
        if boxplot:
            percentages = []
            for numb_graphs in diver_list:
                unique_per = (numb_graphs / numb_g_eval) * 100
                percentages.append(unique_per)
            print(diver_list, (np.mean(diver_list) / numb_g_eval) * 100)
            ax.boxplot(percentages, labels=[legends[i]], positions=[i])

    set_ax_off(ax, True)
    ax.grid(True)
    plt.legend(loc=1)

    plt.title("Novelty (dataset-model)")
    plt.ylabel("Unique graphs not in dataset")
    plt.xlabel("Model")

    if save:
        fig.savefig(os.path.join(save_dir, "novelty"), dpi=300)

    return fig


def cluster_one_img(imgs: [], exp_dir, model_name, epoch):
    if model_name is None:
        mn = "model"
    else:
        mn = model_name[2:-9]
    h_imgs = []

    ref_img = fig2img(imgs[0])
    for idx in range(0, len(imgs), 2):
        img1 = fig2img(imgs[idx])
        img1 = img1.resize(
            (int(img1.width * ref_img.height / img1.height), ref_img.height),
            resample=Image.BICUBIC,
        )
        img1 = img1.resize(
            (ref_img.width, int(img1.height * ref_img.width / img1.width)),
            resample=Image.BICUBIC,
        )
        img2 = fig2img(imgs[idx + 1])
        img2 = img2.resize(
            (int(img2.width * ref_img.height / img2.height), ref_img.height),
            resample=Image.BICUBIC,
        )
        img2 = img2.resize(
            (ref_img.width, int(img2.height * ref_img.width / img2.width)),
            resample=Image.BICUBIC,
        )
        h_ = Image.new("RGB", (img1.width + 1 + img2.width, img1.height))
        h_.paste(img1, (0, 0))
        h_.paste(img2, (img1.width + 1, 0))

        h_imgs.append(h_)

    final_img = Image.new("RGB", (h_imgs[0].width, h_imgs[0].height * 2))
    for idx in range(0, len(h_imgs), 2):
        final_img.paste(h_imgs[idx], (0, 0))
        final_img.paste(h_imgs[idx + 1], (0, h_imgs[idx].height))

    draw = ImageDraw.Draw(final_img)
    fonts_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils/fonts")
    font = ImageFont.truetype(os.path.join(fonts_path, "Archivo-SemiBold.ttf"), 32)
    draw.text((200, 30), "Parameters: " + mn, (0, 0, 0), font=font)

    # Make transparent image (whites to transparent)
    # img = final_img.convert("RGBA")
    # datas = img.getdata()
    # newData = []
    # for item in datas:
    #     if item[0] == 255 and item[1] == 255 and item[2] == 255:
    #         newData.append((255, 255, 255, 0))
    #     else:
    #         newData.append(item)
    #
    # img.putdata(newData)
    # final_img = img

    # Save figure
    final_img.save(os.path.join(exp_dir, str(epoch) + ".pdf"))
    return final_img
