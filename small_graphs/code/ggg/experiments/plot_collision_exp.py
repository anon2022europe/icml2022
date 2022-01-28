import os
from collections import defaultdict
import numpy as np
import matplotlib
import torch as pt
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
from ggg.utils.load_tensor_scalars import get_metrics

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 28})


def get_runs(run_dir):
    """

    :param run_dir: sacred logdir from the collision experiment
    :return: model name,config of the run, run path, event file path
    """
    run_dir = os.path.abspath(run_dir)
    outs = []
    dirs = os.listdir(run_dir)
    logging.debug(dirs)
    for r in dirs:
        rp = os.path.join(run_dir, r)
        if "_sources" in rp or not os.path.isdir(rp):
            continue
        with open(os.path.join(rp, "config.json")) as f:
            c = json.load(f)
        model = c["hpars"]["model"]
        for dp, dn, fn in os.walk(rp):
            ep = None
            for f in fn:
                if "events" in f:
                    ep = os.path.join(dp, f)
                    outs.append((model, c, rp, ep))
                    break
            if ep is not None:
                break

    return outs


def tsplot(
    ax,
    x,
    y,
    n=20,
    percentile_min=1,
    percentile_max=99,
    color="r",
    plot_mean=False,
    plot_median=True,
    line_color="k",
    label=None,
    **kwargs,
):
    # from https://github.com/arviz-devs/arviz/issues/2
    # calculate the lower and upper percentile groups, skipping 50 percentile
    perc1 = np.percentile(
        y, np.linspace(percentile_min, 50, num=n, endpoint=False), axis=0
    )
    perc2 = np.percentile(y, np.linspace(50, percentile_max, num=n + 1)[1:], axis=0)

    if "alpha" in kwargs:
        alpha = kwargs.pop("alpha")
    else:
        alpha = 1 / n
    if plot_mean:
        ax.plot(
            x,
            np.mean(y, axis=0),
            color=line_color,
            label=label,
            linestyle=kwargs.get("linestyle", None),
        )
    elif plot_median:
        ax.plot(
            x,
            np.median(y, axis=0),
            color=line_color,
            label=label,
            linestyle=kwargs.get("linestyle", None),
        )
    # fill lower and upper percentile groups
    for p1, p2 in zip(perc1, perc2):
        ax.fill_between(x, p1, p2, alpha=alpha, color=color, edgecolor=None)

    return ax


EXP_PATH = (
    "/home/nada/Documents/phd/anon/collaborations/graph-gan-main/code/collision_demo"
)


def load_mses(EXP_PATH):
    runs = get_runs(EXP_PATH)
    logging.info(f"Found {len(runs)} runs in {EXP_PATH}")
    model_mses = defaultdict(list)
    model_steps = defaultdict(list)
    max_len = 0
    for m, _, _, ep in tqdm(runs, desc="Loading mses"):
        steps, vals = get_metrics(ep, "mse")
        if len(steps) > max_len:
            max_len = len(steps)
        model_mses[m].append(vals)
        model_steps[m].append(steps)
    for k in model_mses.keys():
        model_mses[k] = np.stack(
            [
                np.concatenate([x, np.ones(max_len - len(x)) * x[-1]])
                for x in model_mses[k]
            ],
            0,
        )
    steps = np.arange(max_len)
    steps: np.ndarray
    return model_mses, max_len, steps


def m_to_ind(mn):
    # col=0 if "rand" in mn else 1
    if "mlp" in mn:
        col = 0
    elif "ds" in mn:
        col = 1
    else:
        col = 2
    return col


def make_plot(steps, model_mses, figsize=None):
    figs = dict()
    axs = dict()
    names = dict()
    for i in range(3):
        figs[i] = plt.figure(figsize=figsize)  # W,H
        f = figs[i]
        f: plt.Figure
        axs[i] = f.add_subplot()
    for i, ((k, vals), c) in enumerate(
        zip(model_mses.items(), plt.rcParams["axes.prop_cycle"].by_key()["color"])
    ):
        ax = axs[m_to_ind(k)]
        fig = figs[m_to_ind(k)]
        if "rand" in k:
            c = "green"
            label = "random set"
        elif "traj" in k:
            c = "orange"
            label = "random context"
        ax: plt.Axes
        vals: np.ndarray
        tsplot(
            ax,
            steps,
            vals,
            n=4,
            percentile_min=0,
            percentile_max=100,
            label=label,
            color=c,
            line_color=c,
            plot_median=True,
            linestyle="-" if "traj" in k else "--",
        )
        # ax.plot(steps,m,label=label)
        # ax.fill_between(steps,m-std,m+std,alpha=0.3)
        if "mlp" in k:
            name = "RGG"
        elif "ds" in k:
            name = "DS"
            ax.legend()
        else:
            name = "Att"
        # ax.set_title(name)
        names[m_to_ind(k)] = name
    for k, fig in figs.items():
        # plt.show(fig)
        ax = axs[k]
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE")
        ax.set_ylim(-0.1, 0.5)
        fig: plt.Figure
        fig.tight_layout(pad=0)
        fig.savefig(f"CollisionDecentiles-{names[k]}.pdf", bbox_inches="tight")


if __name__ == "__main__":
    import sys

    FRESH_PLOT = False
    CACHE_FILE = "/tmp/mses.pkl"
    if not os.path.exists(CACHE_FILE) or FRESH_PLOT:
        model_mses, max_len, steps = load_mses(EXP_PATH)
        with open(CACHE_FILE, "wb") as f:
            pt.save([model_mses, steps], f)
    else:
        with open(CACHE_FILE, "rb") as f:
            model_mses, steps = pt.load(f)
    make_plot(
        steps,
        model_mses,
        None if len(sys.argv) < 3 else ([float(x) for x in sys.argv[1:3]]),
    )
