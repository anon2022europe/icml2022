import pandas as pd
import os
from collections import defaultdict
import numpy as np
import matplotlib
import torch as pt
from tqdm import tqdm
import logging

from ggg.models.sbm_readout import SBMCheckPars

logging.basicConfig(level=logging.INFO)
from ggg.utils.load_tensor_scalars import get_metrics, get_runs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# plt.rcParams.update({'font.size': 28})


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


def load_metrics(EXP_PATH):
    """

    :param EXP_PATH: path in which to find the sacred 0,1,2...files
    :return:
    """
    runs = get_runs(EXP_PATH)
    logging.info(f"Found {len(runs)} runs in {EXP_PATH}")

    max_len = 0
    outs = []
    for c, _, ep in tqdm(runs, desc="Loading mses"):
        steps, vals = get_metrics(ep, "mse")
        if len(steps) > max_len:
            max_len = len(steps)
        outs.append((c, steps, vals))
    return outs


def m_to_ind(mn):
    # col=0 if "rand" in mn else 1
    if "mlp" in mn:
        col = 0
    elif "ds" in mn:
        col = 1
    else:
        col = 2
    return col


def make_name(c):
    c = c["hpars"]
    name = f'{c["readout"]}-λ{c["temperature"]}-disc{c["discretization"]=="relaxed_bernoulli"}-F{c["F"]}'
    subc = {k: c[k] for k in ["readout", "temperature", "discretization", "F"]}
    return name, subc


def make_plot(outs, figsize=None):
    K = "mse"
    keys = set()
    model_mses = []
    for c, steps, vals in tqdm(outs, desc="Out"):
        if len(steps) == 0:
            continue
        name, subc = make_name(c)
        if isinstance(vals, dict):
            steps = steps[K]
            vals = vals[K]
        assert len(steps) == len(vals)
        keys.add(name)
        name = [name] * len(steps)
        model_mses.append(
            pd.DataFrame.from_dict(dict(steps=steps, mse=vals, name=name, **subc))
        )
    df = pd.concat(model_mses)
    print(f"Have {len(keys)} exps")
    make_joint(df, figsize, keys)
    make_joint(df, figsize, keys, zero_only=True)
    make_overview(df, figsize, keys)


def make_joint(df, figsize, keys, zero_only=False):
    N_EXP = len(keys)
    A = int(np.ceil(np.sqrt(N_EXP)))
    fig, ax = plt.subplots(figsize=figsize)  # W,H
    f = fig
    for i, (g, dfg) in tqdm(enumerate(df.groupby("name")), desc="Plot"):
        label = g
        ax: plt.Axes
        vals: np.ndarray
        dfgs = dfg.groupby("steps")
        std = dfgs["mse"].std()
        m = dfgs["mse"].mean()
        if zero_only and (m[-10:] > 0.0).any():
            continue
        steps = m.index
        # tsplot(ax, steps, vals, n=4, percentile_min=0, percentile_max=100, label=label, color=c, line_color=c,plot_median=True,linestyle="-" if "traj" in k else "--")
        ax.set_title(label)
        ax.plot(steps, m, label=label)
        # ax.fill_between(steps, m - std, m + std, alpha=0.3)
        # ax.set_title(name)
        # plt.show(fig)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE")
    ax.legend()
    # ax.set_ylim(-0.1,0.5)
    fig.tight_layout(pad=0)
    fig.savefig(f"SBM-Readout-joint_zeroonly{zero_only}.pdf", bbox_inches="tight")


def make_overview(df, figsize, keys):
    axs = dict()
    N_EXP = len(keys)
    A = int(np.ceil(np.sqrt(N_EXP)))
    fig = plt.figure(figsize=figsize)  # W,H
    f = fig
    for i in range(N_EXP):
        axs[i] = f.add_subplot(
            A,
            A,
            i + 1,
            sharey=axs[0] if len(axs) > 0 else None,
            sharex=axs[0] if len(axs) > 0 else None,
        )
    for i, (g, dfg) in tqdm(enumerate(df.groupby("name")), desc="Plot"):
        ax = axs[i]
        label = g
        ax: plt.Axes
        vals: np.ndarray
        dfgs = dfg.groupby("steps")
        std = dfgs["mse"].std()
        m = dfgs["mse"].mean()
        steps = m.index
        # tsplot(ax, steps, vals, n=4, percentile_min=0, percentile_max=100, label=label, color=c, line_color=c,plot_median=True,linestyle="-" if "traj" in k else "--")
        ax.set_title(label)
        ax.plot(steps, m, label=label)
        ax.fill_between(steps, m - std, m + std, alpha=0.3)
        # ax.set_title(name)
        # plt.show(fig)
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE")
        ax.legend()
        # ax.set_ylim(-0.1,0.5)
    fig.tight_layout(pad=0)
    fig.savefig(f"SBM-Readout-overview.pdf", bbox_inches="tight")


if __name__ == "__main__":
    import sys

    FRESH_PLOT = os.environ.get("FRESH", False)
    CACHE_FILE = "/tmp/sbm-mses.pkl"
    EXP_PATH = "/home/anon/code/graph-gan-main/code/sbm_run/smb_exp"
    if not os.path.exists(CACHE_FILE) or FRESH_PLOT:
        outs = load_metrics(EXP_PATH)  # List[config,steps,vals]
        with open(CACHE_FILE, "wb") as f:
            pt.save(outs, f)
    else:
        with open(CACHE_FILE, "rb") as f:
            outs = pt.load(f)
    make_plot(outs, None if len(sys.argv) < 3 else ([float(x) for x in sys.argv[1:3]]))
