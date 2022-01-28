import os

import numpy
import numpy as np
from matplotlib import pyplot as plt
import torch as pt
from matplotlib.lines import Line2D
import io
import matplotlib.pyplot as plt
import PIL.Image
from torchvision.transforms import ToTensor

from ggg_utils.utils.logging import easy_hist


def plot_grad_flow(
    named_parameters,
    epoch,
    exp_name,
    model_n,
    g_=False,
    to_tensor=False,
    to_file=True,
    ylim=None,
    disc=False
):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow.

    https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10"""

    ave_grads = []
    max_grads = []
    min_grads = []
    layers = []
    for n, p in named_parameters:
        if (p.requires_grad) and ("bias" not in n):
            if p.grad is None or np.prod(p.shape) == 0:
                continue
            layers.append(n)
            pabs = p.grad.abs()
            ave_grads.append(pabs.mean().cpu())
            max_grads.append(pabs.max().cpu())
            min_grads.append(pabs.min().cpu())
    x_lgd = []
    for l_ in layers:
        try:
            x_lgd.append(l_.split(".")[0] + l_.split(".")[1][0])
        except:
            x_lgd.append(l_)

    if to_file:
        plot_to_file(
            ave_grads,
            epoch,
            exp_name,
            g_,
            max_grads,
            model_n,
            x_lgd,
            ylim=min(ylim, min(max_grads))
            if ylim is not None and min(min_grads) * 1e3 < min(ave_grads)
            else None,
        )
        plt.close()
    n="disc" if disc else "gen"
    if len(ave_grads)>0:
        easy_hist(f"ave_grad_flow{n}",pt.stack(ave_grads))
    if to_tensor:
        return plot_to_tensor(ave_grads, epoch, exp_name, g_, max_grads, model_n, x_lgd)
    return None


def plot_to_file(ave_grads, epoch, exp_name, g_, max_grads, model_n, x_lgd, ylim=None):
    if model_n is None:
        model_n = "model"
    plt.figure(figsize=(8, 5))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), x_lgd, rotation="vertical", fontsize=4)
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
            Line2D([0], [0], color="r", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient", epoch],
    )
    if ylim is not None:
        plt.ylim(-10, ylim)

    if g_:
        grad_path = os.path.join(os.getcwd(), exp_name, model_n, "Ggrads/")
        os.makedirs(grad_path, exist_ok=True)
        plt.savefig(grad_path + "{}_grad".format(epoch), dpi=200)
    else:
        grad_path = os.path.join(os.getcwd(), exp_name, model_n, "Dgrads/")
        os.makedirs(grad_path, exist_ok=True)
        plt.savefig(grad_path + "{}_grad".format(epoch), dpi=200)


def plot_to_tensor(ave_grads, epoch, exp_name, g_, max_grads, model_n, x_lgd):
    fig, ax = plt.subplots(figsize=(8, 5))
    fig: plt.Figure
    ax: plt.Axes
    ax.bar(np.arange(len(max_grads)), max_grads, alpha=0.5, lw=1, color="c")
    ax.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="b")
    ax.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    ax.set_xticks(range(0, len(ave_grads), 1))
    ax.set_xticklabels(x_lgd)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation="vertical", fontsize=4)
    # plt.xlim(left=0, right=len(ave_grads))
    # plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    ax.set_xlabel("Layers")
    ax.set_ylabel("average gradient")
    name = "Gen" if g_ else "Dis"
    fig.suptitle(f"{model_n}")
    ax.set_title(f"{name} Gradient flow e={epoch}")
    ax.grid(True)
    ax.legend(
        [
            Line2D([0], [0], color="c", lw=4),
            Line2D([0], [0], color="b", lw=4),
            Line2D([0], [0], color="k", lw=4),
            Line2D([0], [0], color="r", lw=4),
        ],
        ["max-gradient", "mean-gradient", "zero-gradient", epoch],
    )
    # plt.savefig(grad_path + "{}_grad".format(epoch), dpi=200)
    return fig, ax
