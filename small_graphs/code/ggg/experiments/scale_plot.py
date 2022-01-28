import os
import matplotlib.pyplot as plt
import attr
import json
import torch as pt
import numpy as np
import pandas as pd
import seaborn as sb
from ipdb import set_trace

sb.set(style="whitegrid")

bigg_extracted = np.array(
    [
        [float(y.strip()) for y in x.strip().split(", ")]
        for x in """100, 0.04028563509429296
199.90786249713972, 0.07810950087815954
502.62584157790434, 0.19972350755326213
1000, 0.44911805554853335
5026.258415779043, 3.0547701681075288
10000, 6.473792844856703
20086.51365410108, 12.929686141479667
50745.109136286475, 32.735686973776964
100000.0000000002, 69.3747956277823""".splitlines()
    ]
)
print(bigg_extracted.shape)
bigg_X = np.round(bigg_extracted[:, 0])
bigg_T = np.round(bigg_extracted[:, 1], decimals=5) * 60  # convert to seconds
print(bigg_X.shape, bigg_T.shape)


@attr.s
class Run:
    config = attr.ib()
    data = attr.ib()

    @classmethod
    def load(cls, dir):
        data = None
        config = None
        for fp in os.listdir(dir):
            fp = os.path.join(dir, fp)
            if "config" in fp:
                with open(fp) as f:
                    config = json.load(f)
            if os.path.splitext(fp)[-1] == ".pt":
                with open(fp, "rb") as f:
                    data = pt.load(f)
        return Run(config, data)

    def plot(self, ax, name):
        times = self.data["times"]
        x = self.data["num_nodes"]
        ax.boxplot(times.T)
        ax.set_xticklabels(x, rotation=90)
        ax.set_ylabel(f" Generation time (seconds)")
        ax.set_xlabel("# Nodes in graph")


BASE_PATH = "/home/nada/Documents/phd/anon/collaborations/graph-gan-main/scaling_data"

GRAPH_RNN_PATH = os.path.join(BASE_PATH, "scale_plot_graphrnn")
CONDGEN_PATH = os.path.join(BASE_PATH, "scale_condgen")
OUR_PATHS = [
    os.path.join(BASE_PATH, x)
    for x in "PEAWGAN_SCALABILTIY  PWGSCALE004  scale_plot".split()
]
graprnn_runs = [
    Run.load(os.path.join(GRAPH_RNN_PATH, x)) for x in os.listdir(GRAPH_RNN_PATH)
]
condgen_runs = [
    Run.load(os.path.join(CONDGEN_PATH, x)) for x in os.listdir(CONDGEN_PATH)
]
gg_runs = []
for our in OUR_PATHS:
    gg_runs.extend([Run.load(os.path.join(our, x)) for x in os.listdir(our)])


def filter_runs(runs):
    return [
        r
        for r in runs
        if r.data
        and "times" in r.data
        and np.isfinite(r.data["times"]).any()
        and r.data["num_samples"] > 1
    ]


graprnn_runs = filter_runs(graprnn_runs)
gg_runs = filter_runs(gg_runs)
condgen_runs = filter_runs(condgen_runs)


def make_df(runs):
    dfs = []
    for r in runs:
        times = r.data["times"]
        steps = r.data["num_nodes"]
        name = r.data["model_name"]
        if "cuda" in name:
            n0 = name.split("cuda")[0]
            name = f"{n0}cuda"
        if "RNN" in name:
            name = "graphRNN"
            print(r.config)
        elif "condgen" in name:
            name = "CondGen"
        elif "cuda" in name:
            name = "GG-GAN"
        else:
            name = "GG-GAN (cpu)"
        for s in range(times.shape[-1]):
            d = dict(times=times[:, s], Model=[name] * len(steps), steps=steps)
            df = pd.DataFrame.from_dict(d)
            df = df[df.times != np.inf]
            dfs.append(df)
            if "RNN" in name:
                dfgran = df.copy()
                dfgran["Model"] = "GRAN (est.)"
                dfgran["times"] = df["times"] / 10
                dfs.append(dfgran)
    # dfs.append(pd.DataFrame.from_dict(dict(
    #    times=[2.2],
    #    steps=[400],
    #    Model=["GRAN (suppl.)"]
    # )
    # ))
    dfs.append(
        pd.DataFrame.from_dict(
            dict(times=bigg_T, steps=bigg_X, Model=["bigg (plot)"] * len(bigg_T))
        )
    )
    return pd.concat(dfs)


df = make_df(graprnn_runs + gg_runs + condgen_runs)


def make_plot(df, scale="log", boxen=False):
    fig, ax = plt.subplots(figsize=[8, 4])
    if boxen:
        sb.boxenplot(
            x="steps",
            y="times",
            hue="Model",
            data=df,
            # palette="Set3",
            ax=ax,
            palette="colorblind",
        )
    else:
        ax: plt.Axes
        for tick in ax.get_xticklabels():
            tick.set_rotation(90)
        for (g, gdf), c, m in zip(
            df.groupby(["Model"]),
            sb.color_palette("colorblind"),
            ["<", "x", "o", "v", "s", "P", "d", "p"],
        ):
            t = gdf.groupby("steps")["times"]
            med = t.median()
            x = med.index
            v = med.values
            if "bigg" in g:
                e = 7
                x = x[:e]
                v = v[:e]
            ax.plot(
                x,
                v,
                color=c,
                marker=m,
                linestyle="--" if "est" in g or "plot" in g else "-",
                label=g,
            )
            # for i, (s, sdf) in enumerate(gdf.groupby("steps")):
            #    val = np.array(sdf["times"])
            #    steps = s * np.ones(len(val))
            i = 0
            # ax.scatter(med.index,med, color=c, marker=m)# label=None if i > 0 else g)
    ax.legend()
    ax.set_yscale(scale)
    # ax.set_xscale(scale)
    ax.set_ylabel("generation time (seconds)")
    # ax.grid(axis="y")
    ax.set_xlabel("# Nodes")
    fig.savefig(f"scale_joint{boxen}.pdf", bbox_inches="tight")
    plt.show(fig)


medians = df.groupby(["Model", "steps"]).median()["times"]
medians: pd.DataFrame
print(medians.to_markdown())
medians.to_csv("scale_meds.csv")
make_plot(df, boxen=False)
make_plot(df, boxen=True)
