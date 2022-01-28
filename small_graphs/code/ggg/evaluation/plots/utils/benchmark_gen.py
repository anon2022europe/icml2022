import attr
import time
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os


@attr.s
class BenchmarkData:
    times = attr.ib()
    num_samples = attr.ib()
    max_N = attr.ib()
    num_nodes = attr.ib()
    model_name = attr.ib(default=None)

    def save(self, savedir):
        d = attr.asdict(self)
        for k in d.keys():
            if "model_name" in k:
                continue
                d[k] = torch.tensor(d[k])
        mn = "" if self.model_name is None else f"{self.model_name}_"
        with open(os.path.join(savedir, f"{mn}benchmark_time.pt"), "wb") as f:
            torch.save(d, f)


def benchmark_graph_gen(
    create_model,
    sample_func,
    num_nodes=(1, 10, 100, 1000, 2000, 4000, 6000, 8000, 10000, 12000, 14000, 16000),
    num_samples=100,
    warm_up=10,
    name=None,
    raise_except=False,
    timeout_abort=300,
) -> BenchmarkData:
    times = []
    steps = []
    max_num_nodes = max(num_nodes)
    for n in tqdm(num_nodes, desc="Bechnmarking num_nodes"):
        if n == 0:
            n = 1
        steps.append(n)
        outs = []
        node_times = []
        model = create_model(n)
        with torch.no_grad():
            for _ in range(warm_up):
                try:
                    _ = sample_func(model)
                except BaseException as e:
                    if raise_except:
                        raise e
            for _ in tqdm(range(num_samples), desc=f"Measuring for N={n}"):
                try:
                    before = time.perf_counter()
                    o = sample_func(model)
                    after = time.perf_counter()
                    outs.append(o)
                    node_times.append(after - before)
                except BaseException as e:
                    node_times.append(np.inf)
                    if raise_except:
                        raise e
            time_arr = np.array(node_times)
            times.append(time_arr)
            if np.median(time_arr) > timeout_abort:
                break

    return BenchmarkData(
        np.stack(times, 0), num_samples, max_num_nodes, np.array(steps), name
    )


def plot_benchmark(benchmark_data: BenchmarkData, save_dir=None, style="box"):
    times = benchmark_data.times
    x = benchmark_data.num_nodes
    tmax = times.max(-1)
    tmin = times.max(-1)
    mu = times.mean(-1)
    med = np.median(times, -1)
    fith = np.percentile(times, 5, -1)
    ninetyfifth = np.percentile(times, 95, -1)
    sig = times.std(-1)

    fig, ax = plt.subplots()
    fig: plt.Figure
    ax: plt.Axes
    if style == "box":
        ax.boxplot(times.T)
        ax.set_xticklabels(x, rotation=90)
        ax.set_ylabel(f"Inference time in s, avg of {benchmark_data.num_samples}")
        ax.set_xlabel("# Nodes in graph")
        name = (
            benchmark_data.model_name if benchmark_data.model_name is not None else ""
        )
        name = f"{name}_inference_box.pdf"
    elif style == "line":
        for resize in [True, False]:
            ax.plot(x, mu, label="Mean")
            ax.plot(x, med, label="Median")
            ax.plot(x, tmax, label="Max")
            ax.plot(x, tmin, label="Min")
            ax.plot(x, fith, label="5th percentile")
            ax.plot(x, ninetyfifth, label="95th percentile")
            ax.fill_between(x, mu + sig, np.maximum(mu - sig, 0), alpha=0.5)
            ax.set_ylabel(f"Inference time in s, avg of {benchmark_data.num_samples}")
            ax.set_xlabel("# Nodes in graph")
            ax.set_xlim(0, benchmark_data.max_N)
            if resize:
                ax.set_ylim(-0.01, ninetyfifth.max() * 1.1)
            ax.legend()
            name = (
                benchmark_data.model_name
                if benchmark_data.model_name is not None
                else ""
            )
            name = f"{name}_inference_ylim{resize}.pdf"
    else:
        raise NotImplementedError("Only box and line style is implemented")
    if save_dir is not None:
        fig.savefig(os.path.join(save_dir, name))
    return fig, ax
