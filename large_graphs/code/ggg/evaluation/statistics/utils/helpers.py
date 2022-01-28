import os
import time
import pickle
import datetime
from collections import defaultdict
from typing import List, Dict
from warnings import warn

import torch as pt
import numpy as np
import networkx as nx
from ipdb import set_trace
from tqdm import tqdm

# imports for stat tests
import pyemd
import concurrent.futures
from functools import partial
from scipy.linalg import toeplitz
from tqdm import tqdm
import attr
from ggg.utils.utils import kcycles


### General helpers
from ggg.utils.utils import ensure_tensor, zero_diag


class ProgressBar:
    """Class for progress bar visualization"""

    def __init__(self, length, max_value):
        assert length > 0 and max_value > 0
        self.length, self.max_value, self.start = length, max_value, time.time()

    def update(self, value):
        assert 0 < value <= self.max_value
        delta = (time.time() - self.start) * (self.max_value - value) / value
        format_spec = [
            value / self.max_value,
            value,
            len(str(self.max_value)),
            self.max_value,
            len(str(self.max_value)),
            "#" * int((self.length * value) / self.max_value),
            self.length,
            datetime.timedelta(seconds=int(delta))
            if delta < 60 * 60 * 10
            else "-:--:-",
        ]

        print("\r{:=5.0%} ({:={}}/{:={}}) [{:{}}] ETA: {}".format(*format_spec), end="")


@attr.s
class Samples:
    """
    Simple helper class to carry around observations+their counts, to be used in the weighted hist for the kde
    to allow for countwise *large* *sparse* observations
    """

    values = attr.ib()
    counts = attr.ib()

    @classmethod
    def from_dict(cls, d):
        values = np.array(sorted(list(d.keys())))
        counts = np.array([d[k] for k in values])
        return cls(values, counts)

    def mmd(self, other, with_torch=False):
        # TODO: could probably still do this smarter...but try this for now
        s1 = pt.cat(
            [pt.ones(count) * value for value, count in zip(self.values, self.counts)]
        )
        s2 = pt.cat(
            [pt.ones(count) * value for value, count in zip(other.values, other.counts)]
        )
        if with_torch:
            mmd = torch_mmd(samples1=s1, samples2=s2, kernel=gaussian_kernel)
        else:
            mmd = compute_mmd(s1.numpy(), s2.numpy(), gaussian, is_hist=False)
        return mmd


def get_dists(flags, graphs, mmd_plot=False, name="") -> Dict[str, pt.Tensor]:
    """Function to get vector representing the
    a K distribution of a sample of graphs.

    K can be edge | cycle |"""

    disable = False
    if mmd_plot:
        disable = True

    vectors = defaultdict(list)
    max_len = defaultdict(lambda: 0)
    for g in tqdm(graphs, leave=not disable, desc=f"Getting {flags} dist {name},graph"):
        A = nx.adjacency_matrix(g).todense()
        np.fill_diagonal(A, 0)
        # Transform graph to matrix A not accounting for self connection
        if "degree" in flags:
            k = "degree"
            # degree vectors

            v = pt.from_numpy(A.sum(-1)).flatten()
            max_len[k] = max(len(v), max_len[k])
            vectors[k].append(v)
        if "cycles" in flags:
            kc = kcycles()
            k = "cycles"
            # will be vector where 0th element is 3 cycles, 1st is 4,2nd is 5,3rd is 6
            k_cycles = pt.cat(
                [
                    pt.zeros(3),
                    kc.k_cycles(pt.tensor(A).clone(), verbose=mmd_plot).flatten(),
                ]
            )
            max_len[k] = max(len(k_cycles), max_len[k])
            vectors[k].append(k_cycles)

    vectors = {
        k: pt.stack(
            [
                x
                if len(x) == max_len
                else pt.cat([x, pt.zeros(max_len[k] - len(x), dtype=x.dtype)])
                for x in vecs
            ]
        )
        for k, vecs in vectors.items()
    }
    return vectors


def get_dist(flag_, graphs, mmd_plot=False) -> pt.Tensor:
    """Function to get vector representing the
    a K distribution of a sample of graphs.

    K can be edge | cycle |"""

    disable = False
    if mmd_plot:
        disable = True

    vectors = []
    max_len = 0
    if flag_ == "degree":
        for g in tqdm(graphs, leave=not disable, desc="Getting degree dist,graph"):
            # Transform graph to matrix A not accounting for self connection
            A = nx.adjacency_matrix(g).todense()
            np.fill_diagonal(A, 0)

            # degree vectors

            v = pt.from_numpy(A.sum(-1)).flatten()
            max_len = max(len(v), max_len)
            vectors.append(v)
    elif flag_ == "cycles":

        kc = kcycles()
        for g in tqdm(graphs, leave=not disable, desc="Getting cycle dist,graph"):
            temp_dist = []
            A = nx.adjacency_matrix(g).todense()
            np.fill_diagonal(A, 0)
            # will be vector where 0th element is 3 cycles, 1st is 4,2nd is 5,3rd is 6
            k_cycles = pt.cat(
                [pt.zeros(3), kc.k_cycles(pt.tensor(A).clone())]
            ).flatten()
            max_len = max(len(k_cycles), max_len)
            vectors.append(k_cycles)

    else:
        raise RuntimeError("flag name : {} is not viable".format(flag_))

    vectors = pt.stack(
        [
            x if len(x) == max_len else pt.cat([x, pt.zeros(max_len - len(x))])
            for x in vectors
        ]
    )

    return vectors


def list_from_pickle(dir_):
    """Read list from pickle file"""

    with open(dir_, "rb") as f:
        pkl_list = pickle.load(f)

    return pkl_list


def metric_to_use(_config):
    """Get suffix for file referent to metric desired"""

    if _config["metric"] == "degree":
        suffix_ = "_degreeD.pkl"
    elif _config["metric"] == "cycles":
        suffix_ = "_cycleD.pkl"

    return suffix_


def check_dir(directory):
    return os.path.exists(directory)


### Stat tests helpers
def emd(x, y, distance_scaling=1.0):
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)  # diagonal-constant matrix
    distance_mat = d_mat / distance_scaling
    x, y = process_tensor(x, y)

    emd_value = pyemd.emd(x, y, distance_mat)
    return np.abs(emd_value)


def l2(x, y):
    dist = np.linalg.norm(x - y, 2)
    return dist


def gaussian_emd(x, y, sigma=1.0, distance_scaling=1.0):
    """Gaussian kernel with squared distance in exponential term replaced by EMD
    Args:
      x, y: 1D pmf of two distributions with the same support
      sigma: standard deviation
    """
    emd_value = emd(x, y, distance_scaling)
    return np.exp(-emd_value * emd_value / (2 * sigma * sigma))


def gaussian_kernel(d, sigma=1.0):
    d = ensure_tensor(d)
    return pt.exp(-d * d / (2 * sigma ** 2))


def gaussian(x, y, sigma=1.0):
    if pt.is_tensor(x):
        x = x.float()
        y = y.float()
    else:
        x = x.astype(np.float)
        y = y.astype(np.float)
    x, y = process_tensor(x, y)
    dist = np.linalg.norm(x - y, 2)
    return gaussian_kernel(dist, sigma).numpy()


def gaussian_tv(x, y, sigma=1.0):
    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    x, y = process_tensor(x, y)

    dist = np.abs(x - y).sum() / 2.0
    return np.exp(-dist * dist / (2 * sigma * sigma))


def kernel_parallel_unpacked(x, samples2, kernel):
    d = 0
    for s2 in samples2:
        d += kernel(x, s2)
    return d


def kernel_parallel_worker(t):
    return kernel_parallel_unpacked(*t)


def disc(samples1, samples2, kernel, is_parallel=True, *args, **kwargs):
    """Discrepancy between 2 samples"""
    d = 0
    if not is_parallel:
        for s1 in tqdm(samples1):
            for s2 in samples2:
                d += kernel(s1, s2, *args, **kwargs)
    else:
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for dist in executor.map(
                kernel_parallel_worker,
                [(s1, samples2, partial(kernel, *args, **kwargs)) for s1 in samples1],
            ):
                d += dist

    d /= len(samples1) * len(samples2)
    return d


def torch_mmd(samples1, samples2, kernel, hist=True, **kwargs):
    samples1 = ensure_tensor(samples1).float()
    samples2 = ensure_tensor(samples2).float()
    if hist:
        samples1 = samples1 / (samples1.sum(-1)[:, None] + 1e-19)
        samples2 = samples2 / (samples2.sum(-1)[:, None] + 1e-19)
    kernel = partial(kernel, **kwargs)
    s1pdist = pt.cdist(samples1, samples1)
    s2pdist = pt.cdist(samples2, samples2)
    s1ks = kernel(s1pdist).mean()
    s2ks = kernel(s2pdist).mean()
    cdists = pt.cdist(samples1, samples2)
    kernels = kernel(cdists).mean()
    # print(s1ks,s2ks,kernels)
    mmd_dist = s1ks + s2ks - 2 * kernels
    return mmd_dist


def compute_mmd_cycle(
    samples1: pt.Tensor, samples2: pt.Tensor, kernel, is_hist=True, *args, **kwargs
):
    """MMD between two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = samples1 / (
            samples1.sum(-1)[:, None] + 1e-10
        )  # [s1 / np.sum(s1) for s1 in samples1]
        samples2 = samples2 / (
            samples2.sum(-1)[:, None] + 1e-10
        )  # [s2 / np.sum(s2) for s2 in samples2]

    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
    # print('===============================')
    return (
        disc(samples1, samples1, kernel, *args, **kwargs)
        + disc(samples2, samples2, kernel, *args, **kwargs)
        - 2 * disc(samples1, samples2, kernel, *args, **kwargs)
    )


def compute_mmd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """MMD between two samples"""
    # normalize histograms into pmf
    if is_hist:
        samples1 = [s1 / np.sum(s1) for s1 in samples1]
        samples2 = [s2 / np.sum(s2) for s2 in samples2]

    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
    # print('===============================')
    d1 = disc(samples1, samples1, kernel, *args, **kwargs)
    d2 = disc(samples2, samples2, kernel, *args, **kwargs)
    dcross = disc(samples1, samples2, kernel, *args, **kwargs)
    print(d1, d2, dcross)
    return d1 + d2 - 2 * dcross


def compute_emd(samples1, samples2, kernel, is_hist=True, *args, **kwargs):
    """EMD between average of two samples"""

    # normalize histograms into pmf
    if is_hist:
        samples1 = [np.mean(samples1)]
        samples2 = [np.mean(samples2)]
    # print('===============================')
    # print('s1: ', disc(samples1, samples1, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('s2: ', disc(samples2, samples2, kernel, *args, **kwargs))
    # print('--------------------------')
    # print('cross: ', disc(samples1, samples2, kernel, *args, **kwargs))
    # print('===============================')
    return disc(samples1, samples2, kernel, *args, **kwargs), [samples1[0], samples2[0]]


def pad_to(x, support_size):
    if not pt.is_tensor(x):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    else:
        x = pt.cat([x, pt.zeros(support_size - len(x))])
    return x


def process_tensor(x, y):
    support_size = max(len(x), len(y))
    if len(x) < len(y):
        x = pad_to(x, support_size)
    elif len(y) < len(x):
        y = pad_to(y, support_size)
    return x, y


import torch as pt
import torch.distributions as td


def discretize(A, temperature):
    """Discretize the continuous adjacency matrix. We use a Bernoulli

    :param A:adjacency matrix
    :return A:discretize adj matrix
    """
    relaxedA = td.RelaxedBernoulli(temperature, probs=A).rsample()
    # hard relaxed bernoulli= create 0 vector with gradients attached, add to rounded values
    grads_only = relaxedA - relaxedA.detach()
    Ar = relaxedA.round() + grads_only
    # rezero and resymmetrize
    Az = zero_diag(Ar)
    Atriu = pt.triu(Az)
    A = Atriu + Atriu.permute(0, 2, 1)

    return A


if __name__ == "__main__":
    s1 = Samples(np.arange(5), np.random.randint(0, 100, 5))
    s2 = Samples(np.arange(5), np.random.randint(0, 100, 5))
    mmd = s1.mmd(s2)
    mmd_torch = s1.mmd(s2, with_torch=True)
    print(mmd, mmd_torch, mmd - mmd_torch)
