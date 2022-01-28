import matplotlib

matplotlib.use("Agg")

from ggg.evaluation.plots.utils.benchmark_gen import (
    benchmark_graph_gen,
    plot_benchmark,
)

from pprint import pprint

from graph_stat import *
from collections import defaultdict
import os
import numpy as np
import os
import attr
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict
from pprint import pprint
from collections import defaultdict

from graph_stat import *
import networkx as nx
from torch.utils.data import Dataset
from options import Options, AttrOptions
from GVGAN import *
from utils import *
from tqdm import tqdm
from pprint import pprint
from sacred import Experiment

ex = Experiment("CondgenEvalScale")

from models import *
from train import load_data, load_model_parts, G_Wrap, load_our_data, train, sacred_copy


@ex.config
def conf():
    hpars = attr.asdict(AttrOptions())
    device = "cuda:1"
    batch_size = 1
    max_num_nodes = 1000
    step = 50


@ex.automain
def run(hpars, _run, _config, device, batch_size, max_num_nodes, step):
    if "cuda" in device:
        torch.cuda.set_device(device)
    batch_size = 1
    print(_config)
    opt = AttrOptions(**sacred_copy(hpars))

    print("=========== OPTIONS ===========")
    pprint(opt)
    opt.gpu = int(device.split(":")[-1])
    print(" ======== END OPTIONS ========\n\n")

    # os.environ['CUDA_VISIBLE_DEVICES'] = f"{opt.gpu}"

    print("Got data")
    # output_dir = opt.output_dir
    def create_func(n):
        train_adj_mats = torch.stack(
            [
                torch.from_numpy(
                    nx.to_numpy_array(
                        nx.connected_watts_strogatz_graph(n, min(n, 3), 0.1, tries=1000)
                    )
                )
            ],
            0,
        )
        train_attr_vecs = torch.rand(1, opt.av_size)
        test_adj_mats = train_adj_mats
        test_attr_vecs = train_attr_vecs
        opt.max_epochs = 0
        D, G = train(
            train_adj_mats,
            test_adj_mats,
            train_attr_vecs,
            test_attr_vecs,
            opt=opt,
            batch_size=batch_size,
        )
        G.eval()
        G = G.to(device)
        train_adj_mats, train_attr_vecs = [
            x.to(device) if x is not None else x
            for x in [train_adj_mats, train_attr_vecs]
        ]
        Gw = G_Wrap(
            G, train_adj_mats, opt.z_size, opt.d_size, attr_vecs=train_attr_vecs
        )
        return Gw

    Gw: G_Wrap

    def sample_func(model):
        model: G_Wrap
        return model.sample(device=device, spectral_emb=False)

    logdir = _run.observers[0].dir
    plots_save_dir = os.path.join(logdir, "plots")
    os.makedirs(plots_save_dir, exist_ok=True)
    benchmark_data = benchmark_graph_gen(
        create_func,
        sample_func,
        num_nodes=(
            2,
            10,
            100,
            1000,
            2000,
            4000,
            6000,
            8000,
            10000,
            12000,
            14000,
            16000,
            18000,
            20000,
        ),
        num_samples=100,
        warm_up=0,
        name=f"condgen_{device}",
        raise_except=True,
    )
    benchmark_data.save(logdir)
    plot_benchmark(benchmark_data, plots_save_dir)
