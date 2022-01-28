# -*- coding: utf-8 -*-
from multiprocessing.pool import Pool
import subprocess as sp

from ggg.models.ggg_model import GGG_Hpar
import attr
from uuid import uuid4

EXP_PATH = "/home/anon/graph-gan-main/code/ggg/experiments/GGG.py"


def with_dict(d, prefix=None):
    withs = []
    for k, v in d.items():
        if type(v) is dict:
            withs = with_dict(v, prefix=k)
            if prefix is not None:
                withs = [f"{prefix}.{x}" for x in withs]
            withs.extend(withs)
        else:
            w = f"{k}={v}"
            if prefix is not None:
                w = f"{prefix}.{w}"
            withs.append(w)
    return withs


if __name__ == "__main__":

    TOTAL_GPUS = 72
    NUM_GPUS = 4
    GPU_MEM = 32510
    ESTIMATED_MEM_USAGE = 2000  # 1.1k observed=> use 2k
    GPU_PER_JOB = ESTIMATED_MEM_USAGE / GPU_MEM
    NUM_PARALLEL = int(NUM_GPUS / GPU_PER_JOB)

    def run_config(config, named_configs):

        args = ["python", EXP_PATH, "-F"]
        args.append("PaperExperiments")
        args.append("with")
        dataset = config["hyper"]["dataset"]
        config["model_n"] = f"{dataset}_{uuid4()}"
        args.extend(named_configs)
        args.extend(with_dict(config))
        print(f"Starting {dataset}")
        print(f"Running {args}")
        run = sp.run(args)

    hyper = GGG_Hpar()

    named_configs = []
    DATASETS = [
        "MolGAN_5k",
        # "MolGAN_kC4","MolGAN_kC5","MolGAN_kC6",
        "anu_graphs_chordal_45789",
        "CommunitySmall_12",
        "CommunitySmall_20",
    ]
    # TODO: launch more here

    configs = []
    gpu = 0
    for dataset in DATASETS:
        hyper.dataset = dataset
        hyper.num_workers = 8
        hdict = attr.asdict(hyper)
        config = dict(hyper=hdict, device=f"cuda:{gpu}")
        configs.append((config, named_configs))
        gpu = (gpu + 1) % NUM_GPUS

    print(f" Running {len(configs)} on {NUM_PARALLEL} workers")
    tp = Pool(processes=NUM_PARALLEL)
    runs = tp.starmap_async(run_config, configs)

    tp.close()
    runs.wait()
