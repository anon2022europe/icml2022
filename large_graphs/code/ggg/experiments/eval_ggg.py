from logging import warning
import torch as pt
from typing import Optional
from warnings import warn

from ipdb import set_trace
from pytorch_lightning.profiler import AdvancedProfiler

import matplotlib
matplotlib.use("Agg")
from uuid import uuid4
import json

import copy


def sacred_copy(o):
    """Perform a deep copy on nested dictionaries and lists.
    If `d` is an instance of dict or list, copies `d` to a dict or list
    where the values are recursively copied using `sacred_copy`. Otherwise, `d`
    is copied using `copy.deepcopy`. Note this intentionally loses subclasses.
    This is useful if e.g. `d` is a Sacred read-only dict. However, it can be
    undesirable if e.g. `d` is an OrderedDict.
    :param o: (object) if dict, copy recursively; otherwise, use `copy.deepcopy`.
    :return A deep copy of d."""
    if isinstance(o, dict):
        return {k: sacred_copy(v) for k, v in o.items()}
    elif isinstance(o, list):
        return [sacred_copy(v) for v in o]
    else:
        return copy.deepcopy(o)


from sacred import Experiment
from sacred.observers import FileStorageObserver

EXPNAME = "GG_GAN_eval"
ex = Experiment(EXPNAME)
if len(ex.observers) == 0:
    ex.observers.append(FileStorageObserver(EXPNAME))
from pprint import pprint
from uuid import uuid4
import os
import tempfile
from itertools import chain
from ggg.evaluation.plots.utils.post_experiment_plots import plot_mmd_wrapper, main_run_MMD, generate_graphs
from sacred.observers import FileStorageObserver
from ggg.models.ggg_model import (
    GGG,
    GGG_Hpar,
)
from ggg.utils.hooks import forward_clip_hook, backward_trace_hook, backward_clean_hook

from ggg.data.dense.GGG_DenseData import GGG_DenseData
from pytorch_lightning import Trainer

from pytorch_lightning.loggers import TensorBoardLogger
import torch as pt

# TODO(adam): delete this once Sacred issue #498 is resolved
from sacred.run import Run

@ex.config
def conf():
    ckpt_path=None
    numb_graphs=512
    gpu="cuda:0"
    epoch=None


@ex.automain
def run(
    ckpt_path,
    numb_graphs,
    gpu,
    epoch,
    _run: Run,
    _config,
):
    save_dir=_run.observers[-1].dir

    os.makedirs(save_dir,exist_ok=True)
    print("loading checkpoint")
    model=GGG.load_from_checkpoint(checkpoint_path=ckpt_path)
    print("moving to gpu")
    model=model.to(gpu)
    print("on gpu")
    model.eval()
    hparams=model.hpars.to_dict()

    gen_graphs, dataset_graphs = generate_graphs(
        model,
        current_epoch=epoch,
        device=model.device,
        batch_size=model.hpars.batch_size,
        dataset=model.validation_set(),
        numb_graphs=numb_graphs,
        save_dir=save_dir,
    )
    ret=main_run_MMD(
        epoch,
        csv_dir=save_dir,
        model_graphs=gen_graphs,
        dataset_graphs=dataset_graphs,
        numb_graphs=numb_graphs,
        save=True,
        dataset_name=hparams["dataset_hpars"]["dataset"],
        model_name=None,
    )
    (degree_metric,
    clustering_metric,
    cycle_metric,
    degree_assortativity_metric,
    algebraic_connectivity_metric)=ret[:5]
    metrics=dict(
        degree_metric=degree_metric,
        clustering_metric=clustering_metric,
        cycle_metric=cycle_metric,
        degree_assortativity_metric=degree_assortativity_metric,
        algebraic_connectivity_metric=algebraic_connectivity_metric
    )
    if len(ret)>5:
        if len(ret)==7:
            ds_ll,m_ll=ret[-2:]
            metrics.update(**{"ds_ll":ds_ll,"m_ll":m_ll})
        else:
            warn(f"Expected len 7, got {len(ret)}")
    for k,v in metrics.items():
        _run.log_scalar(k,v)
    dataset_name=hparams["dataset_hpars"]["dataset"]
    values=list(metrics.values())
    keys=list(metrics.keys())
    d={
        "GG-GAN":{
        dataset_name:{"vals":values,"keys":keys}
        }
    }
    def rec_item(d):
        if isinstance(d,dict):
            return {k:rec_item(v) for k,v in d.items()}
        elif pt.is_tensor(d):
            return d.item()
        elif hasattr(d,"__getitem__") and hasattr(d,"__len__") and len(d)==1:
            return d[0]
        else:
            return d
    with open(os.path.join(save_dir,"model_val.json"),"wt") as f:
        json.dump(rec_item(d),f)

