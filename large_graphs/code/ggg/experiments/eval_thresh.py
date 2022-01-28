from logging import warning
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

EXPNAME = "thresholds"
ex = Experiment(EXPNAME)
if len(ex.observers) == 0:
    ex.observers.append(FileStorageObserver(EXPNAME))
from pprint import pprint
from uuid import uuid4
import os
import tempfile
from itertools import chain
from ggg.evaluation.plots.utils.post_experiment_plots import plot_mmd_wrapper, main_run_MMD, generate_graphs, \
    get_dataset_graphs
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
    config_path=None
    numb_graphs=512


@ex.automain
def run(
    config_path,
    numb_graphs,
    _run: Run,
    _config,
):
    save_dir=_run.observers[-1].dir

    os.makedirs(save_dir,exist_ok=True)
    with open(config_path,"r") as f:
        c=json.load(f)["hyper"]
    hpars=GGG_Hpar.from_dict(c)
    ds1=hpars.dataset_hpars.make()
    ds2=hpars.dataset_hpars.make_val()
    ds1=get_dataset_graphs(ds1)
    ds2=get_dataset_graphs(ds2)
    ret=main_run_MMD(
        current_epoch=0,
        csv_dir=save_dir,
        model_graphs=ds1,
        dataset_graphs=ds2,
        numb_graphs=numb_graphs,
        save=True,
        dataset_name=c["dataset_hpars"]["dataset"],
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
    dataset_name=c["dataset_hpars"]["dataset"]
    values=metrics.values()
    keys=metrics.keys()
    d={
        "threshold":{
        dataset_name:{"vals":list(values),"keys":list(keys)}
        }
    }
    with open(os.path.join(save_dir,"model_val.json"),"wt") as f:
        json.dump(d,f)

