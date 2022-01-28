import matplotlib


matplotlib.use("Agg")
from uuid import uuid4

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

EXPNAME = "GG_GAN_Train"
ex = Experiment(EXPNAME)
if len(ex.observers) == 0:
    ex.observers.append(FileStorageObserver(EXPNAME))


def get_model_n_uuid(hyper):
    ds = hyper["dataset"]
    return f"{ds}_{uuid4()}"


def get_model_n(hyper):
    return (
        "Arch="
        + str(hyper["architecture"])
        + "_Z0dim="
        + str(hyper["embed_dim"])
        + "_CVdim="
        + str(hyper["finetti_dim"])
        + "_Trainable="
        + str(hyper["finetti_trainable"])
        + "FixCV="
        + str(hyper["finetti_train_fix_context"])
        + "DynamicCreation="
        + str(hyper["dynamic_finetti_creation"])
        + "FlipFinetti="
        + str(hyper["flip_finetti"])
        + "_Model="
        + str(hyper["cycle_opt"])
        + "_AttentionMode="
        + str(hyper["attention_mode"])
        + "_EdgeReadout="
        + str(hyper["edge_readout"])
    )
