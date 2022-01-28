import matplotlib

matplotlib.use("TkAgg")
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sacred import Experiment
import attr
import os
import torch as pt

from pytorch_lightning.callbacks import EarlyStopping
from ggg.data.dense.GGG_DenseData import GGG_DenseData
from ggg.models.collisiondemo import CollisionPars, CollisionDemo

matplotlib.use("TkAgg")

ex = Experiment("CollisionDemo")


@ex.config
def conf():
    hpars = attr.asdict(CollisionPars())
    early_stopping = 5


@ex.automain
def run(_run, hpars, _seed, early_stopping):
    print(f"Seed {_seed}")
    hpars = CollisionPars.from_sacred(hpars)
    if hpars.max_N is None and hpars.dataset != "kernel-custom":
        ds = GGG_DenseData(dataset=hpars.dataset, inner_kwargs=hpars.inner_kwargs)
        hpars.max_N = ds.max_N
    if len(_run.observers) > 0:
        save_dir = _run.observers[0].dir
    else:
        save_dir = "collision_save"
    print(hpars)
    model = CollisionDemo(hpars)
    trainer = Trainer(
        max_steps=hpars.max_steps,
        logger=TensorBoardLogger(save_dir=save_dir, name=f"{hpars.model}"),
        early_stop_callback=early_stopping
        if early_stopping is None
        else EarlyStopping(monitor="loss", patience=early_stopping, verbose=True),
        track_grad_norm=2,
    )
    trainer.fit(model)
    model: CollisionDemo
    if "traj" in hpars.model:
        d = {"trained": model.Z0_fixed, "init": model.Z0_fixed}
        with open(os.path.join(save_dir, "Z0s.pt"), "wb") as f:
            pt.save(d, f)
    model.make_plots(save_dir=save_dir, show=True)
