import matplotlib

from ggg.models.sbm_readout import SBMCheckPars

BACKEND = "Agg"
matplotlib.use(BACKEND)
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from sacred import Experiment
import attr
import os
import torch as pt

from pytorch_lightning.callbacks import EarlyStopping
from ggg.data.dense.GGG_DenseData import GGG_DenseData
from ggg.models.sbm_readout import SBMCheckPars, SBMCheck

matplotlib.use(BACKEND)
ex = Experiment("SBMCheck")


@ex.config
def conf():
    hpars = attr.asdict(SBMCheckPars())
    early_stopping = None


@ex.automain
def run(_run, hpars, _seed, early_stopping):
    print(f"Seed {_seed}")
    hpars = SBMCheckPars.from_sacred(hpars)
    if len(_run.observers) > 0:
        save_dir = _run.observers[0].dir
    else:
        save_dir = "sbm_save"
    print(hpars)
    model = SBMCheck(hpars)
    trainer = Trainer(
        max_steps=hpars.max_steps // hpars.batch_size,
        logger=TensorBoardLogger(save_dir=save_dir, name=f"{hpars.readout}"),
        early_stop_callback=early_stopping
        if early_stopping is None
        else EarlyStopping(monitor="loss", patience=early_stopping, verbose=True),
        track_grad_norm=2,
    )
    trainer.fit(model)
    model.make_plots(save_dir=save_dir, show=True)
