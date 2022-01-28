from logging import warning

from ipdb import set_trace
from pytorch_lightning.profiler import AdvancedProfiler

from ggg.experiments.ggan_exp.base import ex, get_model_n_uuid, sacred_copy
from pprint import pprint
from uuid import uuid4
import os
import tempfile
from itertools import chain
from ggg.evaluation.plots.utils.callback import ModelCheckPointWithPlots
from ggg.evaluation.plots.utils.post_experiment_plots import plot_mmd_wrapper
from pytorch_lightning.callbacks import LearningRateLogger
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


@ex.main
def run(
    hyper,
    save_dir,
    epochs,
    track_norm,
    log_k,
    overfit_pct,
    ckpt_period,
    detect_anomaly,
    forward_clip,
    backward_clean,
    clip_grad_val,
    val_every,
    gpus,
    _run: Run,
    _config,
):
    pprint(_config)
    if clip_grad_val>0.0:
        # https://pytorch-lightning.readthedocs.io/en/stable/training_tricks.html
        warning(f"Using gradient clipping with val{clip_grad_val}")
    hyper, epochs, overfit_pct = [
        sacred_copy(o)
        for o in [
            hyper,
            epochs,
            overfit_pct,
        ]
    ]
    exp_dir = _run.observers[0].dir
    if save_dir is None:
        save_dir = os.path.join(exp_dir, "lightning_log")

    old_hyper = hyper
    changes = dict(
        save_dir=save_dir,
    )
    # finalize updated hparams
    hparams = {
        k: changes[k] if k in changes else old_hyper[k]
        for k in chain(old_hyper.keys(), changes.keys())
    }
    if gpus is None:
        gpus = int(pt.cuda.is_available())

    print("Initializing model with parameters")
    pprint(hparams)
    model = GGG(hparams)

    for name, module in model.named_modules():
        if forward_clip:
            module.register_forward_hook(forward_clip_hook)
        if backward_clean:
            module.register_backward_hook(backward_clean_hook)

    try:
        # try to use a logger with a nicer folder structure
        from pytorch_lightning.loggers import TestTubeLogger

        tblogger = TestTubeLogger(save_dir)
    except:
        tblogger = TensorBoardLogger(save_dir)

    # TODO add toggle if user wants to define directory to save
    plots_save_dir = os.path.join(save_dir, "plots")

    checkpoint_callback = ModelCheckPointWithPlots(
        save_top_k=-1,
        period=ckpt_period,
        verbose=True,
        numb_graphs=512,
        plot_dataset=True,
        loss_dir=save_dir,
        plot_dir=plots_save_dir,
        mmd=False,
        lcc=hparams.get("plot_lcc", False),
    )

    print("Starting training")
    trainer = Trainer(
        progress_bar_refresh_rate=50,
        max_epochs=epochs,
        weights_summary="full",
        logger=tblogger,
        log_save_interval=1,
        benchmark=True,
        checkpoint_callback=checkpoint_callback,
        callbacks=[LearningRateLogger()],
        overfit_pct=overfit_pct,
        num_nodes=1,
        gpus=gpus,
        gradient_clip_val=clip_grad_val,
        check_val_every_n_epoch=val_every,
        row_log_interval=log_k,
        track_grad_norm=track_norm,
        precision=hparams["precision"]
        # profiler=AdvancedProfiler()
    )

    with pt.autograd.set_detect_anomaly(detect_anomaly):
        trainer.fit(model)
    # fpath = os.path.join("/tmp", f"{uuid4()}")
    # trainer.save_checkpoint(fpath)
    # _run.add_artifact(fpath, "finalCheckpoint.ckpt")
    # _run.add_artifact(fpath)
    # tempdir = tempfile.mkdtemp()

    ## Plot function
    # os.makedirs(plots_save_dir, exist_ok=True)
    # model.eval()

    ## MMD
    # plot_mmd_wrapper(
    #    hparams=hparams,
    #    current_epoch=trainer.current_epoch,
    #    plots_save_dir=plots_save_dir,
    #    dataset_name=hparams["dataset_hpars"]["dataset"],
    #    dataset_used=model.train_set,
    #    pyl_log_dir=save_dir,
    #    numb_g_eval=5000,
    #    numb_g_mmd=5000,
    #    model=model,
    #    allow_greater=True,
    # )
