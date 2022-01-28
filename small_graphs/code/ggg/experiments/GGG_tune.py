import os
from itertools import chain
from pathlib import Path

import optuna

from ggg.models.ggg_model import (
    GGG,
    GGG_Hpar,
)
from pytorch_lightning import Trainer, Callback

from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_logger import LearningRateLogger
import torch as pt
import attr


class WGANPrun(Callback):
    def __init__(self, trial, batch_size):
        super().__init__()
        self.trial = trial
        self.batch_size = batch_size

    def on_batch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "W1" in metrics:
            FoM = metrics["W1"] - (metrics["fake_score"] + metrics["real_score"]) / 2
            self.trial.report(FoM, trainer.total_batch_idx * self.batch_size)
            if self.trial.should_prune():
                raise optuna.TrialPruned()


def run(trial: optuna.Trial):
    parent_dir = str(str(Path(os.path.abspath(__file__)).parents[2]))

    hyper_raw = GGG_Hpar()
    data_dir = parent_dir + "/ggg/data/QM9/"

    epochs = 300
    model_n = "DeepAttQQ_D_AXreadout"
    filename = "QM9_5000.sparsedataset"
    exp_name = "/ggg/experiments/"
    overfit_pct = 0.1
    gpus = [1]
    ckpt_period = 10
    detect_anomaly = False
    deep = False
    deep_disc = False
    deep_gen = False

    node_dist_weights = pt.tensor(
        [
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            0.0000,
            6.00720865e-04,
            9.99399279e-01,
        ]
    )  # computed on 5000 QM9 dataset

    # start optuna
    hyper_raw: GGG_Hpar

    lr = trial.suggest_categorical("lrs", [(1e-4, 1e-4), (3e-4, 1e-4)])
    betars = trial.suggest_categorical("betas", [(0.5, 0.9999), (0, 0.9)])
    wdec = trial.suggest_categorical("weight_decay", [0.0, 1e-3])
    gp = trial.suggest_categorical("GP", ["ZP", True, False])
    hyper_raw.LP = gp
    am = trial.suggest_categorical("attention_mode", ["QK", "QQ"])
    er = trial.suggest_categorical(
        "edge_readout",
        [
            ("biased_sigmoid", "scalar"),
            ("biased_sigmooid", "nodes"),
            ("rescaled_softmax", "add"),
            ("rescaled_softmax", "mult"),
            ("rescaled_softmax", False),
        ],
    )
    hyper_raw.attention_mode = am
    hyper_raw.edge_readout = er[0]
    hyper_raw.edge_bias_mode = er[1]
    if er[1]:
        hyper_raw.edge_bias_hidden = trial.suggest_int(
            "edge_bias_hdim", low=64, high=256, step=64
        )
    hyper_raw.generator_every = trial.suggest_int("generator_every", low=1, high=10)
    hyper_raw.disc_spectral_norm = trial.suggest_categorical(
        "disc_sn", [None, "nondiff", "diff"]
    )
    dw = trial.suggest_int("disc_width", 32, 256, 32)
    hyper_raw.disc_conv_channels = [
        dw for _ in range(trial.suggest_int("disc_depth", 2, 6))
    ]
    hyper_raw.n_attention_layers = trial.suggest_int("gen_depth", 2, 6)
    hyper_raw.deep_gen_out_act = trial.suggest_categorical(
        "gen_act", [None, "swish", "relu"]
    )
    hyper_raw.disc_readout_hidden = trial.suggest_int(
        "disc_readout_hidden", 32, 256, 32
    )
    hyper_raw.disc_optim_args = dict(
        lr=lr[0], betas=betars, eps=1e-8, weight_decay=wdec, ema=False, ema_start=100
    )  # recommended setting from WGAN-GP/optimistic Adam:wq paper, half the learning rate tho
    hyper_raw.n_optim_args = dict(
        lr=lr[1], betas=betars, eps=1e-8, weight_decay=wdec, ema=False, ema_start=100
    )  # recommended setting from WGAN-GP/optimistic Adam:wq paper, half the learning rate tho
    hyper_raw.disc_dropout = trial.suggest_categorical("dropout", [None, 0.1, 0.2, 0.3])
    hyper_raw.temperature = trial.suggest_loguniform("temperature", 2 / 3, 5.0)
    hyper_raw.penalty_lambda = trial.suggest_uniform("penalty_lambda", 5, 100)
    hyper_raw.embed_dim = trial.suggest_int("embed_dim", 7, 31)
    hyper_raw.num_workers = 0
    # end optuna
    pyl_log_dir = os.path.join(exp_name, "lightning_log")
    old_hyper = attr.asdict(hyper_raw)
    changes = dict(
        node_count_weights=node_dist_weights,
        data_dir=data_dir,
        filename=filename,
        base_dir=".",
        model_n=model_n,
        deep=deep,
        deep_disc=deep_disc,
        deep_gen=deep_gen,
        save_dir=pyl_log_dir,
    )
    # finalize updated hparams
    hparams = {
        k: changes[k] if k in changes else old_hyper[k]
        for k in chain(old_hyper.keys(), changes.keys())
    }

    model = GGG(hparams)
    try:
        # try to use a logger with a nicer folder structure
        from pytorch_lightning.loggers import TestTubeLogger

        tblogger = TestTubeLogger(exp_name[1:-1], model_n)
    except:
        tblogger = TensorBoardLogger(exp_name[1:-1], model_n)

    checkpoint_callback = ModelCheckpoint(
        save_top_k=-1, period=ckpt_period, verbose=True
    )
    lr_logger = LearningRateLogger()

    trainer = Trainer(
        progress_bar_refresh_rate=1,
        max_epochs=epochs + 1,
        # track_grad_norm=2, works, but needs a small fix to pytorch lightning (should use 0 tensor, not float)
        logger=tblogger,
        log_save_interval=1,
        # checkpoint_callback=checkpoint_callback,
        overfit_pct=overfit_pct,
        num_nodes=1,
        gpus=gpus,
        callbacks=[lr_logger, WGANPrun(trial, hyper["batch_size"])],
    )
    with pt.autograd.set_detect_anomaly(detect_anomaly):
        trainer.fit(model)
    dl = model.train_dataloader()
    model: PEAWGANDeep
    W1 = 0.0
    fs = 0.0
    rs = 0.0
    count = 0
    for realX, realA in dl:
        fake_nodes, fake_adj, fake_score, real_score = model.forward(
            real_data=(realX, realA)
        )
        W1 = W1 + real_score - fake_score
        fs = fs + fake_score
        rs = rs + real_score
        count += 1
    W1 /= count
    fs /= count
    rs /= count
    FoM = W1 - (fs + rs) * 0.5
    return FoM


if __name__ == "__main__":
    study_name = "pewagan_tune"  # Unique identifier of the study.
    pruner = optuna.pruners.MedianPruner()
    study = optuna.create_study(
        study_name=study_name,
        storage="sqlite:///example.db",
        direction="minimize",
        pruner=pruner,
    )
    study.optimize(run, timeout=3600, n_jobs=5)  # in seconds
