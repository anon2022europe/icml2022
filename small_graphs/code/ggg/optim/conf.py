import math
from logging import warning
from typing import Optional, Dict

import attr
import torch
from attr.validators import in_
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from ggg.models.components.abstract_conf import AbstractConf
from ggg.optim.debug_adam import DebugAdam
from ggg.optim.optim import ExtraAdam
from ggg.utils.utils import kwarg_create
from ggg.warmup import GradualWarmupScheduler


@attr.s
class OptSchedHpars(AbstractConf):
    name = attr.ib(default="linear")
    # for linear
    max_epochs = attr.ib(default=100)
    epoch_batches = attr.ib(default=None)
    # for exponential/step
    reduce_every = attr.ib(default=100)
    lr_gamma = attr.ib(default=0.1)
    interval = attr.ib(default="epoch")
    warmup = attr.ib(default=None)

    def make(self, opt: torch.optim.Optimizer, parent_opt_hpars: "OptHpars"):
        if self.name == "step":
            sched = StepLR(opt, step_size=self.reduce_every, gamma=self.lr_gamma)
        elif self.name == "linear":
            raise NotImplementedError()
        else:
            raise NotImplementedError()
        if self.warmup == "recommended":
            # https://arxiv.org/abs/1910.04209
            disc_warm_factor = int(
                math.ceil(2 / (1 - parent_opt_hpars.betas[1]))
            )  # "learning period...just use epoch"
            disc_warm_ep = disc_warm_factor

            gen_warm_factor = int(
                math.ceil(2 / (1 - parent_opt_hpars.betas[1]))
            )  # "learning period...just use epoch"
            gen_warm_ep = gen_warm_factor
            sched = GradualWarmupScheduler(
                opt,
                multiplier=1.0,
                total_epoch=self.max_epochs,
                after_scheduler=sched,
            )
        elif self.warmup:
            # use "recommended" to use 2/1-\beta2 as in https://arxiv.org/abs/1910.04209
            sched = GradualWarmupScheduler(
                opt,
                multiplier=1.0,
                total_epoch=self.max_epochs,
                after_scheduler=sched,
            )
        return sched


@attr.s
class OptHpars(AbstractConf):
    name = attr.ib(default="adam", validator=in_({"adam", "extra_adam"}))
    lr = attr.ib(default=1e-5)
    betas = attr.ib(default=(0.5, 0.9999))
    weight_decay = attr.ib(default=1e-4)
    eps = attr.ib(default=1e-8)

    ema = attr.ib(default=False)
    ema_decay = attr.ib(default=0.9999)
    ema_start = attr.ib(default=100)

    sched = attr.ib(default=None, type=Optional[OptSchedHpars])
    every = attr.ib(default=1)
    amsgrad = attr.ib(default=True)

    @classmethod
    def children(cls) -> Dict:
        return dict(scheduler=OptSchedHpars)

    def make(self, params):
        kwargs = self.to_dict()
        kwargs["params"] = params
        if self.name == "adam":
            opt = kwarg_create(DebugAdam, kwargs)
        elif self.name == "extra_adam":
            warning("Fix extra adam")
            opt = kwarg_create(ExtraAdam, kwargs)
        else:
            raise NotImplementedError(f"Not implemented {self.name}")
        return opt

    @classmethod
    def disc_default(cls):
        # ema motivated by https://arxiv.org/pdf/1806.04498.pdf
        # recommended setting from WGAN-GP/optimistic Adam:wq paper, half the learning rate tho
        return cls(
            name="adam",
            lr=3e-4,
            betas=(0.0, 0.9),
            weight_decay=1e-3,
            eps=1e-8,
        )

    @classmethod
    def gen_default(cls):
        return cls(
            name="adam",
            lr=1e-4,
            betas=(0.0, 0.9),
            weight_decay=1e-3,
            eps=1e-8,
        )
