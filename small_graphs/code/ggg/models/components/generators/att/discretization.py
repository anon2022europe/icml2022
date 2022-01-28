from inspect import signature

import attr
import torch
from attr.validators import in_
from torch import nn as nn, distributions as td

from ggg.models.components.abstract_conf import AbstractConf
from ggg.utils.utils import zero_diag, kwarg_create, zero_and_symmetrize


class ClampingDiscretize(nn.Module):
    def __init__(self, straight_through=False):
        super(ClampingDiscretize, self).__init__()
        self.straight_through = straight_through

    def forward(self, A):
        if self.straight_through:
            Agrad = A - A.detach()
            A = A.detach()
        Ac = A.clamp(0, 1)
        if self.straight_through:
            Ac = Ac + Agrad
        Az = zero_diag(Ac)
        Atriu = torch.triu(Az)
        A = Atriu + Atriu.permute(0, 2, 1)
        return A


class RoundingDiscretize(nn.Module):
    def __init__(self, straight_through=False):
        super(RoundingDiscretize, self).__init__()
        self.straight_through = straight_through

    def forward(self, A):
        if self.straight_through:
            Agrad = A - A.detach()
            A = A.detach()
        Ar = torch.round(A.clamp(0, 1))
        if self.straight_through:
            Ar = Ar + Agrad
        Az = zero_diag(Ar)
        Atriu = torch.triu(Az)
        A = Atriu + Atriu.permute(0, 2, 1)
        return A


class BernoulliDiscretize(nn.Module):
    def __init__(self, hard=False, temperature=0.66):
        super(BernoulliDiscretize, self).__init__()
        self.hard = hard
        self.temperature = temperature

    def forward(self, A):
        relaxedA = td.RelaxedBernoulli(self.temperature, probs=A).rsample()
        # hard relaxed bernoulli= create 0 vector with gradients attached, add to rounded values
        if self.hard:
            grads_only = relaxedA - relaxedA.detach()
            # adding grads might not be necessary?
            # Ar = relaxedA.round().detach() + grads_only TODO: revert/reveisit
            # 100% match to anons working code
            Ar = relaxedA.round() + grads_only
        else:
            Ar = relaxedA
        # rezero and resymmetrize
        A = zero_and_symmetrize(Ar)
        return A


class NoDiscretization(nn.Module):
    def __init__(self):
        super(NoDiscretization, self).__init__()

    def forward(self, A):
        A = zero_and_symmetrize(A)
        return A


@attr.s
class DiscretizationHpars(AbstractConf):
    OPTIONS = dict(
        bernoulli=BernoulliDiscretize,
        rounding=RoundingDiscretize,
        clamp=ClampingDiscretize,
        ident=NoDiscretization,
    )
    name = attr.ib(default="bernoulli", validator=in_(OPTIONS))
    hard = attr.ib(default=True)
    straight_through = attr.ib(default=False)
    temperature = attr.ib(default=0.66)

    def make(self):
        cls = DiscretizationHpars.OPTIONS[self.name]
        kwargs = attr.asdict(self)
        return kwarg_create(cls, kwargs)
