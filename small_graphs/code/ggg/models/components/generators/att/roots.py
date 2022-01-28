import abc
import math
from inspect import signature
from typing import Tuple, Optional

import attr
import torch as pt
from attr.validators import in_
from ipdb import set_trace
from torch import nn as nn

import numpy as np
from torch.nn.modules.module import T

from ggg.models.components.abstract_conf import AbstractConf
from ggg.utils.utils import zero_mask_nodes, kwarg_create, ensure_tensor, pdf


def get_dist(dist):
    if isinstance(dist, pt.distributions.Distribution):
        return dist
    elif dist == "normal":
        return pt.distributions.Normal(0.0, 1.0)
    else:
        raise ValueError(f"Don't know dist {dist}")


class GeneratorRoot(nn.Module):
    def __init__(
        self,
        n_node_dist: pt.distributions.Categorical,
        node_embedding_dim=32,
        noise_dist="normal",
    ):
        super(GeneratorRoot, self).__init__()
        self.node_n_dist = n_node_dist
        self.noise_dist = get_dist(noise_dist)
        self.max_N = int(self.node_n_dist.param_shape[0])
        self.node_embedding_dim = node_embedding_dim

    @abc.abstractmethod
    def forward(
        self, batch_size, device, Z=None, N=None
    ) -> Tuple[pt.Tensor, Optional[pt.Tensor], Optional[pt.Tensor], pt.Tensor]:
        """

        Parameters
        ----------
        batch_size
        device

        Returns
        -------

        """

    def sample(self, batch_size=1, device=None, Z=None, N=None):
        return self.forward(batch_size, device, Z, N)


class RandomRoot(GeneratorRoot):
    def forward(self, batch_size, device, Z=None, N=None):
        if N is None:
            N = self.node_n_dist.sample([batch_size]).to(device) + 1
        if Z is None:
            Z = self.noise_dist.sample(
                [batch_size, self.max_N, self.node_embedding_dim]
            )
        Zm = zero_mask_nodes(Z, N)
        return Zm, None, N, Z


class RandomJoinedRoot(GeneratorRoot):
    def __init__(
        self,
        n_node_dist: pt.distributions.Categorical,
        node_embedding_dim=32,
        context_dim=32,
        noise_dist="normal",
        joined_noise_dist="normal",
    ):
        super().__init__(n_node_dist, node_embedding_dim, noise_dist)
        self.joined_noise_dist = joined_noise_dist
        self.context_dim = context_dim

    def forward(self, batch_size, device, Z=None, N=None):
        if N is None:
            N = self.node_n_dist.sample([batch_size]).to(device) + 1
        if Z is None:
            Z = self.noise_dist.sample(
                [batch_size, self.max_N, self.node_embedding_dim]
            )
            joined_embed: pt.Tensor = self.joined_noise_dist.sample(
                [batch_size, 1, self.context_dim]
            )
            Z = pt.cat([Z, joined_embed.expand(-1, self.max_N, -1)])
        noise = zero_mask_nodes(Z, N)
        return noise, None, N, Z


class GeometricRoot(GeneratorRoot):
    def __init__(
        self,
        n_node_dist: pt.distributions.Categorical,
        node_embedding_dim=16,
        context_dim=16,
        embedding_batch_size=1,
        allow_greater=False,
        trainable=True,
        noise_dist="normal",
        joined_noise_dist="normal",
        extra_features=0,
    ):
        super().__init__(n_node_dist, node_embedding_dim, noise_dist)
        self.joined_noise_dist = get_dist(joined_noise_dist)
        self.context_dim = context_dim
        self.embedding_batch_size = embedding_batch_size
        self.allow_greater = allow_greater
        self.extra_features = extra_features
        self.node_embeddings = pt.nn.Parameter(
            self.noise_dist.sample(
                [embedding_batch_size, self.max_N, self.node_embedding_dim]
            ),
            requires_grad=trainable,
        )
        self.trainable = trainable

    def forward(self, batch_size, device, Z=None, N=None):
        if not self.trainable:
            # always turn this of in case it gets turned on by accident
            self.node_embeddings.requires_grad = False
        if N is None:
            N = self.node_n_dist.sample([batch_size]).to(device) + 1
        if Z is None:
            noise = self.node_embeddings.clone()
            if batch_size > self.embedding_batch_size:
                if self.allow_greater:
                    repetition = int(
                        math.ceil(batch_size / float(self.embedding_batch_size))
                    )
                    noise = noise.repeat(repetition, 1, 1)[:batch_size]
                else:
                    raise ValueError(
                        f"Requested batch size {batch_size} with fixed embedding size {self.embedding_batch_size} and no expansion allowed"
                    )
            else:
                noise = noise[:batch_size]

            joined_embed: pt.Tensor = self.joined_noise_dist.sample(
                [batch_size, 1, self.context_dim]
            ).to(noise.device)
            # B 1 CF
            Z = pt.cat([noise, joined_embed.expand(-1, self.max_N, -1)], -1)
            # B N CF
        N=N.to(Z.device)
        if self.extra_features:
            node_mask = (
                (pt.ones(Z.shape[0], Z.shape[1], 1, device=Z.device).cumsum(dim=1) <= self.max_N)
                    .float()
                    .max(-1, keepdim=True)
                    .values
            )
            N_app = node_mask * pt.tensor(self.max_N).reshape(-1, 1, 1).repeat([1, Z.shape[1], 1])
            Z = pt.cat([Z, N_app], dim=-1)

        noise = zero_mask_nodes(Z, N)
        return noise, None, N, Z


class FixedContextRoot(GeneratorRoot):
    def __init__(
        self,
        n_node_dist: pt.distributions.Categorical,
        node_embedding_dim=32,
        context_dim=32,
        embedding_batch_size=1,
        allow_greater=False,
        trainable=True,
        noise_dist=pt.distributions.Normal(0.0, 1.0),
        joined_noise_dist=pt.distributions.Normal(0.0, 1.0),
    ):
        super().__init__(n_node_dist, node_embedding_dim, noise_dist)
        self.context_noise_dist = get_dist(joined_noise_dist)
        self.context_dim = context_dim
        self.embedding_batch_size = embedding_batch_size
        self.allow_greater = allow_greater
        self.fixed_context = pt.nn.Parameter(
            self.context_noise_dist.sample([embedding_batch_size, 1, self.context_dim]),
            requires_grad=trainable,
        )
        self.trainable = trainable

    def train(self, mode: bool = True):
        ret = super().train(mode)
        self.fixed_context.requires_grad = self.trainable
        return ret

    def forward(self, batch_size, device, Z=None, N=None):
        if self.training:
            assert self.fixed_context.requires_grad == self.trainable
        if N is None:
            N = self.node_n_dist.sample([batch_size]).to(device) + 1
        if Z is None:
            context = self.fixed_context.clone()
            if batch_size > self.embedding_batch_size:
                if self.allow_greater:
                    context.expand(batch_size, -1, -1)
                else:
                    raise ValueError(
                        f"Requested batch size {batch_size} with fixed embedding size {self.embedding_batch_size} and no expansion allowed"
                    )
            else:
                context = context[:batch_size]

            noise: pt.Tensor = self.context_noise_dist.sample(
                [batch_size, self.max_N, self.context_dim]
            )
            Z = pt.cat([noise, context.expand(-1, self.max_N, -1)])
        noise = zero_mask_nodes(Z, N)
        return noise, None, N, Z


@attr.s
class GenRootHpars(AbstractConf):
    OPTIONS = dict(
        fixed_context=FixedContextRoot,
        random=RandomRoot,
        random_joind=RandomJoinedRoot,
        geometric=GeometricRoot,
    )
    name = attr.ib(default="geometric", validator=in_(OPTIONS.keys()))
    noise_dist = attr.ib(default="normal")  # dist or normal
    joined_noise_dist = attr.ib(default="normal")  # dist or normal
    node_embedding_dim = attr.ib(default=16)
    context_dim = attr.ib(default=16)
    embedding_batch_size = attr.ib(default=1)
    allow_greater = attr.ib(default=True)
    trainable = attr.ib(default=True)
    extra_features = attr.ib(default=0)

    def make(self, n_node_weights: pt.Tensor):
        cls = GenRootHpars.OPTIONS[self.name]
        kwargs = attr.asdict(self)
        kwargs["n_node_dist"] = pt.distributions.Categorical(
            ensure_tensor(n_node_weights)
        )
        return kwarg_create(cls, kwargs)
