import attr
import torch
import torch as pt
from attr.validators import in_

from ggg.models.components.generators.att.models import Generator
from ggg.models.components.abstract_conf import AbstractConf


@attr.s
class JNF:
    def jnf_rejection_sample(
        self, X, A, Z, ctx, Q, given_N=None, batch_size=None, device=None
    ):
        with torch.no_grad():
            perturbed_samples = [
                self._inner_sample(
                    batch_size=batch_size,
                    given_N=given_N,
                    device=device,
                    external_context_vec=ctx + pt.randn_like(ctx) * self.jnf.eps,
                    in_jnf=True,
                )
                for _ in range(self.jnf.N)
            ]
            base_logits = self.discriminator.forward(X, A, mode="graph")
            sample_logits = [
                self.discriminator.forward(x[0], x[1], mode="graph")
                for x in perturbed_samples
            ]
            dist_est = pt.stack(
                [
                    pt.norm(s - base_logits, p="fro", dim=s.shape[1:])
                    for s in sample_logits
                ],
                -1,
            ).mean(-1)
            assert dist_est.dim() == 1
            _, indices = pt.topk(
                dist_est, k=int(batch_size * self.jnf.ratio), largest=False
            )
        # select from indices and make sure we still have batch dim
        Xsel, Asel, Zsel, ctxsel, Qsel = [
            x[indices].reshape(-1, *x.shape[1:]) for x in [X, A, Z, ctx, Q]
        ]
        assert Xsel.dim() == X.dim()
        return Xsel, Asel, Zsel, ctxsel, Qsel


class SimpleSampler(pt.nn.Module):
    def forward(
        self,
        generator: Generator,
        batch_size=None,
        Z=None,
        X=None,
        A=None,
        N=None,
        device=None,
    ):
        X, A, N, Z = generator.forward(batch_size, Z=Z, X=X, A=A, N=N, device=device)
        return X, A, N, Z


@attr.s
class SamplerHpars(AbstractConf):
    OPTIONS = dict(simple=SimpleSampler)
    name = attr.ib(default="simple", validator=in_(OPTIONS))
