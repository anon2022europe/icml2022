from typing import List, Tuple

import attr
import torch as pt
import logging

from ggg.models.components.abstract_conf import AbstractConf
from ggg.utils.utils import kwarg_create, maybe_assert, asserts_enabled


def aggregate_tensor(pens, agg="mean"):
    if type(pens) is list:
        pens = pt.stack(pens)
    if agg == "lse":
        new_penalty = pens.flatten().logsumexp(-1)
    elif agg == "max":
        new_penalty = pens.max()
    elif agg == "mean":
        new_penalty = pens.mean()
    elif agg == "sum":
        new_penalty = pens.sum()
    else:
        raise NotImplementedError(f"Don't know agg {agg}")
    return new_penalty


def perturb_adj(A, percentage):
    A_triu = pt.triu(A)
    # roll an erasure dice on every node
    erase_chance = pt.rand_like(A_triu) * A_triu
    erased = percentage > erase_chance
    # mask out erased nodes
    erased_neg = erased.logical_not()
    A_triu = A_triu * erased_neg
    # roll a dice for adding an edge
    # roll a dice on all empty edges which aren't empty because we erased them
    add_chance = pt.rand_like(erase_chance)
    added = (percentage > add_chance) * (1 - A_triu).abs() * erased_neg
    # add the new connections
    A_triu = A_triu + added
    # resymmetriue
    A = A_triu + A_triu.permute(0, 1, -1, -2)
    return A




class GradPenalty(pt.nn.Module):
    _SUPPORTED = {"GP", "LP", "ZP", "maxP", "maxPlse", "simple"}  # penalty methods
    _ON = {
        "int",
        "rec-row",
        "real",
        "fake",
        "real-perturbed",
        "fake-perturbed",
    }  # what to penalize

    def __init__(
        self,
        modes=None,
        weights=None,
        input_agg="mean",
        on=None,
        sample_agg="mean",
        penalty_agg="mean",
        perturbation_percentage=0.3,
        squash_nans=False
    ):
        super(GradPenalty, self).__init__()
        # default: original WGAN-GP, on interpolated fake/real samples
        self.perturbation_percentage = perturbation_percentage
        if modes is None:
            modes = ["GP"]
        if on is None:
            on = ["int" for _ in modes]
        if weights is None:
            weights = [1.0 for _ in modes]
        assert len(weights) == len(modes) == len(on)
        self.squash_nans=squash_nans
        self.weights = weights
        self.on = on
        self.modes = modes
        self.input_agg = input_agg
        self.sample_agg = sample_agg
        self.penalty_agg = penalty_agg

    def forward(self, D, *pairs: List[Tuple[pt.Tensor, pt.Tensor]]):
        # assumes ordering [(fake1,real1),(fake2,real2)] to be fed into discrimantor as D(*[t1,t2,t3])
        penalties = []
        for mode, on, w in zip(self.modes, self.on, self.weights):
            if mode == "simple":
                # ignore on,input_agg in this instance, since we are doing a *very* crude estimation
                real = self.get_penalty_target("real", *pairs)
                fake = self.get_penalty_target("fake", *pairs)
                diff_Dx = pt.norm(1e-5+(D(*real) - D(*fake)), "fro")
                diff_x = pt.stack(
                    [pt.norm(r - f+1e-5, "fro") for r, f in zip(real, fake)]
                ).min()  # smallest change=> overestimate a bit
                penalty = diff_Dx / (diff_x + 1e-9)
            else:
                d_inputs = self.get_penalty_target(on, *pairs)
                d_out = D(*d_inputs)
                gradients = get_gradients(d_out, *d_inputs)
                if self.squash_nans:
                    for g in gradients:
                        nans=pt.isfinite(g)==False
                        if nans.any():
                            logging.warning(f"Squashing {len(nans.nonzero())} nans in gradpen")
                            g[nans]=0.0

                # Get gradient w.r.t. target specified by on
                pens = penalty_from_gradient(gradients, mode, self.sample_agg)
                # aggregate penalty w.r.t to different input tensors
                penalty = aggregate_tensor(pens, self.input_agg)
                if asserts_enabled():
                    if not pt.isfinite(penalty).all():
                        d_inputs2=[]
                        for x in d_inputs:
                            if x is None:
                                y =None
                            else:
                                y=x
                                if not getattr(y,"requires_grad",False) :
                                    y.requires_grad=True
                            d_inputs2.append(y)
                        with pt.autograd.detect_anomaly():
                            nd_out = D(*d_inputs2)
                            ngradients = get_gradients(nd_out, *d_inputs2)
                            # Get gradient w.r.t. target specified by on
                            npens = penalty_from_gradient(ngradients, mode, self.sample_agg)
                            # aggregate penalty w.r.t to different input tensors
                            npenalty = aggregate_tensor(npens, self.input_agg)
                            npenalty.mean().backward()
                    maybe_assert(func=lambda:pt.isfinite(penalty).all())

                # weight this penalty in the total sum
            penalties.append(penalty * w)

        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            logging.debug(
                f"Pe-agg penalties \n{'-'.join([str(m)+str(o) for m,o in zip(self.modes,self.on)])}\n{'-'.join([f'{p}' for p in penalties])}"
            )
        penalty = aggregate_tensor(penalties, self.penalty_agg)
        return penalty

    def get_penalty_target(self, on, *pairs):
        # expects pairs to be (fakeA realA) (fakeB realB) etc
        if on == "int":
            return convex_combination(*pairs)
        elif on == "rec-row":
            # only works for graph gan but works with API so poka
            return recombine_rows(*pairs)
        elif on == "fake":
            fakes = [o[0].clone() if o[0] is not None else o[0] for o in pairs]
            for f in fakes:
                if f is not None:
                    f.requires_grad = True
            return fakes
        elif on == "real":
            reals = [o[1] for o in pairs]
            for r in reals:
                if r is not None:
                    r.requires_grad = True
            return reals
        elif on == "fake-perturbed":
            fakes = [o[0].clone() if o[0] is not None else o[0] for o in pairs]
            # assuming 2nd element in pair is adj
            fakes[1] = perturb_adj(fakes[1], self.perturbation_percentage)
            for f in fakes:
                if f is not None and not f.requires_grad:
                    f.requires_grad = True
            return fakes
        elif on == "real-perturbed":
            reals = [o[1] for o in pairs]
            reals[1] = perturb_adj(reals[1], self.perturbation_percentage)
            for r in reals:
                if r is not None and not r.requires_grad:
                    r.requires_grad = True
            return reals
        else:
            raise NotImplementedError(f"Unkown target {on}")


def compute_gradient_penalty(
    D,
    real_samples,
    fake_samples,
    LP=True,
    onreal=False,
    onfake=False,
    add_recombined=True,
    mode="interpolate_Adj",
    penalty_agg="sum",
):
    """Calculates the gradient penalty loss for WGAN GP, adapt for WGAN-LP https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/wgan_gp/wgan_gp.py"""
    # Random weight term for interpolation between real and fake samples
    realX, realA = real_samples
    fakeX, fakeA = fake_samples
    # Get random interpolation between real and fake samples
    if mode == "interpolate_Adj":
        penalty = interpolated_grad_pen(
            D, realX, realA, fakeX, fakeA, LP=LP, penalty_agg=penalty_agg
        )
    elif mode == "avg_grads":
        penalty = grad_avg_grad_pen(
            D, realX, realA, fakeX, fakeA, LP=LP, penalty_agg=penalty_agg
        )
    if add_recombined:
        rec_penalty = recombined_grad_pen(
            D, realX, realA, fakeX, fakeA, LP=LP, penalty_agg=penalty_agg
        )
    else:
        rec_penalty = 0.0
    fake_pen = (
        simple_grad_pen(D, fakeX, fakeA, LP=LP, penalty_agg=penalty_agg)
        if onfake
        else 0.0
    )
    real_pen = (
        simple_grad_pen(D, realX, realA, LP=LP, penalty_agg=penalty_agg)
        if onreal
        else 0.0
    )
    penalty = penalty + fake_pen + real_pen + rec_penalty
    return penalty


def recombine_rows(a1, a2, p=0.3, x1=None, x2=None):
    """

    Recombine two graphs in a *discrete* way, by recombining nodes and their connectivities
    p: probability of taking a row from a1
    """
    if a1.dim() == 2:
        from_1 = pt.rand([a1.shape[-2]], device=a1.device, dtype=a1.dtype) >= p
        idx_r = from_1.nonzero(as_tuple=True)
        (r,) = idx_r
        idx = (r, r)
    elif a1.dim() == 3:
        from_1 = (
            pt.rand([a1.shape[0], a1.shape[-2]], device=a1.device, dtype=a1.dtype) >= p
        )
        idx_r = from_1.nonzero(as_tuple=True)
        (
            b,
            r,
        ) = idx_r
        idx = (b, r, r)
    elif a1.dim() == 4:
        B, pac, N, _ = a1.shape
        if p.dim() == 2:
            p = p.unsqueeze(1).repeat([1, pac, 1])
        from_1 = pt.rand([B, pac, N], device=a1.device, dtype=a1.dtype) >= p
        idx_r = from_1.nonzero(as_tuple=True)
        (
            b,
            p,
            r,
        ) = idx_r
        idx = (b, p, r, r)
    else:
        raise NotImplementedError(
            f"Don't know how to deal with A shape {a1.shape} {a2.shape}"
        )
    # in place shit still gives errors with gradients I think
    # new_a=a2.clone()
    # new_a[idx]=a1[idx]
    a1_sel = pt.zeros_like(a2)
    a1_sel[idx] = 1.0
    a2_sel = pt.ones_like(a1_sel) - a1_sel
    new_a = a1 * a1_sel + a2_sel * a2
    if x1 is not None:
        # in place shit still gives errors with gradients I think
        # new_x=x2.clone()
        # new_x[idx]=x1[idx]
        x1_sel = pt.zeros_like(x2)
        x1_sel[idx_r] = 1.0
        x2_sel = pt.ones_like(x1_sel) - x1_sel
        new_x = x1 * x1_sel + x2_sel * x2
        return new_a, new_x
    else:
        return new_a


def recombined_grad_pen(D, realX, realA, fakeX, fakeA, LP=False, penalty_agg="sum"):
    old_reqs = []
    for x in [fakeX, fakeA, realX, realA]:
        old_reqs.append(x.requires_grad)
        x.requires_grad = True
    if realX.dim() == 3:
        alpharow = pt.rand(realX.shape[0], realX.shape[1], 1, device=fakeA.device)
    else:
        alpharow = pt.rand(realX.shape[0], 1, device=fakeA.device)
    recA, recX = recombine_rows(fakeA, realA, alpharow, x2=realX, x1=fakeX)

    d_interpolates = D(recX, recA)
    gradients = get_gradients(d_interpolates, recA, recX)
    assert len(gradients) == 2
    # Get gradient w.r.t. interpolates
    penalty = 0.0
    penalty = pen_from_grad(LP, gradients, penalty, penalty_agg=penalty_agg)
    for i, x in enumerate([fakeX, fakeA, realX, realA]):
        x.requires_grad = old_reqs[i]
    return penalty


def interpolated_grad_pen(D, realX, realA, fakeX, fakeA, LP=False, penalty_agg="sum"):
    interpolateA, interpolateX = convex_combination((fakeA, realA), (fakeX, realX))

    d_interpolates = D(interpolateX, interpolateA)
    gradients = get_gradients(d_interpolates, interpolateA, interpolateX)
    assert len(gradients) == 2
    # Get gradient w.r.t. interpolates
    penalty = 0.0
    penalty = pen_from_grad(LP, gradients, penalty, penalty_agg=penalty_agg)
    return penalty


def convex_combination(*pairs):
    # expects pairs to be (fakeA realA) (fakeB realB) etc
    interpolated = []
    for (fake, real) in pairs:
        if fake is not None and real is not None:
            alpha = pt.rand_like(real).type_as(real)
            interpolate = (real * alpha + ((1.0 - alpha) * fake)).requires_grad_(True)
            interpolated.append(interpolate)
        else:
            interpolated.append(None)
    return interpolated


def grad_avg_grad_pen(D, realX, realA, fakeX, fakeA, LP=False, penalty_agg="sum"):
    old_reqs = []
    for x in [fakeX, fakeA, realX, realA]:
        old_reqs.append(x.requires_grad)
        x.requires_grad = True
    d_out_real = D(realX, realA)
    d_out_fake = D(fakeX, fakeA)
    gradients_real = get_gradients(d_out_real, realX, realA)
    gradients_fake = get_gradients(d_out_fake, fakeX, fakeA)
    gradients = [
        0.5 * (g_real + g_fake)
        for g_real, g_fake in zip(gradients_real, gradients_fake)
    ]
    assert len(gradients) == 2
    # Get gradient w.r.t. interpolates
    penalty = 0.0
    penalty = pen_from_grad(LP, gradients, penalty, penalty_agg=penalty_agg)
    for i, x in enumerate([fakeX, fakeA, realX, realA]):
        x.requires_grad = old_reqs[i]
    return penalty


def simple_grad_pen(D, X, A, LP=False, penalty_agg="sum"):
    old_reqs = []
    for x in [X, A]:
        old_reqs.append(x.requires_grad)
        x.requires_grad = True
    d_out_real = D(X, A)
    gradients = get_gradients(d_out_real, X, A)
    assert len(gradients) == 2
    # Get gradient w.r.t. interpolates
    penalty = 0.0
    penalty = pen_from_grad(LP, gradients, penalty, penalty_agg=penalty_agg)
    for i, x in enumerate([X, A]):
        x.requires_grad = old_reqs[i]
    return penalty


def pen_from_grad(pen_type, gradients, penalty, penalty_agg="sum"):
    old_penalty = penalty
    pens = penalty_from_gradient(gradients, pen_type)
    if penalty_agg == "lse":
        penalty = old_penalty + pt.stack(pens).flatten().logsumexp(-1)
    elif penalty_agg == "max":
        penalty = old_penalty + pt.stack(pens).max()
    elif penalty_agg == "mean":
        penalty = old_penalty + pt.stack(pens).mean()
    elif penalty_agg == "sum":
        penalty = old_penalty + pt.stack(pens).sum()
    else:
        raise NotImplementedError(f"Don't know agg {penalty_agg}")
    return penalty


def penalty_from_gradient(gradients, pen_type, sample_agg="mean"):
    pens = []
    for grads in gradients:
        grads = grads.reshape(grads.size(0), -1)
        # add a small offset for sqrt shit..
        grads=grads+1e-5
        if pen_type == "ZP":
            # from https://openreview.net/forum?id=ByxPYjC5KQ
            diff = grads.norm(2, dim=1)
            penalty_samples = diff ** 2
        elif pen_type == "maxP":
            # https://github.com/ZhimingZhou/LGANs-for-reproduce/blob/master/code/gan_synthetic4.py
            penalty_samples = grads.pow(2.0).max(-1).values
        elif pen_type == "maxPlse":
            diff = grads.norm(2, dim=1).flatten().logsumexp(-1)
            penalty_samples = diff ** 2
        elif pen_type == "LP" or pen_type is True:
            diff = grads.norm(2, dim=1) - 1
            penalty_samples = pt.max(diff, pt.zeros_like(diff)) ** 2
        elif pen_type == "GP" or pen_type is False:
            penalty_samples = (grads.norm(2, dim=1) - 1) ** 2
        else:
            raise NotImplementedError(f"Don't know penalty {pen_type}")
        penalty = aggregate_tensor(penalty_samples, sample_agg)
        pens.append(penalty)
    return pens


def get_gradients(discriminator_output, *discriminator_inputs: pt.Tensor):
    grads = [
        pt.ones(x.shape[0], 1, requires_grad=False).type_as(x)
        if x.dim() > 0
        else pt.ones([], requires_grad=False)
        for x in [discriminator_output]
    ]
    # Get gradient w.r.t. interpolates
    discriminator_inputs = [x for x in discriminator_inputs if x is not None]
    gradients = pt.autograd.grad(
        outputs=discriminator_output,
        inputs=discriminator_inputs,
        grad_outputs=grads,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    maybe_assert(func=lambda: pt.isfinite(discriminator_output).all())
    for x in discriminator_inputs:
        if x is not None:
            maybe_assert(func=lambda:pt.isfinite(x).all())
    for g in gradients:
        maybe_assert(func=lambda:pt.isfinite(g).all())
    return gradients


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.DEBUG)
    lin = pt.nn.Parameter(pt.randn(1, 2), requires_grad=True)
    x = pt.randn(2, 1, requires_grad=True)
    f = pt.ones(2, 1, requires_grad=True)
    D = lambda x: (lin @ (x)).sigmoid().sum()

    pen = GradPenalty(modes=["GP", "ZP", "LP", "maxP"])
    p = pen(D, [x, f])
    print(p)


@attr.s
class GradPenHpars(AbstractConf):
    penalty_agg = attr.ib(
        default="mean", validator=attr.validators.in_({"sum", "mean", "max", "lse"})
    )
    input_agg = attr.ib(
        default="sum", validator=attr.validators.in_({"sum", "mean", "max", "lse"})
    )
    sample_agg = attr.ib(
        default="mean", validator=attr.validators.in_({"sum", "mean", "max", "lse"})
    )
    on = attr.ib(
        default=("real", "fake")
    )  # ,"real-perturbed","fake-perturbed"))  # interable of int, real, fake
    modes = attr.ib(
        default=("LP", "LP")
    )  # ,"LP","LP"))  # interable of int, real, fake
    penalty_lambda = attr.ib(default=10)
    weights = attr.ib(default=None)
    perturbation_percentage = attr.ib(default=0.3)
    squash_nans=attr.ib(default=True)

    def make(self):
        kwargs = self.to_dict()
        pen = kwarg_create(GradPenalty, kwargs)
        return pen
