from typing import List, Tuple, Optional, Dict

import attr
import torch as pt
import logging

import torch_geometric as pg
from attr.validators import in_

from ggg_utils.abstract_conf import AbstractConf
from ggg_utils.utils.logging import summarywriter, global_step, log_hists


@attr.s
class LapW2Conf(AbstractConf):
    enabled=attr.ib(default=False)
    def make(self):
        return  LapW2(self)
    # TODO: maybe ensure unique eigval by adding something to the diag
from ggg.utils.utils import (
    kwarg_create,
    maybe_assert,
    asserts_enabled,
    EMA,
    squash_nonfinite, Graph,
    squash_nan, squash_infs, get_laplacian
)
from pdb import set_trace

@attr.s
class LapW2:
    conf=attr.ib(type=LapW2Conf,default=LapW2Conf())

    def norm(self,adj:pt.Tensor)->pt.Tensor:
        L=get_laplacian(adj)
        n=adj.shape[-1]
        # compute singular values since we need only them for both tr(L^dagger+1) and tr(sqrt(L^dagger))
        # compute reciprocals, except for zeros
        use_naive=False
        if use_naive:
            # S=pt.linalg.svdvals(L) # svdvals is always numerically stable, but syncs to CPU => might be slow?
            # Sinv[mask]=S[mask].reciprocal() # TODO: check inline modification for backward errors...
            Ldagger=pt.pinverse(L) # compute pseudo inverse, TODO might be possible to be done cheaper by computing *only* eigenvalues and taking inverses...
            # TODO: also check if the power thing works for singular values as well
            norm=(Ldagger.diagonal(dim1=-1,dim2=-2)+1).sum(-1)-2*((pt.linalg.eigvals(Ldagger)+1e-9).pow(0.5).sum(-1)).real
        else:
            # assuming eigsvalh is the appropriate thing to use here
            eigs=pt.linalg.eigvals(L) # this returns complex values...
            mask=eigs!=0
            #mask=pt.logical_or(mask,eigs.real.abs()<=1e-4)
            new_eigs=pt.zeros_like(eigs)
            new_eigs+=(mask==False).float()*1e-9
            new_eigs[mask]=eigs[mask].reciprocal()
            emax=None
            # we only care about the real part anyway, so go back to real here (calling backward would ignore it anyway)
            if emax is not None:
                norm=(new_eigs.real.clamp(-emax,emax)+1).sum(-1)-2*new_eigs.pow(0.5).real.clamp(-emax,emax).sum(-1)  # which makes this part much more chill (can always take sqrt of complex)
            else:
                norm=(new_eigs+1).sum(-1)-2*new_eigs.pow(0.5).sum(-1)  # which makes this part much more chill (can always take sqrt of complex)
                norm=norm.real
        return norm

def aggregate_tensor(pens:List[pt.Tensor], agg="mean")->pt.Tensor:
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
        "ppr-diffusion",
        "heat-diffusion"
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
        squash_nans=False,
        squash_inf=False,
        ema_clip=False,
        ema_alpha=0.7,  # 7 steps to go from 1 to \approx 0.1
        ema_clip_n=5,
        ema_clip_eps=1e-2,
        lapW2:Optional[LapW2Conf]=None,
        force_L1=False
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
        self.ema_clip = ema_clip
        self.ema_clip_n = ema_clip_n
        self.ema_clip_eps = ema_clip_eps
        self.ema_alpha = ema_alpha

        self.squash_nans = squash_nans
        self.squash_inf = squash_inf
        self.weights = weights
        self.on = on
        self.modes = modes
        self.input_agg = input_agg
        self.sample_agg = sample_agg
        self.penalty_agg = penalty_agg
        self.ema = EMA(shape=[], smoothing=self.ema_alpha, trainable=False)
        self.lap2W=None if lapW2 is None else LapW2Conf.from_dict(lapW2).make()
        self.force_L1=force_L1

    def forward(self, D:'Discriminator', fake:Graph,real:Graph,suffix=None)->pt.Tensor:
        if suffix is None:
            suffix=""
        # assumes ordering [(fake1,real1),(fake2,real2)] to be fed into discrimantor as D(*[t1,t2,t3])
        penalties = []
        pairs=fake,real
        for mode, on, w in zip(self.modes, self.on, self.weights):
            if mode == "simple":
                # ignore on,input_agg in this instance, since we are doing a *very* crude estimation
                raise NotImplementedError(f"{mode} not shifted to Graph API yet")
                real = self.get_penalty_target("real", *pairs)
                fake = self.get_penalty_target("fake", *pairs)
                diff_Dx = pt.norm(1e-5 + (D(*real) - D(*fake)), "fro")
                diff_x = pt.stack(
                    [pt.norm(r - f + 1e-5, "fro") for r, f in zip(real, fake)]
                ).min()  # smallest change=> overestimate a bit
                penalty = diff_Dx / (diff_x + 1e-9)
            else:
                if self.force_L1:
                    rG,fG=pairs
                    with pt.no_grad():
                        xs_r,xs_f=[D.readout_in(x) for x in [rG,fG]]
                        xs_r:pt.Tensor
                    xs_r.requires_grad=True
                    xs_f.requires_grad=True
                    alpha=pt.rand_like(xs_f)
                    d_inputs=xs_r*alpha+xs_f*(1-alpha)
                    d_out = D.readout_mlp(d_inputs)
                    gradients = get_gradients(
                        d_out, d_inputs, disable_assert=self.squash_nans
                    )
                    gradients=[gradients]*len(mode)
                else:
                    d_inputs = self.get_penalty_target(on, *pairs)
                    d_inputs.requires_grad_()
                    d_out = D(d_inputs)
                    gradients = get_gradients(
                        d_out, d_inputs, disable_assert=self.squash_nans
                    )

                if any([not pt.isfinite(x).all() for x in gradients]):
                    #set_trace()
                    pass
                if self.squash_nans and self.squash_inf:
                    for g in gradients:
                        squash_nonfinite(g, name="grad-pen"+suffix)
                elif self.squash_inf:
                    for g in gradients:
                        squash_infs(g, name="grad-pen"+suffix)
                elif self.squash_nans:
                    for g in gradients:
                        squash_nan(g, name="grad-pen"+suffix)

                # Get gradient w.r.t. target specified by on
                pens = penalty_from_gradient(gradients, mode, self.sample_agg,lapW2=self.lap2W)
                # aggregate penalty w.r.t to different input tensors
                penalty = aggregate_tensor(pens, self.input_agg)
                if asserts_enabled():
                    if not pt.isfinite(penalty).all():
                        d_inputs.clear_grad()
                        with pt.autograd.detect_anomaly():
                            nd_out = D(d_inputs)
                            ngradients = get_gradients(nd_out, d_inputs)
                            # Get gradient w.r.t. target specified by on
                            npens = penalty_from_gradient(
                                ngradients, mode, self.sample_agg,lapW2=self.lap2W
                            )
                            # aggregate penalty w.r.t to different input tensors
                            npenalty = aggregate_tensor(npens, self.input_agg)
                            npenalty.mean().backward()
                    maybe_assert(func=lambda: pt.isfinite(penalty).all())

                # weight this penalty in the total sum
            penalties.append(penalty * w)

        if logging.getLogger().getEffectiveLevel() <= logging.DEBUG:
            logging.debug(
                f"Pe-agg penalties \n{'-'.join([str(m)+str(o) for m,o in zip(self.modes,self.on)])}\n{'-'.join([f'{p}' for p in penalties])} {suffix}"
            )
        penalty = aggregate_tensor(penalties, self.penalty_agg)
        if self.ema_clip:
            with pt.no_grad():
                clip_range = self.ema_clip_n * (penalty.abs() + self.ema_clip_eps)
            penalty = pt.clip(penalty, -clip_range, clip_range)
            # update ema with post-clip penalty
            self.ema(penalty)
        return penalty

    def get_penalty_target(self, on, real:Graph,fake:Graph)->Graph:
        # expects pairs to be (fakeA realA) (fakeB realB) etc
        if on == "int":
            return Graph.convex_combination(real,fake).detach()
        elif on == "rec-row":
            return Graph.recombine_rows(real,fake,self.perturbation_percentage).detach()
        elif on == "fake":
            return fake.detach()
        elif on == "real":
            return real.detach()
        elif on == "fake-perturbed":
            return fake.detach().perturb_edges(self.perturbation_percentage)
        elif on == "real-perturbed":
            return real.detach().perturb_edges(self.perturbation_percentage)
        elif on in {"ppr-diffusion","heat-diffusion"}:# same thing for heat perturbation
            assert real.typ==fake.typ
            real_inner=real.to_torch_sparse(batch_list=True) if real.dense or real.coo_sparse else real
            fake_inner=fake.to_torch_sparse(batch_list=True) if fake.dense or fake.coo_sparse else fake
            targets=[]
            for f,r in fake_inner,real_inner:
                if "ppr" in on:
                    # alphas
                    alpha=pt.rand()*0.25 -0.05
                    #seems to not be batcheable => loop, or maybe we can just draw 1 diffussion coefficient at each batch...
                    # for connections: erdos renyi connections between graph nodes, or simply sample number of cross nodes (min 1? min n?)
                    # and do random matching...or global node?
                    # TODO: might want to try and batch this...but not sure if possible
                    avg_degree_sum=f.avg_degree()+r.avg_degree()
                    # we want to at most double the average degree?
                    low=max(1,0.5*avg_degree_sum)
                    high=max(low+1,avg_degree_sum)
                    avg_degree_target=pt.randint(low,high)
                    rewired= Graph.multi_rewire([f,r], intra_graph_density=0.1, diffusion_strength=alpha, diff_avg_degree=avg_degree_target)
                    targets.append(rewired)
                else:
                    raise NotImplementedError("Heat diffusion requries exact computation and might not be scalable")
                # diffuse both the real-fake pair and the real-real pair with a random coefficient, then apply
                # to both unperturbed and perturbed
            if real.dense:
                targets=[g.to_dense() for g in targets]
            elif real.coo_sparse:
                targets=[g.to_coo_sparse() for g in targets]
            elif real.torch_sparse:
                pass # all good, no conversion required
            else:
                raise NotImplementedError(f"Unknown graph repr{real.typ}")
            return Graph.to_batch(targets)
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


def penalty_from_gradient(gradients:List[pt.Tensor], pen_type, sample_agg="mean",lapW2:Optional[LapW2]=None):
    pens = []
    is_matrix_list=[len(grads.shape)>=2 and grads.shape[-1]==grads.shape[-2] for grads in gradients]
    assert sum(is_matrix_list)>=len(is_matrix_list)/2.0, "Need at least 1 adjacency matrix per node features!"
    for grads,is_matrix in zip(gradients,is_matrix_list):
        if is_matrix:
            if lapW2 is not None and lapW2.conf.enabled:
                # THIS COULD BE the case of the norm as well: in the case of adjmatrix, we need to take the norm of the *matrix*, not the *vector*
                # short circuit the matrix norm into our custom "norm"
                norm= lambda x:lapW2.norm(x)
            else:
                norm=lambda x:pt.linalg.matrix_norm(x,ord="fro",dim=(-2,-1))
                # TODO: verify this doesn't mess things up!; keep same as before now
                #norm = lambda x: pt.linalg.vector_norm(x, ord=2, dim=-1)
                # corresponds to l2 for node features, TODO might want to try nuclear norm here? what does it imply?
            #grads = grads.reshape(grads.size(0), -1)
            # sanity check for matrix gradient dims
            assert len(grads.shape) in {2,3}
        else:
            # sanity check for matrix gradient dims
            assert len(grads.shape) in {2,3}
            # go from B,N,F to B,N*F since we care about the sensitivity to each *individual* input
            grads = grads.reshape(grads.size(0), -1)
            norm=lambda x:pt.linalg.vector_norm(x,ord=2,dim=-1)
        # add a small offset to avoid numerical problems when there is a sqrt of 0
        grads = grads + 1e-7
        if pen_type == "ZP":
            # from https://openreview.net/forum?id=ByxPYjC5KQ
            diff = norm(grads)
            penalty_samples = diff ** 2
        elif pen_type == "maxP":
            # https://github.com/ZhimingZhou/LGANs-for-reproduce/blob/master/code/gan_synthetic4.py
            # this is the same for matrix or vector?
            penalty_samples = grads.pow(2.0).max(-1).values
        elif pen_type == "maxPlse":
            diff = norm(grads).flatten().logsumexp(-1)
            penalty_samples = diff ** 2
        elif pen_type == "LP" or pen_type is True:
            diff = (norm(grads) - 1)
            penalty_samples = diff.clamp_min(0.0) ** 2
        elif pen_type == "GP" or pen_type is False:
            penalty_samples = (norm(grads) - 1) ** 2
        else:
            raise NotImplementedError(f"Don't know penalty {pen_type}")
        if log_hists():
            summarywriter().add_histogram("gpen",penalty_samples,global_step=global_step())
        penalty = aggregate_tensor(penalty_samples, sample_agg)
        pens.append(penalty)
    return pens


def get_gradients(
    discriminator_output, input_graph:Graph, disable_assert=False
)->List[pt.Tensor]:
    grads = [
        pt.ones(x.shape[0], 1, requires_grad=False).type_as(x)
        if x.dim() > 0
        else pt.ones([], requires_grad=False)
        for x in [discriminator_output]
    ]
    # Get gradient w.r.t. interpolates
    if not (getattr(input_graph,"dense",True) or isinstance(input_graph,pt.Tensor)):
        raise NotImplementedError("Haven't checked the gradient calculation of sparse batches yet...")
    discriminator_inputs = input_graph.effective_inputs() if not isinstance(input_graph,pt.Tensor) else input_graph
    gradients = pt.autograd.grad(
        outputs=discriminator_output,
        inputs=discriminator_inputs,
        grad_outputs=grads,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )
    if not disable_assert:
        maybe_assert(func=lambda: pt.isfinite(discriminator_output).all())
        for x in discriminator_inputs:
            if x is not None:
                maybe_assert(func=lambda: pt.isfinite(x).all())
        for g in gradients:
            maybe_assert(func=lambda: pt.isfinite(g).all())
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
    @staticmethod
    def children() -> Dict:
        return dict(lapW2=LapW2Conf)
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
        default=("real", "fake", "real-perturbed", "fake-perturbed")
    )  # )  # interable of int, real, fake
    modes = attr.ib(
        default=("LP", "LP", "LP", "LP")
    )  # )  # interable of int, real, fake
    penalty_lambda = attr.ib(default=10)
    weights = attr.ib(default=None)
    perturbation_percentage = attr.ib(default=0.3)
    squash_nans = attr.ib(default=True)
    squash_inf = attr.ib(default=True)
    ema_clip = attr.ib(default=False)
    ema_alpha = attr.ib(default=0.7)  # 7 steps to go from 1 to \approx 0.1
    ema_clip_n = attr.ib(default=5)
    ema_clip_eps = attr.ib(default=1e-2)
    lapW2=attr.ib(default=LapW2Conf(),type=Optional[LapW2Conf])
    force_L1=attr.ib(default=False)

    def make(self)->GradPenalty:
        kwargs = self.to_dict()
        pen = kwarg_create(GradPenalty, kwargs)
        return pen
