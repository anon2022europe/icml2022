import attr
import torch
import torch as pt


@attr.s
class JNFPars:
    eps = attr.ib(default=1e-2)
    N = attr.ib(default=10)
    ratio = attr.ib(default=0.7)
    preserve_batch = attr.ib(default=True)
    noise = attr.ib(
        default="normal", validator=attr.validators.in_({"normal", "uniform"})
    )

    def jnf_rejection_sample(self, Z, generator, feature_func):
        N_sampled = 0
        Xs = []
        noise_func = pt.randn_like if self.noise == "normal" else pt.randn_like
        while True:
            X = generator(Z)
            base_logits = feature_func(X)
            with torch.no_grad():
                perturbed_samples = [
                    generator(Z + noise_func(Z) * self.eps) for _ in range(self.N)
                ]
                sample_logits = [feature_func(x) for x in perturbed_samples]
                dist_est = pt.stack(
                    [
                        pt.norm(s - base_logits, p="fro", dim=s.shape[1:])
                        for s in sample_logits
                    ],
                    -1,
                ).mean(-1)
                assert dist_est.dim() == 1
                _, indices = pt.topk(
                    dist_est, k=int(X.shape[0] * self.ratio), largest=False
                )
            # select from indices and make sure we still have batch dim
            Xsel = X[indices].reshape(-1, *X.shape[1:])
            assert Xsel.dim() == X.dim()
            N_sampled += Xsel.shape[0]
            Xs.append(Xsel)
            if N_sampled > Z.shape[0]:
                break
            else:
                Z = noise_func(Z)

        return pt.cat(Xs, 0)[: Z.shape[0]]
