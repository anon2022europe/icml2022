import torch as pt
from torch import nn
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


SPECTRAL_NORM_OPTIONS = {None, "diff", "nondiff"}


def sn_wrap(x, spectral_norm, name="weight"):
    if spectral_norm == "diff":
        return SpectralNorm(x, name=name)
    elif spectral_norm == "nondiff":
        return SpectralNormNonDiff(x, name=name)
    elif spectral_norm is None or spectral_norm == False:
        return x
    else:
        raise ValueError(f"Dunno what to do with{spectral_norm}")


class SpectralNorm(nn.Module):
    """
    # from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    Differentiable version?
    """

    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(pt.mv(pt.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(pt.mv(w.view(height, -1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


class SpectralNormNonDiff(nn.Module):
    """
    # from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
    """

    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNormNonDiff, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations

    def _update_u_v(self):
        if not self._made_params():
            self._make_params()
        w = getattr(self.module, self.name)
        u = getattr(self.module, self.name + "_u")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v = l2normalize(pt.mv(pt.t(w.view(height, -1).data), u))
            u = l2normalize(pt.mv(w.view(height, -1).data, v))

        setattr(self.module, self.name + "_u", u)
        w.data = w.data / pt.dot(u, pt.mv(w.view(height, -1).data, v))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = l2normalize(w.data.new(height).normal_(0, 1))

        self.module.register_buffer(self.name + "_u", u)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)
