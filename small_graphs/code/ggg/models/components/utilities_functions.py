# from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan
import torch as pt


# from ggg.models.components.utilities_classes import SpectralNorm, SpectralNormNonDiff


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


# non-differentiable spectral normalization module
# weight tensors are normalized directly
def l2normalizenonDiff(v, eps=1e-12):
    return v / (v.norm() + eps)


def triangles_(adj_matrix, k_, prev_k=None):
    if prev_k is None:
        k_matrix = pt.matrix_power(adj_matrix.float(), k_)
    else:
        k_matrix = prev_k @ adj_matrix.float()
    egd_l = pt.diagonal(k_matrix, dim1=-2, dim2=-1)
    return egd_l, k_matrix
