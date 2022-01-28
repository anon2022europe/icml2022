from _warnings import warn

import attr
import torch as pt

from ggg_utils.utils.anon_eig_features import extract_canonical_k_eigenfeat
from ggg_utils.utils.eigen import our_small_symeig, approx_small_symeig
from ggg_utils.utils.logging import easy_hist
from ggg_utils.utils.utils import ensure_tensor, asserts_enabled, get_laplacian


@attr.s
class StructuralFeatures:
    # mode can be given as all, None, list of valid modes or eigen-gsize-degree style string
    SUPPORTED_MODES = {"all", None, "eigen", "gsize", "degree"}
    # use the egenvectors belonging to the k smallest eigen values as features
    k_smallest_eigen = attr.ib(default=4)
    # use the laplacian (D-A) instead of adjacency matrix
    use_laplacian = attr.ib(default=True)
    # whether to use an apprximation for node number N>10e3
    large_N_approx = attr.ib(default=False)
    # offset to get the offset to offset+kth smallest eigenfeats
    offset = attr.ib(default=0)
    # in order to increase the chance of nonunique eigenvalues our sym eig adds noise with std thisto the diagonal
    noise_scale = attr.ib(default=0.1)
    # zero pad up to max_N TODO FIXME
    zero_pad = attr.ib(default=False)
    # maximum number of nodes in use for zeropad
    max_N = attr.ib(default=None)
    # whether to fake out the eigenvalue calculation
    fake_eigen = attr.ib(default=False)

    @staticmethod
    def prec_suported(inst, att, val):
        modes = StructuralFeatures.SUPPORTED_MODES
        assert val is None or (isinstance(val, str) and val in modes) or any(m in modes for m in val) or (
                isinstance(val, str) and any(m in modes for m in val.split("-")))

    def _deprecated_dominant_sym_eig(self, target):
        if self.large_N_approx and target.shape[-1] > 10000:
            return approx_small_symeig(target, self.k_smallest_eigen)
        else:
            return our_small_symeig(
                target,
                self.k_smallest_eigen,
                offset=self.offset,
                noise_scale=self.noise_scale,
            )

    def get_eigen(self,mode):
        return "all" in mode or "eigen" in mode or any("eigen" in m for m in mode)
    def get_degree(self,mode):
        return "all" in mode or "degree" in mode or any("degree" in m for m in mode)
    def get_gsize(self,mode):
        feat="gsize"
        return "all" in mode or feat in mode or any(feat in m for m in mode)

    def get_struct_feats(self, A,mode="all",which=None):
        # TODO: adopt to batchwise
        if not pt.is_tensor(A):
            A = pt.from_numpy(A).float()
        assert A.dim() in {3, 2}  # no pac stuff here
        if A.dim() == 2:
            # simplify dim thinking by always operating in batch mode
            A = A.unsqueeze(0)

        degree_feat = self.degree_feat(A,which) if self.get_degree(mode) else None

        graph_size_feat, num = self.graph_size_feat(A,which) if self.get_gsize(mode) else None
        if self.get_eigen(mode):
            if self.fake_eigen:
                k_eigen_feats = pt.randn(
                    A.shape[0], A.shape[-1], self.k_smallest_eigen, device=A.device
                ).type_as(A)
            else:
                k_eigen_feats = self.k_eigen_feat(A)
        else:
            k_eigen_feats=None
        # cat eigenfeat,degrees, graph_size
        all_feats=[k_eigen_feats, degree_feat, graph_size_feat]
        struct_feats = pt.cat([x for x in all_feats if x is not None], dim=-1)
        return struct_feats

    def maybe_pad(self, struct_feats, A):
        N_pad = self.max_N - struct_feats.shape[-2]
        if self.zero_pad and N_pad > 0:
            struct_feats = pt.nn.functional.pad(
                struct_feats, (0, 0, 0, N_pad), "constant", 0
            )
            A = pt.nn.functional.pad(A, (0, N_pad, 0, N_pad), "constant", 0)
            assert A.shape[-1] == self.max_N
        return struct_feats, A

    def degree_feat(self, A,which=None):
        degrees = A.sum(
            -1, keepdims=False
        )  # undirected graphs, in and out degree shttps://github.com/buwantaiji/DominantSparseEigenAD/blob/master/examples/TFIM_vumps/symmetric.pyame
        # put the degree size between 0 and 1 to be on the same scale as the eigenfeats
        if which is not None:
            easy_hist(f"{which}_mu_degres",degrees.mean(-1),force=False)
        degrees = 1.0 / (1 + ensure_tensor(degrees).unsqueeze(-1))
        return degrees

    def k_eigen_feat(self, A,which=None):
        if self.use_laplacian:
            target = get_laplacian(A)
        else:
            target = A
        if asserts_enabled():
            k_eigen_feats = extract_canonical_k_eigenfeat(
                target,
                offset=self.offset,
                k=self.k_smallest_eigen,
                noise_scale=self.noise_scale,
            )
        else:
            try:
                k_eigen_feats = extract_canonical_k_eigenfeat(
                    target,
                    offset=self.offset,
                    k=self.k_smallest_eigen,
                    noise_scale=self.noise_scale,
                )
            except:
                warn(f"Getting eigen values failed, so replaceing it with zeros")
                k_eigen_feats = pt.zeros(
                    A.shape[0], A.shape[-1], self.k_smallest_eigen
                ).type_as(A)
        # zero pad to full k
        if k_eigen_feats.shape[-1] < self.k_smallest_eigen:
            zero_pad = pt.zeros(
                list(k_eigen_feats.shape[:-1])+[self.k_smallest_eigen - k_eigen_feats.shape[-1]],
                device=k_eigen_feats.device,
            )
            k_eigen_feats = pt.cat([k_eigen_feats, zero_pad], -1)
        assert k_eigen_feats.shape[-1] == self.k_smallest_eigen
        # k_eigen_feats=k_eigen_feats/((1e-3+k_eigen_feats).norm(dim=-1,keepdim=True))
        return k_eigen_feats

    def graph_size_feat(self, A,which):
        # Graph size feature
        num = A.max(dim=-1).values.sum(dim=-1)  # [B]
        node_mask = (
            (pt.ones(A.shape[0], A.shape[1], 1, device=A.device).cumsum(dim=1) <= num)
            .float()
            .max(-1, keepdim=True)
            .values
        )
        graph_size_feat = node_mask * pt.sqrt(
            1e-3 + num.reshape(-1, 1, 1).repeat([1, A.shape[1], 1])
        )  # sqrt to keep the size manageable
        return graph_size_feat, num
