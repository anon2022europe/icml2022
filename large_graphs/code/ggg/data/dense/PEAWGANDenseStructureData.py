from logging import warning

from ggg.data.dense.GGG_DenseData import GGG_DenseData
import torch as pt
from warnings import warn

from ggg.data.dense.utils.features import (
    our_small_symeig,
    approx_small_symeig,
)
from ggg.utils.utils import ensure_tensor
import attr


@attr.s
class StructuralFeatures:
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
    fake_eigen=attr.ib(default=False)

    def dominant_sym_eig(self, target):
        if self.large_N_approx and target.shape[-1] > 10000:
            return approx_small_symeig(target, self.k_smallest_eigen)
        else:
            return our_small_symeig(
                target,
                self.k_smallest_eigen,
                offset=self.offset,
                noise_scale=self.noise_scale,
            )

    def get_struct_feats(self, A):
        # TODO: adopt to batchwise
        if not pt.is_tensor(A):
            A = pt.from_numpy(A).float()
        assert A.dim() in {3, 2}  # no pac stuff here
        if A.dim() == 2:
            # simplify dim thinking by always operating in batch mode
            A = A.unsqueeze(0)

        degree_feat = self.degree_feat(A)

        graph_size_feat, num = self.graph_size_feat(A)
        if self.fake_eigen:
            k_eigen_feats = pt.randn(A.shape[0], A.shape[-1], self.k_smallest_eigen,device=A.device).type_as(A)
        else:
            k_eigen_feats = self.k_eigen_feat(A)
        # cat eigenfeat,degrees, graph_size
        struct_feats = pt.cat([k_eigen_feats, degree_feat, graph_size_feat], dim=-1)
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

    def degree_feat(self, A):
        degrees = A.sum(
            -1, keepdims=False
        )  # undirected graphs, in and out degree shttps://github.com/buwantaiji/DominantSparseEigenAD/blob/master/examples/TFIM_vumps/symmetric.pyame
        # put the degree size between 0 and 1 to be on the same scale as the eigenfeats
        degrees = 1.0/(1+ensure_tensor(degrees).unsqueeze(-1))
        return degrees

    def k_eigen_feat(self, A):
        if self.use_laplacian:
            degrees = A.sum(
                -1, keepdims=False
            )  # undirected graphs, in and out degree shttps://github.com/buwantaiji/DominantSparseEigenAD/blob/master/examples/TFIM_vumps/symmetric.pyame
            D = pt.diag_embed(ensure_tensor(degrees))
            L = D - A
            target = L
        else:
            target = A
        try:
            k_eigenval, k_eigen_feats = self.dominant_sym_eig(target)
        except:
            warn(f"Getting eigen values failed, so replaceing it with zeros")
            k_eigenval, k_eigen_feats = (
                pt.zeros(A.shape[0], self.k_smallest_eigen).type_as(A),
                pt.zeros(A.shape[0], A.shape[-1], self.k_smallest_eigen).type_as(A),
            )
        # zero pad to full k
        if k_eigen_feats.shape[-1]<self.k_smallest_eigen:
            zero_pad=pt.zeros(k_eigen_feats.shape[0],self.k_smallest_eigen-k_eigen_feats.shape[-1],device=k_eigen_feats)
            k_eigen_feats=pt.cat([k_eigen_feats,zero_pad],-1)
        assert k_eigen_feats.shape[-1]==self.k_smallest_eigen
        # torch tensor as anu_graphs features are arrays
        k_eigen_feats = ensure_tensor(k_eigen_feats)
        #k_eigen_feats=k_eigen_feats/((1e-3+k_eigen_feats).norm(dim=-1,keepdim=True))
        return k_eigen_feats

    def graph_size_feat(self, A):
        # Graph size feature
        num = A.max(dim=-1).values.sum(dim=-1)  # [B]
        node_mask = (
            (pt.ones(A.shape[0], A.shape[1], 1, device=A.device).cumsum(dim=1) <= num)
            .float()
            .max(-1, keepdim=True)
            .values
        )
        graph_size_feat = node_mask *pt.sqrt(1e-3+num.reshape(-1, 1, 1).repeat([1, A.shape[1], 1])) # sqrt to keep the size manageable
        return graph_size_feat, num


class PEAWGANDenseStructureData(GGG_DenseData):
    """
    Has access to the same datasets as PEAWGANDenseData, but replaces the node features with strictly graph-structural features.
    Currently these are (in order): node degree, the kth first entries of eigenvectors.
    Uses torch.symeig by default, with the option to switch to an approximate version for large graphs on which this would become infeasible
    """

    def __init__(
        self,
        data_dir=None,
        filename=None,
        k_eigenvals=4,
        dataset="CommunitySmall",
        print_statistics=True,
        remove_zero_padding=None,
        inner_kwargs=None,
        use_laplacian=True,
        large_N_approx=False,
        zero_pad=True,
        cut_train_size=False,
        dropout_ps=None,
        fake_eigen=False,
        repeat=None
    ):
        self.fake_eigen=fake_eigen
        self.k_eigenvals = k_eigenvals
        self.use_laplacian = use_laplacian
        # get k smallest eig impl
        self.large_N_approx = large_N_approx
        warning(
            f"Using vectors of k *smallest* eigenvals of {'L' if self.use_laplacian else 'A'}"
        )
        if large_N_approx:
            self.dominant_symeig = approx_small_symeig
        else:
            self.dominant_symeig = our_small_symeig
        self._cache = {}
        super().__init__(
            data_dir=data_dir,
            filename=filename,
            dataset=dataset,
            print_statistics=print_statistics,
            remove_zero_padding=remove_zero_padding,
            inner_kwargs=inner_kwargs,
            zero_pad=zero_pad,
            cut_train_size=cut_train_size,
            dropout_ps=dropout_ps,
            repeat=repeat
        )

    def __getitem__(self, idx):
        if idx not in self._cache:
            x, A, N = super().__getitem__(idx)
            if self.zero_pad:
                assert A.shape[-1] == self.max_N
                assert A.shape[-2] == self.max_N
            self._cache[idx] = (x, A, N)
        else:
            x, A, N = self._cache[idx]
        return x, A, N

    def get_structural_node_features(self, A, only_feat=False):
        """
        Extracted so we can use it in the generator as well
        :param A: [B,N,N] or [N,N]
        :param x:  [B,N,F] or [ N,F]
        :return:
        """

        use_laplacian = self.use_laplacian
        large_N_approx = self.large_N_approx
        max_N = self.max_N
        k_eigenvals = self.k_eigenvals
        zero_pad = self.zero_pad

        was_tensor = pt.is_tensor(A)
        assert A.dim() <= 3
        was_batch = A.dim() == 3
        struct_getter = StructuralFeatures(
            k_smallest_eigen=k_eigenvals,
            large_N_approx=large_N_approx,
            use_laplacian=use_laplacian,
            zero_pad=zero_pad,
            max_N=max_N,
            fake_eigen=self.fake_eigen
        )
        struct_feats, A, num = struct_getter.get_struct_feats(A)
        if not was_batch:
            struct_feats, A = struct_getter.maybe_pad(struct_feats, A)
        if not was_batch and A.dim() == 3:
            A = A[0]
            struct_feats = struct_feats[0]
        if not was_tensor:
            struct_feats = struct_feats.numpy()
            A = A.numpy()
        if only_feat:
            ret = struct_feats
        else:
            ret = struct_feats, A, int(num.item())
        return ret


if __name__ == "__main__":
    ds = PEAWGANDenseStructureData(dataset="egonet", inner_kwargs={"num_graphs": 20})
    As = [x[1] for x in ds]
    xs = [x[0] for x in ds]
