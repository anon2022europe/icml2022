from logging import warning
from typing import Union, List, Optional, Tuple

import torch as pt

from ggg_data.dense.GGG_DenseData import GGG_DenseData
from ggg_data.dense.curriculum import DegreeCurriculumScheduler
from ggg_utils.utils.eigen import our_small_symeig, approx_small_symeig
from ggg_utils.utils.structural_features import StructuralFeatures


# from anon


class PEAWGANDenseStructureData(GGG_DenseData):
    """
    Has access to the same datasets as PEAWGANDenseData, but replaces the node features with strictly graph-structural features.
    Currently these are (in order): node degree, the kth first entries of eigenvectors.
    Uses torch.symeig by default, with the option to switch to an approximate version for large graphs on which this would become infeasible
    """

    def __init__(
        self,
        dataset: Union[str, List[str]] = "CommunitySmall",
        data_dir=None,
        filename=None,
        k_eigenvals=4,
        print_statistics=True,
        remove_zero_padding=None,
        inner_kwargs=None,
        use_laplacian=True,
        large_N_approx=False,
        zero_pad=True,
        cut_train_size=False,
        dropout_ps=None,
        fake_eigen=False,
        repeat=None,
        force_fresh=False,
        curriculum=False,
        first_train=False,
        partition="train",
            limit=None,
            limitpath=None,
        schedule: Optional[List[Tuple[float, int]]] = None
    ):
        self.curriculum = curriculum
        self.first_train = first_train
        if self.curriculum:
            raise NotImplementedError("General wrapper for curriculum not implemented.")

        self.fake_eigen = fake_eigen
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
            repeat=repeat,
            force_fresh=force_fresh,
            partition=partition,
            schedule=None,# handle this here, with different name
            limit=limit,
            limitpath=limitpath,
        )
        self.structured_scheduler=DegreeCurriculumScheduler(schedule) if schedule is not None else None
    def __getitem__(self, idx):
        if idx not in self._cache:
            x, A, N = super().__getitem__(idx)
            if self.zero_pad:
                assert A.shape[-1] == self.max_N
                assert A.shape[-2] == self.max_N
            self._cache[idx] = (x, A, N)
        else:
            x, A, N = self._cache[idx]
            if (A.shape[-1] == self.max_N and A.shape[-2] == self.max_N) != self.zero_pad:
                x, A, N = super().__getitem__(idx)
                self._cache[idx] = (x, A, N)
        if self.structured_scheduler:
            x,A,N=self.structured_scheduler(x,A,N,idx)
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
            fake_eigen=self.fake_eigen,
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
