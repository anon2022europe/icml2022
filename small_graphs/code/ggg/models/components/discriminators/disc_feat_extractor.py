from logging import warning
from typing import Optional

import torch as pt
from torch import nn as nn

from ggg.data.dense.PEAWGANDenseStructureData import StructuralFeatures
from ggg.utils.utils import node_mask


class DiscFeatExtractor(nn.Module):
    def __init__(self, mode,fake_eigen=False,n_rand_feats=0):
        super(DiscFeatExtractor, self).__init__()
        SUPPORTED_MODES = {"all", None}
        assert mode in SUPPORTED_MODES, f"Got {mode} only know {SUPPORTED_MODES}"
        self.mode = mode
        if fake_eigen:
            warning(f"Faking eigenvalues with 0")
        self.structure_extract = StructuralFeatures(fake_eigen=fake_eigen)
        self.n_rand_feats=n_rand_feats

    def feat_dim(self):
        if self.mode is None:
            return 0
        else:
            return 1 + 1 + self.structure_extract.k_smallest_eigen+self.n_rand_feats

    def forward(self, A, N=None) -> Optional[pt.Tensor]:
        if self.mode is None:
            return None
        elif self.mode == "all":
            n = A.shape[-1]
            Xf = self.structure_extract.get_struct_feats(A.reshape(-1, n, n))
            if self.n_rand_feats>0:
                rand_feats=pt.randn(Xf.shape[0],Xf.shape[1],self.n_rand_feats,device=Xf.device)
                Xf=pt.cat([Xf,rand_feats],-1)
            if N is not None:
                Xf=node_mask(Xf,N)*Xf
            assert pt.isfinite(Xf).all()
            if A.dim() == 4:
                # pac reshape
                Xf = Xf.reshape(A.shape[0], A.shape[1], n, -1)
            return Xf
        else:
            raise NotImplementedError(f"Don't know {self.mode}")