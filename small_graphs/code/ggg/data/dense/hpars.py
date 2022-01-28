import os

import attr
from attr.validators import in_
from torch.utils.data import ChainDataset, ConcatDataset

from ggg.data.dense.GGG_DenseData import GGG_DenseData
from ggg.data.dense.PEAWGANDenseStructureData import PEAWGANDenseStructureData
from ggg.models.components.abstract_conf import AbstractConf


@attr.s
class DatasetHpars(AbstractConf):
    filename = attr.ib(default=None)
    data_dir = attr.ib(default=os.path.expanduser("~/.datasets/"))
    dataset = attr.ib(
        default="CommunitySmall_20", validator=in_(GGG_DenseData.SUPPORTED_DATASETS)
    )
    structured_features = attr.ib(default=False)
    k_eigenvals = attr.ib(default=4)
    use_laplacian = attr.ib(default=True)
    large_N_approx = attr.ib(default=False)
    dataset_kwargs = attr.ib(factory=dict)
    cut_train_size = attr.ib(default=None)
    num_labels = attr.ib(default=20)
    fake_eigen=attr.ib(default=False)
    repeat=attr.ib(default=None)
    force_fresh=attr.ib(default=False)

    def make(self):
        if self.structured_features:
            ds= PEAWGANDenseStructureData(
                data_dir=self.data_dir,
                filename=self.filename,
                dataset=self.dataset,
                k_eigenvals=self.k_eigenvals,
                use_laplacian=self.use_laplacian,
                large_N_approx=self.large_N_approx,
                inner_kwargs=self.dataset_kwargs,
                cut_train_size=self.cut_train_size,
                zero_pad=True,
                dropout_ps=None,
                fake_eigen=self.fake_eigen,
                repeat=self.repeat
            )
        else:
            ds= GGG_DenseData(
                data_dir=self.data_dir,
                filename=self.filename,
                dataset=self.dataset,
                inner_kwargs=self.dataset_kwargs,
                one_hot=self.num_labels,
                cut_train_size=self.cut_train_size,
                dropout_ps=None,
                zero_pad=True,
                repeat=self.repeat
            )
        return ds
