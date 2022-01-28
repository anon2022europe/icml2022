import os
from typing import Optional, Tuple, List

import attr
from torch.utils.data import Subset

from ggg_data.dense.GGG_DenseData import GGG_DenseData
from ggg_data.dense.PEAWGANDenseStructureData import PEAWGANDenseStructureData
from ggg_utils.abstract_conf import AbstractConf
import torch

def supported_dataset(instance, attribute, value):
    values = [value] if isinstance(value,str) else value
    for value in values:
        if not (GGG_DenseData.is_supported(value) or value in GGG_DenseData.SUPPORTED_DATASETS):
            raise NotImplementedError(f"No Support for {value}")

@attr.s
class DatasetHpars(AbstractConf):
    filename = attr.ib(default=None)
    data_dir = attr.ib(default=os.path.expanduser("~/.datasets/"))
    dataset = attr.ib(
        default="CommunitySmall_20", validator=supported_dataset
    )
    structured_features = attr.ib(default=False)
    k_eigenvals = attr.ib(default=4)
    use_laplacian = attr.ib(default=True)
    large_N_approx = attr.ib(default=False)
    dataset_kwargs = attr.ib(factory=dict)
    cut_train_size = attr.ib(default=None)
    num_labels = attr.ib(default=None)
    fake_eigen = attr.ib(default=False)
    repeat = attr.ib(default=None)
    limit_train=attr.ib(default=None,type=Optional[int])
    limit_val=attr.ib(default=None,type=Optional[int])
    limit_path_train=attr.ib(default=None,type=Optional[int])
    limit_path_val=attr.ib(default=None,type=Optional[int])
    force_fresh = attr.ib(default=False)
    curriculum = attr.ib(default=False)
    first_train = attr.ib(default=False)
    schedule=attr.ib(default=None,type=Optional[List[Tuple[float,int]]])

    def make(self):
        user_home=os.path.expanduser("~")
        ddir=self.data_dir
        if "/home/" in ddir:
            relparts = []
            head,tail = os.path.split(ddir)
            relparts.append(tail)
            while "/home" in head:
                relparts.append(tail)
                head,tail = os.path.split(head)

            # use everything except the username we just removed
            self.data_dir=os.path.join(user_home,*list(reversed(relparts[:-1])))


        if self.structured_features:
            ds = PEAWGANDenseStructureData(
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
                repeat=self.repeat,
                force_fresh=self.force_fresh,
                curriculum=self.curriculum,
                first_train=self.first_train,
                schedule=self.schedule,
                limit=self.limit_train,
                limitpath = self.limit_path_train,
            )
        else:
            ds = GGG_DenseData(
                data_dir=self.data_dir,
                filename=self.filename,
                dataset=self.dataset,
                inner_kwargs=self.dataset_kwargs,
                one_hot=self.num_labels,
                cut_train_size=self.cut_train_size,
                dropout_ps=None,
                zero_pad=True,
                repeat=self.repeat,
                force_fresh=self.force_fresh,
                schedule=self.schedule,
                limit=self.limit_train,
            limitpath = self.limit_path_train,
            )
        return ds
    def make_val(self):
        if self.structured_features:
            ds = PEAWGANDenseStructureData(
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
                repeat=self.repeat,
                force_fresh=self.force_fresh,
            partition="val",
            limit=self.limit_val,
            limitpath = self.limit_path_val,
            )
        else:
            ds = GGG_DenseData(
                data_dir=self.data_dir,
                filename=self.filename,
                dataset=self.dataset,
                inner_kwargs=self.dataset_kwargs,
                one_hot=self.num_labels,
                cut_train_size=self.cut_train_size,
                dropout_ps=None,
                zero_pad=True,
                repeat=self.repeat,
                force_fresh=self.force_fresh,
                partition="val",
                limit=self.limit_val,
                limitpath=self.limit_path_val,
            )
        return ds
