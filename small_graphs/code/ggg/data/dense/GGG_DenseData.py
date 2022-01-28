from collections import Counter
from typing import Union, Optional

import os
import torch
import numpy as np
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

from ggg.data.dense.SBM import SBM
from ggg.data.dense.RandTree import RandTrees
from ggg.data.dense.CommunitySmall import CommSmall
from ggg.data.dense.QM9.MolGAN_QM9 import QM9preprocess
from ggg.data.dense.anu_graphs.anudataset import ANUDataset

from ggg.data.dense.GeometricDenseWrapper import geom_data, GeometricDenseWrapper
from ggg.data.condgen import CondgenTCGA, CondgenDBLP
import torch as pt

import numpy as np

from ggg.data.dense.egonet import EgonetSnap
from ggg.data.dense.product import Product_Categories
from ggg.utils.utils import expand_item, zero_diag
import torch_geometric as tg
from ggg.data.dense.nx_graphs import NX_CLASSES,NXGraphWrapper

def mkfloat(x):
    if pt.is_tensor(x):
        return x.float()
    elif isinstance(x, np.ndarray):
        return x.astype(np.float)
    else:
        return float(x)


def ensure_tensor(x):
    if pt.is_tensor(x):
        return x
    elif isinstance(x, np.ndarray):
        return pt.from_numpy(x)
    else:
        return pt.tensor(x)


class RemoveConditional(Dataset):
    def __init__(self, ds):
        super().__init__()
        self.ds = ds

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        (x, A), _ = self.ds[idx]
        return x, A, A.shape[-1]


def dense_adj_dropout(A, p):
    if p > 0.0:
        A = A * torch.bernoulli(A * (1 - p))
    return A


class GGG_DenseData(Dataset):
    """
    Wraps the following datasets in a handy wrapper such that they can be retrieved by name:
    - Our custom MolGAN5k, MolGAN_kC{4-6},CommunitySmal
    - ALL of the https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html (installation separately)

    For ours:
    Gets a dataset from a given data directory+file, each sample consists of the tuple (X,A)
    Where X are the node features and A is the adjacency matrix and both are dense Tensors.
    If no specific datadir and filename are given, uses the dataset name as dir to download the relevant data, and creates the file in an auto-generated name based on the dataset name.

    For torch_geometric ones: ignores data_dir,filename and everything except the "dataset" argument and
    the inner_kwargs which is expected to be a dictionary that is passed to the underlying class.
    Once instantiated, retrives the Data objects from dataset and extracts dense (x,A), which are then returned.
    The following

    For ANUGraphs, can select either anu_graphs_chordal, anu_graphs_all or anu_graphs and specify which ones by passing a list
    of the names as given in anudataset.py

    For condgen datasets, can select condgen_tcga,condgen_dblp
    """

    SUPPORTED_DATASETS = [x for x in geom_data.__all__] + [
        "SBM10",
        "SBM20",
        "SBM30",
        "SBM50",
        "Trees09",
        "Trees12",
        "Trees20",
        "Trees40",
        "MolGAN_5k",
        "MolGAN_kC4",
        "MolGAN_kC5",
        "MolGAN_kC6",
        "RandMolGAN_5k",
        "anu_graphs_chordal9",
        "anu_graphs_chordal9_rand",
        "anu_graphs_chordal_45789",
        "anu_graphs_all",
        "anu_graphs",
        "CommunitySmall_12",
        "CommunitySmall_20",
        "CommunitySmall_50",
        "CommunitySmall_100",
        "CommunitySmall_20_rand1",
        "CommunitySmall_20_rand2",
        "CommunitySmall_20_rand3",
        "egonet",  # full condgen
        "egonet20-1",  # 20 points in the condegen dataset
        "egonet20-2",
        "egonet20-3",
        "egonet20-4",
        "egonet20-5",
        "egonet20-6",
        "egonet-rand-100",
        "condgen_tcga",
        "condgen_dblp",
        "product5k1",  # first 5k unique seed edges with N<=1e3
        "product5k2",  # second 5k unique seed edges
        "product100",  # first 100 unique seed edges
    ]+list(NX_CLASSES.keys())

    def __init__(
        self,
        data_dir=None,
        filename=None,
        dataset="CommunitySmall_20",
        print_statistics=True,
        remove_zero_padding=None,
        inner_kwargs=None,
        zero_pad=False,
        one_hot: Optional[int] = None,
        cut_train_size=False,
        dropout_ps=None,
        repeat=None,
        force_fresh=False,
        **kwargs,
    ):
        assert dataset in GGG_DenseData.SUPPORTED_DATASETS
        self.force_fresh=force_fresh
        self.repeat=repeat
        if inner_kwargs is None:
            for k in kwargs:
                if "args" in k and type(kwargs[k]) is not dict:
                    inner_kwargs = kwargs[k]
        if inner_kwargs is None:
            inner_kwargs = {}
        user_root = os.path.expanduser("~/.datasets")
        os.makedirs(user_root, exist_ok=True)
        self.dropout_ps = dropout_ps
        if data_dir is None:
            data_dir = os.path.join(user_root, dataset.lower())
        self.zero_pad = zero_pad
        self.one_hot = one_hot
        self.dataset = dataset
        self.data_dir = data_dir
        if remove_zero_padding is None:
            remove_zero_padding = "MolGAN" in self.dataset
        self.remove_zero_padding = remove_zero_padding

        if "MolGAN" in dataset:
            if dataset == "MolGAN_5k":
                k_ = None
                size = 5000
                name = "5k"
                create_rand = False
            elif dataset == "MolGAN_kC4":
                k_ = 4
                size = None
                name = "kc4"
                create_rand = False
            elif dataset == "MolGAN_kC5":
                k_ = 5
                size = None
                name = "kc5"
                create_rand = False
            elif dataset == "MolGAN_kC6":
                k_ = 6
                size = None
                name = "kc6"
                create_rand = False
            elif "RandMolGAN_5k" in dataset:
                k_ = None
                size = 5000
                name = "rand"
                if filename == "QM9_rand1.sparsedataset":
                    name = "rand1"
                elif filename == "QM9_rand2.sparsedataset":
                    name = "rand2"
                elif filename == "QM9_rand3.sparsedataset":
                    name = "rand3"
                create_rand = True
            else:
                raise ValueError(
                    f"UnknownConfiguration {dataset} only know MolGAN_5k, and MolGAN_kCn for n in [4,5,6]"
                )
            QM9_MolGAN = QM9preprocess(
                data_dir, filename, k_=k_, size=size, name=name, create_rand=create_rand
            )
            if k_ is not None:
                self._data = QM9_MolGAN.Py_data[-300:]
            else:
                if cut_train_size:
                    self._data = QM9_MolGAN.Py_data[90:590]
                else:
                    self._data = QM9_MolGAN.Py_data[90:4990]
        elif "anu_graphs" in dataset:
            if dataset == "anu_graphs_chordal9":
                chordal = ANUDataset(
                    data_dir=self.data_dir,
                    datasets=["chordal"],
                    exclude_files=[
                        f"chordal{i}.g6{'.gz' if i > 11 else ''}"
                        for i in filter(lambda x: x not in {9}, range(4, 14))
                    ],
                )
                self._data = chordal
            elif dataset == "anu_graphs_chordal9_rand":
                create_rand = True
                chordal = ANUDataset(
                    data_dir=self.data_dir,
                    create_rand=create_rand,
                    datasets=["chordal"],
                    exclude_files=[
                        f"chordal{i}.g6{'.gz' if i > 11 else ''}"
                        for i in filter(lambda x: x not in {9}, range(4, 14))
                    ],
                )
                self._data = chordal
            elif dataset == "anu_graphs_chordal_45789":
                chordal = ANUDataset(
                    data_dir=self.data_dir,
                    datasets=["chordal"],
                    exclude_files=[
                        f"chordal{i}.g6{'.gz' if i > 11 else ''}"
                        for i in filter(
                            lambda x: x not in {4, 5, 7, 8, 9}, range(4, 14)
                        )
                    ],
                )
                self._data = chordal
            elif dataset == "anu_graphs_all":
                self._data = ANUDataset(data_dir=self.data_dir)
            elif dataset == "anu_graphs":
                self._data = ANUDataset(data_dir=self.data_dir, **inner_kwargs)
            else:
                raise ValueError(
                    f"Unkown key for the anu dataset, know only anu_graphs_chordal,anu_graphs_all,anu_graphs (+optional anu_graphs_keys argument),got {dataset}"
                )
        elif "CommunitySmall" in dataset:
            if dataset == "CommunitySmall_12":
                CommSmallgen = CommSmall(data_dir=self.data_dir, filename=filename,
                                         min_nodes=12, max_nodes=12)
                self._data = CommSmallgen.Py_data

            elif dataset == "CommunitySmall_50":
                CommSmallgen = CommSmall(data_dir=self.data_dir, filename=filename,
                                         min_nodes=50, max_nodes=50)
                self._data = CommSmallgen.Py_data

            elif dataset == "CommunitySmall_100":
                CommSmallgen = CommSmall(data_dir=self.data_dir, filename=filename,
                                         min_nodes=100, max_nodes=100)
                self._data = CommSmallgen.Py_data

            elif dataset == "CommunitySmall_200":
                CommSmallgen = CommSmall(data_dir=self.data_dir, filename=filename,
                                         min_nodes=200, max_nodes=200)
                self._data = CommSmallgen.Py_data

            elif dataset == "CommunitySmall_20":
                CommSmallgen = CommSmall(data_dir=self.data_dir, filename=filename)
                if cut_train_size:
                    self._data = CommSmallgen.Py_data[90:590]
                else:
                    # dataset is not 5k exactly and lighthning gives error if len(data)%batch!=0
                    self._data = CommSmallgen.Py_data[90:4990]
            elif dataset == "CommunitySmall_20_biggertrain":
                CommSmallgen = CommSmall(
                    data_dir=self.data_dir, filename=filename, size=10000
                )
                self._data = CommSmallgen.Py_data[90:9990]
            elif "CommunitySmall_20_rand" in dataset:
                if "rand1" in dataset:
                    CommSmallgen = CommSmall(
                        data_dir=self.data_dir,
                        filename=filename,
                        create_rand=True,
                        name="rand1",
                    )
                    self._data = CommSmallgen.Py_data
                if "rand2" in dataset:
                    CommSmallgen = CommSmall(
                        data_dir=self.data_dir,
                        filename=filename,
                        create_rand=True,
                        name="rand2",
                    )
                    self._data = CommSmallgen.Py_data
                if "rand3" in dataset:
                    CommSmallgen = CommSmall(
                        data_dir=self.data_dir,
                        filename=filename,
                        create_rand=True,
                        name="rand3",
                    )
                    self._data = CommSmallgen.Py_data
            else:
                raise ValueError(
                    f"Unkown key for the CommunitySmall, know CommunitySmall_12,got {dataset}"
                )

        elif "SBM" in dataset:
            if "50" in dataset:
                SBM_gen = SBM(data_dir=self.data_dir, filename=filename, sizes_of_blocks=[20, 18, 12],
                              p_inter=[[0.7, 0.015, 0.03], [0.015, 0.4, 0.015], [0.03, 0.015, 0.5]])
            elif "10" in dataset:
                SBM_gen = SBM(data_dir=self.data_dir, filename=filename, sizes_of_blocks=[2, 5, 3],
                              p_inter=[[0.7, 0.015, 0.03], [0.015, 0.4, 0.015], [0.03, 0.015, 0.5]])
                self._data = SBM_gen.Py_data
            elif "20" in dataset:
                SBM_gen = SBM(data_dir=self.data_dir, filename=filename, sizes_of_blocks=[4, 10, 6],
                              p_inter=[[0.7, 0.015, 0.03], [0.015, 0.4, 0.015], [0.03, 0.015, 0.5]])
                self._data = SBM_gen.Py_data
            elif "30" in dataset:
                SBM_gen = SBM(data_dir=self.data_dir, filename=filename, sizes_of_blocks=[6, 14, 10],
                              p_inter=[[0.7, 0.015, 0.03], [0.015, 0.4, 0.015], [0.03, 0.015, 0.5]])
                self._data = SBM_gen.Py_data


        elif "Tree" in dataset:
            if "09" in dataset:
                RT = RandTrees(sizes_of_trees=9, filename=filename, data_dir=self.data_dir)
                self._data = RT.Py_data
            elif "12" in dataset:
                RT = RandTrees(sizes_of_trees=12, filename=filename, data_dir=self.data_dir)
                self._data = RT.Py_data
            elif "20" in dataset:
                RT = RandTrees(sizes_of_trees=20, filename=filename, data_dir=self.data_dir)
                self._data = RT.Py_data
            elif "40" in dataset:
                RT = RandTrees(sizes_of_trees=40, filename=filename, data_dir=self.data_dir)
                self._data = RT.Py_data

        elif "egonet" == dataset:
            self._data = EgonetSnap(**inner_kwargs)
        elif "egonet20" in dataset:
            num = int(dataset.split("-")[-1])
            inner_kwargs["preset20_num"] = num
            self._data = EgonetSnap(**inner_kwargs)
        elif "egonet-rand" in dataset:
            num = int(dataset.split("-")[-1])
            inds = np.random.permutation(948)[:num]
            inner_kwargs["select_inds"] = inds
            self._data = EgonetSnap(**inner_kwargs)
        elif hasattr(geom_data, dataset):
            self._data = GeometricDenseWrapper(dataset, **inner_kwargs)
        elif dataset in NX_CLASSES:
            self._data=NXGraphWrapper(dataset,**inner_kwargs)
        elif "condgen" in dataset:
            if dataset == "condgen_tcga":
                self._data = RemoveConditional(CondgenTCGA(**inner_kwargs))
            elif dataset == "condgen_dblp":
                self._data = RemoveConditional(CondgenDBLP(**inner_kwargs))
            else:
                raise ValueError(
                    f"Unkown dataset {dataset}, try one of \n CommunitySmall,MolGAN_5k,MolGAN_kC4/5/6 or one of {geom_data.__all__}"
                )
        elif "product" in dataset:
            maxN = int(1e3)
            if dataset == "product5k1":
                seed_edge_offset = 0
                max_len = int(5e3)
            elif dataset == "product5k2":
                seed_edge_offset = int(5e3)
                max_len = int(5e3)
            elif dataset == "product100":
                seed_edge_offset = 0
                max_len = 100
            else:
                raise NotImplementedError(
                    "Only 100,5k1 and 5k2 implemented for product"
                )
            self._data = Product_Categories(
                mode="different",
                k=inner_kwargs.get("k", 1),
                max_len=max_len,
                seed_edge_offset=seed_edge_offset,
                **inner_kwargs,
                size_limit=inner_kwargs.get("size_limit", maxN),
            )
        else:
            raise ValueError(
                f"Unkown dataset {dataset}, try one of \n CommunitySmall,MolGAN_5k,MolGAN_kC4/5/6,condgen_{{condgen_tcga,condgen_dblp,}}, egonet or one of {geom_data.__all__}"
            )

        self.max_N = 0
        if print_statistics or zero_pad:
            count = []
            stat_cache_file = os.path.join(user_root, f"{dataset}_stat_cache.pt")
            if self.force_fresh or not os.path.exists(stat_cache_file):
                self.zero_pad = False
                for i in tqdm(range(len(self)), desc="Collecting stats", leave=False):
                    # x,A=self[i]
                    if any(x in dataset for x in ["anu_graphs", "egonet", "product","nx_"]):
                        (X, A) = self._data[i]
                    else:
                        g = self._data[i]
                        A = g.A
                    # count.append(len(x[x != 0]))
                    count.append(A.shape[0])
                    self.max_N = max(self.max_N, A.shape[0])
                self.zero_pad = zero_pad
                if self.repeat is not None:
                    count=count*repeat
                with open(stat_cache_file, "wb") as f:
                    pt.save(dict(count=count, max_N=self.max_N), f)
            else:
                with open(stat_cache_file, "rb") as f:
                    d = pt.load(f)
                    count = d["count"]
                    self.max_N = d["max_N"]

            if print_statistics:
                unique, counts = np.unique(np.array(count), return_counts=True)
                print(
                    "These are the graph size occurrences in the dataset \n {}".format(
                        np.asarray((unique, counts)).T
                    )
                )
                print(
                    "These are the relative occurrences {}".format(
                        counts / len(self._data)
                    )
                )
            print(f"Maximum node number:{self.max_N}")
        if self.repeat is not None:
            self._data=ConcatDataset([self._data]*self.repeat)
    def __len__(self):
        L = len(self._data)
        if self.dropout_ps is not None:
            L = L * (1 + len(self._data))
        return L

    def __getitem__(self, idx):
        # prepare to remove _pydatahelper where possible, for the datasets here we don't need it
        in_idx = idx
        L = len(self._data)
        dropout_idx = idx // L if idx > L else None
        idx = idx % L if self.dropout_ps else idx
        if any(
            isinstance(self._data, d) or (isinstance(self._data,ConcatDataset) and isinstance(self._data.datasets[0],d))
            for d in [ANUDataset, RemoveConditional, EgonetSnap, Product_Categories,NXGraphWrapper]
        ):
            x, A = [mkfloat(ensure_tensor(x)) if x is not None else None for x in self._data[idx]]
            N = A.shape[-1]
        else:
            graph = self._data[idx]
            x = mkfloat(graph.x)
            A = mkfloat(graph.A)
            N = A.shape[-1]
        # NOTE: ensure that we don't have any self-loops in the graph
        if pt.is_tensor(A):
            A = zero_diag(A).detach()
        else:
            np.fill_diagonal(A, 0)
        if self.one_hot is not None:
            was_numpyx = isinstance(x, np.ndarray)
            was_numpyA = isinstance(A, np.ndarray)
            x, A = expand_item(x, A, self.max_N, self.one_hot)
            if was_numpyx and pt.is_tensor(x):
                x = x.numpy()
            if was_numpyA and pt.is_tensor(A):
                A = A.numpy()
        N_pad = self.max_N - A.shape[0]
        if self.zero_pad and N_pad > 0:
            if pt.is_tensor(x):
                x = pt.nn.functional.pad(x, (0, 0, 0, N_pad), "constant", 0)
            elif isinstance(x, np.ndarray):
                x = np.pad(x, [(0, N_pad), (0, 0)])
            elif x is None:
                pass
            else:
                raise TypeError(f"Unkown {type(x)!a}")
            if pt.is_tensor(A):
                A = pt.nn.functional.pad(A, (0, N_pad, 0, N_pad), "constant", 0)
            elif isinstance(A, np.ndarray):
                A = np.pad(A, [(0, N_pad), (0, N_pad)])
            else:
                raise TypeError(f"Unkown {type(x)!a}")

        if self.remove_zero_padding and not self.zero_pad:
            # Remove pad 0's from MolGAN's dataset creation (don't do it if we pad ourselves)
            if x is not None:
                x = x[x != 0]

            # Remove pad 0's from MolGAN's dataset creation
            remcols_A = A[A.sum(dim=1) != 0]
            A = remcols_A.T[remcols_A.T.sum(dim=1) != 0]
            if dropout_idx is not None and self.dropout_ps is not None:
                A = dense_adj_dropout(A, self.dropout_ps[dropout_idx])
        if x is None:
            x=-1
        return x, A, N

    def node_dist_weights(self):
        """
        Compute and return node_dist_weights
        :return:
        """
        old_zero_pad = self.zero_pad
        self.zero_pad = False
        counts = []
        for i in range(len(self)):
            x, A, N = self[i]
            counts.append(N)
        unique, counts = np.unique(np.array(counts), return_counts=True)
        rel_freq = np.zeros(self.max_N)
        for u, c in zip(unique, counts):
            rel_freq[u - 1] = c / len(self)
        self.zero_pad = old_zero_pad

        # counts=Counter(pt.sum((ensure_tensor(self[i][0])!=0).any(dim=-1)) for i in range(len(self)))
        # counts=[counts[k] for k in sorted(counts)]
        # counts=[c/sum(counts) for c in counts]

        return rel_freq


if __name__ == "__main__":
    d = GGG_DenseData(dataset="egonet", inner_kwargs=dict(), print_statistics=True)
    # d = PEAWGANDenseData(
    #    dataset="product100", inner_kwargs=dict(verbose=True), print_statistics=True
    # )
    # d = PEAWGANDenseData(
    #    dataset="product5k1", inner_kwargs=dict(verbose=True), print_statistics=True
    # )
    # d = PEAWGANDenseData(
    #    dataset="product5k2", inner_kwargs=dict(verbose=True), print_statistics=True
    # )
