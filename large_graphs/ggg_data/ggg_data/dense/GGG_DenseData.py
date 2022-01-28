import os
from collections import defaultdict
from typing import Union, Optional, List, Tuple, Dict
from warnings import warn

import numpy as np
import torch as pt
from ipdb import set_trace
from torch.utils.data import Dataset, ConcatDataset, Subset, DataLoader
from tqdm import tqdm

from ggg_data.dense.condgen import CondgenTCGA, CondgenDBLP
from ggg_data.dense.gran_dataset import Enzymes
from ggg_data.dense.CommunitySmall import CommSmall
from ggg_data.dense.GeometricDenseWrapper import geom_data, GeometricDenseWrapper, OGBWrapper, CircuitWrapper
from ggg_data.dense.QM9.MolGAN_QM9 import QM9preprocess
from ggg_data.dense.RandTree import RandTrees
from ggg_data.dense.SBM import SBM
from ggg_data.dense.anu_graphs.anudataset import ANUDataset
from ggg_data.dense.curriculum import DegreeCurriculumScheduler
from ggg_data.dense.egonet import EgonetSnap
from ggg_data.dense.house_floorplans.HouseDataset import HousePlans
from ggg_data.dense.nx_graphs import NX_CLASSES, NXGraphWrapper
from ggg_data.dense.product import Product_Categories
from ggg_utils.utils.utils import expand_item, zero_diag


def get_or_create_inds(root, DS_NAME, N, limit,fpath=None):
    if fpath is None:
        fpath = os.path.join(root, f"{DS_NAME}_{N}_{limit}.pt")
        warn(f"No fpath given, random indices in  {fpath}, for reproducibility, you should copy this file and pass fpath on any reproduction attempt")
    if os.path.exists(fpath):
        with open(fpath, "rb") as f:
            inds = pt.load(f)["inds"]
    else:
        inds = pt.randperm(N)[:limit]
        with open(fpath, "wb") as f:
            pt.save(dict(inds=inds),f)
    return inds


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
        A = A * pt.bernoulli(A * (1 - p))
    return A


class GGG_DenseData(Dataset):
    """
    Wraps the following datasets in a handy wrapper such that they can be retrieved by name:

    * ALL of the datasets from `<https://pytorch-geometric.readthedocs.io/en/latest/modules/datasets.html>`_, using the names
      given in that documentation
    * ALL of the datasets from `<OGB>`_, using the names
      given in that documentation
    * Stochastic Block Model SBM`N` with `N` being the number of nodes
       * SBM10
       * SBM20
       * SBM30
       * SBM50
    * Trees
       * `Trees_rand1`
    * QM9 dataset similar to `<>`_
       * `MolGAN_5k`
       * `MolGAN_kC4`
       * `MolGAN_kC5`
       * `MolGAN_kC6`
       * `RandMolGAN_5k`
    * Graphs from the ANU dataset
       * Chordal graphs
          * `anu_graphs_chordal9`
          * `anu_graphs_chordal9_rand`
          * `anu_graphs_chordal_45789`
       * `anu_graphs_all`
       * `anu_graphs`
    * The CommunitySmall dataset from
       * `CommunitySmall_12`
       * `CommunitySmall_20`
       * `CommunitySmall_50`
       * `CommunitySmall_100`
       * `CommunitySmall_100_rand1`
       * `CommunitySmall_100_rand2`
       * `CommunitySmall_100_rand3`
    * The HouseFloorplan dataset from `<>`_
       * `HouseFloorplan`
       * `HouseFloorplan_1024`
       * `HouseFloorplan_rand1`
       * `HouseFloorplan_rand2`
       * `HouseFloorplan_rand3`

    * SNAP egonet
        * `egonet20-1`
        * `egonet20-2`
        * `egonet20-3`
        * `egonet20-4`
        * `egonet20-5`
        * `egonet20-6`
        * `egonet-rand-100`
    * Condgen datasets
       * condgen_tcga
       * condgen_dblp
    * Amazon Product network
       * `product5k1`
       * `product5k2`
       * `product100`
    * Electronic circuits represented as tripartite Graphs
        * `Circuit_500`
    * Networkx generated graphs
       * `nx_star`
       * `nx_circ_ladder`
       * `nx_lollipop`
       * `nx_roc`
       * `nx_combo`
       * `nx_triangle`
       * `nx_square`

    For our

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

    SUPPORTED_DATASETS = (
        [x for x in geom_data.__all__]
        + [
            "SBM10",
            "SBM20",
            "SBM30",
            "SBM50",
            "Trees",
            "Trees_rand1",
            "Trees_curriculum",
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
            "CommunitySmall_200",
            "CommunitySmall_400",
            "CommunitySmall_100_rand1",
            "CommunitySmall_100_rand2",
            "CommunitySmall_100_rand3",
            "HouseFloorplan",
            "HouseFloorplan_1024",
            "HouseFloorplan_rand1",
            "HouseFloorplan_rand2",
            "HouseFloorplan_rand3",
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
            "Circuit_5000",
            "gran_dataset_DD",
            "gran_dataset_FIRSTMM"
        ]
        + list(NX_CLASSES.keys())+list(OGBWrapper.DATASETS)
    )

    @classmethod
    def is_supported(cls, d):
        if any((name in d and d.split(split)[-1].isnumeric()) for name,split in [("CommunitySmall_","_"),("Trees","Trees"),("Circuits_","_")]):
            return True
        else:
            return False

    def __init__(
        self,
        dataset: Union[str, List[str]] = "CommunitySmall_20",
        data_dir=None,
        filename=None,
        print_statistics=True,
        remove_zero_padding=None,
        inner_kwargs=None,
        zero_pad=True,
        one_hot: Optional[int] = None,
        cut_train_size=False,
        dropout_ps=None,
        repeat=None,
        force_fresh=False,
        partition="train",
        limit=None,
        limitpath=None,
        schedule: Optional[List[Tuple[float, int]]]=None,
        **kwargs,
    ):
        assert partition in {"train","val","test",None}
        user_root = os.path.expanduser("~/.datasets")
        data_dir=os.path.expanduser(data_dir if data_dir is not None else user_root)
        os.makedirs(user_root, exist_ok=True)
        if isinstance(dataset, str):
            assert  (
                dataset in GGG_DenseData.SUPPORTED_DATASETS or GGG_DenseData.is_supported(dataset)
            ), f"Unkown dataset {dataset}, know only {GGG_DenseData.SUPPORTED_DATASETS}"
            dataset = [dataset]
        else:
            for i, d in enumerate(dataset):
                assert GGG_DenseData.SUPPORTED_DATASETS or GGG_DenseData.is_supported(
                    d
                ), f"Unkown dataset {d} at position {i}, know only {GGG_DenseData.SUPPORTED_DATASETS}"
        dataset = sorted(dataset)
        self.DS_NAME = "-".join(sorted([d.lower() for d in dataset]))
        if data_dir is None:
            data_dir = os.path.join(user_root, self.DS_NAME)
        if inner_kwargs is None:
            all_inner_kwargs = [{}] * len(dataset)
        else:
            if not isinstance(inner_kwargs,list):
                all_inner_kwargs=[inner_kwargs]


        self.print_statistics = print_statistics
        self.force_fresh = force_fresh
        self.repeat = repeat
        self.dropout_ps = dropout_ps
        self.zero_pad = zero_pad
        self.one_hot = one_hot
        self.counts_computed=False
        self.node_rel_freq=None
        self.edge_rel_freq_dict=None
        self.edge_map_dict=None
        self.dataset = dataset
        self.data_dir = data_dir
        self.remove_zero_padding = remove_zero_padding

        datasets = tqdm(dataset)
        self.individual_datasets = []
        for inner_kwargs,dataset in zip(all_inner_kwargs,datasets):
            datasets.set_description(f"{dataset}")
            if "MolGAN" in dataset:
                remove_zero_padding = (
                    True
                    if self.remove_zero_padding is None
                    else self.remove_zero_padding
                )
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
                    name = "rand3"
                    create_rand = True
                else:
                    raise ValueError(
                        f"UnknownConfiguration {dataset} only know MolGAN_5k, and MolGAN_kCn for n in [4,5,6]"
                    )
                QM9_MolGAN = QM9preprocess(
                    data_dir,
                    filename,
                    k_=k_,
                    size=size,
                    name=name,
                    create_rand=create_rand or partition=="val",
                )
                if k_ is not None:
                    self._data = QM9_MolGAN.Py_data[-300:]
                else:
                    if cut_train_size:
                        self._data = QM9_MolGAN.Py_data[90:590]
                    else:
                        self._data = QM9_MolGAN.Py_data[90:4990]
            elif dataset in OGBWrapper.DATASETS:
                self._data=OGBWrapper(dataset,**inner_kwargs)
            elif "anu_graphs" in dataset:
                if dataset == "anu_graphs_chordal9":
                    chordal = ANUDataset(
                        data_dir=self.data_dir,
                        datasets=["chordal"],
                        create_rand=partition=="val",
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
                if dataset == "CommunitySmall_100":
                    CommSmallgen = CommSmall(
                        data_dir=self.data_dir,
                        filename=filename,
                    )
                    self._data = CommSmallgen.Py_data
                elif dataset == "CommunitySmall_20":
                    CommSmallgen = CommSmall(data_dir=self.data_dir, filename=filename, min_nodes=20, max_nodes=20)
                    self._data = CommSmallgen.Py_data
                elif dataset == "CommunitySmall_50":
                    CommSmallgen = CommSmall(data_dir=self.data_dir, filename=filename, min_nodes=50, max_nodes=50)
                    self._data = CommSmallgen.Py_data
                elif dataset == "CommunitySmall_400":
                    CommSmallgen = CommSmall(data_dir=self.data_dir, filename=filename, size=1000, min_nodes=400, max_nodes=400)
                    self._data = CommSmallgen.Py_data
                elif "CommunitySmall_100_rand" in dataset:
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
                
                elif len(dataset.split("_")) == 2 and dataset.split("_")[1].isnumeric():
                    N = int(dataset.split("_")[1])
                    CommSmallgen = CommSmall(
                        data_dir=self.data_dir,
                        filename=filename,
                        min_nodes=N,
                        max_nodes=N,
                    )
                    self._data = CommSmallgen.Py_data
                else:
                    raise ValueError(
                        f"Unkown key for the CommunitySmall, know CommunitySmall_12,got {dataset}"
                    )

            elif "SBM" in dataset:
                if "50" in dataset:
                    SBM_gen = SBM(
                        data_dir=self.data_dir,
                        filename=filename,
                        sizes_of_blocks=[20, 18, 12],
                        p_inter=[
                            [0.7, 0.015, 0.03],
                            [0.015, 0.4, 0.015],
                            [0.03, 0.015, 0.5],
                        ],
                    )
                    self._data = SBM_gen.Py_data
                elif "10" in dataset:
                    SBM_gen = SBM(
                        data_dir=self.data_dir,
                        filename=filename,
                        sizes_of_blocks=[2, 5, 3],
                        p_inter=[
                            [0.7, 0.015, 0.03],
                            [0.015, 0.4, 0.015],
                            [0.03, 0.015, 0.5],
                        ],
                    )
                    self._data = SBM_gen.Py_data
                elif "20" in dataset:
                    SBM_gen = SBM(
                        data_dir=self.data_dir,
                        filename=filename,
                        sizes_of_blocks=[4, 10, 6],
                        p_inter=[
                            [0.7, 0.015, 0.03],
                            [0.015, 0.4, 0.015],
                            [0.03, 0.015, 0.5],
                        ],
                    )
                    self._data = SBM_gen.Py_data
                elif "30" in dataset:
                    SBM_gen = SBM(
                        data_dir=self.data_dir,
                        filename=filename,
                        sizes_of_blocks=[6, 14, 10],
                        p_inter=[
                            [0.7, 0.015, 0.03],
                            [0.015, 0.4, 0.015],
                            [0.03, 0.015, 0.5],
                        ],
                    )
                    self._data = SBM_gen.Py_data
            elif "Circuits_" in dataset:
                Nmax=int(dataset.split("_")[-1])
                self._data=CircuitWrapper(Nmax=Nmax)
            elif "gran" in dataset:
                if "DD" in dataset:
                    EZ = Enzymes(min_nodes=100, max_nodes=500, name="DD", node_attributes=False, graph_labels=True)
                    self._data = EZ.Py_data
                elif "FIRSTMM" in dataset:
                    EZ = Enzymes(min_nodes=0, max_nodes=10000, name="FIRSTMM", node_attributes=False, graph_labels=True)
                    self._data = EZ.Py_data
            elif "Tree" in dataset:
                if dataset.split("Trees")[-1].isnumeric():
                    RT = RandTrees(
                        sizes_of_trees=int(dataset.split("Trees")[-1]), filename=filename, data_dir=self.data_dir
                    )
                    self._data = RT.Py_data
                elif "curriculum" in dataset:
                    RT = RandTrees(
                        **inner_kwargs, filename=filename, data_dir=self.data_dir, create_rand=True, curriculum=True,
                        name=dataset.split("_")[-1], sizes_of_trees=50
                    )
                    self._data = RT.Py_data
                elif "rand" in dataset:
                    RT = RandTrees(
                        **inner_kwargs, filename=filename, data_dir=self.data_dir, create_rand=True,
                        name=dataset.split("_")[-1]
                    )
                    self._data = RT.Py_data
                else:
                    RT = RandTrees(
                        **inner_kwargs, filename=filename, data_dir=self.data_dir,
                    )
                    self._data = RT.Py_data
            elif "House" in dataset:
                if "rand" in dataset:
                    RT = HousePlans(filename=filename, data_dir=self.data_dir,
                                    create_rand=True, name=dataset.split("_")[1])
                    self._data = RT.Py_data
                if "1024" in dataset:
                    RT = HousePlans(filename=filename, data_dir=self.data_dir, size=1024,
                                    create_rand=False, name=dataset.split("_")[1])
                    self._data = RT.Py_data
                else:
                    RT = HousePlans(filename=filename, data_dir=self.data_dir)
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
                self._data = NXGraphWrapper(dataset, **inner_kwargs)
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
            self.individual_datasets.append((dataset, self._data))
        self._data = ConcatDataset([ds for _, ds in self.individual_datasets])
        if limit is not None:
            inds=get_or_create_inds(user_root,self.DS_NAME,len(self._data),limit,fpath=limitpath)
            self._data=Subset(self._data,inds)
        self.max_N = 0
        count = []
        stat_cache_file = os.path.join(user_root, f"{self.DS_NAME}_stat_cache.pt")
        if print_statistics or zero_pad:
            if self.force_fresh or not os.path.exists(stat_cache_file):
                ds_bar = tqdm(self.individual_datasets, leave=True)
                for dataset, _data in ds_bar:
                    ds_bar.set_description(f"{dataset}")
                    for i in tqdm(
                        range(len(_data)), desc="Collecting stats", leave=True
                    ):
                        # x,A=self[i]
                        if any(
                            x in dataset
                            for x in ["anu_graphs", "egonet", "product", "nx_", "ogb"]
                        ):
                            (X, A) = _data[i]
                        else:
                            g = _data[i]
                            A = g.A
                        # count.append(len(x[x != 0]))
                        #N_counted=(A.abs()!=0).float().sum(-1)
                        count.append(A.shape[-1])
                        self.max_N = max(self.max_N, A.shape[-1])
                ds_bar.close()
                if self.repeat is not None:
                    count = count * repeat
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
                "These are the graph size occurrences in the dataset {} \n {}".format(
                    self.DS_NAME, np.asarray((unique, counts)).T
                )
            )
            print(
                "These are the relative occurrences {} {}".format(
                    self.DS_NAME, counts / len(self._data)
                )
            )
        print(f"Maximum node number {self.DS_NAME}:{self.max_N}")
        if self.repeat is not None:
            self._data = ConcatDataset([self._data] * self.repeat)
        if schedule is not None:
            self.schedules=schedule
            self.scheduler=DegreeCurriculumScheduler(schedule)
        else:
            self.scheduler=None

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
            isinstance(self._data, d)
            or (
                isinstance(self._data, ConcatDataset)
                and isinstance(self._data.datasets[0], d)
            ) or (isinstance(self._data,Subset) and isinstance(self._data.dataset,d))
            for d in [
                ANUDataset,
                RemoveConditional,
                EgonetSnap,
                Product_Categories,
                NXGraphWrapper,
                OGBWrapper,
                #GeometricDenseWrapper
                #CircuitWrapper
            ]
        ):
            x, A = [
                mkfloat(ensure_tensor(x)) if x is not None else None
                for x in self._data[idx]
            ]
            N = A.shape[-1]
        else:
            graph = self._data[idx]
            x = mkfloat(graph.x)
            A = mkfloat(graph.A)
            A=A.reshape(A.shape[-2:])
            x=x.reshape(x.shape[-2:])
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
        if x is not None:
            x = x.reshape(A.shape[0],-1) # ensure we have a 2d node feat tensor N F for padding
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
            x = np.ones((A.shape[-1],1))*-1
            x[N:,:]=0
            if pt.is_tensor(A):
                x=pt.from_numpy(x).to(A.dtype)
            else:
                x=x.astype(A.dtype)

        if self.scheduler:
            x,A,N=self.scheduler(x,A,N,idx)
        return x, A, N

    def node_dist_weights(self) -> np.ndarray:
        """
        Compute and return the graph size (number of nodes) distribution

        :return: number of nodes distribution, numpy array
        """
        if not self.counts_computed:
            self.compute_counts()
        rel_freq=self.node_rel_freq

        return rel_freq

    def compute_counts(self):
        old_zero_pad = self.zero_pad
        self.zero_pad = False
        node_counts = []
        edge_counts = defaultdict(list)
        L=len(self)
        for i in tqdm(range(L),total=L,leave=False,desc=f"Computing node counts for {self.DS_NAME}"):
            x, A, N = self[i]
            node_counts.append(N)
            m = pt.triu(A).sum().item()
            edge_counts[N].append(m)
        unique, node_counts = np.unique(np.array(node_counts), return_counts=True)
        edge_count_dict={}
        edge_map_dict={}
        for n,edge_count in edge_counts.items():
            muniuq,m_counts=np.unique(np.array(edge_count),return_counts=True)
            map={i:c for i,c in enumerate(muniuq)}
            C=m_counts.sum()
            if C==0:
                raise ValueError("Unexpected? shouldn't there be *some* edges?")
            rel_freq=m_counts/m_counts.sum()
            edge_count_dict[n]=ensure_tensor(rel_freq)
            edge_map_dict[n]=map


        rel_freq = np.zeros(self.max_N)
        for u, c in zip(unique, node_counts):
            rel_freq[u - 1] = c / len(self)
        self.zero_pad = old_zero_pad
        self.node_rel_freq = rel_freq
        self.edge_rel_freq_dict=edge_count_dict
        self.edge_map_dict=edge_map_dict
        self.counts_computed=True
        # counts=Counter(pt.sum((ensure_tensor(self[i][0])!=0).any(dim=-1)) for i in range(len(self)))
        # counts=[counts[k] for k in sorted(counts)]
        # counts=[c/sum(counts) for c in counts]

    def edge_dist_weights_dict(self)->Tuple[Dict[int,pt.Tensor],Dict[int,Dict[int,int]]]:
        if not self.counts_computed:
            self.compute_counts()
        return self.edge_rel_freq_dict,self.edge_map_dict


if __name__ == "__main__":
    d = GGG_DenseData(
        dataset="Trees12",
        schedule=[(0.001,0),(0.75,10),(1.0,20)],
    )
    d=Subset(d,list(range(50)))
    oldn=None
    loader=DataLoader(d,batch_size=10,shuffle=True,num_workers=3,persistent_workers=True)
    for e in range(50):
        bar=tqdm(iter(loader))
        for i,b in enumerate(bar):
            X,A,n=b
            bar.set_description(f"Visit  nodes  {e}{i}{n[0].item()}")
            if oldn!=n[0].item():
                bar.write(f"Old {oldn} new {n[0].item()} {e}{i}")
                oldn=n[0].item()
        if oldn==12:
            break


    # d = PEAWGANDenseData(
    #    dataset="product100", inner_kwargs=dict(verbose=True), print_statistics=True
    # )
    # d = PEAWGANDenseData(
    #    dataset="product5k1", inner_kwargs=dict(verbose=True), print_statistics=True
    # )
    # d = PEAWGANDenseData(
    #    dataset="product5k2", inner_kwargs=dict(verbose=True), print_statistics=True
    # )
