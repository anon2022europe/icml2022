import torch as pt
from torch.utils.data import Dataset
import torch_geometric.datasets as geom_data
from torch_geometric.data import Dataset as GeomDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import os

from ggg.data.dense.utils.helpers import _data_helper


class GeometricDenseWrapper(Dataset):
    def __init__(
        self, clsname, *dataset_args, root_dir="./geometric_data", **dataset_kwargs
    ):
        self.cls = getattr(geom_data, clsname)
        self.dataset_root_path = os.path.join(root_dir, clsname)
        assert (
            "root" not in dataset_kwargs
        ), f"We automatically set the rootdir for {clsname} to {self.dataset_root_path}, don't pass it additionally"
        self.data: GeomDataset = self.cls(
            self.dataset_root_path, *dataset_args, **dataset_kwargs
        )

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        G: Data = self.data[item]
        xdense = G.x.to_dense() if G.x.is_sparse else G.x
        Adense = to_dense_adj(G.edge_index)  # we throw away edge features
        return _data_helper(xdense, Adense)
