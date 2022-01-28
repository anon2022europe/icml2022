from warnings import warn

import torch as pt
try:
    from circuit_data.base import DATA_BASE_PATH
    from circuit_data.spice.toy_circuits import GithubSmall
except:
    warn("Couldn't find circuit_data, circuit dataset won't be available")
from torch.utils.data import Dataset, Subset
import torch_geometric.datasets as geom_data
from torch_geometric.data import Dataset as GeomDataset
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj
import os

from ggg_data.dense.utils.helpers import _data_helper
class OGBWrapper(Dataset):
    DATASETS={"ogbg-ppa","ogbg-molhiv","ogbg-molpcba","ogbg-code2"}
    def __init__(
            self, dataset,*dataset_args,first_N=5000,max_size=None,  **dataset_kwargs
    ):

        from ogb.graphproppred.dataset_pyg import PygGraphPropPredDataset
        self.data = Subset(PygGraphPropPredDataset(dataset),range(first_N))
        if max_size is not None:
            inds=[]
            for i in range(len(self.data)):
                g=self.data[i]
                if g.num_nodes<=max_size:
                    inds.append(i)
            self.data=Subset(self.data,inds)

    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        G: Data = self.data[item]
        Adense = to_dense_adj(G.edge_index).squeeze(0)  # we throw away edge features
        return None, Adense

class GeometricDenseWrapper(Dataset):
    def __init__(
        self, clsname, *dataset_args, root_dir="./geometric_data",max_size=None, **dataset_kwargs
    ):
        self.cls = getattr(geom_data, clsname)
        self.dataset_root_path = os.path.join(root_dir, clsname)
        assert (
            "root" not in dataset_kwargs
        ), f"We automatically set the rootdir for {clsname} to {self.dataset_root_path}, don't pass it additionally"
        self.data: GeomDataset = self.cls(
            self.dataset_root_path, *dataset_args, **dataset_kwargs
        )
        if max_size is not None:
            inds=[]
            for i in range(len(self.data)):
                g=self.data[i]
                if g.num_nodes<=max_size:
                    inds.append(i)
            self.data=Subset(self.data,inds)


    def __len__(self):
        return self.data.__len__()

    def __getitem__(self, item):
        G: Data = self.data[item]
        xdense = G.x.to_dense() if G.x.is_sparse else G.x
        Adense = to_dense_adj(G.edge_index)  # we throw away edge features
        return _data_helper(xdense, Adense)

class CircuitWrapper(Dataset):
    def __init__(self,name="github_small",Nmax=5000,dense=True):
        assert name=="github_small"
        self.inner=GithubSmall(dense=dense,max_nodes=Nmax,cache_path=os.path.join(DATA_BASE_PATH,f"{Nmax}-{name}.pt"))
    def __len__(self):
        return len(self.inner)
    def __getitem__(self, item):
        adj,x= [pt.from_numpy(y).float() for y in self.inner.__getitem__(item)[:2]]
        return _data_helper(x,adj)


