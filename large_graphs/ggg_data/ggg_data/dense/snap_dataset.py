import os
import os.path as osp

import torch
import pandas
import numpy as np
from tqdm import tqdm
from torch_sparse import coalesce
from torch_geometric.data import (Data, download_url,
                                  extract_gz, extract_tar)
from torch.utils.data import Dataset
from torch_geometric.data.makedirs import makedirs


class EgoData(Data):
    def __inc__(self, key, item):
        if key == 'circle':
            return self.num_nodes
        elif key == 'circle_batch':
            return item.max().item() + 1 if item.numel() > 0 else 0
        else:
            return super(EgoData, self).__inc__(key, item)


def get_ego_data(circles_file,edges_file,egofeat_file,feat_file,featnames_file,name,all_featnames):
    x = None
    if name != 'gplus':  # Don't read node features on g-plus:
        x_ego = pandas.read_csv(egofeat_file, sep=' ', header=None,
                                dtype=np.float32)
        x_ego = torch.from_numpy(x_ego.values)

        x = pandas.read_csv(feat_file, sep=' ', header=None,
                            dtype=np.float32)
        x = torch.from_numpy(x.values)[:, 1:]

        x_all = torch.cat([x, x_ego], dim=0)

        # Reorder `x` according to `featnames` ordering.
        x_all = torch.zeros(x.size(0), len(all_featnames))
        with open(featnames_file, 'r') as f:
            featnames = f.read().split('\n')[:-1]
            featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
        indices = [all_featnames[featname] for featname in featnames]
        x_all[:, torch.tensor(indices)] = x
        x = x_all

    idx = pandas.read_csv(feat_file, sep=' ', header=None, dtype=str,
                          usecols=[0], squeeze=True)

    idx_assoc = {}
    for i, j in enumerate(idx):
        idx_assoc[j] = i

    circles = []
    circles_batch = []
    with open(circles_file, 'r') as f:
        for i, circle in enumerate(f.read().split('\n')[:-1]):
            circle = [idx_assoc[c] for c in circle.split()[1:]]
            circles += circle
            circles_batch += [i] * len(circle)
    circle = torch.tensor(circles)
    circle_batch = torch.tensor(circles_batch)

    row = pandas.read_csv(edges_file, sep=' ', header=None, dtype=str,
                          usecols=[0], squeeze=True)
    col = pandas.read_csv(edges_file, sep=' ', header=None, dtype=str,
                          usecols=[1], squeeze=True)

    row = torch.tensor([idx_assoc[i] for i in row])
    col = torch.tensor([idx_assoc[i] for i in col])

    N = max(int(row.max()), int(col.max())) + 2
    N = x.size(0) if x is not None else N

    row_ego = torch.full((N - 1, ), N - 1, dtype=torch.long)
    col_ego = torch.arange(N - 1)

    # Ego node should be connected to every other node.
    row = torch.cat([row, row_ego, col_ego], dim=0)
    col = torch.cat([col, col_ego, row_ego], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    edge_index, _ = coalesce(edge_index, None, N, N)
    return Data(x=x, edge_index=edge_index, circle=circle,
                circle_batch=circle_batch)
def read_ego(files, name,max_size=None):
    all_featnames = []
    files = [
        x for x in files if x.split('.')[-1] in
        ['circles', 'edges', 'egofeat', 'feat', 'featnames']
    ]
    for i in range(4, len(files), 5):
        featnames_file = files[i]
        with open(featnames_file, 'r') as f:
            featnames = f.read().split('\n')[:-1]
            featnames = [' '.join(x.split(' ')[1:]) for x in featnames]
            all_featnames += featnames
    all_featnames = sorted(list(set(all_featnames)))
    all_featnames = {key: i for i, key in enumerate(all_featnames)}

    data_list = []
    total_nodes=0
    avg_nodes=0
    file_list=[]
    if max_size is None:
        bar=range(0, len(files), 5)       
    else:
        bar=tqdm(range(0, len(files), 5),desc=f"Filtering for max_size={max_size}")
    for i in bar:
        circles_file = files[i]
        edges_file = files[i + 1]
        egofeat_file = files[i + 2]
        feat_file = files[i + 3]
        featnames_file = files[i + 4]
        if max_size is not None:
            data=get_ego_data(circles_file,edges_file,egofeat_file,feat_file,featnames_file,name,all_featnames)
            if data.edge_index.shape[-1]>max_size:
                continue
        file_list.append((circles_file,edges_file,egofeat_file,feat_file,featnames_file))
        if max_size is not None:
            bar.set_description(f"Filtering for max_size={max_size}, retained {len(file_list)}/{i//5}")



    return file_list,len(file_list),name,all_featnames


def read_soc(files, name):
    skiprows = 4
    if name == 'pokec':
        skiprows = 0

    edge_index = pandas.read_csv(files[0], sep='\t', header=None,
                                 skiprows=skiprows, dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()
    num_nodes = edge_index.max().item() + 1
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


def read_wiki(files, name):
    edge_index = pandas.read_csv(files[0], sep='\t', header=None, skiprows=4,
                                 dtype=np.int64)
    edge_index = torch.from_numpy(edge_index.values).t()

    idx = torch.unique(edge_index.flatten())
    idx_assoc = torch.full((edge_index.max() + 1, ), -1, dtype=torch.long)
    idx_assoc[idx] = torch.arange(idx.size(0))

    edge_index = idx_assoc[edge_index]
    num_nodes = edge_index.max().item() + 1
    edge_index, _ = coalesce(edge_index, None, num_nodes, num_nodes)

    return [Data(edge_index=edge_index, num_nodes=num_nodes)]


class SNAPDataset(Dataset):
    r"""A variety of graph datasets collected from `SNAP at Stanford University
    <https://snap.stanford.edu/data>`_.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string): The name of the dataset.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    url = 'https://snap.stanford.edu/data'

    available_datasets = {
        'ego-facebook': ['facebook.tar.gz'],
        'ego-gplus': ['gplus.tar.gz'],
        'ego-twitter': ['twitter.tar.gz'],
        'soc-epinions1': ['soc-Epinions1.txt.gz'],
        'soc-livejournal1': ['soc-LiveJournal1.txt.gz'],
        'soc-pokec': ['soc-pokec-relationships.txt.gz'],
        'soc-slashdot0811': ['soc-Slashdot0811.txt.gz'],
        'soc-slashdot0922': ['soc-Slashdot0902.txt.gz'],
        'wiki-vote': ['wiki-Vote.txt.gz'],
    }

    def __init__(self, name, max_size=None,transform=None, pre_transform=None,
                 pre_filter=None,root=os.path.expanduser("~/.datasets/SNAPDataset/")):
        super().__init__()
        self.root=root
        self.name = name.lower()
        assert self.name in self.available_datasets.keys()

        self._download()
        self.file_list,self.N,self.name,self.all_featnames = self.process(max_size)
       
    
    def __len__(self):
        return self.N

    @property
    def raw_dir(self):
        return osp.join(self.root, self.name, 'raw')

    def __getitem__(self,i):
        return get_ego_data(*self.file_list[i],self.name,self.all_featnames)



    def _download(self):
        if osp.isdir(self.raw_dir) and len(os.listdir(self.raw_dir)) > 0:
            return

        makedirs(self.raw_dir)
        self.download()

    def download(self):
        for name in self.available_datasets[self.name]:
            path = download_url('{}/{}'.format(self.url, name), self.raw_dir)
            print(path)
            if name.endswith('.tar.gz'):
                extract_tar(path, self.raw_dir)
            elif name.endswith('.gz'):
                extract_gz(path, self.raw_dir)
            os.unlink(path)

    def process(self,max_size=None):
        raw_dir = self.raw_dir
        filenames = os.listdir(self.raw_dir)
        if len(filenames) == 1 and osp.isdir(osp.join(raw_dir, filenames[0])):
            raw_dir = osp.join(raw_dir, filenames[0])

        raw_files = sorted([osp.join(raw_dir, f) for f in os.listdir(raw_dir)])
        if self.name[:4] == 'ego-':
            return read_ego(raw_files, self.name[4:],max_size=max_size)
        else:
            raise NotImplementedError
        return file_list,N,name,all_featnames
        #elif self.name[:4] == 'soc-':
        #    data_list = read_soc(raw_files, self.name[:4])
        #elif self.name[:5] == 'wiki-':
        #    data_list = read_wiki(raw_files, self.name[5:])
        #else:
        #    raise NotImplementedError

        #if len(data_list) > 1 and self.pre_filter is not None:
        #    data_list = [data for data in data_list if self.pre_filter(data)]

        #if self.pre_transform is not None:
        #    data_list = [self.pre_transform(data) for data in data_list]

        #torch.save(self.collate(data_list), self.processed_paths[0])

    def __repr__(self):
        return 'SNAP-{}({})'.format(self.name, len(self))
