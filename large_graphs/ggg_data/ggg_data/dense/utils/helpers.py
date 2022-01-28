import numpy as np
import torch as pt
import networkx as nx


class _data_helper(object):
    """Helper class to build standard (x,A) structure for dataset"""

    def __init__(self, x=None, A=None):
        self.x = x
        self.A = A


from collections import namedtuple

TripartiteGraphCacheFake = namedtuple("TripartiteGraphCacheFake", "x adj")


def graph_dump(Py_data, dir_=''):
    graph_dict = {}

    all_g_edges = []
    all_g_features = []
    for py_g in Py_data:
        adj_m = py_g.A.cpu().detach().numpy()
        g = nx.from_numpy_matrix(adj_m)
        g_edge_list = nx.to_edgelist(g)
        all_g_edges.append(g_edge_list)

        all_g_features.append(py_g.x)

    graph_dict["edges"] = all_g_edges
    graph_dict["features"] = all_g_features

    pt.save(graph_dict, dir_)


def graph_load(dir_=None):
    Py_data = []
    graph_dict = pt.load(dir_)

    for idx, edge_list in enumerate(graph_dict["edges"]):
        if len(list(edge_list)[0]) == 3:
            n = max(max(nd1, nd2) for nd1, nd2, weight in edge_list) + 1

            A = np.zeros((n, n))
            for nd1, nd2, _ in edge_list:
                A[nd1][nd2] = 1
                A[nd2][nd1] = 1
        else:
            n = max(max(nd1, nd2) for nd1, nd2 in edge_list) + 1

            A = np.zeros((n, n))
            for nd1, nd2 in edge_list:
                A[nd1][nd2] = 1
                A[nd2][nd1] = 1

        x = graph_dict["features"][idx]

        # make sure disconnected nodes are taken into account
        # MolGAN had a case like this, but it's unusual, should be all connected graphs
        if len(x) > A.shape[0]:
            N_pad = len(x) - A.shape[0]
            A = np.pad(A, [(0, N_pad), (0, N_pad)])

        graph = _data_helper()
        graph.x = x.clone().detach()
        graph.A = pt.from_numpy(A)
        Py_data.append(graph)

    return Py_data


if __name__ == "__main__":
    g = nx.erdos_renyi_graph(5, 0.25)

    data_A = []
    data_X = []
    A = pt.tensor(nx.to_numpy_matrix(g))
    # Get attributes X as degree of nodes in A
    X = A.sum(dim=1)

    data_X.append(X)
    data_A.append(A)

    graph = _data_helper()
    graph.x = pt.tensor(data_X[0])
    graph.A = pt.tensor(data_A[0])
    Py_data = [graph]

    graph_dump(Py_data, dir_='test.pt')
    loaded_Py_dat = graph_load(dir_='test.pt')
    print(A)
    print(loaded_Py_dat[0].A)
