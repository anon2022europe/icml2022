# from https://github.com/JiaxuanYou/graph-generation/blob/master/baselines/baseline_simple.py
from collections import Counter
from warnings import warn

import attr
from scipy.linalg import toeplitz
from tqdm import tqdm
import pyemd
import scipy.optimize as opt
import numpy as np
import networkx as nx
import torch as pt


def Graph_generator_baseline_train_rulebased(graphs, generator="BA"):
    graph_nodes = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    graph_edges = [graphs[i].number_of_edges() for i in range(len(graphs))]
    parameter = {}
    for i in range(len(graph_nodes)):
        nodes = graph_nodes[i]
        edges = graph_edges[i]
        # based on rule, calculate optimal parameter
        parameter_temp = get_rule_param(edges, generator, nodes)
        # update parameter list
        if nodes not in parameter.keys():
            parameter[nodes] = parameter_temp
        else:
            count = parameter[nodes][-1]
            parameter[nodes] = [
                (parameter[nodes][i] * count + parameter_temp[i]) / (count + 1)
                for i in range(len(parameter[nodes]))
            ]
            parameter[nodes][-1] = count + 1
    # print(parameter)
    return parameter


import torch.distributions as td


@attr.s
class JointSampler:
    params = attr.ib()
    freqs = attr.ib()
    _dist = attr.ib(default=None, init=False)

    def __attrs_post_init__(self):
        if len(self.freqs) > 1:
            self._dist = td.Categorical(logits=pt.from_numpy(self.freqs).float())
        else:
            self._dist = None

    def sample(self, N):
        if self._dist:
            idx = self._dist.sample(N)
        else:
            idx = np.zeros(N, dtype=np.int64).tolist()
        return [self.params[i] for i in idx]


def fit_params(graphs, generator="BA"):
    parameters = []
    max_N = 0
    for i in tqdm(range(len(graphs)), desc="Fitting"):
        graph = graphs[i]
        nodes = graph.number_of_nodes()
        edges = graph.number_of_edges()
        # based on rule, calculate optimal parameter
        try:
            parameter_temp = get_rule_param(generator, nodes, edges)
        except ValueError as e:
            try:
                parameter_temp = get_param_opt(graph, generator, metric="degree")
            except ValueError as e:
                warn(f"Got error {e}, skipping graph")
                continue
        parameters.append(parameter_temp)
        max_N = max(max_N, nodes)
    # print(parameter)
    counts = Counter(parameters)
    counts = counts.most_common()
    params = [x[0] for x in counts]
    freqs = np.array([x[1] for x in counts])
    return params, freqs, max_N


def get_rule_param(generator, nodes, edges):
    if generator == "BA":
        # BA optimal: nodes = n; edges = (n-m)*m
        n = nodes
        m = (n - np.sqrt(n ** 2 - 4 * edges)) / 2
        parameter_temp = (int(n), int(m))
    elif generator == "Gnp":
        # Gnp optimal: nodes = n; edges = ((n-1)*n/2)*p
        n = nodes
        p = float(edges) / ((n - 1) * n / 2)
        parameter_temp = (int(n), p)
    else:
        raise ValueError(f"Unkown generator{generator}")
    return parameter_temp


def Graph_generator_baseline(graph_train, pred_num=1000, generator="BA"):
    graph_nodes = [graph_train[i].number_of_nodes() for i in range(len(graph_train))]
    graph_edges = [graph_train[i].number_of_edges() for i in range(len(graph_train))]
    repeat = pred_num // len(graph_train)
    graph_pred = []
    for i in range(len(graph_nodes)):
        nodes = graph_nodes[i]
        edges = graph_edges[i]
        # based on rule, calculate optimal parameter
        if generator == "BA":
            # BA optimal: nodes = n; edges = (n-m)*m
            n = nodes
            m = int((n - np.sqrt(n ** 2 - 4 * edges)) / 2)
            for j in range(repeat):
                graph_pred.append(nx.barabasi_albert_graph(n, m))
        if generator == "Gnp":
            # Gnp optimal: nodes = n; edges = ((n-1)*n/2)*p
            n = nodes
            p = float(edges) / ((n - 1) * n / 2)
            for j in range(repeat):
                graph_pred.append(nx.fast_gnp_random_graph(n, p))
    return graph_pred


def emd_distance(x, y, distance_scaling=1.0):
    support_size = max(len(x), len(y))
    d_mat = toeplitz(range(support_size)).astype(np.float)
    distance_mat = d_mat / distance_scaling

    # convert histogram values x and y to float, and make them equal len
    x = x.astype(np.float)
    y = y.astype(np.float)
    if len(x) < len(y):
        x = np.hstack((x, [0.0] * (support_size - len(x))))
    elif len(y) < len(x):
        y = np.hstack((y, [0.0] * (support_size - len(y))))

    emd = pyemd.emd(x, y, distance_mat)
    return emd


# def Loss(x,args):
#     '''
#
#     :param x: 1-D array, parameters to be optimized
#     :param args: tuple (n, G, generator, metric).
#     n: n for pred graph;
#     G: real graph in networkx format;
#     generator: 'BA', 'Gnp', 'Powerlaw';
#     metric: 'degree', 'clustering'
#     :return: Loss: emd distance
#     '''
#     # get argument
#     generator = args[2]
#     metric = args[3]
#
#     # get real and pred graphs
#     G_real = args[1]
#     if generator=='BA':
#         G_pred = nx.barabasi_albert_graph(args[0],int(np.rint(x)))
#     if generator=='Gnp':
#         G_pred = nx.fast_gnp_random_graph(args[0],x)
#
#     # define metric
#     if metric == 'degree':
#         G_real_hist = np.array(nx.degree_histogram(G_real))
#         G_real_hist = G_real_hist / np.sum(G_real_hist)
#         G_pred_hist = np.array(nx.degree_histogram(G_pred))
#         G_pred_hist = G_pred_hist/np.sum(G_pred_hist)
#     if metric == 'clustering':
#         G_real_hist, _ = np.histogram(
#             np.array(list(nx.clustering(G_real).values())), bins=50, range=(0.0, 1.0), density=False)
#         G_real_hist = G_real_hist / np.sum(G_real_hist)
#         G_pred_hist, _ = np.histogram(
#             np.array(list(nx.clustering(G_pred).values())), bins=50, range=(0.0, 1.0), density=False)
#         G_pred_hist = G_pred_hist / np.sum(G_pred_hist)
#
#     loss = emd_distance(G_real_hist,G_pred_hist)
#     return loss


def Loss(x, n, G_real, generator, metric="degree"):
    """

    :param x: 1-D array, parameters to be optimized
    :param
    n: n for pred graph;
    G: real graph in networkx format;
    generator: 'BA', 'Gnp', 'Powerlaw';
    metric: 'degree', 'clustering'
    :return: Loss: emd distance
    """
    # get argument

    # get real and pred graphs
    if generator == "BA":
        G_pred = nx.barabasi_albert_graph(n, int(np.rint(x)))
    elif generator == "Gnp":
        G_pred = nx.fast_gnp_random_graph(n, x)
    else:
        raise ValueError()

    # define metric
    if metric == "degree":
        G_real_hist = np.array(nx.degree_histogram(G_real))
        G_real_hist = G_real_hist / np.sum(G_real_hist)
        G_pred_hist = np.array(nx.degree_histogram(G_pred))
        G_pred_hist = G_pred_hist / np.sum(G_pred_hist)
    elif metric == "clustering":
        G_real_hist, _ = np.histogram(
            np.array(list(nx.clustering(G_real).values())),
            bins=50,
            range=(0.0, 1.0),
            density=False,
        )
        G_real_hist = G_real_hist / np.sum(G_real_hist)
        G_pred_hist, _ = np.histogram(
            np.array(list(nx.clustering(G_pred).values())),
            bins=50,
            range=(0.0, 1.0),
            density=False,
        )
        G_pred_hist = G_pred_hist / np.sum(G_pred_hist)
    else:
        NotImplementedError()

    loss = emd_distance(G_real_hist, G_pred_hist)
    return loss


def optimizer_brute(x_min, x_max, x_step, n, G_real, generator, metric):
    loss_all = []
    x_list = np.arange(x_min, x_max, x_step)
    for x_test in x_list:
        if generator == "BA" and x_test <= n:
            continue
        loss_all.append(Loss(x_test, n, G_real, generator, metric))
    x_optim = x_list[np.argmin(np.array(loss_all))]
    return x_optim


def Graph_generator_baseline_train_optimizationbased(
    graphs, generator="BA", metric="degree"
):
    graph_nodes = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    parameter = {}
    for i in range(len(graphs)):
        print("graph ", i)
        graph = graphs[i]

        nodes, parameter_temp = get_param_opt(graph, generator, metric)

        parameter_temp = [*parameter_temp, 1]

        # update parameter list
        if nodes not in parameter.keys():
            parameter[nodes] = parameter_temp
        else:
            count = parameter[nodes][2]
            parameter[nodes] = [
                (parameter[nodes][i] * count + parameter_temp[i]) / (count + 1)
                for i in range(len(parameter[nodes]))
            ]
            parameter[nodes][2] = count + 1
    print(parameter)
    return parameter


def get_param_opt(graph, generator, metric="degree"):
    nodes = graph.number_of_nodes()
    if generator == "BA":
        x_min, x_max, x_step = 1, 10, 1
        n = nodes
        m = optimizer_brute(x_min, x_max, x_step, nodes, graph, generator, metric)
        parameter_temp = (n, m, 1)
    elif generator == "Gnp":
        n = nodes
        x_min, x_max, x_step = 1e-6, 1, 0.01
        p = optimizer_brute(x_min, x_max, x_step, nodes, graph, generator, metric)
        ## if use evolution
        # result = opt.differential_evolution(Loss,bounds=[(0,1)],args=(nodes, graphs[i], generator, metric),maxiter=1000)
        # p = result.x
        parameter_temp = (n, p)
    else:
        raise NotImplementedError()
    return nodes, parameter_temp


def Graph_generator_baseline_test(graph_nodes, parameter, generator="BA"):
    graphs = []
    for i in range(len(graph_nodes)):
        nodes = graph_nodes[i]
        if not nodes in parameter.keys():
            nodes = min(parameter.keys(), key=lambda k: abs(k - nodes))
        if generator == "BA":
            n = int(parameter[nodes][0])
            m = int(np.rint(parameter[nodes][1]))
            print(n, m)
            graph = nx.barabasi_albert_graph(n, m)
        if generator == "Gnp":
            n = int(parameter[nodes][0])
            p = parameter[nodes][1]
            print(n, p)
            graph = nx.fast_gnp_random_graph(n, p)
        graphs.append(graph)
    return graphs


def ensure_nx(X):
    if pt.is_tensor(X) or pt.is_tensor(X[0]) or type(X) == np.ndarray:
        return [
            nx.from_numpy_array(x.int().numpy())
            for x in tqdm(X, desc="Nxification", leave=False)
        ]
    else:
        return X


class ClassicalBaseline(pt.nn.Module):
    def __init__(self, training_graphs, generator="BA"):
        super().__init__()
        training_graphs = ensure_nx(training_graphs)
        self.gen = generator
        assert generator in {"BA", "Gnp"}
        params, freqs, self.max_N = fit_params(training_graphs, generator)
        self.param_dist = JointSampler(params, freqs)

    def forward(self, batch_size=1, mode="pt"):
        return self.sample(batch_size, ret_mode=mode)

    def sample(self, batch_size, ret_mode="pt", device=None):
        assert ret_mode in {"nx", "np", "pt"}
        params = self.param_dist.sample([batch_size])
        if self.gen == "BA":
            A = [nx.barabasi_albert_graph(*p) for p in params]
        else:
            A = [nx.fast_gnp_random_graph(*p) for p in params]
        if ret_mode == "nx":
            return A
        else:
            A = np.stack([self.pad_to_max(nx.to_numpy_array(g), self.max_N) for g in A])
            X = np.ones([A.shape[0], A.shape[-1], 1])
            if ret_mode == "pt":
                A = pt.from_numpy(A)
                X = pt.from_numpy(X)
            return X, A

    def pad_to_max(self, A, N_max):
        N = A.shape[-1]
        if N < N_max:
            p = N_max - N
            A = np.pad(A, [(0, p), (0, p)])
        return A


if __name__ == "__main__":
    graphs = [
        nx.barabasi_albert_graph(np.random.randint(100, 102), 5) for _ in range(5)
    ]
    cb = ClassicalBaseline(graphs, "Gnp")
    print(cb.param_dist)
    print(cb.sample(1))
