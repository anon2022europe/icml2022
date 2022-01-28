from warnings import simplefilter

from ggg.models.ggg_model import GGG

simplefilter(action="ignore", category=FutureWarning)

import os
import pickle
import numpy as np
import networkx as nx

import pyemd
from scipy.linalg import toeplitz

from ggg.evaluation.statistics.utils.helpers import ProgressBar, get_dist


class graph_gen(object):
    def __init__(self, config, checkpoint_path):
        """Initialize configurations."""

        # Model parameters
        self.z_dim = config["z_dim"]
        self.dataset = config["dataset"]
        self.node_dist = config["node_dist"]
        self.batch_size = config["batch_size"]
        self.generator_arch = config["generator_arch"]

        # Graph generation file parameters
        self.numb_graphs = config["numb_graphs"]

        self.version = config["version_exp"]
        self.model_epoch = config["model_epoch"]
        self.model_struct = config["model_struct"]
        self.specific_exp = config["specific_exp"]

        # Relative paths
        self.exps_dir = config["exps_dir"]
        self.dist_dir = config["dist_dir"]
        self.graphs_dir = config["graphs_dir"]
        self.dataset_dist_dir = config["dataset_dist_dir"]
        self.dataset_graphs_dir = config["dataset_graphs_dir"]
        self.checkpoint_path = checkpoint_path

        self.model = self.get_trained_model()

    def get_trained_model(self):
        """Load structure and weights of trained model"""

        if self.checkpoint_path:
            self._full_model = GGG.load_from_checkpoint(self.checkpoint_path)
            self._full_model: GGG
            return self._full_model.generator
        else:
            return None

    def get_distribution(self, file_, flag_):
        """Function to get metrics from graphs
        metrics --> degree/cycles/etc"""

        with open(file_, "rb") as f:
            pkl_list = pickle.load(f)

        dist_ = get_dist(flag_, pkl_list)

        return dist_

    def emd_distance(self, x, y, distance_scaling=1.0):
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

    def loss_(self, x_test, G_real, metric="degree"):

        graphs_ = self.create_graphs(brute_opt=True)
        G_pred = get_dist(metric, graphs_)

        G_pred = np.random.choice(G_pred, size=min(len(G_pred), 1000), replace=False)
        G_real = np.random.choice(G_real, size=min(len(G_pred), 1000), replace=False)

        # define metric
        if np.sum(G_pred) == 0:
            loss = np.inf
        else:
            G_real_hist = np.array(G_real)
            G_pred_hist = np.array(G_pred)

            loss = self.emd_distance(G_real_hist, G_pred_hist)
        return loss

    def optimizer_brute(self, x_min, x_max, x_step, G_real, metric="degree"):
        loss_all = []
        x_list = np.arange(x_min, x_max, x_step)
        pr = ProgressBar(60, len(x_list))
        for i, x_test in enumerate(x_list):
            loss_all.append(self.loss_(x_test, G_real, metric))
            pr.update(i + 1)
        x_optim = x_list[np.argmin(np.array(loss_all))]
        return x_optim

    def pickle_to_dataset(self):
        return NotImplementedError

    def dataset_to_dist(self, _data, already_pkl=False):
        """Generate networkx graphs from datasets (ex. QM9)"""
        # TODO: Converting .sparsedataset --> graph pickle

        if not already_pkl:
            pkl_file = self.dataset_graphs_dir + self.dataset + ".pkl"

            gen_graphs = []

            for graph in _data:
                adj_m = graph.A.detach().numpy()

                a_np = np.clip(adj_m, 0, 1)
                np.fill_diagonal(a_np, 0)
                g = nx.from_numpy_matrix(a_np)
                gen_graphs.append(g)

            # Save generated networkx graphs pickle list
            with open(pkl_file, "wb") as f:
                pickle.dump(gen_graphs, f)

        else:
            pkl_file = self.dataset_graphs_dir + self.dataset + ".pkl"

        cycle_dist_pkl = self.dataset_dist_dir + self.dataset + "_cycleD.pkl"
        degree_dist_pkl = self.dataset_dist_dir + self.dataset + "_degreeD.pkl"

        # Save vector of degree of generated graphs
        d_dist = self.get_distribution(pkl_file, "degree")
        with open(degree_dist_pkl, "wb") as f:
            pickle.dump(d_dist, f)

        # Save vector of cycle counts of generated graphs
        c_dist = self.get_distribution(pkl_file, "cycles")
        with open(cycle_dist_pkl, "wb") as f:
            pickle.dump(c_dist, f)

    def sample(self, batch_size=None):
        X, A = self.model.sample(batch_size=batch_size)

        return X, A

    def create_graphs(self, brute_opt=False):
        gen_graphs = []

        if brute_opt:
            numb_graphs = 250
            z_ori_list = []
        else:
            pr = ProgressBar(60, self.numb_graphs)
            numb_graphs = self.numb_graphs
            z_ori_list = []

        while len(gen_graphs) < numb_graphs:
            nodes_logits, edges_hat = self.sample(batch_size=self.batch_size)

            for b in range(self.batch_size):
                A = edges_hat[b]

                G = nx.from_numpy_matrix(A.detach().numpy())
                gen_graphs.append(G)

                if not brute_opt:
                    pr.update(len(gen_graphs))

                if len(gen_graphs) >= self.numb_graphs:
                    break

        if brute_opt:
            return gen_graphs
        else:
            return gen_graphs, z_ori_list

    def analysis(self):
        """Generate graphs from saved models."""

        original_file_dir = self.dataset_dist_dir + self.dataset + "_degreeD.pkl"

        with open(original_file_dir, "rb") as f:
            original_file = pickle.load(f)

        pkl_file = self.graphs_dir + self.specific_exp + ".pkl"
        noise_vec_pkl = self.dist_dir + self.specific_exp + "_noise_vec.pkl"
        cycle_dist_pkl = self.dist_dir + self.specific_exp + "_cycleD.pkl"
        degree_dist_pkl = self.dist_dir + self.specific_exp + "_degreeD.pkl"

        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)

        # Optimized threshold
        gen_graphs, z_noise = self.create_graphs()

        # Save generated networkx graphs pickle list
        with open(pkl_file, "wb") as f:
            pickle.dump(gen_graphs, f)

        # Save vector of degree of generated graphs
        d_dist = self.get_distribution(pkl_file, "degree")
        with open(degree_dist_pkl, "wb") as f:
            pickle.dump(d_dist, f)

        # Save vector of cycle counts of generated graphs
        c_dist = self.get_distribution(pkl_file, "cycles")
        with open(cycle_dist_pkl, "wb") as f:
            pickle.dump(c_dist, f)
