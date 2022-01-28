import os
from typing import Union

import numpy as np
import torch as pt
from torch.utils.data import Dataset, Subset

from ggg_data.dense.GGG_DenseData import GGG_DenseData
from ggg.models.components.overlapping_com_detection.overlapping_com_detection import GGG_Chunker


class ChunkedDenseWrapper(Dataset):
    def __init__(
            self,
            dense_data: Union[GGG_DenseData, Subset],
            graph_chunker: GGG_Chunker,
    ):
        self.dense_data = dense_data
        self.dense_data_accessor = self.dense_data if not type(self.dense_data) is Subset else self.dense_data.dataset
        self.graph_chunker = graph_chunker

        self.graphs_partitioned = False
        self.partition_counts_computed = False
        self.partition_data = None
        self.overlap_region_rel_freq = None

        self.max_N = self.dense_data_accessor.max_N
        self.max_K = 0
        self.max_N_chunk = 0
        self.max_ORS = 0

        self.chunk_dist_weights()

    def __len__(self):
        return len(self.dense_data)

    def __getitem__(self, idx):
        x, A, N = self.dense_data[idx]

        N_chunk = pt.zeros(self.max_K).long()
        OR = pt.zeros(self.max_K, self.max_K).long()
        M = pt.zeros(self.max_N, self.max_K).long()

        _, membership_matrix = self.partition_data[idx]
        K = membership_matrix.size(1)
        N_chunk[:K] = pt.sum(membership_matrix, dim=0).long()
        p = pt.sum(membership_matrix, dim=1)
        overlap_cardinality = (p.view(-1, 1) == pt.arange(1, K + 1)).float()
        OR[:K, :K] = (overlap_cardinality.T @ membership_matrix).long()
        M[:N, :K] = membership_matrix.long()

        return x, A, N, K, N_chunk, OR, M

    def node_dist_weights(self):
        return self.dense_data_accessor.node_dist_weights()

    def chunks(self):
        """
        Compute and return the overlap-agnostic partitioning of the graphs in the dataset

        :return: overlap-agnostic partition data, see output of :func:`~ggg.model.components.overlapping_com_detection.overlapping_com_detection.GGG_Chunker.process_dataset`
        """
        if not self.graphs_partitioned:
            self.partition_data = []
            old_zero_pad = self.dense_data_accessor.zero_pad
            self.dense_data_accessor.zero_pad = False

            for chunk_list, M in self.graph_chunker.process_dataset(self.dense_data):
                K = len(chunk_list)
                self.max_K = max(self.max_K, K)
                for _, _, N_chunk, _ in chunk_list:
                    self.max_N_chunk = max(self.max_N_chunk, N_chunk)
                self.partition_data.append((chunk_list, M))
            for _, M in self.partition_data:
                p = pt.sum(M, dim=1)
                overlap_cardinality = (p.view(-1, 1) == pt.arange(1, self.max_K + 1)).float()
                self.max_ORS = max(self.max_ORS, pt.amax(overlap_cardinality.T @ M, dim=(0, 1)).long().item())

            self.graphs_partitioned = True
            self.dense_data_accessor.zero_pad = old_zero_pad

        return self.partition_data

    def chunk_dist_weights(self) -> np.ndarray:
        """
        Compute and return the joint distribution of chunk number and size + the joint distribution of overlapping region number and size

        :return: joint distribution of chunk number, overlapping region cardinality and size, numpy array
        """
        stat_cache_file = os.path.join(os.path.expanduser("~/.datasets"),
                                       f"{self.dense_data_accessor.DS_NAME}_chunks_{self.graph_chunker.algorithm}_stat_cache.pt")
        if not self.partition_counts_computed:
            if self.dense_data_accessor.force_fresh or not os.path.exists(stat_cache_file):
                self.chunks()
                self.compute_partition_counts()
                with open(stat_cache_file, "wb") as f:
                    pt.save(dict(partition_data=self.partition_data,
                                 overlap_region_rel_freq=self.overlap_region_rel_freq,
                                 max_K=self.max_K, max_N_chunk=self.max_N_chunk,
                                 max_ORS=self.max_ORS), f)
            else:
                with open(stat_cache_file, "rb") as f:
                    d = pt.load(f)
                    self.partition_data = d["partition_data"]
                    self.overlap_region_rel_freq = d["overlap_region_rel_freq"]
                    self.max_K = d["max_K"]
                    self.max_N_chunk = d["max_N_chunk"]
                    self.max_ORS = d["max_ORS"]
                self.graphs_partitioned = True
                self.partition_counts_computed = True

        return self.overlap_region_rel_freq

    def compute_partition_counts(self):
        old_zero_pad = self.dense_data_accessor.zero_pad
        self.dense_data_accessor.zero_pad = False
        possible_cardinalities = pt.arange(1, self.max_K + 1)
        overlap_region_counts = np.zeros((self.max_N, self.max_K, self.max_K,
                                          self.max_N + 1, self.max_K, self.max_ORS + 1))

        for i in range(len(self.dense_data)):
            _, _, N = self.dense_data[i]
            _, M = self.partition_data[i]

            p = pt.sum(M, dim=1)
            overlap_cardinality = (p.view(-1, 1) == possible_cardinalities).float()
            OR = (overlap_cardinality.T @ M).long()
            K = pt.count_nonzero(OR, dim=1)

            N_cardinality = pt.sum(OR, dim=1) // possible_cardinalities
            K_cardinality = pt.count_nonzero(K)

            for cardinality in range(1, M.size(1) + 1):
                for chunk in range(M.size(1)):
                    overlap_region_counts[N - 1, K_cardinality - 1, cardinality - 1,
                                          N_cardinality[cardinality - 1], K[cardinality - 1] - 1,
                                          OR[cardinality - 1, chunk]] += 1

        overlap_region_rel_freq = overlap_region_counts / np.sum(overlap_region_counts)

        if self.dense_data_accessor.print_statistics:
            np.set_printoptions(suppress=True)
            overlap_region_counts_non_zero = np.nonzero(overlap_region_counts)
            print(
                "These are the overlapping region size occurrences in the dataset {} \n {}".format(
                    self.dense_data_accessor.DS_NAME,
                    np.hstack((np.argwhere(overlap_region_counts) + np.array([1, 1, 1, 0, 1, 0]),
                               overlap_region_counts[overlap_region_counts_non_zero].reshape((-1, 1))))
                )
            )
            print(
                "These are the relative occurrences {} {}".format(
                    self.dense_data_accessor.DS_NAME, overlap_region_rel_freq[overlap_region_counts_non_zero]
                )
            )
        print(f"Maximum chunk number {self.dense_data_accessor.DS_NAME}:{self.max_K}")
        print(f"Maximum chunk size {self.dense_data_accessor.DS_NAME}:{self.max_N_chunk}")
        print(f"Maximum overlapping region size {self.dense_data_accessor.DS_NAME}:{self.max_ORS}")

        self.dense_data_accessor.zero_pad = old_zero_pad
        self.overlap_region_rel_freq = overlap_region_rel_freq
        self.partition_counts_computed = True
