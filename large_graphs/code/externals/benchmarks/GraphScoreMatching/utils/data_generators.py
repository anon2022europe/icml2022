import json
import logging
import os
import pickle

import networkx as nx
import numpy as np
from ggg.data.dense.GGG_DenseData import GGG_DenseData
from ggg.data.dense.PEAWGANDenseStructureData import PEAWGANDenseStructureData


def n_community(num_communities, max_nodes, p_inter=0.05):
    c_sizes = [max_nodes // num_communities] * num_communities
    max_nodes = max_nodes // num_communities * num_communities
    p_inter = (p_inter * max_nodes) / (
        num_communities
        * (num_communities - 1)
        // 2
        * (max_nodes // num_communities) ** 2
    )
    print(num_communities, max_nodes, end=" ")
    graphs = [nx.gnp_random_graph(c_sizes[i], 0.7, seed=i) for i in range(len(c_sizes))]
    # for i, graph in enumerate(graphs):
    #     temp_arr = np.zeros(len(graphs))
    #     temp_arr[i] = 1.0
    #     nx.set_node_attributes(graph, temp_arr.tolist(), 'feature')
    G = nx.disjoint_union_all(graphs)
    communities = list(nx.connected_component_subgraphs(G))
    add_edge = 0
    for i in range(len(communities)):
        subG1 = communities[i]
        nodes1 = list(subG1.nodes())
        for j in range(i + 1, len(communities)):
            subG2 = communities[j]
            nodes2 = list(subG2.nodes())
            has_inter_edge = False
            for n1 in nodes1:
                for n2 in nodes2:
                    if np.random.rand() < p_inter:
                        G.add_edge(n1, n2)
                        has_inter_edge = True
                        add_edge += 1
            if not has_inter_edge:
                G.add_edge(nodes1[0], nodes2[0])
                add_edge += 1
    print(
        "connected comp: ",
        len(list(nx.connected_component_subgraphs(G))),
        "add edges: ",
        add_edge,
    )
    print(G.number_of_edges())
    return G


NAME_TO_NX_GENERATOR = {
    "community": n_community,
    "grid": nx.generators.grid_2d_graph,  # grid_2d_graph(m, n, periodic=False, create_using=None)
    "gnp": nx.generators.fast_gnp_random_graph,  # fast_gnp_random_graph(n, p, seed=None, directed=False)
    "ba": nx.generators.barabasi_albert_graph,  # barabasi_albert_graph(n, m, seed=None)
    "pow_law": lambda **kwargs: nx.configuration_model(
        nx.generators.random_powerlaw_tree_sequence(**kwargs, gamma=3, tries=2000)
    ),
    "except_deg": lambda **kwargs: nx.expected_degree_graph(**kwargs, selfloops=False),
    "cycle": nx.cycle_graph,
    "c_l": nx.circular_ladder_graph,
    "lobster": nx.random_lobster
    # 'ego': nx.generators.ego_graph  # ego_graph(G, n, radius=1, center=True, undirected=False, distance=None)
}


class GraphGenerator:
    def __init__(self, graph_type="grid", possible_params_dict=None, corrupt_func=None):
        if possible_params_dict is None:
            possible_params_dict = {}
        assert isinstance(possible_params_dict, dict)
        self.count = {k: 0 for k in possible_params_dict}
        self.possible_params = possible_params_dict
        self.corrupt_func = corrupt_func
        self.nx_generator = NAME_TO_NX_GENERATOR[graph_type]

    def __call__(self):
        params = {}
        for k, v_list in self.possible_params.items():
            params[k] = np.random.choice(v_list)
        graph = self.nx_generator(**params)
        graph = nx.relabel.convert_node_labels_to_integers(graph)
        if self.corrupt_func is not None:
            graph = self.corrupt_func(self.corrupt_func)
        return graph


def gen_graph_list(
    graph_type="grid",
    possible_params_dict=None,
    corrupt_func=None,
    length=1024,
    save_dir=None,
    file_name=None,
    max_node=None,
    min_node=None,
):
    params = locals()
    logging.info("gen data: " + json.dumps(params))
    if file_name is None:
        file_name = graph_type + "_" + str(length)
    file_path = os.path.join(save_dir, file_name)
    graph_generator = GraphGenerator(
        graph_type=graph_type,
        possible_params_dict=possible_params_dict,
        corrupt_func=corrupt_func,
    )
    graph_list = []
    i = 0
    max_N = 0
    while i < length:
        graph = graph_generator()
        if max_node is not None and graph.number_of_nodes() > max_node:
            continue
        if min_node is not None and graph.number_of_nodes() < min_node:
            continue
        print(i, graph.number_of_nodes(), graph.number_of_edges())
        max_N = max(max_N, graph.number_of_nodes())
        if graph.number_of_nodes() <= 1:
            continue
        graph_list.append(graph)
        i += 1
    if save_dir is not None:
        with open(file_path + ".pkl", "wb") as f:
            pickle.dump(obj=graph_list, file=f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(file_path + ".txt", "w") as f:
            f.write(json.dumps(params))
            f.write(f"max node number: {max_N}")
    print(max_N)
    return graph_list


def load_ggg(
    filename,
    dataset=None,
    data_dir="data",
    k_eign=4,
    use_lpl=False,
    largN=False,
    structure=False,
    need_set=False,
):
    if structure:
        untreated_graphs = PEAWGANDenseStructureData(
                data_dir=data_dir,
                filename=filename,
                dataset=dataset,
                k_eigenvals=k_eign,
                use_laplacian=use_lpl,
                large_N_approx=largN,
                inner_kwargs=None,
                cut_train_size=False,
                zero_pad=True,
                dropout_ps=None,
                fake_eigen=False
            )
        # with open(file_path, "rb") as f:
        #     untreated_graphs = pickle.load(f)
    else:
        untreated_graphs = GGG_DenseData(
            data_dir=data_dir,
            filename=filename,
            dataset=dataset,
            one_hot=5,
            zero_pad=True,
        )

    graph_list = []
    feature_set = set()

    for (X, A, N) in untreated_graphs:
        A_np = A.cpu().detach().numpy()
        np.fill_diagonal(A_np, 0)
        G = nx.from_numpy_matrix(A_np)
        for i, node in enumerate(G.nodes()):
            G.nodes[i]["feature"] = X[i].detach().numpy()
        if need_set:
            assert isinstance(G, nx.Graph)
            feature_values = nx.get_node_attributes(G, "feature").values()
            feature_set.update(map(lambda x: tuple(x), feature_values))
        graph_list.append(G)


    return graph_list, feature_set


def load_dataset(data_dir="data", file_name=None, need_set=False):
    if file_name == "community_small20":
        filename = "community_N_nodes5000_maxN20_minN20.sparsedataset"

        graph_list, feature_set = load_ggg(
            filename, dataset="CommunitySmall_20", structure=True
        )

    elif file_name == "MolGAN_5k":
        filename = "QM9_5k.sparsedataset"

        graph_list, feature_set = load_ggg(
            filename, dataset="MolGAN_5k", structure=False
        )

    elif file_name == "chordal9":
        filename = "chordal.npz"

        graph_list, feature_set = load_ggg(
            filename,
            dataset="anu_graphs_chordal9",
            data_dir="data/ANU_graphs",
            structure=True,
        )

    elif file_name == "Trees09":
        filename = "chordal.npz"

        graph_list, feature_set = load_ggg(
            data_dir=os.path.expanduser("~/.datasets/"),
            filename=None,
            dataset="Trees09",
            k_eign=4,
            use_lpl=True,
            largN=False,
            structure=True,
            need_set=False,
        )

    else:
        file_path = os.path.join(data_dir, file_name)
        feature_set = set()
        with open(file_path + ".pkl", "rb") as f:
            graph_list = pickle.load(f)
        if need_set:
            for g in graph_list:
                assert isinstance(g, nx.Graph)
                feature_values = nx.get_node_attributes(g, "feature").values()
                print(feature_values)
                exit()
                feature_set.update(map(lambda x: tuple(x), feature_values))
        with open(file_path + ".txt", "r") as f:
            info = f.read()
        logging.info("load dataset: " + info)
        # print('features set:', feature_set)

    return graph_list, feature_set


if __name__ == "__main__":
    res_graph_list = gen_graph_list(
        graph_type="grid",
        possible_params_dict={"m": [2, 3], "n": [4, 5]},
        corrupt_func=None,
        length=4,
        save_dir=None,
    )
