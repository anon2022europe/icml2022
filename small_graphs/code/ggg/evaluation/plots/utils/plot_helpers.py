from PIL import Image

from tqdm import tqdm

import os
import torch
import pickle
import numpy as np
from tqdm import tqdm
import networkx as nx

# GG-GAN model imports
from ggg.models.ggg_model import GGG
from ggg.utils.utils import ensure_tensor
from ggg.data.dense.GGG_DenseData import GGG_DenseData


def list_from_pickle(dir_):
    """Read list from pickle file"""

    with open(dir_, "rb") as f:
        pkl_list = pickle.load(f)

    return pkl_list


def metric_to_use(metric):
    """Get suffix for file referent to desired metric"""

    if metric == "degree":
        suffix_ = "_degreeD.pkl"
        title = "Degree distribution"
        xlabel = "Degree"
    elif metric == "cycles":
        suffix_ = "_cycleD.pkl"
        title = "Cycle distribution"
        xlabel = "Cycles"

    return suffix_, title, xlabel


def set_ax_off(ax, first=False):
    """Set invisible axes"""

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    if first:
        pass
    else:
        ax.axes.get_yaxis().set_visible(False)


def calculate_iso_classes(graph_list):
    isomorphism_graphs = []
    isomorphism_classes = []
    for i, g1 in tqdm(enumerate(graph_list)):
        iso = False
        same_class = []
        for j, g2 in enumerate(graph_list[i + 1 :]):
            if nx.is_isomorphic(g1, g2):
                iso = True
                if g1 not in isomorphism_graphs:
                    isomorphism_graphs.append(g1)
                    same_class.append(g1)
                if g2 not in isomorphism_graphs:
                    isomorphism_graphs.append(g2)
                    same_class.append(g2)

        if len(same_class) != 0:
            isomorphism_classes.append(same_class)
        elif iso == False and g1 not in isomorphism_graphs:
            same_class.append(g1)
            isomorphism_classes.append(same_class)
    return isomorphism_classes


def check_isomorphism_dataset(graph_list, dataset, chordal=False):
    # Check if such classes are not in the dataset
    not_in_dataset = []
    in_dataset = 0
    chordal_count = 0
    for i, g1 in tqdm(enumerate(graph_list)):
        iso = False
        for j, g2 in enumerate(dataset):
            if nx.is_isomorphic(g1, g2):
                iso = True
                in_dataset += 1
                break
        if chordal and nx.is_chordal(g1):
            chordal_count += 1
            if not iso:
                not_in_dataset.append(g1)
        elif not iso and not chordal:
            not_in_dataset.append(g1)

    print("Graphs in dataset {}".format(in_dataset))
    print("Chordal graphs {}".format(chordal_count))
    return not_in_dataset


def isog_generated(pkl_list, number_g, total_iso_graphs: []):
    """Count isomorph graphs in generated samples"""

    rand_graph_list = []
    rand_choice = np.random.choice(len(pkl_list), number_g, replace=False)
    for num in rand_choice:
        rand_graph_list.append(pkl_list[num])

    isomorphism_classes = calculate_iso_classes(rand_graph_list)

    total_iso_graphs.append(len(isomorphism_classes))

    return total_iso_graphs, len(isomorphism_classes)


def isog_novelty2(
    pkl_list, number_g, total_iso_graphs: [], dataset=None, chordal=False
):
    """Count isomorph graphs between model and dataset"""

    graph_list = []
    rand_choice = np.random.choice(len(pkl_list), number_g, replace=False)
    for num in rand_choice:
        graph_list.append(pkl_list[num])

    # TODO: improve isomorphism count (do not need to iterate twice fully)
    if chordal:
        not_in_dataset = check_isomorphism_dataset(graph_list, dataset, chordal=True)
        print(
            "There are {} chordal graphs not in training dataset".format(
                len(not_in_dataset)
            )
        )
        isomorphism_classes = calculate_iso_classes(not_in_dataset)
    else:
        not_in_dataset = check_isomorphism_dataset(graph_list, dataset)
        isomorphism_classes = calculate_iso_classes(not_in_dataset)

    total_iso_graphs.append(len(isomorphism_classes))

    return total_iso_graphs, len(isomorphism_classes)


def get_isog(df, pkl_graphs: [], number_g=100, diver_list=[]):

    diver_list, val = isog_generated(pkl_graphs, number_g, diver_list)

    return df, diver_list


def get_novelty(
    df, pkl_graphs: [], number_g=None, dataset=None, diver_list=[], chordal=False
):

    diver_list, val = isog_novelty2(
        pkl_graphs, number_g, diver_list, dataset=dataset, chordal=chordal
    )

    return df, diver_list


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w, h = fig.canvas.get_width_height()
    buf = np.fromstring(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (w, h, 4)

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    return buf


def fig2img(fig):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data(fig)
    w, h, d = buf.shape
    return Image.frombytes("RGBA", (w, h), buf.tostring())


def get_epoch_graphs(
    epoch,
    g_dir,
    model_n="None",
    all_models_g=[],
    number_g=None,
    log_dir=None,
    model=None,
    batch_idx=None,
    allow_greater=False,
):
    if "GraphRNN" in model_n:
        generated_graphs = []
        with open(g_dir + f"_5000g_ep{epoch}.pkl", "rb") as f:
            graphs = pickle.load(f)
            for adj in graphs:
                g = nx.from_numpy_matrix(adj)
                generated_graphs.append(g)
        all_models_g.append(generated_graphs)

    elif "MolGAN" in model_n:
        # Baseline MolGAN imports
        from externals.benchmarks.MolGAN.example import trainer
        from externals.benchmarks.MolGAN.example import (
            model,
            optimizer,
            _test_update,
        )

        generated_graphs = get_MolGAN_graphs(
            trainer,
            300,
            number_g=number_g,
            model=model,
            optimizer=optimizer,
            _test_update=_test_update,
            dir_=log_dir,
        )
        all_models_g.append(generated_graphs)
    elif "ScrM" in model_n:
        with open(g_dir, "rb") as f:
            generated_graphs = pickle.load(f)
        all_models_g.append(generated_graphs)
    elif "GG-GAN" in model_n or "PointMLP" in model_n:
        if model is None:
            model = get_GGGAN_model(epoch, log_dir=log_dir)
            device = "cpu"
        else:
            device = model.device
        generated_graphs = []
        # TODO batch size changed given error in root (check with anon): Requested batch size 20 with fixed embedding size 2 and no expansion allowed
        if allow_greater:
            if hasattr(model.generator.root, "allow_greater"):
                old_allow = (
                    model.generator.root.allow_greater
                )  # save it here, restore later
            model.generator.root.allow_greater = True
        batch_size = 20

        for _ in tqdm(range(int(number_g / batch_size) + 1)):
            # Only 4 returns, given Z and context are concatenated
            X_out, A_out, _, _ = model.generator.sample(
                batch_size=batch_size, device=device
            )
            if batch_idx is not None:
                adj_m = A_out[batch_idx].cpu().detach().numpy()

                g = nx.from_numpy_matrix(adj_m)
                generated_graphs.append(g)

                if len(generated_graphs) > (number_g + 1):
                    break
            else:
                for b in range(batch_size):
                    adj_m = A_out[b].cpu().detach().numpy()

                    g = nx.from_numpy_matrix(adj_m)
                    generated_graphs.append(g)

                    if len(generated_graphs) > (number_g + 1):
                        break

            if len(generated_graphs) > (number_g + 1):
                break
        all_models_g.append(generated_graphs)

        # restore default values of model.generator
        if allow_greater:
            if hasattr(model.generator.root, "allow_greater"):
                model.generator.root.allow_greater = old_allow
    return all_models_g


def get_dataset_epochs_graphs(g_dir, dataset_g=[], post_exp=False, dataset=None):

    if post_exp:
        dat_g = dataset
        gen_graphs = []
        for (_, A) in dat_g:
            adj_m = A.detach().numpy()

            np.fill_diagonal(adj_m, 0)
            g = nx.from_numpy_matrix(adj_m)
            gen_graphs.append(g)

    else:
        if "Mgan_feat" in dataset:
            with open(g_dir, "rb") as f:
                dat_g = pickle.load(f)
            gen_graphs = []
            for A in dat_g["data_A"]:
                adj_m = A
                np.fill_diagonal(adj_m, 0)
                g = nx.from_numpy_matrix(adj_m)
                gen_graphs.append(g)
        elif "chordal" in dataset:
            _, A = load_npz_keys(["X", "A"], g_dir)
            gen_graphs = []
            for adj_m in A:
                np.fill_diagonal(adj_m, 0)
                g = nx.from_numpy_matrix(adj_m)
                gen_graphs.append(g)

        elif (
            "MolGAN_5k" in dataset
            or "Community" in dataset
            or "QM9" in dataset
            or "Tree" in dataset
            or "SBM" in dataset
        ):
            with open(g_dir, "rb") as f:
                dat_g = pickle.load(f)
            gen_graphs = []
            for graph in dat_g:
                adj_m = graph.A.detach().numpy()
                np.fill_diagonal(adj_m, 0)
                g = nx.from_numpy_matrix(adj_m)
                gen_graphs.append(g)

    dataset_g.append(gen_graphs)

    return dataset_g


def check_dataset_file(filepath, dataset):
    # check if datasets exist
    if os.path.isfile(filepath):
        pass
    else:
        if dataset == "MolGAN_5k":
            dataset = "MolGAN_5k"
            filename = "QM9_5k.sparsedataset"
        if dataset == "RandMolGAN_5k":
            dataset = "MolGAN_5k"
            filename = "QM9_rand1.sparsedataset"
        if dataset == "RandMolGAN_5k":
            dataset = "MolGAN_5k"
            filename = "QM9_rand2.sparsedataset"
        if dataset == "RandMolGAN_5k":
            dataset = "MolGAN_5k"
            filename = "QM9_rand3.sparsedataset"

        elif dataset == "CommunitySmall_20":
            dataset = "CommunitySmall_20"
            filename = None
        elif dataset == "CommunitySmall_20_rand1":
            dataset = "CommunitySmall_20_rand1"
            filename = None
        elif dataset == "CommunitySmall_20_rand2":
            dataset = "CommunitySmall_20_rand3"
            filename = None
        elif dataset == "CommunitySmall_20_rand3":
            dataset = "CommunitySmall_20_rand3"
            filename = None

        elif dataset == "anu_graphs_chordal9":
            dataset = "anu_graphs_chordal9"
            filename = "chordal.npz"
        elif dataset == "anu_graphs_chordal9_rand":
            dataset = "anu_graphs_chordal9_rand"
            filename = "chordal_test.npz"
        ds = GGG_DenseData(dataset=dataset, filename=filename, data_dir="ggg/data")


def get_MolGAN_graphs(
    trainer,
    epoch,
    number_g=1024,
    model=None,
    optimizer=None,
    _test_update=None,
    dir_=None,
):
    trainer.load(directory=dir_, model=epoch)

    _, _, e, z_noise = _test_update(
        model=model,
        optimizer=optimizer,
        batch_dim=32,
        test_batch=None,
        create_G_data=True,
    )

    gen_graphs = []
    while len(gen_graphs) < number_g:
        for matrix in e:
            a = np.clip(matrix, 0, 1)
            G = nx.from_numpy_matrix(a)
            gen_graphs.append(G)
            if len(gen_graphs) >= number_g:
                break

    return gen_graphs


def get_GGGAN_model(epoch, log_dir=None):
    ckpt_path = os.path.join(log_dir, f"{str(epoch).zfill(4)}/state.ckpt")
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"))

    # TODO figure out why in cluster I need to go for hyper_param still, the git is up to date there...
    checkpoint['hyper_parameters']['dataset_hpars']['data_dir'] = os.path.expanduser("~/.datasets")
    model = GGG(checkpoint['hyper_parameters'])
    # for key in ["generator.edge_readout.attn.q_prenorm.norm.weight", "generator.edge_readout.attn.q_prenorm.norm.bias",
    #             "generator.edge_readout.attn.out_prenorm.norm.weight", "generator.edge_readout.attn.out_prenorm.norm.bias"]:
    #     del checkpoint['state_dict'][key]
    model.load_state_dict(checkpoint['state_dict'])
    # checkpoint['hyper_parameters']['device'] = "cpu"
    # checkpoint['hyper_parameters']['disc_readout_hidden'] = 64
    # model = PEAWGAN(checkpoint['hyper_parameters'])
    # model.load_state_dict(checkpoint['state_dict'])
    return model


def load_npz_keys(keys, file):
    """
    Small utility to directly load an npz_compressed file
    :param keys:
    :param file:
    :return:
    """
    out = []
    with np.load(file) as d:
        for k in keys:
            out.append(d[k])
    return tuple(out) if len(out) > 1 else out[0]
