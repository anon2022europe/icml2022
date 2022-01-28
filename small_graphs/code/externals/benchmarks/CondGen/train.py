import matplotlib

matplotlib.use("Agg")

from externals.benchmarks.CondGen.utils import sacred_copy
from ggg.data.dense.QM9.MolGAN_QM9 import QM9preprocess
from ggg.data.dense.GGG_DenseData import GGG_DenseData
from ggg.evaluation.plots.utils.post_experiment_plots import (
    generate_graphs,
    main_run_plot,
    main_run_MMD,
)
from ggg.utils.utils import pad_to_max

from pprint import pprint

from graph_stat import *
from collections import defaultdict
import os
import numpy as np
import os
import networkx as nx
import attr
import torch
import time
import random
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
from collections import defaultdict
from pprint import pprint
from collections import defaultdict

from graph_stat import *
from torch.utils.data import Dataset
from options import Options, AttrOptions
from GVGAN import *
from utils import *
from tqdm import tqdm
from pprint import pprint
from sacred import Experiment

ex = Experiment("Condgen")
import torch.nn as nn
import torch.optim as optim
import warnings

from models import *
from ggg.data.condgen import CondgenTCGA, CondgenDBLP
import sklearn
import sklearn.manifold


def ensure_tensor(x):
    if torch.is_tensor(x):
        return x
    else:
        return torch.from_numpy(x)


def get_spectral_embedding(adj, d, solver="arpack"):  # , ‘lobpcg’, or ‘amg’"):
    """
    Given adj is N*N, return its feature mat N*D, D is fixed in model
    :param adj:
    :return:
    """

    adj_ = np.array(adj)
    emb = sklearn.manifold.SpectralEmbedding(
        n_components=d, eigen_solver=solver
    )  # use lopcg since otherwis ethis errors now?
    res = emb.fit_transform(adj_)
    x = res.astype(np.float32)
    return x


def load_our_data(dataset):
    if dataset in {"MolGAN_5k", "CommunitySmall_20", "anu_graphs_chordal9"}:
        if dataset == "MolGAN_5k":
            dataset = "MolGAN_5k"
            filename = "QM9_5k.sparsedataset"
        elif dataset == "CommunitySmall_20":
            dataset = "CommunitySmall_20"
            filename = "community_N_nodes5000_maxN20_minN20.sparsedataset"
        elif dataset == "anu_graphs_chordal9":
            dataset = "anu_graphs_chordal9"
            filename = "chordal.npz"
        ds = GGG_DenseData(
            dataset=dataset,
            filename=filename,
            data_dir="externals/benchmarks/CondGen/data",
        )
    else:
        raise ValueError(f"Don't know {dataset}")
    adj_mats = []
    for i in tqdm(range(len(ds)), desc="Unpacking"):
        (_, A) = ds[i]
        if torch.is_tensor(A):
            A = A.cpu().numpy()
        adj_mats.append(A)
    train_adj_mats = adj_mats[: int(len(adj_mats) * 0.8)]
    test_adj_mats = adj_mats[int(len(adj_mats) * 0.8) :]
    assert len(train_adj_mats) > 0
    assert len(test_adj_mats) > 0
    return train_adj_mats, test_adj_mats


def load_data(DATA_DIR):
    if "gene" in DATA_DIR or "tcga" in DATA_DIR:
        ds = CondgenTCGA(DATA_DIR)
    else:
        ds = CondgenDBLP(DATA_DIR)
    adj_mats = []
    attr_vecs = []
    for i in tqdm(range(len(ds)), desc="Unpacking"):
        (_, A), attr_vec = ds[i]
        adj_mats.append(A)
        attr_vecs.append(attr_vec)
    attr_vecs = np.stack(attr_vecs, 0)
    train_adj_mats = adj_mats[: int(len(adj_mats) * 0.8)]
    test_adj_mats = adj_mats[int(len(adj_mats) * 0.8) :]
    train_attr_vecs = attr_vecs[: int(len(attr_vecs) * 0.8)]
    test_attr_vecs = attr_vecs[int(len(attr_vecs) * 0.8) :]
    assert len(train_adj_mats) > 0
    assert len(test_adj_mats) > 0
    print(
        f"Loaded {DATA_DIR} with {len(train_adj_mats)} train and {len(test_adj_mats)} test"
    )
    return train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs


def load_dblp(DATA_DIR):
    # script for loading NWE dblp
    # folder structure
    # - this.ipynb
    # - $DATA_DIR - *.txt

    mat_names = []  # e.g. GSE_2304
    adj_mats = []  # essential data, type: list(np.ndarray)
    attr_vecs = []  # essential data, type: list(np.ndarray)
    id_maps = []  # map index to gene name if you need

    for f in os.listdir(DATA_DIR):
        if not f.startswith(("nodes", "links", "attrs")):
            continue
        else:
            mat_names.append("_".join(f.split(".")[0].split("_")[1:]))
    mat_names = sorted([it for it in set(mat_names)])
    print("Test length", len(mat_names))
    for mat_name in mat_names:
        node_file = "nodes_" + mat_name + ".txt"
        link_file = "links_" + mat_name + ".txt"
        attr_file = "attrs_" + mat_name + ".txt"
        node_file_path = os.path.join(DATA_DIR, node_file)
        link_file_path = os.path.join(DATA_DIR, link_file)
        attr_file_path = os.path.join(DATA_DIR, attr_file)

        id_to_item = {}
        with open(node_file_path, "r") as f:
            for i, line in enumerate(f):
                author = line.rstrip("\n")
                id_to_item[i] = author
        all_ids = set(id_to_item.keys())

        with open(attr_file_path, "r") as f:
            attr_vec = np.loadtxt(f).T.flatten()
            attr_vecs.append(attr_vec)

        links = defaultdict(set)
        with open(link_file_path, "r") as f:
            for line in f:
                cells = line.rstrip("\n").split(",")
                from_id = int(cells[0])
                to_id = int(cells[1])
                if from_id in all_ids and to_id in all_ids:
                    links[from_id].add(to_id)

        N = len(all_ids)
        adj = np.zeros((N, N))
        for from_id in range(N):
            for to_id in links[from_id]:
                adj[from_id, to_id] = 1
                adj[to_id, from_id] = 1

        adj -= np.diag(np.diag(adj))
        id_map = [id_to_item[i] for i in range(N)]

        # Remove small component
        # adj = remove_small_conns(adj, keep_min_conn=4)

        # Keep large component
        adj = keep_topk_conns(adj, k=1)
        adj_mats.append(adj)
        id_maps.append(id_map)

        if int(np.sum(adj)) == 0:
            adj_mats.pop(-1)
            id_maps.pop(-1)
            mat_names.pop(-1)
            attr_vecs.pop(-1)

    train_adj_mats = adj_mats[: int(len(adj_mats) * 0.8)]
    test_adj_mats = adj_mats[int(len(adj_mats) * 0.8) :]
    train_attr_vecs = attr_vecs[: int(len(attr_vecs) * 0.8)]
    test_attr_vecs = attr_vecs[int(len(attr_vecs) * 0.8) :]
    assert len(train_adj_mats) > 0
    assert len(test_adj_mats) > 0
    print(f"Loaded DBLP with {len(train_adj_mats)} train and {len(test_adj_mats)} test")
    return train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs


def load_genes(DATA_DIR):
    NODE_FILE = "node.dat"
    LINK_FILE = "link.dat"
    LABEL_FILE = "label.dat"
    ATTR_FILE = "attribute.dat"
    mat_names = []  # e.g. GSE_2304
    adj_mats = []  # essential data, type: list(np.ndarray)
    attr_vecs = []  # essential data, type: list(np.ndarray)
    id_maps = []  # map index to gene name if you need

    ATTR_LEN = 8

    for folder in tqdm(os.listdir(DATA_DIR)):
        if "." in folder or folder.startswith("_"):
            continue
        mat_names.append(folder)
        with open(os.path.join(DATA_DIR, folder, ATTR_FILE), "r") as f:
            attr_vec = np.loadtxt(f)
            attr_vecs.append(attr_vec[:ATTR_LEN])

        id_to_item = {}
        with open(os.path.join(DATA_DIR, folder, NODE_FILE), "r") as f:
            for i, line in enumerate(f):
                cells = line.split("\t")
                id_to_item[i] = cells[0]

        all_items = set(id_to_item.values())
        all_ids = set(id_to_item.keys())

        links = defaultdict(set)
        with open(os.path.join(DATA_DIR, folder, LINK_FILE), "r") as f:
            for line in f:
                cells = line.rstrip("\n").split("\t")
                from_id = int(cells[0])
                to_id = int(cells[1])
                if from_id in all_ids and to_id in all_ids:
                    links[from_id].add(to_id)

        N = len(all_ids)
        adj = np.zeros((N, N))
        for from_id in range(N):
            for to_id in links[from_id]:
                adj[from_id, to_id] = 1
                adj[to_id, from_id] = 1
        adj -= np.diag(np.diag(adj))
        id_map = [id_to_item[i] for i in range(N)]

        # Remove small component
        # adj = remove_small_conns(adj, keep_min_conn=4)

        # Keep large component
        adj = keep_topk_conns(adj, k=1)

        adj_mats.append(adj)
        id_maps.append(id_map)

        if int(np.sum(adj)) == 0:
            adj_mats.pop(-1)
            id_map.pop(-1)
            mat_names.pop(-1)
            attr_vecs.pop(-1)

    train_adj_mats = adj_mats[: int(len(adj_mats) * 0.8)]
    test_adj_mats = adj_mats[int(len(adj_mats) * 0.8) :]
    train_attr_vecs = attr_vecs[: int(len(attr_vecs) * 0.8)]
    test_attr_vecs = attr_vecs[int(len(attr_vecs) * 0.8) :]

    assert len(train_adj_mats) > 0
    assert len(test_adj_mats) > 0
    print(f"Loaded TCGA with {len(train_adj_mats)} train and {len(test_adj_mats)} test")

    return train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs


def save_model_parts(model_dict, output_dir, epoch):
    for k, m in model_dict.items():
        fpath = os.path.join(output_dir, f"{k}_{epoch!a}.pt")
        with open(fpath, "wb") as f:
            torch.save(m.state_dict(), f)


def load_model_parts(model_dict, output_dir, epoch):
    outs = []
    for k, m in model_dict.items():
        fpath = os.path.join(output_dir, f"{k}_{epoch!a}.pt")
        with open(fpath, "rb") as f:
            d = torch.load(f, map_location=torch.device("cpu"))
            if isinstance(d, dict):
                m.load_state_dict(d)
            else:
                assert type(m) == type(d)
                m.load_state_dict(d.state_dict())


def train(
    train_adj_mats,
    test_adj_mats,
    train_attr_vecs,
    test_attr_vecs,
    opt=None,
    _run=None,
    batch_size=None,
):
    training_index = list(range(0, len(train_adj_mats)))

    z_out_size = opt.z_size + opt.av_size

    # TO MAKE EVERYTHING EASY, NO CLASS GVGAN() HERE...

    G = Generator(
        av_size=opt.av_size,
        d_size=opt.d_size,
        gc_size=opt.gc_size,
        z_size=opt.z_size,
        z_out_size=z_out_size,
        rep_size=opt.rep_size,
    ).cuda()
    num_pars = [np.prod(p.shape) for p in G.parameters()]
    num_pars = np.sum(num_pars)
    print(f"G has {num_pars:0.2e} parameters")
    num_pars = [np.prod(p.shape) for p in G.decoder.parameters()]
    num_pars = np.sum(num_pars)
    print(f"G.decoder has {num_pars:0.2e} parameters")

    D = Discriminator(
        av_size=opt.av_size,
        d_size=opt.d_size,
        gc_size=opt.gc_size,
        rep_size=opt.rep_size,
    ).cuda()

    criterion_bce = nn.BCELoss()
    criterion_bce.cuda()

    # This three are for A A' loss
    loss_MSE = nn.MSELoss()
    loss_MSE.cuda()

    loss_BCE_logits = nn.BCEWithLogitsLoss()  # size_average=False)
    loss_BCE_logits.cuda()

    loss_BCE = nn.BCELoss()  # size_average=False)
    loss_BCE.cuda()

    opt_enc = optim.Adam(G.encoder.parameters(), lr=opt.lr)
    opt_dec = optim.Adam(G.decoder.parameters(), lr=opt.lr)
    opt_dis = optim.Adam(D.parameters(), lr=opt.lr * opt.alpha)

    max_epochs = opt.max_epochs
    for epoch in tqdm(range(max_epochs), desc="Epoch", leave=False):
        D_real_list, D_rec_enc_list, D_rec_noise_list, D_list, Encoder_list = (
            [],
            [],
            [],
            [],
            [],
        )
        # g_loss_list, rec_loss_list, prior_loss_list = [], [], []
        g_loss_list, rec_loss_list, prior_loss_list, aa_loss_list = [], [], [], []
        random.shuffle(training_index)
        for i in tqdm(training_index, desc="train_ind", leave=False):
            ones_label = Variable(torch.ones(1)).cuda()
            zeros_label = Variable(torch.zeros(1)).cuda()
            # adj = Variable(train_adj_mats[i]).cuda()
            adj = Variable(torch.from_numpy(train_adj_mats[i]).float()).cuda()

            # if adj.shape[0] <= d_size + 2 :
            #    continue
            if adj.shape[0] <= opt.d_size + 2:
                continue
            if opt.av_size == 0:
                attr_vec = None
            else:
                # attr_vec = Variable(train_attr_vecs[i, :]).cuda()
                attr_vec = Variable(torch.from_numpy(train_attr_vecs[i]).float()).cuda()

            # edge_num = train_adj_mats[i].sum()
            G.set_attr_vec(attr_vec)
            D.set_attr_vec(attr_vec)

            norm = (
                adj.shape[0]
                * adj.shape[0]
                / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)
            )
            pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
            # print('pos_weight', pos_weight)

            mean, logvar, rec_adj = G(adj)

            noisev = torch.randn(mean.shape, requires_grad=True).cuda()
            noisev = cat_attr(noisev, attr_vec)
            rec_noise = G.decoder(noisev)

            e = int(np.sum(train_adj_mats[i])) // 2

            c_adj = topk_adj(F.sigmoid(rec_adj), e * 2)
            c_noise = topk_adj(F.sigmoid(rec_noise), e * 2)

            # train discriminator
            output = D(adj)
            errD_real = criterion_bce(output, ones_label)
            D_real_list.append(output.data.mean())
            # output = D(rec_adj)
            output = D(c_adj)
            errD_rec_enc = criterion_bce(output, zeros_label)
            D_rec_enc_list.append(output.data.mean())
            # output = D(rec_noise)
            output = D(c_noise)

            errD_rec_noise = criterion_bce(output, zeros_label)
            D_rec_noise_list.append(output.data.mean())

            dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
            # print ("print (dis_img_loss)", dis_img_loss)
            D_list.append(dis_img_loss.data.mean())
            opt_dis.zero_grad()
            dis_img_loss.backward(retain_graph=True)
            opt_dis.step()

            # AA_loss b/w rec_adj and adj
            # aa_loss = loss_MSE(rec_adj, adj)

            loss_BCE_logits = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            loss_BCE_logits.cuda()

            aa_loss = loss_BCE_logits(rec_adj, adj)

            # print(c_adj,c_adj)
            # aa_loss = loss_BCE(c_adj, adj)

            # train decoder
            output = D(adj)
            errD_real = criterion_bce(output, ones_label)
            # output = D(rec_adj)
            output = D(c_adj)

            errD_rec_enc = criterion_bce(output, zeros_label)
            errG_rec_enc = criterion_bce(output, ones_label)
            # output = D(rec_noise)
            output = D(c_noise)

            errD_rec_noise = criterion_bce(output, zeros_label)
            errG_rec_noise = criterion_bce(output, ones_label)

            similarity_rec_enc = D.similarity(c_adj)
            similarity_data = D.similarity(adj)

            dis_img_loss = errD_real + errD_rec_enc + errD_rec_noise
            # print (dis_img_loss)
            # gen_img_loss = norm*(aa_loss + errG_rec_enc  + errG_rec_noise)- dis_img_loss #- dis_img_loss #aa_loss #+ errG_rec_enc  + errG_rec_noise # - dis_img_loss
            gen_img_loss = -dis_img_loss  # norm*(aa_loss) #

            g_loss_list.append(gen_img_loss.data.mean())
            rec_loss = ((similarity_rec_enc - similarity_data) ** 2).mean()
            rec_loss_list.append(rec_loss.data.mean())
            # err_dec =  gamma * rec_loss + gen_img_loss

            err_dec = opt.gamma * rec_loss + gen_img_loss
            opt_dec.zero_grad()
            err_dec.backward(retain_graph=True)
            opt_dec.step()

            # train encoder
            # fix me: sum version of prior loss
            pl = []
            for j in range(mean.size()[0]):
                prior_loss = 1 + logvar[j, :] - mean[j, :].pow(2) - logvar[j, :].exp()
                prior_loss = (-0.5 * torch.sum(prior_loss)) / torch.numel(
                    mean[j, :].data
                )
                pl.append(prior_loss)
            prior_loss_list.append(sum(pl))
            err_enc = (
                sum(pl) + gen_img_loss + opt.beta * (rec_loss)
            )  # + beta2* norm* aa_loss
            opt_enc.zero_grad()
            err_enc.backward()
            opt_enc.step()
            Encoder_list.append(err_enc.data.mean())

        print(
            "[%d/%d]: D_real:%.4f, D_enc:%.4f, D_noise:%.4f, Loss_D:%.4f, Loss_G:%.4f, rec_loss:%.4f, prior_loss:%.4f"
            % (
                epoch,
                max_epochs,
                torch.mean(torch.stack(D_real_list)),
                torch.mean(torch.stack(D_rec_enc_list)),
                torch.mean(torch.stack(D_rec_noise_list)),
                torch.mean(torch.stack(D_list)),
                torch.mean(torch.stack(g_loss_list)),
                torch.mean(torch.stack(rec_loss_list)),
                torch.mean(torch.stack(prior_loss_list)),
            )
        )

        print("Training set")
        for i in range(3):
            base_adj = train_adj_mats[i]

            if base_adj.shape[0] <= opt.d_size:
                continue
            print("Base Adj_size: ", base_adj.shape)
            if opt.av_size == 0:
                attr_vec = None
            else:
                attr_vec = Variable(torch.from_numpy(train_attr_vecs[i]).float()).cuda()

            # add a new line
            G.set_attr_vec(attr_vec)

            print("Show sample")
            sample_adj = gen_adj(
                G,
                base_adj.shape[0],
                int(np.sum(base_adj)) // 2,
                attr_vec,
                z_size=opt.z_size,
            )
            show_graph(
                sample_adj,
                base_adj=base_adj,
                remove_isolated=True,
                epoch=epoch,
                sample=i,
                dataset=os.path.basename(os.path.abspath(opt.DATA_DIR)),
                opt=opt,
            )
            model_dict = dict(G=G, D=D)
            save_model_parts(model_dict, opt.output_dir, epoch)
    return D, G


class G_Wrap:
    def __init__(self, G, train_adj, z_size, d_size, attr_vecs=None):
        self.G = G
        self.train_adj = train_adj
        self.attr_vecs = attr_vecs
        self.z_size = z_size
        self.d_size = d_size

    def sample(
        self, batch_size=1, device="cpu", show=False, opt=None, spectral_emb=True
    ):
        As = []
        xs = []
        Zs = []
        bases = np.random.randint(len(self.train_adj), size=batch_size)
        for i in range(batch_size):
            base_adj = self.train_adj[bases[i]]
            if self.attr_vecs is not None:
                attr_vec = self.attr_vecs[bases[i]]
                if not torch.is_tensor(attr_vec):
                    attr_vec = torch.from_numpy(attr_vec).float()
                attr_vec = Variable(attr_vec).cuda()
            else:
                attr_vec = None
            if torch.is_tensor(base_adj):
                adj_sum = base_adj.sum().int().item()
            else:
                adj_sum = int(np.sum(base_adj))
            A, Z = gen_adj(
                self.G,
                base_adj.shape[0],
                adj_sum // 2,
                attr_vec,
                z_size=self.z_size,
                return_Z=True,
                device=device,
            )
            if show:
                show_graph(
                    A,
                    base_adj=base_adj,
                    remove_isolated=True,
                    epoch=-1,
                    sample=i,
                    dataset=os.path.basename(os.path.abspath(opt.DATA_DIR)),
                    opt=opt,
                    suffix="_Gwrap",
                )
            if spectral_emb:
                try:
                    x = get_spectral_embedding(A.cpu().numpy(), self.d_size)
                except:
                    x = get_spectral_embedding(
                        A.cpu().numpy(), self.d_size, solver="lobpcg"
                    )
            else:
                x = np.random.randn(A.shape[0], self.d_size)
            xs.append(ensure_tensor(x))
            As.append(ensure_tensor(A))
            Zs.append(ensure_tensor(Z))

        if batch_size > 1:
            As = pad_to_max(As, 0, (0, 1))
            xs = pad_to_max(xs, 0, (0,))
            Zs = pad_to_max(Zs, 0, (0,))
            As = torch.stack(As, 0)
            xs = torch.stack(xs, 0)
            Zs = torch.stack(Zs, 0)
        else:
            As = torch.stack(As, 0)
            xs = torch.stack(xs, 0)
            Zs = torch.stack(Zs, 0)
        return xs, As, Zs, None, None


@ex.config
def conf():
    hpars = attr.asdict(AttrOptions())
    our_dataset = None
    batch_size = None


@ex.named_config
def molgan5k():
    our_dataset = "MolGAN_5k"
    hpars = attr.asdict(
        AttrOptions(
            output_dir="molgan5k_out", DATA_DIR="data/molgan/", av_size=0, d_size=5
        )
    )


@ex.named_config
def community20():
    our_dataset = "CommunitySmall_20"
    hpars = attr.asdict(
        AttrOptions(
            output_dir="community20", DATA_DIR="data/community20/", av_size=0, d_size=10
        )
    )


@ex.named_config
def chordal_9():
    our_dataset = "anu_graphs_chordal9"
    hpars = attr.asdict(
        AttrOptions(
            output_dir="chordal_9", DATA_DIR="data/chordal_9/", av_size=0, d_size=5
        )
    )


@ex.named_config
def dblp():
    hpars = attr.asdict(
        AttrOptions(
            output_dir="dblp_out", DATA_DIR="data/data_dblp/", av_size=10, d_size=5
        )
    )


@ex.named_config
def dblp_github():
    hpars = attr.asdict(
        AttrOptions(output_dir="dblp_out", DATA_DIR="data/dblp/", av_size=10, d_size=5)
    )


@ex.named_config
def tcga():
    hpars = attr.asdict(
        AttrOptions(
            output_dir="tcga_out", DATA_DIR="data/data_tcga/", d_size=10, av_size=8
        )
    )


@ex.automain
def run(hpars, _run, _config, our_dataset, batch_size):
    print(_config)
    opt = AttrOptions(**sacred_copy(hpars))
    if len(_run.observers) >= 1:
        print(f"Changing output_dir from {opt.output_dir} to {_run.observers[0].dir}")
        opt.output_dir = _run.observers[0].dir
    else:
        os.makedirs(opt.output_dir)

    print("=========== OPTIONS ===========")
    pprint(opt)
    print(" ======== END OPTIONS ========\n\n")

    os.environ["CUDA_VISIBLE_DEVICES"] = f"{opt.gpu}"

    if our_dataset is None:
        train_adj_mats, test_adj_mats, train_attr_vecs, test_attr_vecs = load_data(
            DATA_DIR=opt.DATA_DIR
        )
    else:
        assert opt.av_size == 0, "Our datasets don't have conditional vectors"
        train_adj_mats, test_adj_mats = load_our_data(our_dataset)
        train_attr_vecs, test_attr_vecs = None, None

    # output_dir = opt.output_dir
    D, G = train(
        train_adj_mats,
        test_adj_mats,
        train_attr_vecs,
        test_attr_vecs,
        opt=opt,
        batch_size=batch_size,
    )
    G.eval()
    Gw = G_Wrap(G, train_adj_mats, opt.z_size, opt.d_size, attr_vecs=train_attr_vecs)
    dataset = os.path.basename(opt.DATA_DIR)
    plots_save_dir = os.path.join(opt.output_dir, "plots")
    data_save_dir = os.path.join(opt.output_dir, "data")
    for x in [plots_save_dir, data_save_dir]:
        os.makedirs(x, exist_ok=True)
    N_GRAPH = 1024
    gen_graphs, _ = generate_graphs(
        Gw,
        current_epoch=opt.max_epochs,
        dataset=None,
        numb_graphs=N_GRAPH,
        save_dir=data_save_dir,
        device="cpu",
        batch_size=1 if batch_size is None else batch_size,
    )
    dataset_graphs = [nx.from_numpy_matrix(g) for g in train_adj_mats]
    main_run_plot(
        opt.max_epochs,
        f"condgen_{dataset}",
        dataset,
        model_graphs=gen_graphs,
        dataset_graphs=dataset_graphs,
        loss_dir=None,
        plots_save_dir=plots_save_dir,
    )
    main_run_MMD(
        opt.max_epochs,
        csv_dir=plots_save_dir,
        model_graphs=gen_graphs,
        dataset_graphs=dataset_graphs,
        numb_graphs=N_GRAPH,
    )
