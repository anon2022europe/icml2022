import torch
import torch.nn as nn
from attr.validators import in_
from torch_geometric.nn import DenseGINConv as DenseGINConv_orig
from tqdm import tqdm

from ggg.models.components.abstract_conf import AbstractConf
from ggg.models.components.discriminators.disc_feat_extractor import DiscFeatExtractor
from ggg.models.components.discriminators.discriminator_readout import DiscriminatorReadout
from ggg.models.components.global_node_adder import GlobalNodeAdder
from ggg.utils.utils import (
    kwarg_create,
    pg_mask_from_N,
    node_mask, maybe_assert,
)
from ggg.models.components.utilities_classes import Swish, NodeFeatNorm
from ggg.models.components.spectral_norm import (
    SpectralNorm,
    SpectralNormNonDiff,
    SPECTRAL_NORM_OPTIONS,
)
from ggg.models.components.pointnet_st import LinearTransmissionLayer
import torch as pt
import attr


class DenseGINConv(DenseGINConv_orig):
    def __init__(
        self,
        model,
        *args,
        transmission_layer=True,
        spectral_norm=None,
        graph_size_norm=None,
        max_nodes=None,
        in_c=None,
        **kwargs,
    ):
        super().__init__(model)
        if graph_size_norm in {
            "graph-size-c",
            "layer",
            "instance",
            "graph-size-v",
            "graph-size-deg",
            "graph-size-maxdeg",
        }:
            self.gs_norm = NodeFeatNorm(in_c, graph_size_norm, max_nodes=max_nodes)
        else:
            self.gs_norm = None
        if spectral_norm == "diff":
            sn = lambda x: SpectralNorm(x, name="B")
        elif spectral_norm == "nondiff":
            sn = lambda x: SpectralNormNonDiff(x, name="B")
        else:
            sn = lambda x: x
        if transmission_layer:
            c_in = None
            for l in model:
                if isinstance(l, SpectralNorm) or isinstance(l, SpectralNormNonDiff):
                    l = l.module
                if hasattr(l, "weight"):
                    c_in = l.weight.shape[1]
                    break
                elif hasattr(l, "weight_bar"):
                    c_in = l.weight_bar.shape[1]
                    break
            c_out = None
            for l in reversed(model):
                if isinstance(l, SpectralNorm) or isinstance(l, SpectralNormNonDiff):
                    l = l.module
                if hasattr(l, "weight"):
                    c_out = l.weight.shape[0]
                    break
                elif hasattr(l, "weight_bar"):
                    c_out = l.weight_bar.shape[0]
                    break
                elif "weight_bar" in l._parameters:
                    c_out = l._parameters["weight_bar"].shape[0]
                    break
            self.transmission_layer = sn(LinearTransmissionLayer(c_in, c_out))
        else:
            self.transmission_layer = None

    def forward(self, x, adj, mask=None, add_loop=True):
        _x = x
        if self.gs_norm is not None:
            maybe_assert(func=lambda:pt.isfinite(x).all())
            x = self.gs_norm(x, adj)
            maybe_assert(func=lambda: pt.isfinite(x).all())
        out = super().forward(x, adj, mask=mask, add_loop=add_loop)
        maybe_assert(func=lambda: pt.isfinite(out).all())
        if self.transmission_layer is not None:
            transmission = self.transmission_layer(x)
            maybe_assert(func=lambda: pt.isfinite(transmission).all())
            out = out + transmission
            maybe_assert(func=lambda: pt.isfinite(out).all())
        return out

    def extra_repr(self) -> str:
        return f"{super().extra_repr()},\n transmission_layer={str(self.transmission_layer)}"

def get_act(swish):
    if swish is True or swish == "swish":
        act = Swish
    elif swish == "leaky":
        act = lambda: torch.nn.LeakyReLU(0.1)
    elif swish == "celu":
        act = torch.nn.CELU
    else:
        act = torch.nn.ReLU
    return act

class GNNPostNorm(torch.nn.Module):
    def __init__(self,model,proj=None,norm=None,rezero=False):
        super(GNNPostNorm, self).__init__()
        self.model=model
        self.proj=proj
        self.norm=norm
        self.gate = pt.nn.Parameter(pt.zeros([])) if rezero else None

    def forward(self,x,A,mask=None):
        skip=self.proj(x,A,mask=mask) if self.proj else x
        xm=self.model(x,A,mask=mask)
        if self.gate is not None:
            xm=xm*self.gate
        xout=skip+xm
        if self.norm:
            xout=self.norm(xout,A)
        return xout,A

class GNNPreNorm(torch.nn.Module):
    """
    Pre-activation GNN layer
    """
    def __init__(self,model,proj=None,norm=None,rezero=False):
        super(GNNPreNorm, self).__init__()
        self.model=model
        self.proj=proj
        self.norm=norm
        self.gate=pt.nn.Parameter(pt.zeros([])) if rezero else None
    def forward(self,x,A,mask=None):
        skip=self.proj(x,A,mask=mask) if self.proj else x
        xn=self.norm(x,A)  if self.norm else x
        xm=self.model(xn,A,mask=mask)
        if self.gate is not None:
            xm=self.gate*xm
        return skip+xm,A

class GINSkip(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        gin_hidden_width=32,
        model=None,
        swish=False,
        spectral_norm=None,
        dropout=None,
        norm_type="layer",
        graph_size_norm=None,
        max_nodes=None,
    ):
        super().__init__()
        act = get_act(swish)
        if spectral_norm == "diff":
            sn = SpectralNorm
        elif spectral_norm == "nondiff":
            sn = SpectralNormNonDiff
        else:
            sn = lambda x: x
        self.act = act()

        self.norm = NodeFeatNorm(out_channels, mode=norm_type, max_nodes=max_nodes,affine="affine" in norm_type)
        if model is None:
            l = [
                sn(torch.nn.Linear(in_channels, gin_hidden_width)),
                # 100% match to working code, TODO: revert/revisit
                # NodeFeatNorm(gin_hidden_width, mode=norm_type,max_nodes=max_nodes),
                act(),
                sn(torch.nn.Linear(gin_hidden_width, out_channels)),
            ]
            model = torch.nn.Sequential(*l)
        self.gcn = DenseGINConv(
            model,
            spectral_norm=spectral_norm,
            graph_size_norm=graph_size_norm,
            max_nodes=max_nodes,
            in_c=in_channels,
        )
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

        if in_channels != out_channels:
            self.proj = DenseGINConv(
                torch.nn.Sequential(sn(torch.nn.Linear(in_channels, out_channels))),
                spectral_norm=spectral_norm,
                graph_size_norm=graph_size_norm,
                max_nodes=max_nodes,
                in_c=in_channels,
            )
        else:
            self.proj = None


    def forward(self, x, A, mask=None):
        if self.dropout:
            x = self.dropout(x)
        if self.proj is not None:
            xskip = self.proj(x, A)
        else:
            xskip = x

        xg = self.gcn(x, A, mask=mask)
        maybe_assert(func=lambda: pt.isfinite(xg).all())
        xg = self.norm(xg, A)
        maybe_assert(func=lambda: pt.isfinite(xg).all())
        maybe_assert(func=lambda: pt.isfinite(xskip).all())
        out = xg + xskip
        maybe_assert(func=lambda: pt.isfinite(out).all())
        return out




class SimpleMLP(pt.nn.Module):
    def __init__(self,
                 in_channels,
                 swish=False,
                 spectral_norm=None,
                 dropout=None,
                 gcn_norm_type="layer",
                 graph_size_norm=None,
                 max_nodes=None,
                 kc_flag=True,
                 eigenfeat4=False,
                 readout_hidden=64,
                 readout_norm_in_type="instance",
                 readout_norm_hidden_type="layer",
                 ):
        super(SimpleMLP, self).__init__()
        act=get_act(swish)
        self.mlp1=pt.nn.Sequential(pt.nn.Linear(in_channels,128),act(),pt.nn.Linear(128,128))
        self.mlp2 = pt.nn.Sequential(pt.nn.Linear(128, 128), act(), pt.nn.Linear(128, 128))
        self.mlp2 = pt.nn.Sequential(pt.nn.Linear(128, 128), act(), pt.nn.Linear(128, 1))

    def forward(self,node,adj):
        x=node
        x=self.mlp1(x)
        x=adj@x
        x=self.mlp2(x)
        x=adj@x
        x=x.mean(1)
        return self.mlp3(x)
class SimpleGIN(pt.nn.Module):
    def __init__(self,in_channels,conv_channels,
                 swish=False,
                 spectral_norm=None,
                 dropout=None,
                 gcn_norm_type="layer",
                 graph_size_norm=None,
                 max_nodes=None,
                 kc_flag=True,
                 eigenfeat4=False,
                 readout_hidden=64,
                 readout_norm_in_type="instance",
                 readout_norm_hidden_type="layer",
                 ):
        super(SimpleGIN, self).__init__()
        for l_ in range(len(conv_channels)):
            self.gcn_layers.append(
                SimpleGCN(
                    in_channels=conv_channels[l_ - 1]
                    if l_ > 0
                    else in_channels,
                    out_channels=conv_channels[l_],
                    swish=swish,
                    spectral_norm=spectral_norm,
                    dropout=dropout,
                    norm_type=gcn_norm_type,
                    max_nodes=max_nodes,
                    graph_size_norm=graph_size_norm,
                )
            )

        self.max_nodes = max_nodes
        readout_feats = sum(conv_channels)
        if self.precat_raw:
            readout_feats += in_channels
        self.read_out = DiscriminatorReadout(
            readout_feats,
            readout_hidden,
            out_dim=1,
            swish=swish,
            spectral_norm=spectral_norm,
            dropout=dropout,
            kc_flag=kc_flag,
            eigenfeat4=eigenfeat4,
            pac=self.pac,
            agg=self.readout_agg,
            norm_in_type=readout_norm_in_type,
            norm_hidden_type=readout_norm_hidden_type,
            max_nodes=max_nodes,
        )
    def forward(self,effect_node_feat,adj,N=None,mode="score"):

        if N is not None:
            pg_mask = pg_mask_from_N(max_N=self.max_nodes, N=N.reshape(-1, 1))
        else:
            pg_mask = None
        zi = effect_node_feat
        if self.precat_raw:
            zis = [zi]
        else:
            zis = []
        for i, gcn in enumerate(self.gcn_layers):
            zi = gcn(zi, adj, mask=pg_mask)
            zis.append(zi)

        zfinal = torch.cat(zis, -1)
        if N is not None:
            zfinal = node_mask(zfinal, N) * zfinal

        out = self.read_out(zfinal, adj,mode=mode)
        return out
class OurGIN(pt.nn.Module):
    def __init__(self,in_channels,conv_channels,
                 swish=False,
                 spectral_norm=None,
                 dropout=None,
                 gcn_norm_type="layer",
                 graph_size_norm=None,
                 max_nodes=None,
                 kc_flag=True,
                 eigenfeat4=False,
                 readout_hidden=64,
                 readout_norm_in_type="instance",
                 readout_norm_hidden_type="layer",
                 precat_raw=True,
                 pac=1,
                 readout_agg="mean"
                 ):
        super(OurGIN, self).__init__()
        self.gcn_layers=pt.nn.ModuleList()
        self.precat_raw=precat_raw
        for l_ in range(len(conv_channels)):
            self.gcn_layers.append(
                GINSkip(
                    in_channels=conv_channels[l_ - 1]
                    if l_ > 0
        else in_channels,
                    out_channels=conv_channels[l_],
                    swish=swish,
                    spectral_norm=spectral_norm,
                    dropout=dropout,
                    norm_type=gcn_norm_type,
                    max_nodes=max_nodes,
                    graph_size_norm=graph_size_norm,
                )
            )

        self.max_nodes = max_nodes
        self.pac=pac
        self.readout_agg=readout_agg
        self.in_channels=in_channels
        readout_feats = sum(conv_channels)
        if self.precat_raw:
            readout_feats += in_channels
        self.read_out = DiscriminatorReadout(
            readout_feats,
            readout_hidden,
            out_dim=1,
            swish=swish,
            spectral_norm=spectral_norm,
            dropout=dropout,
            kc_flag=kc_flag,
            eigenfeat4=eigenfeat4,
            pac=self.pac,
            agg=self.readout_agg,
            norm_in_type=readout_norm_in_type,
            norm_hidden_type=readout_norm_hidden_type,
            max_nodes=max_nodes,
        )
    def forward(self,effect_node_feat,adj,N=None,mode="score"):

        if N is not None:
            pg_mask = pg_mask_from_N(max_N=self.max_nodes, N=N.reshape(-1, 1))
        else:
            pg_mask = None
        zi = effect_node_feat
        if self.precat_raw:
            zis = [zi]
        else:
            zis = []
        for i, gcn in enumerate(self.gcn_layers):
            zi = gcn(zi, adj, mask=pg_mask)
            maybe_assert(pt.isfinite(zi).all())
            zis.append(zi)

        zfinal = torch.cat(zis, -1)
        maybe_assert(func=lambda: pt.isfinite(zfinal).all())
        if N is not None:
            zfinal = node_mask(zfinal, N) * zfinal

        out = self.read_out(zfinal, adj,mode=mode)
        return out

class Discriminator(nn.Module):
    """GCN encoder with residual connections"""
    ARCHITECTURES={"GIN":OurGIN, "MLPSimple":SimpleMLP}
    def __init__(
        self,
        node_attrib_dim,
        conv_channels,
        readout_hidden=64,
        swish=False,
        spectral_norm=None,
        dropout=None,
        kc_flag=True,
        eigenfeat4=False,
        pac=1,
        structural_features="all",
        readout_agg="sum",
        gcn_norm_type="layer",
        graph_size_norm=None,
        readout_norm_in_type="instance",
        readout_norm_hidden_type="layer",
        max_nodes=None,
        add_global_node=True,
        precat_raw=False,
        architecture="kcycleGIN",
        fake_eigen=False,
        n_rand_feats=0
    ):
        super().__init__()

        self.gcn_layers = torch.nn.ModuleList()
        self.feature_extract = DiscFeatExtractor(structural_features,fake_eigen=fake_eigen,n_rand_feats=n_rand_feats)
        self.pac = pac
        self.readout_agg = readout_agg
        self.precat_raw = precat_raw
        self.node_attrib_dim = node_attrib_dim
        self.effective_node_feat_dim = (
            self.node_attrib_dim + self.feature_extract.feat_dim()
        )

        if add_global_node:
            self.global_node_adder = GlobalNodeAdder()
        else:
            self.global_node_adder = None
        # node_feature_dim=+1 #node features + number of nodes feature
        in_channels=self.effective_node_feat_dim + int(add_global_node)
        DISCCLASS=Discriminator.ARCHITECTURES[architecture]
        self.architecture=architecture
        self.disc=DISCCLASS(in_channels,conv_channels,
                      swish=swish,
                      spectral_norm=spectral_norm,
                      dropout=dropout,
                      gcn_norm_type=gcn_norm_type,
                      graph_size_norm=graph_size_norm,
                      max_nodes=max_nodes,
                      kc_flag=kc_flag,
                      eigenfeat4=eigenfeat4,
                      readout_hidden=readout_hidden,
                      readout_norm_in_type=readout_norm_in_type,
                      readout_norm_hidden_type=readout_norm_hidden_type,
                        precat_raw=precat_raw,
                        pac=pac,readout_agg=readout_agg)

    def forward(self, attrib, adj, N=None, mode="score"):
        # x=single_node_featues(x,adj,k_paths=0)
        if self.node_attrib_dim == 0:
            # ignore whatever we pass in
            attrib = None
        if mode not in DiscriminatorReadout._MODES:
            raise ValueError(
                f"Only know discriminator modes {DiscriminatorReadout._MODES}"
            )


        if adj.dim() == 4:
            assert adj.shape[1] == self.pac
            adj = adj.reshape(-1, adj.shape[-2], adj.shape[-1])
        elif adj.dim() == 2:
            # ensure batch size dime exists, otherwise InstanceNorm1D throws errors
            adj = adj.unsqueeze(0)

        assert adj.dim() == 3  # for global node adder
        if attrib is not None:
            assert attrib.shape[-1] == self.node_attrib_dim
            if attrib.dim() == 4:
                assert attrib.shape[1] == self.pac
                attrib = attrib.reshape(-1, attrib.shape[-2], attrib.shape[-1])
            elif attrib.dim() < 3:
                attrib = attrib.unsqueeze(0)
            assert attrib.dim() == 3  # for global node adder

        # extract structural node features if using them, append them to node attribs if they exist
        extracted_feats = self.feature_extract(adj, N=N)
        effect_node_feat = [] if attrib is None else [attrib]
        if extracted_feats is not None:
            effect_node_feat.append(extracted_feats)
        assert (
                len(effect_node_feat) > 0
        ), "Need either attributions in the dataset or structural features enabled"
        effect_node_feat = pt.cat(effect_node_feat, -1)
        assert effect_node_feat.shape[-1] == self.effective_node_feat_dim

        if self.global_node_adder:
            effect_node_feat, adj = self.global_node_adder(effect_node_feat, adj)
        assert effect_node_feat.shape[-2] == adj.shape[-1] == adj.shape[-2]
        assert effect_node_feat.shape[-1] == self.effective_node_feat_dim + (int(self.global_node_adder is not None)        )
        out= self.disc(effect_node_feat,adj,N=N,mode=mode)
        maybe_assert(pt.isfinite(out).all())
        return out


SCORE_FUNCTIONS = {"sigmoid", "softmax", "sparsemax"}
EDGE_READOUTS = {
    "biased_sigmoid",
    "rescaled_softmax",
    "attention_weights",
    "QQ_sig",
    "score_thresh",
    "mlp_ratio_sig",
    "mlp_ratio_sparse",
    "mlp_score_thresh",
}

class GraphAggregation(nn.Module):
    def __init__(self, in_features, out_features, n_dim, dropout_rate=0):
        super(GraphAggregation, self).__init__()
        self.sigmoid_linear = nn.Sequential(
            nn.Linear(in_features + n_dim, out_features), nn.Sigmoid()
        )
        self.tanh_linear = nn.Sequential(
            nn.Linear(in_features + n_dim, out_features), nn.Tanh()
        )
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs, activation):
        i = self.sigmoid_linear(inputs)  # i: BxNx128
        j = self.tanh_linear(inputs)  # j: BxNx128
        output = torch.sum(torch.mul(i, j), 1)  # output: Bx128
        output = activation(output) if activation is not None else output
        output = self.dropout(output)

        return output


class MolGAN_Discriminator(nn.Module):
    def __init__(
        self,
        node_feature_dim,
        conv_channels=None,
        readout_hidden=64,
        swish=False,
        spectral_norm=None,
    ):
        super(MolGAN_Discriminator, self).__init__()

        auxiliary_dim = 128
        self.layers_ = [[node_feature_dim, 128], [128 + node_feature_dim, 64]]

        self.bn = torch.nn.ModuleList()
        self.gcn_layers = torch.nn.ModuleList()
        for l_ in self.layers_:
            self.gcn_layers.append(GraphConvolution(l_[0], l_[1]))

        self.agg_layer = GraphAggregation(64, auxiliary_dim, node_feature_dim)

        # Multi dense layer [128x64]
        layers = []
        for c0, c1 in zip([auxiliary_dim], [64]):
            layers.append(nn.Linear(c0, c1))
            layers.append(nn.Tanh())
        self.linear_layer = nn.Sequential(*layers)  # L1: 256x512 | L2: 512x256

        # Linear map [128x1]
        self.output_layer = nn.Linear(64, 1)

    def forward(self, node, adj):
        h = None
        for l in range(len(self.layers_)):
            h = self.gcn_layers[l](inputs=(adj, h, node))
        annotations = torch.cat((h, node), -1)
        h = self.agg_layer(annotations, torch.nn.Tanh())
        h = self.linear_layer(h)

        output = self.output_layer(h)

        return output


# TODO delete after tests
class Discriminator_test(nn.Module):
    """GCN encoder with residual connections"""

    def __init__(
        self,
        node_attrib_dim,
        conv_channels,
        readout_hidden=64,
        swish=False,
        spectral_norm=None,
        dropout=None,
        kc_flag=True,
        eigenfeat4=False,
        structural_features="all",
        pac=1,
        readout_agg="sum",
        gcn_norm_type="layer",
        graph_size_norm=None,
        readout_norm_in_type="instance",
        readout_norm_hidden_type="layer",
        max_nodes=None,
        add_global_node=True,
        precat_raw=False,
    ):
        super().__init__()
        self.pac = pac
        self.readout_agg = readout_agg
        self.precat_raw = precat_raw

        self.max_nodes = max_nodes
        self.node_attrib_dim = node_attrib_dim
        self.feature_extract = DiscFeatExtractor(structural_features)

        if self.node_attrib_dim == 0:
            self.effective_node_feat_dim = self.node_attrib_dim + self.feature_extract.feat_dim()
        else:
            self.effective_node_feat_dim = self.node_attrib_dim

        nn1 = pt.nn.Linear(self.effective_node_feat_dim, 64)
        # DenseGINConv_orig(nn1))
        self.MLP = nn1 #DenseGINConv_orig(nn1)
        self.read_out = pt.nn.Linear(64, 1)

    def forward(self, attrib, adj, N=None, mode="score"):
        #x=single_node_featues(x,adj,k_paths=0)

        if self.node_attrib_dim != 0:
            x = attrib

            x = self.MLP(x)
            x = x.sum(dim=-2)
            out = self.read_out(x)
        else:
            if self.node_attrib_dim==0:
                # ignore whatever we pass in
                attrib=None

            if N is not None:
                pg_mask=pg_mask_from_N(max_N=self.max_nodes,N=N.reshape(-1,1))
            else:
                pg_mask=None


            if adj.dim()==4:
                assert adj.shape[1]==self.pac
                adj = adj.reshape(-1, adj.shape[-2], adj.shape[-1])
            elif adj.dim()==2:
                # ensure batch size dime exists, otherwise InstanceNorm1D throws errors
                adj = adj.unsqueeze(0)

            assert adj.dim() == 3  # for global node adder
            if attrib is not None:
                assert attrib.shape[-1] == self.node_attrib_dim
                if attrib.dim() == 4:
                    assert attrib.shape[1] == self.pac
                    attrib = attrib.reshape(-1, attrib.shape[-2], attrib.shape[-1])
                elif attrib.dim() < 3:
                    attrib = attrib.unsqueeze(0)
                assert attrib.dim() == 3  # for global node adder

            # extract structural node features if using them, append them to node attribs if they exist
            extracted_feats=self.feature_extract(adj,N=N,mask=pg_mask)
            effect_node_feat=[] if attrib is None else [attrib]
            if extracted_feats is not None:
                effect_node_feat.append(extracted_feats)
            assert len(effect_node_feat)>0, "Need either attributions in the dataset or structural features enabled"
            effect_node_feat=pt.cat(effect_node_feat,-1)

            x = effect_node_feat

            x = self.MLP(x)
            x = x.sum(dim=-2)
            out = self.read_out(x)
        return out

SCORE_FUNCTIONS={"sigmoid", "softmax", "sparsemax"}
EDGE_READOUTS={
                "biased_sigmoid",
                "rescaled_softmax",
                "attention_weights",
                "QQ_sig",
                "score_thresh",
                "mlp_ratio_sig",
                "mlp_ratio_sparse",
                "mlp_score_thresh",
            }
@attr.s
class DiscriminatorHpars(AbstractConf):
    architecture=attr.ib(default="GIN", validator=in_(Discriminator.ARCHITECTURES))
    conv_channels = attr.ib(default=[64, 128, 128, 128])
    node_attrib_dim = attr.ib(default=32)
    add_global_node = attr.ib(default=False)
    disc_conv_channels = attr.ib(default=32)
    kc_flag = attr.ib(default=True)
    eigenfeat4 = attr.ib(default=False)
    dropout = attr.ib(default=None)
    structural_features = attr.ib(default="all", validator=in_({None, "all"}))
    spectral_norm = attr.ib(default=None, validator=in_(SPECTRAL_NORM_OPTIONS))
    readout_hidden = attr.ib(default=128)
    fake_eigen=attr.ib(default=False)
    swish = attr.ib(
        default="leaky",
        validator=attr.validators.in_({False, True, "swish", "leaky", "relu", "celu"}),
    )
    pac = attr.ib(default=1)
    readout_agg = attr.ib(
        default="mean", validator=attr.validators.in_({"sum", "mean", "max", "lse"})
    )
    graph_size_norm = attr.ib(
        default=None, validator=in_({"graph-size-c", "graph-size-deg", None})
    )
    gcn_norm_type = attr.ib(
        default="layer-affine", validator=in_(NodeFeatNorm.SUPPORTED)
    )
    readout_norm_in_type = attr.ib(
        default="identity", validator=in_(NodeFeatNorm.SUPPORTED)
    )
    readout_norm_hidden_type = attr.ib(
        default="layer-affine", validator=in_(NodeFeatNorm.SUPPORTED)
    )
    precat_raw = attr.ib(default=False)

    # TODO delete after use
    simple_disc = attr.ib(default=False)

    def make(self,max_nodes):
        kwargs=self.to_dict()
        kwargs["max_nodes"]=max_nodes
        if kwargs["simple_disc"]:
            D = Discriminator_test
        else:
            D = Discriminator
        return kwarg_create(D, kwargs)
