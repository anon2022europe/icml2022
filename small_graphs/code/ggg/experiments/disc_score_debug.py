import matplotlib

from ggg.utils.grad_penalty import GradPenHpars, GradPenalty

matplotlib.use("Qt5Agg")
import torch as pt
from sacred import Experiment
from sacred.experiment import Run
import matplotlib.pyplot as plt
from tqdm import tqdm

from ggg.data.dense.PEAWGANDenseStructureData import PEAWGANDenseStructureData
from ggg.models.ggg_model import Discriminator,DiscriminatorHpars
import networkx as nx
import numpy as np
import seaborn as sb
sb.set()
import attr
matplotlib.use("Qt5Agg")
ex=Experiment("Disc Curve")
@ex.config
def myconf():
    disc_hpars=DiscriminatorHpars(
        node_attrib_dim=0,
        kc_flag=True,
        conv_channels = [32, 64, 128, 128],  # 128, 128, 128]
        eigenfeat4=True,
        add_global_node=False,
    ).to_dict()
    max_N=100
    step=10
    yscale="linear"

@ex.automain
def run(disc_hpars,max_N,step,_run:Run,yscale):
    disc_hpars=DiscriminatorHpars.from_dict(disc_hpars)
    disc:Discriminator=disc_hpars.make(max_N)
    Ns=np.array(list(range(0,max_N,step)))
    data=[pt.from_numpy(nx.to_numpy_array(nx.circular_ladder_graph(n))).float().unsqueeze(0) for n in Ns]
    disc_scores=[disc.forward(None,adj=a) for a in tqdm(data,desc="Getting scores")]
    grad_pen:GradPenalty=GradPenHpars().make()
    penalties=[grad_pen.forward(disc,(None,None),(a.detach(),a.detach())) for a in tqdm(data,desc="Getting penalties")]
    label1="CircularLadderScores"
    fig,axs=plt.subplots(ncols=2)
    axs[0].plot(Ns*2,disc_scores,label=label1,marker="x")
    axs[0].set_xlabel("Number of nodes")
    axs[0].set_ylabel("Disc score")
    axs[1].plot(Ns*2,penalties,label=label1,marker="x")
    axs[1].set_yscale(yscale)
    axs[1].set_xlabel("Number of nodes")
    axs[1].set_ylabel("Grad penalty")
    fig_path="disc_scores.pdf"
    fig.savefig(fig_path)
    _run.add_artifact(fig_path,"disc_scores.pdf")
    plt.show()

