from warnings import warn

import igraph
import torch
import numpy as np
import torch as pt
from ipdb import set_trace
from torch.nn.functional import one_hot
import networkx as nx
from tqdm import tqdm
try:
    from graph_tool.inference import minimize_blockmodel_dl
    import graph_tool as gt
except:
    warn(f"please install graph tools with 'conda install -c conda-forge graph-tool' to use the sbm ll metric")

def Leiden(A, resolution=1.0, device="cpu", list_only=False):
    """
    Utilizes an implementation of https://doi.org/10.1038/s41598-019-41695-z
    """

    N = A.size(0)
    edges = A.nonzero().numpy()
    graph = igraph.Graph(n=N, edges=edges)
    membership_list = graph.community_leiden(resolution_parameter=resolution, n_iterations=-1).membership
    if list_only:
        return membership_list
    num_com = max(membership_list) + 1
    membership_matrix = one_hot(torch.tensor(membership_list, device=device), num_com).float()
    return membership_matrix, num_com


def flat_triu(x):
    """
    Get the upper triangular part of a matrix as a flattened vector
    :param x:
    :return:
    """
    assert x.dim() == 2
    assert x.shape[0] == x.shape[1]
    i = pt.triu_indices(*x.shape)
    triu= x[i[0],i[1]]
    return triu


def blockll(b, p):
    """
    Block Likelihood: \sum_{e_{ij}\in g} log p(e_{ij}\in g)  + \sum_{e_{ij}\ni g} log p(e_{ij}\ni g)
    with p=p(e_{ij} \in g) and np=p(e_{ij} \ni g) and log (prod (...))=sum(log(...)) becomes
    sum(log(p*edges + (1-edges)*np))=
    sum(log(p*edges + np - edges*np))=
    sum(log(2p*edges  - edges+ 1-p))=
    sum(log((2p-1)*edges +1-p))
    :param b:
    :param p:
    :return:
    """
    return ((2*p-1)*b+1-p).log().sum()

def block_mean(x,diag=False):
    assert x.dim()==2
    assert x.shape[0]==x.shape[1]
    assert pt.diagonal(x).sum()==0.0
    n=x.shape[0]
    if diag:
        return x.mean()
    else:
        return x.sum()/(n**2-n)

def comm_is_connected(g,comm):
    view=gt.Graph(gt.GraphView(g,vfilt=[x in comm for x in g.vertex_index]),prune=True)
    lcc=gt.topology.extract_largest_component(view,prune=True)
    nv=view.num_vertices()
    nlcc=lcc.num_vertices()
    return nv==nlcc

def sbm_comm(a,entropy=False,num_tries=5):
    if pt.is_tensor(a):
        a=a.numpy()
    gtg = gt.Graph(directed=False)
    al = np.stack(a.nonzero()).T
    gtg.add_edge_list(al)
    connected=False
    for i in range(max(num_tries,1)):
        state = minimize_blockmodel_dl(gtg,
                                       state_args=dict(deg_corr=False,B=2),
                                       multilevel_mcmc_args=dict(B_min=2, B_max=2))
        state:gt.inference.BlockState
        blocks = state.get_blocks()
        bar=blocks.get_array()
        assert len(np.unique(bar)) == 2
        c0, c1 = [(bar == b).nonzero()[0] for b in np.unique(bar)]
        connected=comm_is_connected(gtg,c0)
        if connected:
            break

    # https://graph-tool.skewed.de/static/doc/inference.html#graph_tool.inference.minimize_blockmodel_dl
    # see here for documentation
    if entropy:
        if connected or True:
            entropy = state.entropy(degree_dl=False, deg_entropy=False,dense=True,partition_dl=True,degree_dl_kind="uniform",dl=True,
                                    adjacency=True,edges_dl=True)
        else:
            entropy=np.inf
        return (c0,c1),entropy
    else:
        return c0,c1

def nx_fitted_sbm_nll_per_edge(g:nx.Graph, seed=0, comms=None, use_entropy=True) -> pt.Tensor:
    a=pt.from_numpy(nx.to_numpy_array(g))
    return fitted_sbm_nll_per_edge(a,seed,comms,use_entropy)


def fitted_sbm_nll_per_edge(A:pt.Tensor,seed=0,comms=None,use_entropy=True,per_edge=True)->pt.Tensor:
    assert A.dim()==2 # only implemented for single graph for now
    A=A.cpu()
    n=A.shape[-1]
    if per_edge:
        normalizer=A.sum()+1e-9
    else:
        normalizer=0.5*n*(n-1)+1e-9
    if comms is None or use_entropy:
        if use_entropy:
            (c0,c1),e= sbm_comm(A.numpy(),entropy=True)
            return e/normalizer
        else:
            c0,c1=sbm_comm(A.numpy())
    else:
        c0,c1=comms
    # this was my attempt to estimate things based on the definiion of ll and SBM, but I trust the library more
    pa0,pa1=[pt.tensor(list(x)) for x in [c0,c1]]
    b0both=A[pa0]
    b1both=A[pa1]
    b0=b0both[:,pa0]
    b1=b1both[:,pa1]
    c0t1=b0both[:,pa1]
    c1t0=b1both[:,pa0]
    n0=len(b0)
    n1=len(b1)
    #assert n0==n1
    # inter block probabilities, accounting for symmetry
    p0=block_mean(b0)
    p1=block_mean(b1)
    p01=c0t1.sum()/(n0*n1) # does not include diagonal
    p10=c1t0.sum()/(n0*n1) # does not include diagonal, should be identical to p01
    assert p01==p10
    p_intra=(p0+p1)/2
    p_inter=p01

    intra_ll=blockll(flat_triu(b0),p_intra)*2+blockll(flat_triu(b1),p_intra)*2
    inter_ll=blockll(c0t1,p_inter)*2
    ll=intra_ll+inter_ll
    print(f"Intra{p_intra} inter {p_inter} ll{ll}")
    return  -ll/normalizer

if __name__=="__main__":
    import matplotlib.pyplot as plt
    import math
    import numpy as np
    from networkx.algorithms.community.centrality import girvan_newman
    plt.close()
    intra=4
    inter=0.4
    n=24
    ns=max(math.sqrt(n),(intra+1))
    p=[[intra/ns,inter/ns],[inter/ns,intra/ns]]
    print(p[0])
    g=nx.stochastic_block_model([n//2,n//2],p,directed=False,selfloops=False,sparse=False)
    # split the center community into 2 halves or assign it to either of them => skew the probability
    ic=0.2
    groc=nx.ring_of_cliques(12,2)
    g3sbm=nx.stochastic_block_model([8,8,8],[
        [0.9,0.1,0],
        [0.1, 0.9, 0.1],
        [0.0, 0.1, 0.5],
    ],directed=False,sparse=False,selfloops=False)

    a=nx.to_numpy_array(g)
    at=pt.from_numpy(nx.to_numpy_array(g3sbm))
    pos=nx.spring_layout(g)
    r=2
    c=2
    fig,axs=plt.subplots(nrows=r,ncols=c)
    axs=axs.flatten()
    if r==1:
        axs=[axs]
    i=0
    (c0,c1)=sbm_comm(a,entropy=False)
    nx.draw(g,pos=pos,ax=axs[i])
    nx.draw_networkx_nodes(g, pos, nodelist=c0, node_color='b',ax=axs[i])
    nx.draw_networkx_nodes(g, pos, nodelist=c1, node_color='r',ax=axs[i])
    c0,c1=sbm_comm(at,entropy=False)
    i+=1
    pos=nx.spring_layout(g3sbm)
    nx.draw(g3sbm,pos=pos,ax=axs[i])
    nx.draw_networkx_nodes(g3sbm, pos, nodelist=c0, node_color='b',ax=axs[i])
    nx.draw_networkx_nodes(g3sbm, pos, nodelist=c1, node_color='r',ax=axs[i])
    i+=1
    tr=nx.random_tree(n)
    atr=pt.from_numpy(nx.to_numpy_array(tr))
    c0,c1=sbm_comm(atr,entropy=False)
    pos=nx.spring_layout(tr)
    nx.draw(tr,pos=pos,ax=axs[i])
    nx.draw_networkx_nodes(tr, pos, nodelist=c0, node_color='b',ax=axs[i])
    nx.draw_networkx_nodes(tr, pos, nodelist=c1, node_color='r',ax=axs[i])
    i+=1
    aroc=pt.from_numpy(nx.to_numpy_array(tr))
    c0,c1=sbm_comm(aroc,entropy=False)
    pos=nx.spring_layout(groc)
    nx.draw(groc,pos=pos,ax=axs[i])
    nx.draw_networkx_nodes(groc, pos, nodelist=c0, node_color='b',ax=axs[i])
    nx.draw_networkx_nodes(groc, pos, nodelist=c1, node_color='r',ax=axs[i])

    plt.show()
    print(a.shape)
    a=pt.from_numpy(a)
    ll=fitted_sbm_nll_per_edge(a)#,comms=(c0,c1))
    print(f"Min entropy sbm {ll} {ll}")
    ll=fitted_sbm_nll_per_edge(at)#,comms=(c0,c1))
    print(f"Min entropy  sbm3 {ll} {ll}")
    ll=fitted_sbm_nll_per_edge(atr)#,comms=(c0,c1))
    print(f"Min entropy  tree {ll} {ll}")
    ll=fitted_sbm_nll_per_edge(aroc)#,comms=(c0,c1))
    print(f"Min entropy  ring {ll} {ll}")






