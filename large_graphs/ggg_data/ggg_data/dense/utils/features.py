import torch as pt

from ggg.utils.eigen import stable_sym_eigen, our_dom_symeig

# use K *SMALLEST* eigenvalues/corresponding vectors *everywhere*

if __name__ == "__main__":
    """
    Verify that the 2nd highed eigenval of the laplacian is indeed informative of communities
    """
    from ggg_data.dense.utils.helpers import CommSmall
    import matplotlib

    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import networkx as nx

    ds = CommSmall()
    ps = ds[0]
    a = pt.stack([p.A for p in [ds[i] for i in range(3)]])
    deg = a.sum(-1, keepdim=False)
    D = pt.diag_embed(deg)
    # print(D.shape, a.shape)
    L = a - D
    _, feat = stable_sym_eigen(L, True, noise_scale=0.1)
    # print(feat.shape)
    # print(feat[:, :, :3])
    g = nx.from_numpy_array(a[0].numpy())
    print(feat.shape)
    eig_k2_direct = feat[0, :, -2].flatten()
    feat0 = stable_sym_eigen(L[0], True, noise_scale=0.1)[1][:, -2]
    eig_k2 = our_dom_symeig(L[0], 2, noise_scale=0.1)[1].flatten()
    print(eig_k2_direct.shape, eig_k2.shape)
    print(len(eig_k2_direct), len(g.nodes))
    c1 = (eig_k2_direct > 0).nonzero().flatten().tolist()
    c2 = (eig_k2_direct <= 0).nonzero().flatten().tolist()
    pos = nx.spring_layout(g)
    nx.draw_networkx_nodes(g, pos, node_color="red", nodelist=c1)
    nx.draw_networkx_nodes(g, pos, node_color="blue", nodelist=c2)
    nx.draw_networkx_edges(g, pos)
    plt.show()
    c1o = (eig_k2 > 0).nonzero().flatten().tolist()
    c2o = (eig_k2 <= 0).nonzero().flatten().tolist()
    plt.figure()
    nx.draw_networkx_nodes(g, pos, node_color="pink", nodelist=c1)
    nx.draw_networkx_nodes(g, pos, node_color="cyan", nodelist=c2)
    nx.draw_networkx_edges(g, pos)
    plt.show()
    print(eig_k2_direct.shape, feat0.shape, feat.shape, eig_k2.shape)
    print(eig_k2_direct.flatten() - feat0.flatten())
    print(eig_k2_direct.flatten() - eig_k2.flatten())
