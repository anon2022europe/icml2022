# old configs we might or might not need, if we need them, fold them into the Hpars class as class methodsc


@ex.named_config
def condgen_dblp():
    hyper = dict(
        dataset="condgen_dblp",
        node_feature_dim=10,
        dataset_kwargs=dict(DATA_DIR="/home/AUTHOR/graphs/data_dblp"),
        label_one_hot=None,
    )


@ex.named_config
def condgen_tcga():
    hyper = dict(
        dataset="condgen_tcga",
        label_one_hot=None,
        node_feature_dim=10,  # TODO: check, this should not need 10 for the node_feature+1 setup?
        dataset_kwargs=dict(DATA_DIR="/home/AUTHOR/graphs/data_tcga"),
    )


@ex.named_config
def RGG_PointMLP_qm9():
    # is this just pointnet_st?
    hyper = dict(
        dataset="MolGAN_5k",
        device="cpu",
        disc_contrast="fake_fake",
        cut_train_size=False,
        edge_readout="QQ_sig",
        architecture="mlp_row",
        MLP_layers=[128, 256, 512],
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=5,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="standard",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        structured_features=False,
    )


@ex.named_config
def RGG_PointMLP_chordal9():
    # is this just pointnet_st?
    hyper = dict(
        dataset="anu_graphs_chordal9",
        device="cpu",
        cut_train_size=False,
        edge_readout="QQ_sig",
        architecture="mlp_row",
        disc_contrast="fake_fake",
        MLP_layers=[128, 256, 512],
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="standard",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=4,
        use_laplacian=True,
        structured_features=True,
    )


@ex.named_config
def RGG_PointMLP_commsmall20():
    # is this just pointnet_st?
    hyper = dict(
        dataset="CommunitySmall_20",
        device="cpu",
        cut_train_size=False,
        edge_readout="QQ_sig",
        disc_contrast="fake_fake",
        architecture="mlp_row",
        MLP_layers=[128, 256, 512],
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="standard",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=4,
        use_laplacian=True,
        structured_features=True,
    )


@ex.named_config
def RGG_PointMLP_SBM():
    # is this just pointnet_st?
    hyper = dict(
        dataset="SBM",
        device="cpu",
        cut_train_size=False,
        edge_readout="QQ_sig",
        disc_contrast="fake_fake",
        architecture="mlp_row",
        MLP_layers=[128, 256, 512],
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="standard",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=4,
        use_laplacian=True,
        structured_features=True,
    )


@ex.named_config
def PointMLP_qm9():
    # is this just mlp_row?
    hyper = dict(
        dataset="MolGAN_5k",
        device="cpu",
        disc_contrast="fake_fake",
        cut_train_size=False,
        edge_readout="QQ_sig",
        architecture="mlp_row",
        MLP_layers=[128, 256, 512],
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=5,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="finetti_noDS",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        structured_features=False,
    )


@ex.named_config
def PointMLP_chordal9():
    # is this just mlprow
    hyper = dict(
        dataset="anu_graphs_chordal9",
        device="cpu",
        cut_train_size=False,
        edge_readout="QQ_sig",
        architecture="mlp_row",
        disc_contrast="fake_fake",
        MLP_layers=[128, 256, 512],
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="finetti_noDS",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=4,
        use_laplacian=True,
        structured_features=True,
    )


@ex.named_config
def PointMLP_commsmall20():
    # is this just point mlp
    hyper = dict(
        dataset="CommunitySmall_20",
        device="cpu",
        cut_train_size=False,
        edge_readout="QQ_sig",
        disc_contrast="fake_fake",
        architecture="mlp_row",
        MLP_layers=[128, 256, 512],
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="finetti_noDS",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=4,
        use_laplacian=True,
        structured_features=True,
    )


@ex.named_config
def PointMLP_SBM():
    # is this just point mlp
    hyper = dict(
        dataset="SBM",
        device="cpu",
        cut_train_size=False,
        edge_readout="QQ_sig",
        disc_contrast="fake_fake",
        architecture="mlp_row",
        MLP_layers=[128, 256, 512],
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="finetti_noDS",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=4,
        use_laplacian=True,
        structured_features=True,
    )


@ex.named_config
def RGG_attention_qm9():
    # is this  just attention
    hyper = dict(
        dataset="MolGAN_5k",
        device="cpu",
        n_attention_layers=6,
        cut_train_size=False,
        disc_contrast="fake_fake",
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=5,
        embed_dim=25,
        finetti_dim=25,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="standard",
        score_function="softmax",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        node_feature_dim=5,
        finneti_MLP=False,
        structured_features=False,
    )


@ex.named_config
def RGG_attention_chordal9():
    hyper = dict(
        dataset="anu_graphs_chordal9",
        device="cpu",
        n_attention_layers=6,
        cut_train_size=False,
        disc_contrast="fake_fake",
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        embed_dim=50,
        finetti_dim=50,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="standard",
        score_function="sigmoid",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=4,
        use_laplacian=True,
        structured_features=True,
    )


@ex.named_config
def attention_SBM():
    # is tis just attention?
    hyper = dict(
        dataset="SBM",
        device="cpu",
        n_attention_layers=3,
        disc_contrast="fake_fake",
        cut_train_size=False,
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        embed_dim=25,
        finetti_dim=25,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="finetti_noDS",
        score_function="sigmoid",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=2,
        k_eigenvals=2,
        use_laplacian=True,
        structured_features=True,
    )


@ex.named_config
def attention_community_eigen():
    hyper = dict(
        dataset="CommunitySmall_20",
        device="cpu",
        n_attention_layers=3,
        disc_contrast="fake_fake",
        cut_train_size=False,
        edge_readout="attention_weights",
        architecture="attention",
        dataset_kwargs=dict(DATA_DIR="/ggg/data"),
        label_one_hot=None,
        embed_dim=25,
        finetti_dim=25,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128],
        cycle_opt="finetti_noDS",
        score_function="sigmoid",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        node_feature_dim=2,
        disc_eigenfeat4=True,
        use_laplacian=True,
        k_eigenvals=2,
        structured_features=True,
    )


@ex.named_config
def product5k1():
    """
    Notes:
    - sparsemax does a sort, so it won't scale to 1k*1k matrices+gradient
    """
    val_every = 5
    ckpt_period = 5
    forward_clip = True
    hyper = GGG_Hpar(
        None,
        dataset="product5k1",
        device="cuda:0",
        n_attention_layers=16,
        num_workers=12,
        cut_train_size=False,
        edge_readout="attention_weights",
        batch_size=20,
        score_function="softmax",
        readout_score_function="sigmoid",
        architecture="attention",
        dataset_kwargs=None,
        embed_dim=29,
        finetti_dim=50,
        node_feature_dim=4,
        num_heads=10,
        kc_flag=True,
        disc_conv_channels=[32, 64, 64, 64, 128, 128, 128, 256],
        disc_readout_hidden=128,
        cycle_opt="finetti_noDS",
        finetti_trainable=True,
        finetti_train_fix_context=False,
        dynamic_finetti_creation=False,
        replicated_Z=False,
        finneti_MLP=False,
        structured_features=True,
        score_penalty=0.0,
        gen_spectral_norm="nondiff",
        disc_spectral_norm="nondiff",
        temperature=2 / 3.0,  # lower => more discrete, less smooth
        disc_contrast="fake-struct-detach_fake",
        generator_every=1,
        disc_optim_args=dict(
            lr=5e-3,  # TTUR
            betas=(0.5, 0.9999),
            eps=1e-8,
            weight_decay=1e-3,
            ema=False,  # ema seems to cause with egonet
            ema_start=10,
        ),
        gen_optim_args=dict(
            lr=1e-3,
            betas=(0.5, 0.9999),
            eps=1e-8,
            weight_decay=1e-3,
            ema=False,  # ema seems to cause with egonet
            ema_start=10,
        ),
    ).to_dict()
