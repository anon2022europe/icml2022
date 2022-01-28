import os
from ggg.evaluation.plots.utils.post_experiment_plots import *

# # CommunitySmall run 1
# data_dir = "exp/community_small/edp-gnn_community_small20__Sep-13-21-20-02_71479/sample/sample_data"
# filename = "community_small20_[0.1, 0.2, 0.4, 0.6, 0.8, 1.6].pth_0.005_1.0_sample.pkl"

# # CommunitySmall run 2
# data_dir = "exp/community_small/edp-gnn_community_small20__Sep-15-11-54-06_46173/sample/sample_data"
# filename = "community_small20_[0.1, 0.2, 0.4, 0.6, 0.8, 1.6].pth_0.005_1.0_sample.pkl"

# Chordal9 run 1
# data_dir = "exp/chordal9/edp-gnn_chordal9__Sep-15-11-54-01_46087/sample/sample_data"
# filename = "chordal9_[0.1, 0.2, 0.4, 0.6, 0.8, 1.6].pth_0.005_0.1_sample.pkl"

# QM9 run 1
# data_dir = "exp/MolGAN_5k/edp-gnn_MolGAN_5k__Sep-15-11-53-58_46004/sample/sample_data"
# filename = "MolGAN_5k_[0.1, 0.2, 0.4, 0.6, 0.8, 1.6].pth_0.05_1.0_sample.pkl"

with open(os.path.join(data_dir, filename), "rb") as f:
    graphs = pickle.load(f)

dataset_graphs = external_dataset_graphs(
    os.path.join(
        os.getcwd(),
        "data/CommunitySmall/community_N_nodes5000_maxN20_minN20.sparsedataset",
    )
)
externals_main_run_plot(
    graphs,
    dataset_graphs=dataset_graphs,
    loss_dir="None",
    plots_save_dir="evaluation",
    dataset=None,
    baseline_name="ScrM",
    epoch="5000_QM9_5k",
)
main_run_MMD(
    model=None,
    csv_dir=None,
    model_graphs=graphs,
    dataset_graphs=dataset_graphs,
    numb_graphs=1024,
    save=False,
)
