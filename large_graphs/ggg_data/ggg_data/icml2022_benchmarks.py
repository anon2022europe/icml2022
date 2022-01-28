"""
Functions defining Benchmarking datasets that test specific graph generation motifs.
Specifically we check

1. Trees06 => verify we can learn a small dataset (120 graphs I think) without any circles
2. Community20 => Verify we can learn a dataset with 2 different levels of connectivity (dense and sparse)
3. Star5 and Star20 => Verify we can learn a single dataset with a large degree gap
4. Roc4 and Roc6 => verify we can learn a non-chord cycle (4) and the largest we can check automatically (6)
5. Trianglegrid=> veirfy we can learn a regular  graph which consists only of chords
5. Squaregrid=> verify we can learn a regular graph which does *not* have chords
"""
from ggg_data.dense.hpars import DatasetHpars
import os
import yaml



def trees06(n_graph=5000):
    return DatasetHpars(
        dataset="Trees06",
        repeat=50,
        limit_train=n_graph,
        limit_val=n_graph,
        force_fresh=True
    )

def comm20():
    return DatasetHpars(
        dataset="CommunitySmall_20",
        force_fresh=True
    )
def star5(n_graph=5000):
    return DatasetHpars(
        dataset="nx_star",
        dataset_kwargs=dict(N_nodes=5, repeat=n_graph),
        force_fresh=True
    )
def star20(n_graph=5000):
    return DatasetHpars(
        dataset="nx_star",
        dataset_kwargs=dict(N_nodes=20, repeat=n_graph),
        force_fresh=True
    )
def roc4(n_graph=5000):
    return DatasetHpars(
        dataset="nx_roc",
        dataset_kwargs=dict(num_cliques=2,repeat=n_graph),
        force_fresh=True
    )
def roc6(n_graph=5000):
    return DatasetHpars(
        dataset="nx_roc",
        dataset_kwargs=dict(num_cliques=3,repeat=n_graph),
        force_fresh=True
    )
def triangles12(n=12,n_graph=5000):
    return DatasetHpars(
        dataset="nx_triangle",
        dataset_kwargs=dict(N_nodes=n,repeat=n_graph),
        force_fresh=True
    )
def square12(n=12,n_graph=5000):
    return DatasetHpars(
        dataset="nx_square",
        dataset_kwargs=dict(N_nodes=n,repeat=n_graph),
        force_fresh=True
    )
def triangles21(n=21,n_graph=5000):
    return DatasetHpars(
        dataset="nx_triangle",
        dataset_kwargs=dict(N_nodes=n,repeat=n_graph),
        force_fresh=True
    )
def square20(n=20,n_graph=5000):
    return DatasetHpars(
        dataset="nx_square",
        dataset_kwargs=dict(N_nodes=n,repeat=n_graph),
        force_fresh=True
    )
BENCHMARKS = {"square12": square12, "triangles12": triangles12,"square20": square20,"roc4": roc4, "roc6": roc6}
#, "triangles21": triangles21, "roc4": roc4, "roc6": roc6, "star5": star5, "star20": star20,
#              "comm20": comm20, "trees06": trees06}

def write_to_folder(dir="."):
    os.makedirs(dir,exist_ok=True)
    for n,f in BENCHMARKS.items():
        with open(os.path.join(dir,f"{n}.yaml"),"w") as fi:
            yaml.dump(f().to_dict(),fi)

if __name__=="__main__":
    write_to_folder("/tmp/")
