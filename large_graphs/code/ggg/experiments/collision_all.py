from ggg.experiments.collision import ex
from sacred.observers import FileStorageObserver

EXP_PATH = "collision_demo"
ex.observers.append(FileStorageObserver(EXP_PATH))
NUM_SEEDS = 10
for _seed in range(NUM_SEEDS):
    for version in ["rand", "traj"]:
        for depth, model in zip(
            [2, 2, 1], ["attention", "mlp", "ds"]
        ):  # 4 layer MLP sometimes doesn't learn either, 3/4 layer DS/Att doesn't learn=> overfit? takes longer?
            ex.run(
                config_updates=dict(
                    hpars=dict(
                        model=f"{model}-{version}",
                        max_steps=2000,
                        epoch_exp=100,
                        depth=depth,
                    ),
                    early_stopping=None,
                )
            )

from ggg.experiments.plot_collision_exp import make_plot, load_mses

model_mses, max_len, steps = load_mses(EXP_PATH)
if max_len > 0:
    make_plot(steps, model_mses)
