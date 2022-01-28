# Readme for the reviewers

This repository holds the code required to replicate the experiments detailed in our ICML2022 submission. The code for small and large graphs is slightly different so we include the two versions in two different reposities.

In order to run the codes yourself you need to

1. create and acivate a conda environment
2. navigate to the respective directory
3. Install `ggg_utils`,`ggg_data` and `ggg` in this order, by first installing the dependencies see the requirements.txt and the setup.py, note that you'll need to install `pytorch_geometric` with the appropriate method based on the version

After this you should be able to run our experiments based on the configurations in the "runs" directory in each subdirectory.
The respective runs are
- in `large_graphs/runs`:
  - `ggg`,`ggrs` for the community100 reported in the table and together with `gggnecomm100` in the comparison figure
  - `ggg200` and the `SUPP_*`  directory for the additional plots in the appendix on communty 200 and the molecule/protein data
- in `small_graphs/runs`:
  - chordal9,comm20,qm9, for the ggg runs reported in the table.

All of the above can be run with `python XXX/code/ggg/experiments/GGG.py with ${path_to_config.json}` since we use [sacred](https://pypi.org/project/sacred/) for all our experiments and the `config.json` contains all necessary hyperparameters. For the large graphs we also provide pretrained [pytorch lightning checkpoints](https://pytorchlightning.ai/) which can be loaded and evaluated with `ggg/experiments/eval_gg.py` (except ogbg-molpcba which was too large for the github repo).


The baselines (condgen,graphrnn,scorematching,molgan) can be found in the `XXX/externals/benchmarks` section of each directory, they are simply the original author codes slightly tweaked to our dataset.
For the classical baselines, you can run `python XXX/code/ggg/experiments/eval_classical.py with hpars.model=BA/GnP hpars.dataset=${dataset_name}`.

To keep the repository size manageable, we have deleted the `externals/benchmarks/GraphScoreMatching/dataset/` and `code/externals/benchmarks/GraphRNN/dataset`, from the `large_graphs` direcotry, it can be recreated by symlinking the same path in `small_datasets`.
