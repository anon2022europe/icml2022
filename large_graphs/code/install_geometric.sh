source ~/.zshrc
conda activate graph
export CUDA=cu111
export TORCH=1.7.1
pip install -U torch numpy torchvision
pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
pip install torch-geometric
pip install -U --ignore-installed PyYAML
pip install -e .
pip install -U tensorboard
pip install -U networkx seaborn matplotlib
