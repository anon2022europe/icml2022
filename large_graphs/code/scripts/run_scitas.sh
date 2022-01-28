#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --time=00:05:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=out_run_torch_cpu%j-%N.out
 
module purge
#module load gcc mvapich2 openblas python/3.6.5 
VENV_NAME=torch_venv_3_6_5

if [ ! -d $VENV_NAME ]; then
python -m venv --system-site-packages $VENV_NAME
source "$VENV_NAME/bin/activate"
pip3 install torch==1.1.0 torchvision==0.3.0 -f https://download.pytorch.org/whl/cu90/torch_stable.html
pip3 install -r requirements.txt
pip3 install -e .
else
source "$VENV_NAME/bin/activate"
fi
"${@}"
