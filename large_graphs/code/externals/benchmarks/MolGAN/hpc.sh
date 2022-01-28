#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=8192  # memory in Mb
#SBATCH -t 00:15:00  # time requested in hour:minute:second

#SBATCH --partition=gpu

# Prevents `plt.show()` for attempting connecting.
unset DISPLAY XAUTHORITY 

module load cuda

cd ~/MolGAN_master
source activate GraphGAN
echo "... Job beginning"
"$@"
echo "... Job Ended"
