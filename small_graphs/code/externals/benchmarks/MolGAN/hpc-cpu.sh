#!/bin/sh
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 4      # cores requested
#SBATCH --mem=8192  # memory in Mb
#SBATCH -t 24:00:00  # time requested in hour:minute:second

# Prevents `plt.show()` for attempting connecting.
unset DISPLAY XAUTHORITY

cd ~/MolGAN_master
source activate GraphGAN
echo "... Job beginning"
"$@"
echo "... Job Ended"
