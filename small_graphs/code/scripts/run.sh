#!/bin/env bash
#SBATCH -N 1      # nodes requested
#SBATCH -n 1      # tasks requested
#SBATCH -c 12      # cores requested
#SBATCH --mem=100000  # memory in Mb
#SBATCH -t 15:00:00  # time requested in hour:minute:second
#SBATCH --exclude=anon-compute-08,anon-compute-09,anon-compute-10,anon-compute-11,anon-compute-12,anon-compute-13,anon-compute-14,anon-compute-15,anon-compute-16,anon-compute-17,anon-compute-18

unset DISPLAY XAUTHORITY 
echo "Running ${@}"
"${@}"
