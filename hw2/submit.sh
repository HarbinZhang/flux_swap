#!/bin/bash
#SBATCH -N 1
#SBATCH -p RM
#SBATCH --ntasks-per-node 7
#SBATCH -t 00:05:00
# echo commands to stdout
set -x

# run OpenMP program
export OMP_NUM_THREAD=7
./mainp.out
