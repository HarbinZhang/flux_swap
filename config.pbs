#!/bin/sh
#PBS -S /bin/sh
#PBS -N <output_name>
#PBS -A eecs570-w17_flux:6124#PBS -l qos=flux
#PBS -l procs=4,walltime=0:5:0
#PBS -l pmem=100mb
#PBS -q flux#PBS -m abe
#PBS -j oe
#PBS -V
#PBS -M haibinzh@umich.edu
echo "I ran on:"
cat $PBS_NODEFILE
# Let PBS handle your output
cd ./
mpirun -np 4 ./<main> <16>