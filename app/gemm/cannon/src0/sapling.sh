#!/bin/sh
#SBATCH -c 40
#SBATCH -p gpu

export LD_LIBRARY_PATH="$PWD":$LD_LIBRARY_PATH

n=$SLURM_NNODES

echo "nodes $n"
python3 run.py sapling $n --size 2000 --taco
python3 run.py sapling $n --size 2000
