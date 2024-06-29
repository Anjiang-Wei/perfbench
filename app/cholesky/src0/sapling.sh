#!/bin/sh
#SBATCH -c 40
#SBATCH -p gpu

export LD_LIBRARY_PATH="$PWD"

n=$SLURM_NNODES

echo "nodes $n"
# -n $size -p $num_partition
python3 run.py sapling $n 10 4