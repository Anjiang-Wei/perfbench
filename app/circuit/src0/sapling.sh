#!/bin/sh
#SBATCH -c 40
#SBATCH -p gpu

export LD_LIBRARY_PATH="$PWD"

n=$SLURM_NNODES

echo "nodes $n"
python3 run.py sapling $n --npp 5000 --wpp 20000 --p 40 --pps 10