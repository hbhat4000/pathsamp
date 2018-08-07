#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH --job-name=ns_3ddd
#SBATCH --array=0,4%1
#SBATCH --output=./varying_noise/output/ns_3ddd_%a.out
#SBATCH --error=./varying_noise/error/ns_3ddd_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=64G

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python EM_noise.py
