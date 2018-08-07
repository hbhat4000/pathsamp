#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH --job-name=int_2d
#SBATCH --array=9
#SBATCH --output=./varying_subintervals/output/int_2d_%a.out
#SBATCH --error=./varying_subintervals/error/int_2d_%a.err
#SBATCH --time=04:00:00
#SBATCH --mem-per-cpu=64G

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python EM_subint.py
