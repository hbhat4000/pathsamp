#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH --job-name=int_4d
#SBATCH --array=1-10%1
#SBATCH --output=./varying_subintervals/output/int_4d_%a.out
#SBATCH --error=./varying_subintervals/error/int_4d_%a.err
#SBATCH --time=10:00:00
#SBATCH --mem-per-cpu=64G

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python EM_subint.py
