#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH --job-name=tp_1d
#SBATCH --array=11-101:10%1
#SBATCH --output=./random_timepoints/output/tp_1d_%a.out
#SBATCH --error=./random_timepoints/error/tp_1d_%a.err
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=4G

echo "My SLURM_ARRAY_TASK_ID: " $SLURM_ARRAY_TASK_ID

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python EM_random.py
