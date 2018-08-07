#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH -J EM_1d
#SBATCH -o 1d.qlog
#SBATCH -e 1d.qlog
#SBATCH --mem=64G
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python EM_1d.py
