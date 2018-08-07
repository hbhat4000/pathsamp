#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH -J EM_2d
#SBATCH -o 2d.qlog
#SBATCH -e 2d.qlog
#SBATCH --mem=64G
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python EM_2d.py
