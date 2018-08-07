#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH -J EM_3ddd
#SBATCH -o 3ddd.qlog
#SBATCH -e 3ddd.qlog
#SBATCH --mem=64G
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python EM_3ddd.py
