#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH -J EM_3dl
#SBATCH -o 3dl.qlog
#SBATCH -e 3dl.qlog
#SBATCH --mem=64G
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python EM_3dl.py
