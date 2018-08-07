#!/bin/bash
#SBATCH -p appliedmath.q
#SBATCH -J data_3dl
#SBATCH --array=0-7
#SBATCH -o data_3dl.qlog
#SBATCH -e data_3dl.qlog
#SBATCH --mem=64G
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hbhat/lib64:/home/hbhat/lib
/home/hbhat/anaconda3/bin/python createdata_noise.py
