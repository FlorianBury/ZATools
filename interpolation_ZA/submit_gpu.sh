#!/bin/bash

# Submission script for cp3-gpu

#SBATCH --job-name=interpolation
#SBATCH --time=0-05:00:00 # days-hh:mm:ss
#SBATCH --output=gpu_res.txt
#
#SBATCH --mem=60G
#SBATCH --partition=cp3-gpu
#SBATCH --qos=cp3-gpu
#
#SBATCH --mail-user=florian.bury@uclouvain.be
#SBATCH --mail-type=ALL


# Module load #
module load root/6.12.04-sl7_gcc73 
module load boost/1.66.0_sl7_gcc73 
module load gcc/gcc-7.3.0-sl7_amd64 
module load python/python36_sl7_gcc73  
module load cmake/cmake-3.9.6 
module load slurm/slurm_utils

python3 interpolate.py -s $1 -d $1 
echo "Job launched"

