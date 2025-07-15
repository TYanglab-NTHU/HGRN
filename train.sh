#!/bin/bash
#SBATCH --job-name=OMGNN   ## job name
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1            ##number of GPU
#SBATCH --cpus-per-task=4 
#SBATCH --account=MST111483
#SBATCH --partition=gp1d

source ~/.bashrc
conda activate eveline

python train_performance.py 