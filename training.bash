#!/bin/bash
#SBATCH --job-name=training_diffusion
#SBATCH --account=aims
#SBATCH --partition=gpu-rtx6k
#SBATCH --cpus-per-task=10
#SBATCH --mem=10G
#SBATCH --gres=gpu:1
#SBATCH --time=100:00:00
#SBATCH --output="slurm/slurm-%J-%x.out"

conda run -n data_att python train_diffusion.py --dataset="cifar" --n_samples=64 --device="cuda"
