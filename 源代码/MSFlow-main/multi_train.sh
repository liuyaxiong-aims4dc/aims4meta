#!/bin/bash
#SBATCH --job-name=molbert_train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=14G
#SBATCH --partition=gpu_p
#SBATCH --qos=gpu_normal
#SBATCH --gres=gpu:4
#SBATCH -C a100_80gb|h100_80gb
#SBATCH --time=24:00:00
#SBATCH --exclude=supergpu02,supergpu03,supergpu15
#SBATCH --nice=1000

source ~/.bashrc
conda activate flow

# Run your training script
torchrun --nproc-per-node=4 cfg_pretrain.py