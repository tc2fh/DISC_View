#!/bin/bash
#SBATCH -A bii_dsc_community
#SBATCH -p bii-gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --ntasks-per-node=1
#SBATCH --time=72:00:00

module load anaconda
conda activate capstone

python train_unet.py --images /path/to/images --masks /path/to/masks
