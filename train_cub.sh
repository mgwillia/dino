#!/bin/bash

#SBATCH --job-name=train_cub
#SBATCH --output=logs/train_cub.out.%j
#SBATCH --error=logs/train_cub.out.%j
#SBATCH --time=24:00:00
#SBATCH --qos=high
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G

module load cuda/11.0.3

srun bash -c "hostname;"
srun bash -c "python -m torch.distributed.launch --nproc_per_node=4 main_dino.py --arch vit_small --data_path /fs/nexus-scratch/mgwillia/CUB_200_2011/train --output_dir /fs/nexus-scratch/mgwillia/dino/outputs/cub --epochs 5 --warmup_epochs 1;"
