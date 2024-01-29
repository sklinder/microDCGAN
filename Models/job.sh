#!/bin/bash
#SBATCH --partition=gpu_4
#SBATCH --ntasks=20
#SBATCH --time=14:00:00
#SBATCH --mem=72000
#SBATCH --gres=gpu:4

# Get the variables defined above (TIME_LIMIT and GPUS are not in list of SLURM enviroment variables)
TIME_LIMIT=$(grep -oP "^#SBATCH --time=\K.*" < "$0")
GPUS=$(grep -oP "^#SBATCH --gres=gpu:\K.*" < "$0")


python3 microDCGAN.py $SLURM_JOB_PARTITION $SLURM_NTASKS $TIME_LIMIT $SLURM_MEM_PER_NODE $GPUS $SLURM_JOB_ID > training.log
