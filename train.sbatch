#!/bin/bash
#SBATCH --job-name=cnn_train
#SBATCH --output=cnn_train.out
#SBATCH --error=cnn_train.err
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

module load python/3.10
cd /home/pranavverma/CAI6108
source .venv/bin/activate
python train_cnn.py
