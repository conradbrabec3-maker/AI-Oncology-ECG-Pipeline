#!/bin/bash
#SBATCH --job-name=v7_multimodal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=ecg_v7_%j.out
#SBATCH --error=ecg_v7_%j.err

echo "Initializing V7 Multimodal Pipeline..."
source /home/conrad.brabec/software_env/bin/activate
python -u /home/conrad.brabec/detector_v7/train_hybrid_v7.py
