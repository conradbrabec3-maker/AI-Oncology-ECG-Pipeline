#!/bin/bash
#SBATCH --job-name=v9_signal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=ecg_v9_%j.out
#SBATCH --error=ecg_v9_%j.err

echo "Initializing V9 Attention-Driven Pipeline..."
source /home/conrad.brabec/software_env/bin/activate
python -u /home/conrad.brabec/detector_v9/train_hybrid_v9_signal.py
