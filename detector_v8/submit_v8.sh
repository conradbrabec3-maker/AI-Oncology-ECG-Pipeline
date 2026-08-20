#!/bin/bash
#SBATCH --job-name=v8_multimodal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=ecg_v8_%j.out
#SBATCH --error=ecg_v8_%j.err

echo "Initializing V8 Feature-Squeezed Pipeline..."
source /home/conrad.brabec/software_env/bin/activate
python -u /home/conrad.brabec/detector_v8/train_hybrid_v8.py
