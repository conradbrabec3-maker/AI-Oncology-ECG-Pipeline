#!/bin/bash
#SBATCH --job-name=global_xai
#SBATCH --partition=gpu-a100 
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=00:30:00
#SBATCH --output=xai_global_%j.out
#SBATCH --error=xai_global_%j.err

echo "⏳ Starting Global XAI Batch..."
source /home/conrad.brabec/software_env/bin/activate
python -u /home/conrad.brabec/xai/generate_global_xai.py
