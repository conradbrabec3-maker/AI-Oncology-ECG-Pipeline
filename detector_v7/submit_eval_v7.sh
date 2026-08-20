#!/bin/bash
#SBATCH --job-name=eval_v7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=eval_v7_%j.out
#SBATCH --error=eval_v7_%j.err

echo "Initializing V7 Multimodal Evaluation..."
source /home/conrad.brabec/software_env/bin/activate
python -u /home/conrad.brabec/detector_v7/eval_hybrid_v7.py
