#!/bin/bash
#SBATCH --job-name=v9_evaluate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=eval_v9_%j.out
#SBATCH --error=eval_v9_%j.err

echo "Initializing V9 Attention-Driven Inference Engine..."
source /home/conrad.brabec/software_env/bin/activate
python -u /home/conrad.brabec/detector_v9/eval_hybrid_v9_signal.py
