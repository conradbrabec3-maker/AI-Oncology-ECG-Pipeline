#!/bin/bash
#SBATCH --job-name=generate_xai
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8GB
#SBATCH --time=00:10:00
#SBATCH --output=xai_job_%j.out
#SBATCH --error=xai_job_%j.err

echo "⏳ Starting XAI Heatmap Generation..."

# 1. Activate your exact software environment
source /home/conrad.brabec/software_env/bin/activate

# 2. Execute the python script (using -u to stream prints live to the .out file)
python -u /home/conrad.brabec/xai/generate_xai.py

echo "✅ XAI Job Complete!"
