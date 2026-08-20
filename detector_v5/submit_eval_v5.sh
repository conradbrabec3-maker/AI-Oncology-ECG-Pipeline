#!/bin/bash
#SBATCH --job-name=ecg_eval_v5
#SBATCH --partition=gpu-a100      # Or gpu-v100 depending on node availability
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1              # Inference only requires 1 GPU
#SBATCH --mem=32G                 # Safe memory buffer for CSV merging
#SBATCH --time=02:00:00
#SBATCH --output=/home/conrad.brabec/detector_v5/ecg_test_v5_%j.out
#SBATCH --error=/home/conrad.brabec/detector_v5/ecg_test_v5_%j.err

source /home/conrad.brabec/software_env/bin/activate

echo "⏳ Stage 1/3: Staging ZIP file onto local NVMe drive..."
cp /home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip /tmp/local_dataset.zip
echo "✅ ZIP file safely staged locally."

echo "⏳ Stage 2/3: Executing Python AI Inference..."
# Using the -u flag ensures real-time progress bar streaming
python -u /home/conrad.brabec/detector_v5/evaluate_v5.py

echo "⏳ Stage 3/3: Cleaning up node storage..."
rm /tmp/local_dataset.zip
echo "✅ Pipeline safely terminated."
