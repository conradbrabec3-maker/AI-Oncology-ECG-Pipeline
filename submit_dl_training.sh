#!/bin/bash
#SBATCH --job-name=ecg_train
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=dl_train_%j.out
#SBATCH --error=dl_train_%j.err

echo "⏳ Staging ZIP file onto local NVMe drive..."
cp /home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip /tmp/
echo "✅ ZIP file safely staged locally."

echo "⏳ Activating Python Environment..."
source /home/conrad.brabec/software_env/bin/activate

echo "⏳ Executing Phase 3 AI Training..."
python -u /home/conrad.brabec/train_ecg_model.py

echo "⏳ Cleaning up node storage..."
rm /tmp/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip
echo "✅ Pipeline safely terminated."
