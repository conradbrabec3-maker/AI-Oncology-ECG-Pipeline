#!/bin/bash
#SBATCH --job-name=ecg_test_v2
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=ecg_test_v2_%j.out
#SBATCH --error=ecg_test_v2_%j.err

echo "⏳ Staging ZIP file onto local NVMe drive..."
cp /home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip /tmp/
echo "✅ ZIP file safely staged locally."

echo "⏳ Activating Python Environment..."
source /home/conrad.brabec/software_env/bin/activate

echo "⏳ Executing Final AI Exam..."
python -u /home/conrad.brabec/detector_v2/evaluate_v2.py

echo "⏳ Cleaning up node storage..."
rm /tmp/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip
echo "✅ Pipeline safely terminated."
