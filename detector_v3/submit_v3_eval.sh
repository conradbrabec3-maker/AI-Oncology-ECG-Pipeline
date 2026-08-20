#!/bin/bash
#SBATCH --job-name=ecg_eval_v3
#SBATCH --partition=gpu-v100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=ecg_test_v3_%j.out
#SBATCH --error=ecg_test_v3_%j.err

source /home/conrad.brabec/software_env/bin/activate

echo "⏳ Stage 1/3: Staging ZIP file onto local NVMe drive..."
cp /home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip /tmp/
echo "✅ ZIP file safely staged locally."

echo "⏳ Stage 2/3: Executing Python AI Inference..."
python -u /home/conrad.brabec/detector_v3/evaluate_v3.py

echo "⏳ Stage 3/3: Cleaning up node storage..."
rm /tmp/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip
echo "✅ Pipeline safely terminated."
