#!/bin/bash
#SBATCH --job-name=eval_v6
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/home/conrad.brabec/detector_v6/eval_v6_%j.out

source /home/conrad.brabec/software_env/bin/activate
cp /home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip /tmp/local_dataset_v6.zip
python -u /home/conrad.brabec/detector_v6/evaluate_model_v6.py
rm /tmp/local_dataset_v6.zip
