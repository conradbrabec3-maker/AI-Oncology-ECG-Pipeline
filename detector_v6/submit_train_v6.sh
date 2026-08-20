#!/bin/bash
#SBATCH --job-name=train_v6
#SBATCH --partition=gpu-a100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --output=/home/conrad.brabec/detector_v6/train_v6_%j.out

source /home/conrad.brabec/software_env/bin/activate
cp /home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip /tmp/local_dataset_v6.zip
python -u /home/conrad.brabec/detector_v6/train_model_v6.py
rm /tmp/local_dataset_v6.zip
