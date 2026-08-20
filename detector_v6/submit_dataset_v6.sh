#!/bin/bash
#SBATCH --job-name=build_data_v6
#SBATCH --partition=cpu2019      # <--- Fixed: Pointing to the massive CPU queue
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=/home/conrad.brabec/detector_v6/build_data_v6_%j.out

source /home/conrad.brabec/software_env/bin/activate
python -u /home/conrad.brabec/detector_v6/build_ml_dataset_v6.py
