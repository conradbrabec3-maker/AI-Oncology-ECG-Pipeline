#!/bin/bash
#SBATCH --job-name=ecg_degrade
#SBATCH --partition=cpu2021
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --array=0-4                  # 5 jobs running in parallel (tasks 0,1,2,3,4)
#SBATCH --output=%x_%A_%a.out        # %A = array job ID, %a = task index
#SBATCH --error=%x_%A_%a.err

# Each task processes 160,007 records  (800,035 / 5 slices)
# All 5 run simultaneously → full dataset done in ~24h instead of ~96h
#
# Already-written CSVs are skipped automatically so it's safe to resubmit
# if any individual task times out or fails.

echo "🚀  Array task ${SLURM_ARRAY_TASK_ID} of 5 starting..."

# 1. Clean environment and load modules
module purge
module load python/3.12.5

# 2. Activate virtual environment
source /home/conrad.brabec/software_env/bin/activate

# 3. Create output directory
mkdir -p /home/conrad.brabec/ecg_degraded_output

# 4. Run — SLURM_ARRAY_TASK_ID is read automatically by the Python script
echo "⏳ Starting Signal Degradation Pipeline (slice ${SLURM_ARRAY_TASK_ID})..."
ECG_ZIP_PATH=/home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip \
ECG_OUT_DIR=/home/conrad.brabec/ecg_degraded_output \
python3 -u /home/conrad.brabec/SmallModel/ecg_degrade_mimic.py

echo "✅  Slice ${SLURM_ARRAY_TASK_ID} complete."
