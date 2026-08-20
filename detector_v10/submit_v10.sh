#!/bin/bash
#SBATCH --job-name=Train_V10_Model
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=v10_train_%j.out
#SBATCH --error=v10_train_%j.err

echo "🚀 Booting V10 Compute Node..."

# 1. Activate your exact software environment
source /home/conrad.brabec/software_env/bin/activate
module load cuda/11.8 # Update to match your specific ARC CUDA module if different

# 2. The Node-Local Storage Hack
# Create a private temporary directory on the fast local NVMe drive of the compute node
LOCAL_TMP="/tmp/$SLURM_JOB_ID"
mkdir -p $LOCAL_TMP

echo "📦 Staging Monolithic ZIP file to lightning-fast NVMe storage..."
# Copy from slow network drive to fast local drive
cp /home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip $LOCAL_TMP/dataset.zip
echo "✅ Staging Complete."

# 3. Execute the Python Training Pipeline
# Using -u to unbuffer output so tqdm progress bars show up instantly in the .out file
echo "🧠 Launching V10 Neural Network Pipeline..."
python -u /home/conrad.brabec/detector_v10/train_v10.py \
    --train_csv /home/conrad.brabec/detector_v10/train_v10.csv \
    --zip_path $LOCAL_TMP/dataset.zip

# 4. Clean Up (CRITICAL to prevent being banned by cluster admins)
echo "🧹 Purging node-local storage..."
rm -rf $LOCAL_TMP
echo "✅ Pipeline Safely Terminated."
