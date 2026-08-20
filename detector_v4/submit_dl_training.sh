#!/bin/bash
#SBATCH --job-name=ecg_cnn_train
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# 1. Clean environment and load modules
module purge
module load python/3.12.5
module load cuda

# 2. Activate virtual environment
source /home/conrad.brabec/software_env/bin/activate

# 3. Stage the monolithic ZIP file to the node's local NVMe drive
echo "⏳ Stage 1/3: Staging monolithic ZIP file onto local NVMe drive..."
# Path points to your home directory where the zip actually lives
cp /home/conrad.brabec/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0.zip /tmp/local_dataset.zip
echo "✅ ZIP file safely staged locally."

# 4. Run the PyTorch Deep Learning Python Pipeline
echo "⏳ Stage 2/3: Executing Python Model Training..."
# Path points to your new detector_v4 folder
python3 -u /home/conrad.brabec/detector_v4/train_ecg_model.py
echo "✅ Training finished."

# 5. Clean up the node storage
echo "⏳ Stage 3/3: Cleaning up node storage..."
rm -f /tmp/local_dataset.zip
rm -rf /tmp/ecg_micro_buffer
echo "✅ Pipeline safely terminated."
