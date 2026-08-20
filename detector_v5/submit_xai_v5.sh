#!/bin/bash
#SBATCH --job-name=xai_render_v5
#SBATCH --partition=cpu            # Use the CPU queue for faster access
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4          # 4 Cores is plenty for one inference pass
#SBATCH --mem=16G                  # 16 GB RAM
#SBATCH --time=00:30:00            # Should take less than 2 minutes
#SBATCH --output=/home/conrad.brabec/detector_v5/xai_v5_%j.out
#SBATCH --error=/home/conrad.brabec/detector_v5/xai_v5_%j.err

source /home/conrad.brabec/software_env/bin/activate

echo "🚀 Booting V5 XAI Pipeline..."
python -u /home/conrad.brabec/detector_v5/generate_xai_v5.py
echo "✅ XAI Render Complete."
