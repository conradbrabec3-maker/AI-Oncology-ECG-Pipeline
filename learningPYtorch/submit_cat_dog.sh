#!/bin/bash
#SBATCH --job-name=cat_dog_practice      
#SBATCH --output=cat_dog_%j.out          
#SBATCH --error=cat_dog_%j.err           
#SBATCH --time=00:10:00                  
#SBATCH --partition=cpu2025              # <--- Explicitly using your CPU partition list
#SBATCH --nodes=1                        
#SBATCH --ntasks=1                       
#SBATCH --cpus-per-task=2                
#SBATCH --mem=4G                         

echo "🚀 Booting up Cat/Dog Practice Node on cpu2025..."

# 1. Activate your custom PyTorch environment
source /home/conrad.brabec/software_env/bin/activate

# 2. Execute the python script
python -u train_cat_dog.py

echo "✅ Job successfully terminated."
