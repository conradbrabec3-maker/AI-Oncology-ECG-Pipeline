#!/bin/bash
#SBATCH --job-name=sql_cohort_gen
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/home/conrad.brabec/SQL/sql_cohort_%j.out
#SBATCH --error=/home/conrad.brabec/SQL/sql_cohort_%j.err

# Clear modules and load Python
module purge
module load python/3.12.5

# Activate your environment
source /home/conrad.brabec/software_env/bin/activate

# Navigate strictly to the SQL folder
cd /home/conrad.brabec/SQL/

echo "🚀 Starting Precise SQL Database Query on ARC compute node..."

# Run the NEW python script unbuffered
python -u /home/conrad.brabec/SQL/generate_control_cohort.py

echo "✅ SQL Query Completed Successfully."
