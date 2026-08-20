#!/bin/bash
#SBATCH --job-name=sql_query
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                  
#SBATCH --time=02:00:00            
#SBATCH --output=sql_query_%j.out 
#SBATCH --error=sql_query_%j.err   

# Clear modules and load Python
module purge
module load python/3.12.5

# Activate your environment
source /home/conrad.brabec/software_env/bin/activate

# Navigate strictly to the SQL folder
cd /home/conrad.brabec/SQL/

echo "🚀 Starting SQL Database Query on ARC compute node..."

# Run the python script from the SQL folder
python -u /home/conrad.brabec/SQL/query_patients.py

echo "✅ SQL Query Completed Successfully."
