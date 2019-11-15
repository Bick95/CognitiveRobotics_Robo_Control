#!/bin/bash
#SBATCH --time=05:00:00
#SBATCH --job-name=frnk6-5
#SBATCH --mail-type=BEGIN,END,FAILL
#SBATCH --output=/dev/null
#SBATCH --mem=10GB
#SBATCH --nodes=1
#SBATCH --ntasks=12
module load Python/3.6.4-foss-2019a
python main.py -p ParameterSettings/params_6.json
