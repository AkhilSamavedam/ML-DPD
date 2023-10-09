#!/bin/sh
#SBATCH --job-name=performance
#SBATCH --nodes=1
#SBATCH -p v100 
#SBATCH --time=8:00:00
#SBATCH --out=/gpfs/scratch/asamavedam/out/test-out.%j
#SBATCH --error=/gpfs/scratch/asamavedam/err/test-err.%j
cd /gpfs/home/asamavedam/ML-DPD
module load tensorflow2-gpu/2.2.0
python model/performance.py 
