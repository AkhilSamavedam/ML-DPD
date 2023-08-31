#!/bin/sh
#SBATCH --job-name=train_conv_ae
#SBATCH --nodes=1
#SBATCH -p v100 
#SBATCH --time=8:00:00
#SBATCH --out=/gpfs/scratch/asamavedam/out/test-out.%j
#SBATCH --error=/gpfs/scratch/asamavedam/err/test-err.%j
cd /gpfs/scratch/asamavedam/ML-DPD/model
module load tensorflow2-gpu/2.2.0
python autoencoder.py 
