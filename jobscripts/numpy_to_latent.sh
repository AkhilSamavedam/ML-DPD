#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=latent_encoding
#SBATCH -p gpu
#SBATCH --out=/gpfs/scratch/asamavedam/out/test-out.%j
#SBATCH --error=/gpfs/scratch/asamavedam/err/test-err.%j

cd /gpfs/home/asamavedam/ML-DPD

module load tensorflow2-gpu/2.2.0

python data_formatting/create_latent.py
