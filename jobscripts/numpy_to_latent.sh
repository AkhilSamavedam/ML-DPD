#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=latent_encoding
#SBATCH -p gpu 

cd /gpfs/scratch/asamavedam/ML-DPD/data_formatting

module load tensorflow2-gpu/2.2.0

python create_latent.py

