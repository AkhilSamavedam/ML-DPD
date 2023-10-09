#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=download_mnf
#SBATCH -p gpu 
#SBATCH --out=/gpfs/scratch/asamavedam/out/test-out.%j
#SBATCH --error=/gpfs/scratch/asamavedam/err/test-err.%j

cd /gpfs/home/asamavedam/ML-DPD

source /gpfs/scratch/asamavedam/venv/bin/activate
python data_formatting/download_mnf.py
