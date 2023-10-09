#!/bin/bash

#SBATCH --nodes=1
#SBATCH --time=04:00:00
#SBATCH --job-name=np_writing
#SBATCH -p gpu 
#SBATCH --out=/gpfs/scratch/asamavedam/out/test-out.%j
#SBATCH --error=/gpfs/scratch/asamavedam/err/test-err.%j

cd /gpfs/home/asamavedam/ML-DPD

module load mvapich2/gcc/64/2.2rc1
module load lammps/gpu/11Aug17

source /gpfs/scratch/asamavedam/venv/bin/activate
python data_formatting/mnf_dump.py

