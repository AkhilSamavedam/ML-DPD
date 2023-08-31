#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=04:00:00
#SBATCH --job-name=csv_writing
#SBATCH -p short-40core

cd /gpfs/scratch/asamavedam/ML-DPD/data_formatting

module load mvapich2/gcc/64/2.2rc1
module load lammps/gpu/11Aug17

source /gpfs/scratch/asamavedam/venv/bin/activate
python format_ds.py

