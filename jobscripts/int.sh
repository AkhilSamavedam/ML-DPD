module load slurm
module load openfoam
module load tensorflow2-gpu/2.2.0
salloc -N 1 -p gpu 
ssh -X $SLURM_NODELIST
