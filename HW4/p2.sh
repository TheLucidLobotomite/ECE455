#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:01:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gpus-per-task=1
#SBATCH --time=00:01:00
#SBATCH --output=thread_indexing.output

cd $SLURM_SUBMIT_DIR
module load nvidia/cuda
nvcc thread_indexing.cu -o thread_indexing
./thread_indexing
