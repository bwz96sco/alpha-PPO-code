#!/bin/bash
#SBATCH --job-name=N-ex
#SBTAHC --partition=debug  
#SBATCH -N 1 
#SBATCH -n 8
#SBTACH --ntasks-per-node=8
#SBATCH --gres=gpu:1                      
#SBATCH --output=./%j.out
#SBATCH --error=./%j.err
#SBATCH -t 24:00:00                            
#SBATCH --mail-type=end
#SBATCH --mail-user=luopeng69131@sjtu.edu.cn

cd $SLURM_SUBMIT_DIR
echo '20250619'
lspci | grep -i vga
lspci | grep -i nvidia
nvidia-smi -L
bash run.sh