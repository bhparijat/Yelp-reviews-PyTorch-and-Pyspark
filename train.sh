#!/bin/bash
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH -J pytorch_training
#SBATCH --nodes=2
#SBATCH --mem=200G
#SBATCH -o pytorch-train-%j.out
#SBATCH -e pytorch-train-%j.err
#SBATCH --time=7-00:00:00
echo "resource allocation done"
source /nfs/hpc/share/bhattpa/anaconda3/etc/profile.d/conda.sh
conda activate Solitaire
echo "conda environment activated"
echo "starting training for sentiment analysis"
free -g
python3 torch_train.py
