#!/bin/bash
#SBATCH -A eecs
#SBATCH -p dgx2
#SBATCH -J data_cleaning
#SBATCH --nodes=2
#SBATCH --mem=400G
#SBATCH -o data_cleaning-%j.out
#SBATCH -e data_cleaning-%j.err
#SBATCH --time=7-00:00:00
echo "resource allocation done"
source /nfs/hpc/share/bhattpa/anaconda3/etc/profile.d/conda.sh
conda activate Solitaire
echo "conda environment activated"
cd yelp_analysis/
python3 data-cleaning-pytorch.py
echo "data cleaning done"
echo "starting center context pair building"
python3 pytorch-model.py
echo "starting training for embedding"
free -g
python3 train.py
