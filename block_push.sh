#!/bin/bash
#SBATCH --job-name=diffusion_train          # Job name
#SBATCH --output=logs/output_%j.log         # Output log file (%j expands to job ID)
#SBATCH --error=logs/error_%j.log           # Error log file
#SBATCH --account=iris                      # Account name
#SBATCH --partition=iris-hi-interactive     # Partition (queue)
#SBATCH --exclude=iris-hp-z8,iris-hgx-1,iris4,iris1,iris2,iris3  # Exclude certain nodes
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=60G                           # Memory allocation
#SBATCH --time=3-0:00:00                    # Max time (3 days)
#SBATCH --gres=gpu:1                        # Number of GPUs

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate robodiff

# Navigate to the working directory
cd /iris/u/jubayer/diffusion_policy

# Run the python command
python train.py --config-dir=. --config-name=low_dim_block_pushing_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
