#!/bin/bash
#SBATCH --job-name=training_job       # Job name
#SBATCH --output=slurm-%j.out         # Standard output and error log
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=4             # Number of CPU cores per task
#SBATCH --mem=60G                     # Total memory
#SBATCH --gres=gpu:1                  # Request one GPU
#SBATCH --partition=iris-hi           # Partition name
#SBATCH --time=3-00:00:00             # Time limit (3 days)
#SBATCH --account=iris                # Account name
#SBATCH --exclude=iris-hp-z8          # Exclude specific node

# Load your environment (if necessary)
source ~/.bashrc
conda activate robodiff

# Run your training script
python train.py --config-dir=. --config-name=low_dim_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
