#!/bin/bash

############################################################################################################

##SBATCH --partition=iris
#SBATCH --partition=iris-hi

#SBATCH --chdir /iris/u/yuejliu/research/dp_validation
#SBATCH --output slurm/train-dp-%j.out 										
#SBATCH --job-name=train

#SBATCH --time=48:00:00 											# Max job length is 5 days
#SBATCH --cpus-per-task=20 											# Request 8 CPUs for this task
#SBATCH --mem-per-cpu=4G 											# Request 8GB of memory

#SBATCH --nodes=1 													# Only use one node (machine)
#SBATCH --gres=gpu:1 												# Request one GPU
#SBATCH --account=iris

#SBATCH --exclude=iris1,iris2,iris3,iris4,iris5,iris6,iris-hp-z8,iris-hgx-1

date

############################################################################################################

source /sailhome/yuejliu/.bashrc
conda activate bid

cd /iris/u/yuejliu/research/dp_validation

export PYTHONUNBUFFERED=1

# Print the list of nodes allocated to the job
echo "Nodes allocated for this job: $SLURM_NODELIST"

############################################################################################################

# CFGNAME=low_dim_pusht_diffusion_policy_cnn.yaml

CFGNAME=low_dim_square_mh_diffusion_policy_cnn_config.yaml
# CFGNAME=low_dim_lift_mh_diffusion_policy_cnn_config.yaml

python train.py --config-dir=. --config-name=${CFGNAME} training.seed=42 training.device=cuda:0 hydra.run.dir='outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'

date