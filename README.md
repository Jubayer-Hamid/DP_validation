# Diffusion Policy

Please, visit [https://github.com/real-stanford/diffusion_policy](https://github.com/real-stanford/diffusion_policy) for set-up instructions and other details regarding the repository. 

# Changes to the original repo
I updated:
1. ```diffusion_policy/workspace/train_diffusion_unet_lowdim_workspace.py``` where I am saving the data offline in the 'run' function. 
2. ```diffusion_policy/policy/diffusion_unet_lowdim_policy.py``` where I am returning per-sample-loss, the predicted action and extra multiple predictions per state in the 'compute_loss' function. 
3. ```submit_training.sh``` to submit training jobs.


# Overview of steps for training on a particular task

First, follow all the set-up instructions as in the original repository. Now, for a task, say Push-T, you need to (1) get the data and config files (2) set off the training run. The procedure for doing each of these is detailed below. 


# Grabbing data and configs for a task:
Under the repo root, create data subdirectory:
```console
[diffusion_policy]$ mkdir data && cd data
```

Download the corresponding zip file from [https://diffusion-policy.cs.columbia.edu/data/training/](https://diffusion-policy.cs.columbia.edu/data/training/)
```console
[data]$ wget https://diffusion-policy.cs.columbia.edu/data/training/pusht.zip
```

Extract training data:
```console
[data]$ unzip pusht.zip && rm -f pusht.zip && cd ..
```

Grab config file for the corresponding experiment:
```console
[diffusion_policy]$ wget -O image_pusht_diffusion_policy_cnn.yaml https://diffusion-policy.cs.columbia.edu/data/experiments/image/pusht/diffusion_policy_cnn/config.yaml
```

### Running for a single seed on interactive node 

You can run the experiments on an interactive node (using tmux):

Activate conda environment and login to [wandb](https://wandb.ai) (if you haven't already).
```console
[diffusion_policy]$ conda activate robodiff
(robodiff)[diffusion_policy]$ wandb login
```

Launch training with seed 42 on GPU 0.
```console
(robodiff)[diffusion_policy]$ python train.py --config-dir=. --config-name=low_dim_pusht_diffusion_policy_cnn.yaml training.seed=42 training.device=cuda:0 hydra.run.dir='data/outputs/${now:%Y.%m.%d}/${now:%H.%M.%S}_${name}_${task_name}'
```

This will create a directory in format `data/outputs/yyyy.mm.dd/hh.mm.ss_<method_name>_<task_name>` where configs, logs and checkpoints are written to. The policy will be evaluated every 50 epochs with the success rate logged as `test/mean_score` on wandb, as well as videos for some rollouts.

### Running for a single seed via slurm

Alternatively, you can submit the job. I added ```submit_training.sh```. 
NOTE: The only change you need to make is replace the ```--condif_name``` to be the correct config file (that you fetched above) for your task. For example, if you are running on BlockPushing, use ```--config-name=low_dim_block_pushing_diffusion_policy_cnn.yaml```. 

Once you have updated ```submit_training.sh```, run ```sbatch submit_training.sh```. 