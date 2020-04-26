#!/bin/bash

#SBATCH --job-name=fy_run_base        # the name of the job
#SBATCH --output=outputs/slurm-%j.log # where stdout and stderr will write to
#SBATCH -A visualai
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1                   # number of GPUs your job requests
#SBATCH --mem=32G                      # amount of memory needed
#SBATCH --time=20:00:00       

source activate cos529
cd /n/fs/visualai-scr/felixy/Classes/COS529/COS529_Project/
python3 main.py --experiment baseline
