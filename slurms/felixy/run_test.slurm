#!/bin/bash

#SBATCH --job-name=fy_run_base        # the name of the job
#SBATCH --output=outputs/slurm-%j.log # where stdout and stderr will write to
#SBATCH -A visualai
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1                   # number of GPUs your job requests
#SBATCH --mem=32G                      # amount of memory needed
#SBATCH --time=20:00:00       

EXPNAME=$1
INPUTTYPE=$2

log=/n/fs/visualai-scr/felixy/Classes/COS529/COS529_Project/slurms/felixy/reports/${INPUTTYPE}.out
exec &> >(tee -a "$log")

source activate cos529
cd /n/fs/visualai-scr/felixy/Classes/COS529/COS529_Project/ResNet2D/
python3 main.py --experiment ${EXPNAME} --input_type ${INPUTTYPE}
