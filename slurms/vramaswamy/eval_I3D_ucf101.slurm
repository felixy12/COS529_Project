#!/bin/bash

#SBATCH --job-name=I3D_ucf101        # the name of the job
#SBATCH --output=outputs/slurm-%j.log # where stdout and stderr will write to
#SBATCH -A visualai
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gres=gpu:1                   # number of GPUs your job requests
#SBATCH --exclude=node905
#SBATCH --mem=32G                      # amount of memory needed
#SBATCH --time=48:00:00       
#SBATCH --mail-user=vr23@cs.princeton.edu

USE_SCENE=$1
USE_BBOX=$2
DATA_ROOT=$3
MODEL_CHECKPOINT=$4
cd /n/fs/visualai-scr/vramaswamy/COS529_Project/I3D/
bash eval_I3D_ucf101.sh ${USE_SCENE} ${USE_BBOX} ${DATA_ROOT} ${MODEL_CHECKPOINT}
