#!/bin/bash

#SBATCH --job-name=convert_avi # the name of the job
#SBATCH --output=outputs/slurm-%j.log # where stdout and stderr will write to
#SBATCH -A visualai
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --mem=32G                      # amount of memory needed
#SBATCH --time=5:00:00       

source activate cos529
cd /n/fs/visualai-scr/felixy/Classes/COS529/COS529_Project/I3D/utils/
python -m generate_video_jpgs /n/fs/visualai-scr/vramaswamy/COS529_project/data/UCF-101/ /n/fs/visualai-scr/Data/UCF101Images/ ucf101
