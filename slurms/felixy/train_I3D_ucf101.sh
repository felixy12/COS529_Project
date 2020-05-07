#!/bin/bash

for use_scene in 0 1
do
    sbatch train_I3D_ucf101.slurm ${use_scene}
done
