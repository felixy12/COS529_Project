#!/bin/bash

for use_scene in 0 1
do
    for use_bbox in 0 1 
    do
        if [ ${use_bbox} -eq 0 ]
        then
            data_root=/n/fs/visualai-scr/Data/UCF101Images/
        else
            data_root=/n/fs/visualai-scr/Data/UCF101ImagesGreyed/
        fi        
        model_checkpoint=/n/fs/visualai-scr/vramaswamy/COS529_Project/I3D/output/20200509_1737_UseScene${use_scene}_UseBB${use_bbox}
        sbatch eval_I3D_ucf101.slurm ${use_scene} ${use_bbox} ${data_root} ${model_checkpoint} 
    done
done
