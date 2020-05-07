#!/bin/bash

USE_SCENE=$1
DATA_ROOT=/n/fs/visualai-scr/Data/UCF101Images

python train.py \
--video_path=${DATA_ROOT}/ \
--annotation=${DATA_ROOT}/ucfTrainTestlist/ucf101_02.json \
--dropout_keep_prob=0.5 \
--num_scales=1 \
--learning_rate=1e-3 \
--batch_size=4 \
--scene_path=/n/fs/visualai-scr/Data/UCF101PlacesFeatures/ \
--use_scene=${USE_SCENE} \
--finetune_prefixes='' \
