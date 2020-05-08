#!/bin/bash

USE_SCENE=$1
USE_BBOX=$2
DATA_ROOT=$3

python train.py \
--video_path=${DATA_ROOT}/ \
--annotation=${DATA_ROOT}/ucfTrainTestlist/filtered_ucf101_01.json \
--dropout_keep_prob=0.5 \
--num_scales=1 \
--learning_rate=1e-3 \
--batch_size=4 \
--scene_path=/n/fs/visualai-scr/Data/UCF101PlacesFeatures/ \
--use_scene=${USE_SCENE} \
--use_bb=${USE_BBOX} \
