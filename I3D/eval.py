# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-03-01

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
from datetime import datetime

import torch.nn as nn

from transforms.spatial_transforms import Compose, Normalize, RandomHorizontalFlip, MultiScaleRandomCrop, ToTensor, CenterCrop
from transforms.temporal_transforms import TemporalRandomCrop, TemporalCenterCrop
from transforms.target_transforms import ClassLabel

from utils.utils import *
from utils.evaluation_metrics import *
import utils.mean_values
import factory.data_factory as data_factory
import factory.model_factory as model_factory
from config import parse_opts

import pdb
import pickle
####################################################################
####################################################################
# Configuration and logging

config = parse_opts()
#config = prepare_output_dirs(config)
config = init_cropping_scales(config)
config = set_lr_scheduling_policy(config)

config.image_mean = utils.mean_values.get_mean(config.norm_value, config.dataset)
config.image_std = utils.mean_values.get_std(config.norm_value)

print_config(config)

# TensorboardX summary writer
if not config.no_tensorboard:
    from tensorboardX import SummaryWriter
    writer = SummaryWriter(log_dir=config.log_dir)
else:
    writer = None

####################################################################
####################################################################
# Initialize model

device = torch.device(config.device)
#torch.backends.cudnn.enabled = False

# Returns the network instance (I3D, 3D-ResNet etc.)
# Note: this also restores the weights and optionally replaces final layer
model, parameters = model_factory.get_model(config)

print(config.eval_checkpoint)
assert(os.path.isfile(config.eval_checkpoint))
    
model.load_state_dict(torch.load(config.eval_checkpoint)['state_dict'])
print('#'*60)
print('#'*60)

####################################################################
####################################################################
# Setup of data transformations

if config.no_dataset_mean and config.no_dataset_std:
    # Just zero-center and scale to unit std
    print('Data normalization: no dataset mean, no dataset std')
    norm_method = Normalize([0, 0, 0], [1, 1, 1])
elif not config.no_dataset_mean and config.no_dataset_std:
    # Subtract dataset mean and scale to unit std
    print('Data normalization: use dataset mean, no dataset std')
    norm_method = Normalize(config.image_mean, [1, 1, 1])
else:
    # Subtract dataset mean and scale to dataset std
    print('Data normalization: use dataset mean, use dataset std')
    norm_method = Normalize(config.image_mean, config.image_std)

train_transforms = {
    'spatial':  Compose([MultiScaleRandomCrop(config.scales, config.spatial_size),
                         RandomHorizontalFlip(),
                         ToTensor(config.norm_value),
                         norm_method]),
    'temporal': TemporalRandomCrop(config.sample_duration),
    'target':   ClassLabel()
}

# print('WARNING: setting train transforms for dataset statistics')
# train_transforms = {
#     'spatial':  Compose([ToTensor(1.0)]),
#     'temporal': TemporalRandomCrop(64),
#     'target':   ClassLabel()
# }

validation_transforms = {
    'spatial':  Compose([CenterCrop(config.spatial_size),
                         ToTensor(config.norm_value),
                         norm_method]),
    'temporal': TemporalCenterCrop(config.sample_duration),
    'target':   ClassLabel()
}

####################################################################
####################################################################
# Setup of data pipeline

data_loaders = data_factory.get_data_loaders(config, train_transforms, validation_transforms, validation_only=True)
print('#'*60)

####################################################################
####################################################################

model.eval()

# Epoch statistics

full_path = config.eval_checkpoint.split('/')[:-2]
out_dir = ''
for d in full_path:
    out_dir+=(d+'/')
print(out_dir)

steps_in_epoch = int(np.ceil(len(data_loaders['validation'].dataset)/config.batch_size))
losses = np.zeros(steps_in_epoch, np.float32)
accuracies = np.zeros(steps_in_epoch, np.float32)

epoch_start_time = time.time()
scores_all = []
targets_all = []
preds_all = []
for step, (clips, scene_feats, targets) in enumerate(data_loaders['validation']):

    start_time = time.time()

    # Move inputs to GPU memory
    clips   = clips.to(device)
    scene_feats = scene_feats.to(device)
    targets = targets.to(device)
    #if config.model == 'i3d':
    #    targets = torch.unsqueeze(targets, -1)
    
    # Feed-forward through the network
    if model.use_scene:
        logits = model.forward(clips.view((-1, 3, 64, 224, 224)), scene_feats.view(-1, 2048))
    else:
        logits = model.forward(clips.view((-1, 3, 64, 224, 224)), scene_feats)

    
    #print(logits.shape)
    _, preds = torch.max(logits.view(-1, 101), 1)
    targets_all.append(targets.detach().cpu().numpy())
    preds_all.append(preds.detach().cpu().numpy())
    scores_all.append(logits.view(-1, 101).detach().cpu().numpy())
    #print(logits.view(-1, 101).shape)

scores_all= np.concatenate(scores_all)
targets_all = np.concatenate(targets_all)
preds_all = np.concatenate(preds_all)

print('Accuracy = ',np.sum(np.where(preds_all.reshape(-1)==targets_all.reshape(-1), 1, 0))/preds_all.reshape(-1).shape[0])

"""
print('Computing per class accuracies... ')
per_class_acc = per_class_accuracy(targets_all, preds_all)
print('Computing per class average precision... ')
per_class_AP = per_class_average_precision(targets_all, scores_all) 


with open(out_dir+'per_class_acc.pkl', 'wb+') as handle:
    pickle.dump(per_class_acc, handle)
with open(out_dir+'per_class_AP.pkl', 'wb+') as handle:
    pickle.dump(per_class_AP, handle)
"""
