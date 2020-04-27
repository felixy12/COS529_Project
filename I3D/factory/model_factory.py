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
# Date Created: 2018-08-15

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import torch
import torch.nn as nn

from models.i3d import InceptionI3D



def get_model(config):

    assert config.model in ['i3d']
    print('Initializing {} model (num_classes={})...'.format(config.model, config.num_classes))

    if config.model == 'i3d':

        from models.i3d import get_fine_tuning_parameters

        model = InceptionI3D(
            num_classes=config.num_classes,
            spatial_squeeze=True,
            final_endpoint='logits',
            in_channels=3,
            dropout_keep_prob=config.dropout_keep_prob
        )

    if 'cuda' in config.device:

        print('Moving model to CUDA device...')
        # Move model to the GPU
        model = model.cuda()

        if config.checkpoint_path:

            print('Loading pretrained model {}'.format(config.checkpoint_path))
            assert os.path.isfile(config.checkpoint_path)

            checkpoint = torch.load(config.checkpoint_path)
            pretrained_weights = checkpoint
            model.load_state_dict(pretrained_weights)

            # Setup finetuning layer for different number of classes
            # Note: the DataParallel adds 'module' dict to complicate things...
            print('Replacing model logits with {} output classes.'.format(config.finetune_num_classes))

            model.replace_logits(config.finetune_num_classes)

            # Setup which layers to train
            finetune_criterion = config.finetune_prefixes 
            parameters_to_train = get_fine_tuning_parameters(model, finetune_criterion)

            return model, parameters_to_train
    else:
        raise ValueError('CPU training not supported.')

    return model, model.parameters()
