from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import torch

from transforms.spatial_transforms import Normalize
from torch.utils.data import DataLoader

from datasets.ucf101 import UCF101

##########################################################################################
##########################################################################################

def get_training_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['ucf101']

    if config.dataset == 'ucf101':
        training_data = UCF101(
            config.video_path,
            config.annotation_path,
            'training',
            use_scene=config.use_scene,
            scene_path=config.scene_path,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            subsample_rate=config.subsample_rate)
    return training_data


##########################################################################################
##########################################################################################

def get_validation_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['ucf101']

    # Disable evaluation
    if config.no_eval:
        return None

    if config.dataset == 'ucf101':
        validation_data = UCF101(
            config.video_path,
            config.annotation_path,
            'validation',
            use_scene=config.use_scene,
            scene_path=config.scene_path,
            n_samples_for_each_video=config.num_val_samples,
            spatial_transform=spatial_transform,
            temporal_transform=temporal_transform,
            target_transform=target_transform,
            sample_duration=config.sample_duration,
            subsample_rate=config.subsample_rate)
    return validation_data

##########################################################################################
##########################################################################################

def get_test_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['ucf101']
    assert config.test_subset in ['val', 'test']

    if config.test_subset == 'val':
        subset = 'validation'
    elif config.test_subset == 'test':
        subset = 'testing'

    elif config.dataset == 'ucf101':
        test_data = UCF101(
            config.video_path,
            config.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)
    return test_data


##########################################################################################
##########################################################################################

def get_normalization_method(config):
    if config.no_mean_norm and not config.std_norm:
        return Normalize([0, 0, 0], [1, 1, 1])
    elif not config.std_norm:
        return Normalize(config.mean, [1, 1, 1])
    else:
        return Normalize(config.mean, config.std)

##########################################################################################
##########################################################################################

def get_data_loaders(config, train_transforms, validation_transforms=None):

    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

    data_loaders = dict()

    # Define the data pipeline
    dataset_train = get_training_set(
        config, train_transforms['spatial'],
        train_transforms['temporal'], train_transforms['target'])

    data_loaders['train'] = DataLoader(
        dataset_train, config.batch_size, shuffle=True,
        num_workers=config.num_workers, pin_memory=True)

    print('Found {} training examples'.format(len(dataset_train)))

    if not config.no_eval and validation_transforms:

        dataset_validation = get_validation_set(
            config, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])

        print('Found {} validation examples'.format(len(dataset_validation)))

        data_loaders['validation'] = DataLoader(
            dataset_validation, config.batch_size, shuffle=True,
            num_workers=config.num_workers, pin_memory=True)

    return data_loaders
