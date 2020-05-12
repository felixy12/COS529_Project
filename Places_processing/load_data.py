from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image
import torchvision.transforms as T
import pickle
import cv2
import pdb

class ucfimages(Dataset):
    def __init__(self, list_IDs, labels, transform=T.ToTensor(), **kwargs):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        count = int(len(os.listdir(ID))/2)
        
        img = Image.open(ID+'/image_'+str(count).zfill(5)+'.jpg')
        #print(transform)
        X = self.transform(img)
        
        y = self.labels[ID]
        return X,y,ID

        
def create_dataset_images(path_images, path_split, params, dataset, split='', kwargs={}):
    
    classes = open(path_split+'/ucfTrainTestlist/classInd.txt', 'r')

    classes_dict = {}
    
    A = classes.readlines()
    for i, a in enumerate(A):
        a = a.strip().split()
        classes_dict[a[1]] = i
    
    list_ids = []
    attr = {}
    
    if split=='':
        split='train'
        split_file = open(path_split+'/ucfTrainTestlist/filtered_'+split+'list01.txt', 'r').readlines()
        for a in split_file:
            a = a.strip().split()
            idx = a[0].split('.')[0]
            list_ids.append(path_images+'/'+idx)
            attr[path_images+'/'+idx] = int(a[1])-1
    
    split='test'
    split_file = open(path_split+'/ucfTrainTestlist/filtered_'+split+'list01.txt', 'r').readlines()
    for a in split_file:
        a= a.strip().split('.')[0]
        list_ids.append(path_images+'/'+a)
        attr[path_images+'/'+a] = classes_dict[a.split('/')[0]] 
            

    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        normalize
    ])
   
    #print(transform)    
    dset = dataset(list_ids, attr, transform, **kwargs)
    loader = DataLoader(dset, **params)

    return loader

class ucf_places_features(Dataset):
    def __init__(self, list_IDs, labels, **kwargs):
        self.labels = labels
        self.list_IDs = list_IDs

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        
        X = pickle.load(open(ID+'.pkl', 'rb')).squeeze()
        
        y = self.labels[ID]
        return X,y

        
def create_dataset_places_features(path_images, path_split, params, dataset, number=0, split='train', kwargs={}):
    
    classes = open(path_split+'/ucfTrainTestlist/classInd.txt', 'r')
    split_file = open(path_split+'/ucfTrainTestlist/filtered_'+split+'list01.txt', 'r').readlines()

    classes_dict = {}
    
    A = classes.readlines()
    for i, a in enumerate(A):
        a = a.strip().split()
        classes_dict[a[1]] = i
    
    list_ids = []
    attr = {}
    
    if split=='train':
        for a in split_file:
            a = a.strip().split()
            idx = a[0].split('.')[0]
            list_ids.append(path_images+'/'+idx)
            attr[path_images+'/'+idx] = int(a[1])-1
    else:
        for a in split_file:
            a= a.strip().split('.')[0]
            list_ids.append(path_images+'/'+a)
            attr[path_images+'/'+a] = classes_dict[a.split('/')[0]] 
   
    dset = dataset(list_ids, attr, **kwargs)
    loader = DataLoader(dset, **params)

    return loader


if __name__=="__main__":

    params = {'batch_size': 10,
             'shuffle': True,
             'num_workers': 1}

    A = create_dataset_places_features('data/places_features', '/n/fs/visualai-scr/vramaswamy/COS529_project/data/UCF-101', params, ucf_places_features)

    for X, y in A:
        print(X.shape)
        print(y.shape)
        
