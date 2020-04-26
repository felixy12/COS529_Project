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


class MyDataset_latefusion(Dataset):
    def __init__(self, list_IDs, labels, transform=T.ToTensor()):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        cap = cv2.VideoCapture(ID)

        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((2, frameHeight, frameWidth, 3), np.dtype('uint8'))
        ret, buf[0] = cap.read()
        buf[0] = cv2.cvtColor(buf[0], cv2.COLOR_BGR2RGB)
        
        
        X = torch.zeros((2,3,224,224))
        t = self.transform(Image.fromarray(buf[0]))
        
        w, h = t.shape[1:]
        if w>224:
            a1 = np.random.randint(w-224)
            a2 = np.random.randint(h-224)
        else:
            a1=0
            a2=0
        X[0] = t[:, a1:a1+224, a2:a2+224]
        cap.set(cv2.CAP_PROP_FRAME_COUNT, frameCount-2)
        ret, buf[1] = cap.read()
        buf[1] = cv2.cvtColor(buf[1], cv2.COLOR_BGR2RGB)
        t = self.transform(Image.fromarray(buf[1]))
        X[1] = t[:, a1:a1+224, a2:a2+224]

        cap.release()
         
        #X = torch.cat(X)
        y = self.labels[ID]

        #print(X.shape)
        
        return X,y

class MyDataset_randomframe(Dataset):
    def __init__(self, list_IDs, labels, transform=T.ToTensor()):
        self.labels = labels
        self.list_IDs = list_IDs
        self.transform = transform

    def __len__(self):
        return len(self.list_IDs)

    def __getitem__(self, index):
        ID = self.list_IDs[index]
        cap = cv2.VideoCapture(ID)

        frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        buf = np.empty((frameHeight, frameWidth, 3), np.dtype('uint8'))
        
        f = np.random.randint(frameCount-1)
        cap.set(cv2.CAP_PROP_FRAME_COUNT, f)
        ret, buf = cap.read()
        buf = cv2.cvtColor(buf[0], cv2.COLOR_BGR2RGB)
        X = torch.zeros((3,224,224))
        X = self.transform(Image.fromarray(buf))

        y = self.labels[ID]

        return X,y
        
def create_dataset_actual(path, params, dataset, number=0, split='train'):
    
    classes = open(path+'/ucfTrainTestlist/classInd.txt', 'r')
    split_file = open(path+'/ucfTrainTestlist/'+split+'list01.txt', 'r').readlines()

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
            list_ids.append(path+'/'+a[0])
            attr[path+'/'+a[0]] = int(a[1])-1
    else:
        for a in split_file:
            a= a.strip()
            list_ids.append(path+'/'+a)
            attr[path+'/'+a] = classes_dict[a.split('/')[0]] 
            

    
    normalize = T.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    if split=='train':
        transform = T.Compose([
            T.Resize(1024),
            T.ToTensor(),
            normalize
        ])

    else:
        transform = T.Compose([
            T.Resize(1024),
            T.CenterCrop(224),
            T.ToTensor(),
            normalize
        ])
       
    #print(transform)    
    dset = dataset(list_ids, attr, transform)
    loader = DataLoader(dset, **params)

    return loader


if __name__=="__main__":

    params = {'batch_size': 10,
             'shuffle': True,
             'num_workers': 1}

    #A = create_dataset_actual('data/UCF-101', params, MyDataset)

    #for X, y in A:
    #    print(X.shape)
    #    print(y.shape)
