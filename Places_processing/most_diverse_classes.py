# PlacesCNN for scene classification
#
# by Bolei Zhou
# last modified by Bolei Zhou, Dec.27, 2017 with latest pytorch and torchvision (upgrade your torchvision please if there is trn.Resize error)


import torch 
from load_data import ucfimages, create_dataset_images

from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F
import os
from PIL import Image
import pickle
import argparse
from collections import Counter


if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--split_path')
    
    opt = vars(parser.parse_args())
    
    params = {'batch_size': 32,
             'shuffle': False,
             'num_workers': 1}

    A = create_dataset_images(opt['image_path'], opt['split_path'], params, ucfimages, split='test')
    print(opt)
    
    dtype = torch.float32

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    
    # th architecture to use
    arch = 'resnet50'
    
    # load the pre-trained weights
    model_file = '%s_places365.pth.tar' % arch
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_file
        os.system('wget ' + weight_url)

    places_model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    places_model.load_state_dict(state_dict)
    places_model.eval()

    file_name = 'categories_places365.txt'
    if not os.access(file_name, os.W_OK):
        synset_url = 'https://raw.githubusercontent.com/csailvision/places365/master/categories_places365.txt'
        os.system('wget ' + synset_url)
    classes = list()
    with open(file_name) as class_file:
        for line in class_file:
            classes.append(line.strip().split(' ')[0][3:])
    
    classes = tuple(classes)
    not_common_elements = open(opt['split_path']+'/ucfTrainTestlist/uncommon_testlist01.txt','w+')
    common_class_per_activity = pickle.load(open(opt['split_path']+'/ucfTrainTestlist/common_class_per_activity.pkl', 'rb'))

    if device==torch.device('cuda'):
        places_model.cuda()
    per_class_loc = [Counter() for i in range(101)]
    for X, y, idx in A:
        X = X.to(device=device, dtype=dtype)
        y = y.to(device=device, dtype=dtype)

        scores = places_model(X)
        h_x = F.softmax(scores, 1)

        for i in range(X.shape[0]):
            sorted_prob, index = h_x[i].sort(descending=True)
            #print(index[0], idx[i].split('/'))
            if classes[int(index[0])] not in common_class_per_activity[idx[i].split('/')[-2]]: 
                not_common_elements.write(idx[i].split('/')[-2]+'/'+idx[i].split('/')[-1]+'.avi\n') 
    
    not_common_elements.close()
    
    #with open('most_diverse_1.pkl','wb+') as handle:
    #    pickle.dump(per_class_loc, handle)
