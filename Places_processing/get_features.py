import torch 
from load_data import ucfimages, create_dataset_images

from torch.autograd import Variable as V
import torchvision.models as models
from torch.nn import functional as F
import os
from PIL import Image
import pickle
import argparse



if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path')
    parser.add_argument('--split_path')
    
    opt = vars(parser.parse_args())
    
    params = {'batch_size': 32,
             'shuffle': False,
             'num_workers': 1}

    A = create_dataset_images(opt['image_path'], opt['split_path'], params, ucfimages)
    print(opt)
    
    dtype = torch.float32

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    save_folder = 'data/places_features'
    
    if not os.path.isdir('data'):
        os.mkdir('data')
    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)
    
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

    modules=list(places_model.children())[:-1]
    places_model=torch.nn.Sequential(*modules)
    #print(places_model)

    if device==torch.device('cuda'):
        places_model.cuda()
    for X, y, idx in A:
        #print(X.shape)
        #print(y.shape)
        #print(idx)
        X = X.to(device=device, dtype=dtype)
        print(torch.mean(X))
        y = y.to(device=device, dtype=dtype)

        features = places_model(X).detach().cpu().numpy()
        print(features.shape)
        for i in range(X.shape[0]):
            final_path = idx[i].split('/')
            if not os.path.isdir(save_folder+'/'+final_path[-2]):
                os.mkdir(save_folder+'/'+final_path[-2])

            with open(save_folder+'/'+final_path[-2]+'/'+final_path[-1]+'.pkl', 'wb+') as handle:
                pickle.dump(features[i], handle)
    
