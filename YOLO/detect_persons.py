#!/usr/bin/env python
# coding: utf-8

# In[2]:


from models import *
from utils import *
import os, sys, time, datetime, random
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import pdb


# In[3]:


config_path='config/yolov3.cfg'
weights_path='weights/yolov3.weights'
class_path='data/coco.names'

img_dir = '/n/fs/visualai-scr/Data/UCF101Images/'
UCF_classes = os.listdir(img_dir)
UCF_classes.remove('ucfTrainTestlist')
img_size=320
conf_thres=0.3
nms_thres=0.4

# Load model and weights
model = Darknet(config_path, img_size=img_size)
model.load_darknet_weights(weights_path)
model.cuda()
model.eval()
classes = utils.load_classes(class_path)
Tensor = torch.cuda.FloatTensor
print('Done loading in model.')


# In[4]:


def chunk_list(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def load_image(filepath):
    return Image.open(filepath)

def detect_image(img):
    imw = 320
    imh = 240
    img_transforms=transforms.Compose([
         transforms.Pad((max(int((imh-imw)/2),0), 
              max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
              max(int((imw-imh)/2),0)), (128,128,128)),
         transforms.ToTensor(),
         ])
    # convert image to Tensor
    image_tensor = img_transforms(img).float()
    image_tensor = image_tensor.unsqueeze_(0)
    input_img = Variable(image_tensor.type(Tensor))
    # run inference on the model and get detections
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 
                        conf_thres, nms_thres)
    return detections[0]

def detect_images(image_list):
    imw = 320
    imh = 240
    img_transforms=transforms.Compose([
    transforms.Pad((max(int((imh-imw)/2),0), 
         max(int((imw-imh)/2),0), max(int((imh-imw)/2),0),
         max(int((imw-imh)/2),0)), (128,128,128)),
    transforms.ToTensor(),
    ])
    image_tensors = []
    for img in image_list:
        image_tensor = img_transforms(img).float()
        image_tensor = image_tensor.unsqueeze_(0)
        image_tensors.append(image_tensor)
    image_tensors = torch.cat(image_tensors, 0)
    input_img = Variable(image_tensors.type(Tensor))
    with torch.no_grad():
        detections = model(input_img)
        detections = utils.non_max_suppression(detections, 
                        conf_thres, nms_thres)
    return detections

def grey_out_detection(img, detections):
    img_np = np.array(img)
    img_mean = np.mean(img, axis=(0,1))
    mask = np.zeros(img_np.shape[0:2])
    detect_person = 0
    if detections is not None:
        for detection in detections:
            if detection[-1] == 0.0:
                detect_person = 1
                bbox = detection[:4].cpu().numpy().astype(np.uint16)
                bbox[1], bbox[3] = bbox[1] - 40, bbox[3] - 40
                mask[bbox[1]:bbox[3],bbox[0]:bbox[2]] = 1
    
    img_np[mask == 0,:] = img_mean
    return Image.fromarray(img_np), detect_person

def grey_out_detections(image_list, detections):
    greyed_image_list = []
    n_frame_w_person = 0
    for i, image in enumerate(image_list):
        detection = detections[i]
        greyed_image, detect_person = grey_out_detection(image, detection)
        greyed_image_list.append(greyed_image)
        n_frame_w_person += detect_person
    return greyed_image_list, n_frame_w_person

def grey_out_video(frame_dir):
    out_frame_dir = frame_dir.replace('UCF101Images', 'UCF101ImagesGreyed')
    os.makedirs(out_frame_dir, exist_ok=True)
    frame_names = os.listdir(frame_dir)
    if len(os.listdir(out_frame_dir)) == len(frame_names):
        print('Video {} has already been processed. Skipping.'.format(frame_dir.split('/')[-1]))
        return
    
    prev_time = time.time()
    frame_names_chunked = list(chunk_list(frame_names, 16))
    total_frames_w_person = 0
    for chunk in frame_names_chunked:
        image_list = []
        for frame_name in chunk:
            frame_path = os.path.join(frame_dir, frame_name)
            image_list.append(load_image(frame_path))
        detections = detect_images(image_list)
        greyed_image_list, n_frame_w_person = grey_out_detections(image_list, detections)
        total_frames_w_person += n_frame_w_person
        for i, frame_name in enumerate(chunk):
            out_frame_path = os.path.join(out_frame_dir, frame_name)
            greyed_image_list[i].save(out_frame_path)
    inference_time = datetime.timedelta(seconds=time.time() - prev_time)
    print ('Done processing Video {}. Time Taken: {}'.format(frame_dir.split('/')[-1], inference_time), flush=True)
    return float(total_frames_w_person)/len(frame_names)


# In[7]:


print('Creating list of videos to process.')
video_list = []
for class_name in UCF_classes:
    video_dir   = os.path.join(img_dir, class_name)
    video_names = os.listdir(video_dir)
    for video_name in video_names:
        frame_dir   = os.path.join(video_dir, video_name)
        video_list.append(frame_dir)
print('Done. Processing {} videos'.format(len(video_list)), flush=True)


# In[10]:


person_fraction_fname = 'person_fraction.txt'
for frame_dir in video_list:
    person_fraction_file = open(person_fraction_fname, 'a+')
    try:
        total_frames_w_person = grey_out_video(frame_dir)
    except:
        print('Failed to process Video {}. Skipping.'.format(frame_dir.split('/')[-1]))
        total_frames_w_person = 0.0
    if total_frames_w_person is not None: 
        person_fraction_file.write('{}\t{}\n'.format(frame_dir.split('/')[-1], total_frames_w_person))
    person_fraction_file.close()

