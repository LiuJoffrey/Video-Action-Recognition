
from .trainer import Trainer
from torch.backends import cudnn
import os
import sys
import cv2
import  torch
import os
import time
import torch
import datetime
from tqdm import tqdm
import numpy as np
import skimage.io
import skimage
import pickle

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from .models import *
# from .dataset import *

def norm(image):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_input = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Pad((0,40), fill=0, padding_mode='constant'),
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            normalize
                        ])
    return transform_input(image)

def main():

    model = Resnet50(pretrained=True).cuda().eval()
    torch.save(model.state_dict(),os.path.join("p3", 'encode_model.pth'))

    print("Encode Train Feature")

    video_path = "hw4_data/FullLengthVideos/videos/train/"
    category_list = sorted(os.listdir(video_path))

    train_video_feature = []
    for category in category_list:
        print("Category: ", category)
        category_frames = []
        img_file_list = sorted(os.listdir(os.path.join(video_path, category)))
        for img in img_file_list:
            image_rgb = skimage.io.imread(os.path.join(video_path, category,img))
            image_nor = norm(image_rgb).view((1,3,224,224)).cuda()
            feature = model(image_nor).data.cpu().view(2048).unsqueeze(0)
            
            category_frames.append(feature)
        
        category_frames = torch.cat(category_frames, dim=0)
        train_video_feature.append(category_frames)
    
    print("Encode Valid Feature")
    video_path = "hw4_data/FullLengthVideos/videos/valid/"
    category_list = sorted(os.listdir(video_path))

    valid_video_feature = []
    for category in category_list:
        print("Category: ", category)
        category_frames = []
        img_file_list = sorted(os.listdir(os.path.join(video_path, category)))
        for img in img_file_list:
            image_rgb = skimage.io.imread(os.path.join(video_path, category,img))
            image_nor = norm(image_rgb).view((1,3,224,224)).cuda()
            feature = model(image_nor).data.cpu().view(2048).unsqueeze(0)
            
            category_frames.append(feature)
        
        category_frames = torch.cat(category_frames, dim=0)
        valid_video_feature.append(category_frames)
    
    
    with open("p3/train_FullLength_features_resnet.pkl", "wb") as f:
        pickle.dump(train_video_feature, f)
    with open("p3/valid_FullLength_features_resnet.pkl", "wb") as f:
        pickle.dump(valid_video_feature, f)

    with open("p3/train_FullLength_features_resnet.pkl", "rb") as f:
        train_all_video_frame = pickle.load(f)
    with open("p3/valid_FullLength_features_resnet.pkl", "rb") as f:
        valid_all_video_frame = pickle.load(f)
    
    label_path = "hw4_data/FullLengthVideos/labels/train/"
    category_txt_list = sorted(os.listdir(label_path))
    train_category_labels = []
    for txt in category_txt_list:
        file_path = os.path.join(label_path,txt)
        with open(file_path,"r") as f:
            label = [int(w.strip()) for w in f.readlines()]
            train_category_labels.append(label)

    label_path = "hw4_data/FullLengthVideos/labels/valid/"
    category_txt_list = sorted(os.listdir(label_path))
    valid_category_labels = []
    for txt in category_txt_list:
        file_path = os.path.join(label_path,txt)
        with open(file_path,"r") as f:
            label = [int(w.strip()) for w in f.readlines()]
            valid_category_labels.append(label)
    
    cutting_steps = 350
    overlap_steps = 50
    train_cut_features = []
    train_cut_labels = []
    train_cut_lengths = []

    for category_frames, category_labels in zip(train_all_video_frame,train_category_labels):
        
        features, labels, lengths = cut_frames(category_frames,category_labels, 
                                            size = cutting_steps, overlap = overlap_steps)
        train_cut_features += features
        train_cut_labels += labels
        train_cut_lengths += lengths
        print("one category done")


    valid_lengths = [len(s) for s in valid_all_video_frame]
    valid_lengths_2 = [len(s) for s in valid_category_labels]


    with open("p3/train_cut_features_350_50_resnet.pkl" , "wb") as f:
        pickle.dump(train_cut_features,f)
    with open("p3/train_cut_labels_350_50_resnet.pkl", "wb") as f:
        pickle.dump(train_cut_labels,f)
    with open("p3/train_cut_lengths_350_50_resnet.pkl", "wb") as f:
        pickle.dump(train_cut_lengths,f)

    with open("p3/valid_cut_features_no_cut_resnet.pkl", "wb") as f:
        pickle.dump(valid_all_video_frame,f)
    with open("p3/valid_cut_labels_no_cut_resnet.pkl", "wb") as f:
        pickle.dump(valid_category_labels,f)
    with open("p3/valid_cut_lengths_no_cut_resnet.pkl", "wb") as f:
        pickle.dump(valid_lengths,f)


    

def cut_frames(features_per_category, labels_per_category, size = 350, overlap = 50):
    a = torch.split(features_per_category, size-overlap)
    b = torch.split(torch.Tensor(labels_per_category), size-overlap)

    cut_features = []
    cut_labels = []
    for i in range(len(a)):
        if i==0:
            cut_features.append(a[i])
            cut_labels.append(b[i])
        else:
            cut_features.append(torch.cat((a[i-1][-overlap:],a[i])))
            cut_labels.append(torch.cat((b[i-1][-overlap:],b[i])))
    
    lengths = [len(f) for f in cut_labels]

    return cut_features, cut_labels, lengths
    
    

if __name__ == '__main__':
    #print(config)
    main()