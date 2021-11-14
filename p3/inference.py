

from .models import *
from .reader import readShortVideo
from .reader import getVideoList
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from os import listdir
import sys
import os
import pandas as pd
import numpy as np
import pickle

import torchvision
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import skimage.io
import skimage
import pickle

def norm(image):

    transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Pad((0,40), fill=0, padding_mode='constant'),
                            transforms.Resize(224),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
    return transform(image)

def main():
    arg = sys.argv

    video_path = arg[1]
    out_file = arg[2]

    model = Resnet50(pretrained=False).cuda()
    model.load_state_dict(torch.load(os.path.join(
            "p3/final", "encode_model.pth")))
    model.eval()
    classifier = Classifier().cuda()
    classifier.load_state_dict(torch.load(os.path.join(
            #"p3/final", "model_0.574513.pkt")))
            #"p3/final", "model_0.579884.pkt")))
            "p3/final", "model_0.594316.pkt")))
    classifier.eval()

    # out_file = "./output"
    # video_path = "hw4_data/FullLengthVideos/videos/valid/"
    # label_path = "hw4_data/FullLengthVideos/labels/valid/"

    category_list = sorted(os.listdir(video_path))

    valid_video_feature = []
    for category in category_list:
        print("Category: ", category)
        out_txt = os.path.join(out_file, category+".txt")
        category_frames = []
        frame_output = []
        img_file_list = sorted(os.listdir(os.path.join(video_path, category)))


        valid_lengths = len(img_file_list)

        for img in img_file_list:
            image_rgb = skimage.io.imread(os.path.join(video_path, category,img))
            image_nor = norm(image_rgb).view((1,3,224,224)).cuda()
            feature = model(image_nor).data.view(2048).unsqueeze(0)
            
            category_frames.append(feature)
        
        category_frames = torch.cat(category_frames, dim=0)
        category_frames = category_frames.unsqueeze(0)
        
        pred = classifier(category_frames, [category_frames.size(1)])

        prediction = torch.argmax(torch.squeeze(pred.cpu()),1).data.numpy()

        # print(prediction.shape)

        for i in range(len(prediction)):
            if i == 0:
                continue
            elif i > 0 and i<(len(prediction)-1):
                if prediction[i-1] == prediction[i+1]:
                    prediction[i] = prediction[i-1]

        with open(os.path.join(out_txt), "w+") as f:
            for p in prediction:
                f.write(str(p)+'\n')

if __name__ == '__main__':
    #print(config)
    main()