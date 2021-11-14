
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

import torch.nn as nn
from torch.autograd import Variable
from torchvision.utils import save_image

from .models import *
from .dataset import *

def main():

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_input = [transforms.ToPILImage(),
                        transforms.Pad((0,40), fill=0, padding_mode='constant'),
                        transforms.Resize(224),
                        # transforms.CenterCrop(224),
                        # transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize]

    batch_size = 1
    data_path = "./hw4_data/TrimmedVideos/"
    videos_path = os.path.join(data_path, "video", "train")
    labels_path = os.path.join(data_path, "label", "gt_train.csv")
    # train_data = Videodataset(videos_path, labels_path, max_num_frame=-1, transform=[transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # 
    train_data = Videodataset(videos_path, labels_path, max_num_frame=-1, transform=transform_input) # 
    train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=train_data.collate_fn)

    videos_path = os.path.join(data_path, "video", "valid")
    labels_path = os.path.join(data_path, "label", "gt_valid.csv")
    valid_data = Videodataset(videos_path, labels_path, max_num_frame=-1, transform=transform_input) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=valid_data.collate_fn)

    model = Resnet50(pretrained=True).cuda()
    torch.save(model.state_dict(),os.path.join("p2", 'encode_model.pth'))

    model.eval()

    trange = tqdm(enumerate(train_data_loader),
                      total=len(train_data_loader),
                      desc='Training')
                      
    # ================== Train ================== #

    train_feature = []
    train_label = []
    train_length = []

    for step, batch in trange:
        images = batch[0]
        label = batch[1]
        length = batch[2]

        images = Variable(images).type(torch.FloatTensor).cuda()
        label = Variable(label).type(torch.LongTensor).cuda()
        # images = images.view(images.size(0)*images.size(1), images.size(2),images.size(3),images.size(4))
        with torch.no_grad():
            feature = model(images, length)

        train_feature.append(feature)
        train_label.append(label)
        train_length.append(length)

    train_feature = torch.cat(train_feature, dim=0).data.cpu().numpy()
    train_label = torch.cat(train_label, dim=0).data.cpu().numpy()
    train_length = torch.cat(train_length, dim=0).data.cpu().numpy()

    np.save("features/aug_train_feature",train_feature)
    np.save("features/aug_train_label",train_label)
    np.save("features/aug_train_length",train_length)

    print(train_feature.shape)
    print(train_label.shape)
    print(train_length.shape)


    trange = tqdm(enumerate(valid_data_loader),
                      total=len(valid_data_loader),
                      desc='Valid')
    # ================== Valid ================== #

    train_feature = []
    train_label = []
    train_length = []

    for step, batch in trange:
        images = batch[0]
        label = batch[1]
        length = batch[2]

        images = Variable(images).type(torch.FloatTensor).cuda()
        label = Variable(label).type(torch.LongTensor).cuda()
        # images = images.view(images.size(0)*images.size(1), images.size(2),images.size(3),images.size(4))
        with torch.no_grad():
            feature = model(images, length)

        train_feature.append(feature)
        train_label.append(label)
        train_length.append(length)

    train_feature = torch.cat(train_feature, dim=0).data.cpu().numpy()
    train_label = torch.cat(train_label, dim=0).data.cpu().numpy()
    train_length = torch.cat(train_length, dim=0).data.cpu().numpy()

    np.save("features/aug_valid_feature",train_feature)
    np.save("features/aug_valid_label",train_label)
    np.save("features/aug_valid_length",train_length)

    print(train_feature.shape)
    print(train_label.shape)
    print(train_length.shape)


    feature = np.load("features/aug_train_feature.npy")
    label = np.load("features/aug_train_label.npy")
    length = np.load("features/aug_train_length.npy")

    print(feature.shape)
    print(label.shape)
    print(length.shape)

    frame_feature = []
    start = 0
    print(length)
    for l in length: # [2, 4]
        # print(x[start:start+l].mean(dim=0).unsqueeze(0).view(1,-1).size())
        frame_feature.append(feature[start:start+l])
        start += l
    
    
    print(len(frame_feature))
    print(frame_feature[1].shape)
    
    np.save("features/aug_list_train_feature",frame_feature )
    #### ======= ###
    feature = np.load("features/aug_valid_feature.npy")
    label = np.load("features/aug_valid_label.npy")
    length = np.load("features/aug_valid_length.npy")

    print(feature.shape)
    print(label.shape)
    print(length.shape)

    frame_feature = []
    start = 0
    print(length)
    for l in length: # [2, 4]
        # print(x[start:start+l].mean(dim=0).unsqueeze(0).view(1,-1).size())
        frame_feature.append(feature[start:start+l])
        start += l
    
    
    print(len(frame_feature))
    print(frame_feature[1].shape)
    
    np.save("features/aug_list_valid_feature",frame_feature )


if __name__ == '__main__':
    #print(config)
    main()