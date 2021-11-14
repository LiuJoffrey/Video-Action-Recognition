import numpy as np
import csv
import torch
import cv2
import os
import torch.utils.data as data
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
#from scipy.misc import imread, imresize
import random
import pickle
from .reader import *


class Np_featuredataset(data.Dataset):
    def __init__(self, videos_path, labels_path, max_num_frame=-1, transform=[]):
        
        
        # self.videos_path = videos_path
        # self.labels_path = labels_path
        self.videos = np.load(videos_path)
        self.labels = np.load(labels_path)

        
    def __len__(self):
        return len(self.labels)   

    def __getitem__(self, index):
        
        features = self.videos[index]
        label = self.labels[index]

        
        
        return torch.from_numpy(features), label

    def collate_fn(self, datas):
        
        frame_len = []
        new_frames = []
        new_labels = []
        for data in datas:
            frame = data[0]
            label = data[1]
            # if frame.size(0) < max_num_frame:
            # frame = self.pad_to_len(frame, self.max_num_frame)
            # new_datas.append((frame, label))
            frame_len.append(frame.size(0)) 
            new_frames.append(frame)
            new_labels.append(label)

        return torch.cat(new_frames, dim=0), torch.tensor(new_labels), torch.tensor(frame_len)

    def pad_to_len(self, arr, padded_len, padding=0):
        if padded_len < 0:
            return arr

        if padded_len>arr.size(0):
            diff = padded_len - arr.size(0)
            pad = torch.zeros((diff, arr.size(1), arr.size(2), arr.size(3)))
            arr = torch.cat((arr, pad), dim=0)
        else:
            arr = arr[:padded_len, :,:,:]


        return arr

    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

def main():
    
    data_path = "./hw4_data/TrimmedVideos/"
    videos_path = os.path.join(data_path, "video", "train")
    labels_path = os.path.join(data_path, "label", "gt_train.csv")
    
    train_data = Videodataset(videos_path, labels_path, transform=[transforms.ToTensor()])

    dataload = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=False, collate_fn=train_data.collate_fn)
    
    test = {}
    for i, batch in enumerate(dataload):
        #print(i)
        if i > 1:
            break
        print(batch[0].size(), batch[1],  batch[2], type(batch[0]))
        #print(batch)

if __name__ == '__main__':
    print("Dataset")

    main()