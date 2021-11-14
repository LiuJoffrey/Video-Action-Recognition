

from .trainer import Trainer
from torch.backends import cudnn
import os
import sys
import cv2
import  torch
from .dataset import *
from .np_dataset import *

def main():
    # For fast training
    cudnn.benchmark = True
    batch_size = 4
    gradient_accumulation_steps = 1
    batch_size = batch_size // gradient_accumulation_steps


    # data_path = "./hw4_data/TrimmedVideos/"
    # videos_path = os.path.join(data_path, "video", "train")
    # labels_path = os.path.join(data_path, "label", "gt_train.csv")
    # train_data = Videodataset(videos_path, labels_path, max_num_frame=-1, transform=[transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) # 
    # train_dataload = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn, num_workers=16)

    # videos_path = os.path.join(data_path, "video", "valid")
    # labels_path = os.path.join(data_path, "label", "gt_valid.csv")
    # valid_data = Videodataset(videos_path, labels_path, max_num_frame=-1, transform=[transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    # valid_dataload = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=valid_data.collate_fn, num_workers=16)
    
    
    # videos_path = "./list_train_feature.npy"
    videos_path = "./all_train_feature.npy"
    # labels_path = "./train_label.npy"
    labels_path = "./all_train_label.npy"
    train_data = Np_featuredataset(videos_path, labels_path, max_num_frame=-1) # 
    train_dataload = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=train_data.collate_fn, num_workers=8)

    videos_path = "./list_valid_feature.npy"
    labels_path = "./valid_label.npy"
    valid_data = Np_featuredataset(videos_path, labels_path, max_num_frame=-1) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    valid_dataload = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=valid_data.collate_fn, num_workers=8)
                
    trainer = Trainer(train_dataload, valid_dataload, batch_size, gradient_accumulation_steps)
        
    trainer.train()

if __name__ == '__main__':
    #print(config)
    main()