

from .tester import Tester
from torch.backends import cudnn
import os
import sys
import cv2
import  torch
#from .dataset import *
from .test_dataset import *
from .np_dataset import *

def main():
    # For fast training

    arg = sys.argv

    videos_path = arg[1]
    labels_path = arg[2]
    out_file = arg[3]


    cudnn.benchmark = True
    batch_size = 1
    gradient_accumulation_steps = 1
    batch_size = batch_size // gradient_accumulation_steps
    
    # data_path = "./hw4_data/TrimmedVideos/"
    # videos_path = os.path.join(data_path, "video", "valid")
    # labels_path = os.path.join(data_path, "label", "gt_valid.csv")
    valid_data = Videodataset(videos_path, labels_path, max_num_frame=-1, transform=[transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))]) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    valid_dataload = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=valid_data.collate_fn, num_workers=16)
                
    
    tester = Tester(valid_dataload)
        
    tester.test(out_file)

if __name__ == '__main__':
    #print(config)
    main()