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


class Videodataset(data.Dataset):
    def __init__(self, videos_path, labels_path, max_num_frame=-1, transform=[]):
        
        
        self.videos_path = videos_path
        self.labels_path = labels_path
        self.transform = transform
        self.max_num_frame = max_num_frame
        # self.video_category = os.listdir(self.videos_path)
        
        self.all_cate_labels = []
        with open(self.labels_path, 'r') as f:
            rows = csv.DictReader(f)
            for row in rows:
                self.all_cate_labels.append({'Video_path':self.videos_path, 'Video_name':row['Video_name'], 
                                                'Video_category': row['Video_category']})#'label':int(row['Action_labels'])

        # print(self.all_cate_labels[0])
        # print(self.all_cate_labels[1])
        # print(self.all_cate_labels[2])
        # print(self.all_cate_labels[3])
        # print(self.all_cate_labels[4])
        # exit()
        
    def __len__(self):
        return len(self.all_cate_labels)   

    def __getitem__(self, index):
        
        frames = readShortVideo(self.all_cate_labels[index]['Video_path'], self.all_cate_labels[index]['Video_category'], 
                                    self.all_cate_labels[index]['Video_name'], downsample_factor=12)

        

        #label = self.all_cate_labels[index]['label']

        tensor_frames = []
        for img in frames:
            
            for t in self.transform:
                img = t(img)
            
            tensor_frames.append(img.unsqueeze(0))
        
        tensor_frames = torch.cat(tensor_frames, dim=0)
        
        return tensor_frames

        # if img.shape[0] != self.img_size:
        #     print("resize")
        #     img = cv2.resize(img, (self.img_size, self.img_size))

    def random_blur(self, img):
        if random.random() > 0.5:
            img = cv2.GaussianBlur(img,(5,5),0)
        return img

    def random_noise(self, img):
        if random.random()>0.5:
            h, w, c = img.shape
            mean = 0
            var = 3
            sigma = var**0.5
            gaussian = np.random.normal(mean, sigma, (h, w))
            noisy_image = np.zeros(img.shape, np.float32)
            noisy_image[:, :, 0] = img[:, :, 0] + gaussian
            noisy_image[:, :, 1] = img[:, :, 1] + gaussian
            noisy_image[:, :, 2] = img[:, :, 2] + gaussian
            cv2.normalize(noisy_image, noisy_image, 0, 255, cv2.NORM_MINMAX, dtype=-1)
            noisy_image = noisy_image.astype(np.uint8)
            return noisy_image

        if random.random()>0.5:
            output = np.zeros(img.shape,np.uint8)
            thres = 1 - 0.05 
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    rdn = random.random()
                    if rdn < 0.05:
                        output[i][j] = 0
                    elif rdn > 0.99:
                        output[i][j] = 255
                    else:
                        output[i][j] = img[i][j]
            return output

        return img
    
    def random_bright(self, img):
        
        if random.random()>0.5:
            value = int(random.uniform(-10,10))
            hsv = self.RGB2HSV(img)
            #hsv = hsv.astype('float32')
            h, s, v = cv2.split(hsv)
            v = v.astype('float32')
            
            v = v+value
            v = np.clip(v, 0, 255).astype(hsv.dtype)
            final_hsv = cv2.merge((h, s, v))
            #final_hsv = final_hsv.astype('uint8')
            img = self.HSV2RGB(final_hsv)
        
        return img

    def collate_fn(self, datas):
        
        frame_len = []
        new_frames = []
        new_labels = []
        for data in datas:
            frame = data #[0]
            #label = data[1]
            # if frame.size(0) < max_num_frame:
            # frame = self.pad_to_len(frame, self.max_num_frame)
            # new_datas.append((frame, label))
            frame_len.append(frame.size(0)) 
            new_frames.append(frame)
            #new_labels.append(label)

        return torch.cat(new_frames, dim=0), torch.tensor(frame_len) #torch.tensor(new_labels)

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
    def flip(self, img):
        return cv2.flip(img, 1)
    def BGR2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    def RGB2BGR(self,img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
    def RGB2HSV(self,img):
        return cv2.cvtColor(img,cv2.COLOR_RGB2HSV)
    def HSV2RGB(self,img):
        return cv2.cvtColor(img,cv2.COLOR_HSV2RGB)

def main():
    
    data_path = "./hw4_data/TrimmedVideos/"
    videos_path = os.path.join(data_path, "video", "train")
    labels_path = os.path.join(data_path, "label", "gt_train.csv")
    
    train_data = Videodataset(videos_path, labels_path, transform=[transforms.ToTensor()])

    dataload = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False, collate_fn=train_data.collate_fn)
    
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
