from .trainer import Trainer
from torch.backends import cudnn
import os
import sys
import cv2
import  torch
from .dataset import *
from .np_dataset import *
import numpy as np
from .vis_models import *
from tqdm import tqdm
from torch.autograd import Variable

model_save_path = "p2/final"
pretrained_model = 43

batch_size = 1
videos_path = "./list_valid_feature.npy"
labels_path = "./valid_label.npy"
valid_data = Np_featuredataset(videos_path, labels_path, max_num_frame=-1) #, transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
valid_data_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=valid_data.collate_fn, num_workers=8)

classifier = Classifier().cuda().eval()
classifier.load_state_dict(torch.load(os.path.join(
            model_save_path, '{}_model.pth'.format(pretrained_model))))

trange = tqdm(enumerate(valid_data_loader),
                      total=len(valid_data_loader),
                      desc='Evaluate')

all_space = []
all_label = []

for step, batch in trange:
    feature = batch[0]
    label = batch[1]
    length = batch[2]

    all_label.append(label)

    feature = Variable(feature).type(torch.FloatTensor).cuda()
    label = Variable(label).type(torch.LongTensor).cuda()

    preds = classifier(feature, length)

    all_space.append(preds.data.cpu())
    
all_space = torch.cat(all_space).data.numpy()
all_label = torch.cat(all_label).data.numpy()

print(all_space.shape)
print(all_label.shape)

from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, verbose=1, random_state=300, n_iter=1000, perplexity=20)
dim2 = tsne.fit_transform(all_space)
print(dim2.shape)

all_cate = {'0':[],'1':[],'2':[],'3':[],'4':[],'5':[],'6':[],'7':[],'8':[],'9':[], '10':[]}

for i in range(len(dim2)):
    l = str(all_label[i])
    all_cate[l].append(dim2[i])

import matplotlib.pyplot as plt
import matplotlib
plt.figure()

for i in range(len(all_cate)):
    l = str(i)
    data = np.array(all_cate[l])
    #print(data[:2,0])
    #exit()
    

    label = l
    plt.scatter(data[:, 0], data[:, 1], label=label)

plt.legend(prop={'size':5})
plt.show()