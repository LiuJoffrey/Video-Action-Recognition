
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

class Tester(object):
    def __init__(self,valid_data_loader):

        # Data loader
        
        self.valid_data_loader = valid_data_loader
        
        # Model hyper-parameters
        # self.pretrained_model = 34 # 0.43 
        self.pretrained_model = 45 # 0.43 
        self.total_step = 100

        # Path
        self.model_save_path = "p1/final"

        self.build_model()

        # Start with trained model
        if self.pretrained_model != 0:
            self.load_pretrained_model()

        self.criterion = nn.NLLLoss().cuda()

    def _run_feature(self, x, length):
        with torch.no_grad():
            feature = self.model(x, length)
        
        return feature
    
    def _run_classifier(self, x, length):
        preds = self.classifier(x, length)
        return preds

    def _run_epoch(self, training, out_file):
        
        self.model.eval()
        self.classifier.eval()
        n_total = 0
        n_correct = 0
        trange = tqdm(enumerate(self.valid_data_loader),
                    total=len(self.valid_data_loader),
                    desc='Evaluate')

        output = []
        for step, batch in trange:
            
            images = batch[0]
            #label = batch[1]
            length = batch[1]

            images = Variable(images).type(torch.FloatTensor).cuda()
            #label = Variable(label).type(torch.LongTensor).cuda()
            # images = images.view(images.size(0)*images.size(1), images.size(2),images.size(3),images.size(4))
            
            feature = self._run_feature(images, length)

            
            preds = self._run_classifier(feature, length)
            #loss = self.criterion(preds, label)

            pred = preds.data.max(1, keepdim=True)[1]
            pred = pred.view((-1, len(length)))

            out = pred.squeeze()
            #for i in range(len(out)):
            output.append(out.item())

            #n_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

            #n_total += len(label)
            #trange.set_postfix({'loss': loss.item() ,'Acc':n_correct/n_total})

        #valid_acc = n_correct/n_total
        #print("Evalute result: Loss: {}, Acc: {}".format(loss.item(), valid_acc))

        import csv
        submission = open(os.path.join(out_file, "p1_valid.txt"), "w+") # "./{}.csv".format(target_dataset_name)

        with open(os.path.join(out_file, "p1_valid.txt"), "w+") as f:
            for p in output:
                f.write(str(p)+'\n')
        

        #return valid_acc

    def test(self, out_file):
        # Start with testing model
        
        #valid_acc = self._run_epoch("valid", out_file)
        self._run_epoch("valid", out_file)
            
            

    def build_model(self):
        
        # self.model = vgg16bn(pretrained=True, batch_size=self.batch_size).cuda()
        self.model = Resnet50(pretrained=False).cuda()
        self.classifier = Classifier().cuda()
        
        # # freeze the pretrained model
        # for i, child in enumerate(self.model.children()):
        #     if i == 0:
                
        #         for param in child.parameters():
        #             # print(param)
        #             param.requires_grad = False

        # Loss and optimizer
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), 1e-3)
        
    def load_pretrained_model(self):
        self.classifier.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_model.pth'.format(self.pretrained_model))))

        self.model.load_state_dict(torch.load(os.path.join(
            self.model_save_path, 'encode_model.pth')))
        
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.optimizer.zero_grad()
