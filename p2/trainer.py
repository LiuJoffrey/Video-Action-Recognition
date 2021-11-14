
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



class Trainer(object):
    def __init__(self, train_data_loader, valid_data_loader, batch_size, grad_accumulate_steps):

        # Data loader
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self.batch_size = batch_size
        self.grad_accumulate_steps = grad_accumulate_steps

        # Model hyper-parameters
        self.pretrained_model = 0
        self.total_step = 100

        # Path
        self.model_save_path = "p2/checkpoint_att_2"

        self.build_model()

        # Start with trained model
        if self.pretrained_model != 0:
            self.load_pretrained_model()

        self.criterion = nn.NLLLoss().cuda()
        # self.criterion = nn.CrossEntropyLoss().cuda()

        self.train_loss = []
        self.train_accu = []
        self.valid_loss = []
        self.valid_accu = []

    def _run_feature(self, x, length):
        with torch.no_grad():
            feature = self.model(x, length)
        return feature
    
    def _run_classifier(self, x, length):
        preds = self.classifier(x, length)
        return preds

    def _run_epoch(self, training):
        if training == "train":
            # self.model.eval()
            self.classifier.train()

            trange = tqdm(enumerate(self.train_data_loader),
                      total=len(self.train_data_loader),
                      desc='Training')
            
            n_total = 0
            n_correct = 0
            for step, batch in trange:

                # ================== Train ================== #
                feature = batch[0]
                label = batch[1]
                length = batch[2]

                if len(label) == 1:
                    continue

                feature = Variable(feature).type(torch.FloatTensor).cuda()
                label = Variable(label).type(torch.LongTensor).cuda()
                # images = images.view(images.size(0)*images.size(1), images.size(2),images.size(3),images.size(4))
                
                # feature = self._run_feature(images, length)
                preds = self._run_classifier(feature, length)

                loss = self.criterion(preds, label)

                if step % self.grad_accumulate_steps == 0:
                    # TODO: call zero gradient here
                    self.reset_grad()
                
                loss.backward()

                if (step + 1) % self.grad_accumulate_steps == 0:
                    self.optimizer.step()

                # self.reset_grad()
                # loss.backward()
                # self.optimizer.step()

                pred = preds.data.max(1, keepdim=True)[1]
                pred = pred.view((-1, len(label)))
                n_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

                n_total += len(label)
            
                trange.set_postfix({'loss': loss.item() ,'Acc':n_correct/n_total})
            self.train_loss.append(loss.item())
            self.train_accu.append(n_correct/n_total)
            return n_correct/n_total

        else:
            # self.model.eval()
            self.classifier.eval()
            n_total = 0
            n_correct = 0
            trange = tqdm(enumerate(self.valid_data_loader),
                      total=len(self.valid_data_loader),
                      desc='Evaluate')

            for step, batch in trange:
                feature = batch[0]
                label = batch[1]
                length = batch[2]

                feature = Variable(feature).type(torch.FloatTensor).cuda()
                label = Variable(label).type(torch.LongTensor).cuda()
                # images = images.view(images.size(0)*images.size(1), images.size(2),images.size(3),images.size(4))
                
                # feature = self._run_feature(images, length)
                preds = self._run_classifier(feature, length)
                loss = self.criterion(preds, label)

                pred = preds.data.max(1, keepdim=True)[1]
                pred = pred.view((-1, len(label)))
                n_correct += pred.eq(label.data.view_as(pred)).cpu().sum().item()

                n_total += len(label)
                trange.set_postfix({'loss': loss.item() ,'Acc':n_correct/n_total})

            valid_acc = n_correct/n_total
            print("Evalute result: Loss: {}, Acc: {}".format(loss.item(), valid_acc))
            self.valid_loss.append(loss.item())
            self.valid_accu.append(n_correct/n_total)
            return valid_acc

    def train(self):
        # Start with trained model
        if self.pretrained_model != 0:
            start = self.pretrained_model + 1
        else:
            start = 0

        best_acc = 0
        for epoch in range(start, self.total_step):
            print("Epoch: ", epoch)
            train_acc = self._run_epoch("train")
            valid_acc = self._run_epoch("valid")
            
            if valid_acc>best_acc:
                print("New record")
                best_acc = valid_acc
                torch.save(self.classifier.state_dict(),
                            os.path.join(self.model_save_path, '{}_model.pth'.format(epoch + 1)))
        print("Best Acc:", best_acc)  

        train_loss = np.array(self.train_loss)
        train_accu = np.array(self.train_accu)
        valid_loss = np.array(self.valid_loss)
        valid_accu = np.array(self.valid_accu)

        # np.save("p2/train_loss", train_loss)
        # np.save("p2/train_accu", train_accu)
        # np.save("p2/valid_loss", valid_loss)
        # np.save("p2/valid_accu", valid_accu)

    def build_model(self):
        
        # self.model = vgg16bn(pretrained=True, batch_size=self.batch_size).cuda()
        # self.model = Resnet50(pretrained=True).cuda() 
        # self.classifier = Classifier().cuda()
        self.classifier = ATT_Classifier().cuda()


        
        # # freeze the pretrained model
        # for i, child in enumerate(self.model.children()):
        #     if i == 0:
                
        #         for param in child.parameters():
        #             # print(param)
        #             param.requires_grad = False

        # Loss and optimizer
        # self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.classifier.parameters()), 1e-3)
        self.optimizer = torch.optim.Adam(self.classifier.parameters(), 1e-5)
        
    def load_pretrained_model(self):

        
        self.classifier.load_state_dict(torch.load(os.path.join(
            self.model_save_path, '{}_model.pth'.format(self.pretrained_model))))
        
        print('loaded trained models (step: {})..!'.format(self.pretrained_model))

    def reset_grad(self):
        self.optimizer.zero_grad()
