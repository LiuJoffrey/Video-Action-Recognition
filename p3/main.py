from .models import *
from .reader import readShortVideo
from .reader import getVideoList
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from os import listdir
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


def sort_pad(input_feature, input_lengths, input_labels):
    perm_index = np.argsort(input_lengths)[::-1]
    input_feature =  [input_feature[i] for i in perm_index]
    input_labels =  [input_labels[i] for i in perm_index]
    input_lengths = sorted(input_lengths, reverse=True)
    input_feature = nn.utils.rnn.pad_sequence(input_feature, batch_first=True)
    return input_feature, input_labels, input_lengths

def main():
    # load features 
    with open("p3/train_cut_features_350_50_resnet.pkl", "rb") as f:
        train_cut_features = pickle.load(f)
    with open("p3/train_cut_labels_350_50_resnet.pkl", "rb") as f:
        train_cut_labels = pickle.load(f)
    with open("p3/train_cut_lengths_350_50_resnet.pkl", "rb") as f:
        train_cut_lengths = pickle.load(f)
        
    with open("p3/valid_cut_features_no_cut_resnet.pkl", "rb") as f:
        valid_features_all = pickle.load(f)
    with open("p3/valid_cut_labels_no_cut_resnet.pkl", "rb") as f:
        valid_labels_all = pickle.load(f)
    with open("p3/valid_cut_lengths_no_cut_resnet.pkl", "rb") as f:
        valid_lengths_all = pickle.load(f)
    
    feature_size = 2048
    model = Classifier().cuda()
    loss_function = Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    batch_size = 32

    max_accuracy = 0
    
    training_loss_list = []
    validation_acc_list = []

    for epoch in range(20):
        print("Epoch: ", epoch+1)
        CE_loss = 0.0

        model.train()

        total_length = len(train_cut_features)
        perm_index = np.random.permutation(total_length)
        train_X_sfl = [train_cut_features[i] for i in perm_index]
        train_y_sfl = [train_cut_labels[i] for i in perm_index]
        train_lengths_sfl = np.array(train_cut_lengths)[perm_index]

        for idx in range(0, total_length, batch_size):
            if idx+batch_size > total_length:
                break

            optimizer.zero_grad()

            input_X = train_X_sfl[idx:idx+batch_size]
            input_y = train_y_sfl[idx:idx+batch_size]
            input_lengths = train_lengths_sfl[idx:idx+batch_size]

            input_X, input_y, input_lengths = sort_pad(input_X, input_lengths, input_y)

            output = model(input_X.cuda(), input_lengths)

            loss = loss_function(output, input_y,input_lengths)

            loss.backward()
            optimizer.step()
            CE_loss += loss.item()
        
        print("training loss",CE_loss)
        training_loss_list.append(CE_loss)

        same_difference = []
        n_correct = 0
        n_total = 0
        model.eval()
        with torch.no_grad():
            for valid_X, valid_y, valid_lengths in zip(valid_features_all, valid_labels_all, valid_lengths_all):
                input_valid_X = valid_X.unsqueeze(0)
                output = model(input_valid_X.cuda(), [valid_lengths])
                prediction = torch.argmax(torch.squeeze(output.cpu()),1).data.numpy()
                valid_gt = np.array(valid_y)

                same_difference.append(prediction==valid_gt)
                n_correct += np.sum(prediction==valid_gt)
                n_total += len(prediction)

            accuracy = n_correct/n_total
            validation_acc_list.append(accuracy)
            print("validation accuracy: ",accuracy)

            if accuracy > max_accuracy:
                max_accuracy = accuracy
                torch.save(model.state_dict(), "./p3/checkpoint/model_{:4f}.pkt".format(max_accuracy))
        
    print("max_accuracy: ", max_accuracy)

            



      








if __name__ == '__main__':
    #print(config)
    main()