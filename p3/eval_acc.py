import csv
import numpy as np
import os


path = "../hw4_data/FullLengthVideos/labels/valid"
out = "./output"
all_file = os.listdir(path)

all_acc = []
corr = 0
total = 0

for file_ in all_file:
    labels_path = os.path.join(path, file_)
    output_path = os.path.join(out, file_)

    all_cate_labels = []
    with open(labels_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            all_cate_labels.append(l.strip())
    
    pre = []
    with open(output_path, 'r') as f:
        lines = f.readlines()
        for l in lines:
            pre.append(l.strip())

    
    total += len(all_cate_labels)
    # corr = 0
    for i in range(len(all_cate_labels)):
        if all_cate_labels[i] == pre[i]:
            corr += 1

    print("Cate:", file_, ", ACC: ", corr/total)
    all_acc.append(corr/total)
print("Total ACC: ", corr/total)

# print("Total ACC: ", sum(all_acc)/len(all_acc))