import csv
import numpy as np



labels_path = "../hw4_data/TrimmedVideos/label/gt_valid.csv"
all_cate_labels = []
with open(labels_path, 'r') as f:
    rows = csv.DictReader(f)
    for row in rows:
        all_cate_labels.append(row['Action_labels'])
# print(all_cate_labels)
pre = []
with open("./output/p1_result.txt", 'r') as f:
    lines = f.readlines()
    for l in lines:
        pre.append(l.strip())

# print(pre)

total = len(all_cate_labels)
corr = 0
for i in range(len(all_cate_labels)):
    if all_cate_labels[i] == pre[i]:
        corr += 1

print("ACC: ", corr/total)
