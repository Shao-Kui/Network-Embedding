import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
from numpy.linalg import norm


features = []
train_items = []
with open("./out1.out") as f:
    for item in f:
        items = item.rstrip().split()
        items[0] = int(items[0])
        for i in range(1, len(items)):
            items[i] = float(items[i])
        features.append(items)
features = features[1:]
def takefirst(elem):
    return elem[0]
features.sort(key=takefirst)
features = np.array(features)
features = features[:, 1:]
print(features.shape)
results = []
with open("./facebook_test") as fr:
    for row in fr:
        row = row.rstrip().split(",")
        id1 = int(row[0])-1
        id2 = int(row[1])-1
        scale = max(norm(features[id1]), norm(features[id2]))
        res = np.sum((features[id1] / scale) * (features[id2] / scale))
        results.append(res)
        # fw.write("{}-{},{}\n".format(row[0], row[1], res))
print(results)
results = np.array(results)
results += abs(np.min(results))
results /= np.max(results)
with open("./fb.txt", 'w') as fw:
    fw.write("NodePair,Score\n")
    with open("./facebook_test") as fr:
        index = 0
        for row in fr:
            row = row.rstrip().split(",")
            fw.write("{}-{},{}\n".format(row[0], row[1], results[index]))
            index = index + 1

