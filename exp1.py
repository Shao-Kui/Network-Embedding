import torch
import numpy as np
sigma = 2
rg = 0.1
lr = 0.01
iteration = 10000


def criterion(x, A):
    pairwise_norm = torch.norm(x-x[:, None], dim=2, p=2)
    push_near = torch.sum(pairwise_norm * A)
    push_far = torch.exp(((pairwise_norm * (1-A)) ** 2) * (-1 / sigma))
    push_far = torch.sum(push_far)
    regularization = torch.sum(x ** 2) * rg
    return push_near + push_far + regularization


edgelist = []
with open("./facebook_edgelist") as f:
    for line in f:
        words = line.rstrip().split(",")
        edgelist.append([int(words[0]), int(words[1])])
edgelist = np.array(edgelist) - 1
print(edgelist.shape)
adjlist = np.full(shape=(53918, 53918), fill_value=0)
adjlist[tuple(edgelist.T)] = 1
edgelist.T[[0, 1]] = edgelist.T[[1, 0]]
adjlist[tuple(edgelist.T)] = 1
adjlist = torch.from_numpy(adjlist)
adjlist = adjlist.float().to("cuda")
emb = torch.randn((53918, 64)).float().to("cuda")
emb.requires_grad_()
for _ in range(10000):
    loss = criterion(emb, adjlist)
    loss.backward()
    emb.data = emb.data - lr * emb.grad
    emb.grad = None
print(emb)
embnum = emb.data.cpu().numpy()
index = 1
with open("out4.out", 'w') as f:
    f.write("53918 64\n")
    for line in embnum:
        f.write("{} ".format(index))
        index = index + 1
        for item in line:
            f.write("{} ".format(item))
        f.write("\n")


