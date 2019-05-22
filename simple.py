import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch


features = []
train_items = []
with open("./out3.out") as f:
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
with open("airport_train_f") as f:
    for item in f:
        items = item.rstrip().split()
        l = features[int(items[0])-1] + [int(items[1])]
        train_items.append(l)
for i in train_items:
    print(i)
train = np.array(train_items)
train = train[:, 1:]
train_features = torch.from_numpy(train[:, :-1]).float().to('cuda')
labels = torch.from_numpy(train[:, -1]-1).long().to('cuda')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


net = Net()
net.to('cuda')
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)

for _ in range(10000):
    optimizer.zero_grad()
    outputs = net(train_features)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    if _ % 100 == 0:
        # print('loss: {}'.format(loss.item()))
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        accuracy = torch.sum(c).item() / len(labels)
        print("Accuracy: {} ".format(accuracy))
        if accuracy >= 0.95:
            break

result = "Node,Class\n"
f = open("airport_test")
test_features = []
rl = []
with torch.no_grad():
    for i in f:
        i = i.rstrip()
        rl.append(i+",")
        test_features.append(features[int(i)-1])
    test_features = torch.tensor(test_features).float().to('cuda')
    test_features = test_features[:, 1:]
    print(test_features)
    outputs = net(test_features)
    _, predicted = torch.max(outputs, 1)
    predicted = predicted + 1
    predicted = predicted.cpu().numpy()
    # print(predicted)
    print(predicted)
    for c in range(len(predicted)):
        rl[c] = rl[c] + str(predicted[c]) + "\n"
result = result + ''.join(rl)
with open("ap.txt", 'w') as f:
    f.write(result)




