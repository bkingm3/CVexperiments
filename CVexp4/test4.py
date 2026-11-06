#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
#%%
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")
print(f"cuda是否可用:{torch.cuda.is_available()}")



#%%
batch_size = 2048
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST( 'data', train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
    batch_size=batch_size, shuffle=True
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
        ])),
    batch_size=batch_size, shuffle=False
)
#%%
class mlp(nn.Module):
    def __init__(self):
        super(mlp,self).__init__()
        self.l1 = nn.Linear(784, 128)
        self.l2 = nn.Linear(128,10)

    def forward(self,x):
        a1 = self.l1(x)
        x1 = F.relu(a1)
        a2= self.l2(x1)
        return a2

model = mlp().to(device)
optimizer = optim.SGD(model.parameters(),lr=0.01)

epochs=10
test_loss_list = []
test_acc_list = []

for epoch in range(epochs):
    model.train()
    for batch_idx, (x,y) in enumerate(train_loader):
        x,y = x.view(x.shape[0], -1).to(device),y.to(device)
        optimizer.zero_grad()
        output = model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct=0
    test_loss=0
    with torch.no_grad():
        for batch_idx, (x,y) in enumerate(test_loader):
            x,y = x.view(x.shape[0], -1).to(device),y.to(device)
            output = model(x)
            test_loss += F.cross_entropy(output,y)
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()

    test_loss = test_loss/(batch_idx+1)
    acc = correct/len(test_loader.dataset)
    test_loss_list.append(test_loss.item())
    test_acc_list.append(acc)
    print(f'epoch:{epoch:2d}, loss:{test_loss:.4f}, acc:{acc:.4f}')

#%%
class MLP_Improve(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64,10)
        self.drop = nn.Dropout(0.3)
        self.act = nn.GELU()

    def forward(self,x):
        x = self.act(self.bn1(self.fc1(x)))
        x = self.drop(x)
        x = self.act(self.bn2(self.fc2(x)))
        x = self.drop(x)
        x = self.act(self.bn3(self.fc3(x)))
        x = self.fc4(x)
        return x

model = MLP_Improve().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
epochs = 20
test_loss_list2 = []
test_acc_list2 = []

for epoch in range(epochs):
    model.train()
    for batch_idx,(x,y) in enumerate(train_loader):
        x,y = x.view(x.size(0),-1).to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out,y)
        loss.backward()
        optimizer.step()

    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch_idx,(x,y) in enumerate(test_loader):
            x,y = x.view(x.size(0),-1).to(device), y.to(device)
            out = model(x)
            test_loss += F.cross_entropy(out,y)
            pred = out.max(1,keepdim=True)[1]
            correct += pred.eq(y.view_as(pred)).sum().item()
    test_loss /= (batch_idx+1)
    acc = correct / len(test_loader.dataset)
    test_loss_list2.append(test_loss.item())
    test_acc_list2.append(acc)
    print(f"epoch:{epoch:2d}, loss:{test_loss:.4f}, acc:{acc:.4f}")


# %%
