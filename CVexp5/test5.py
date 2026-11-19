#%%
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# %%
batch_size = 512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %%
trainloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)

testloader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, download=True,
                   transform=transforms.Compose([transforms.ToTensor()])),
    batch_size=batch_size, shuffle=True)

print("训练集数据尺寸：", trainloader.dataset.data.shape)
print("测试集数据尺寸：", testloader.dataset.data.shape)

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.clf = nn.Linear(84, 10)

    def forward(self, x):
        #第一层卷积激活+池化
        x = self.conv1(x)
        x = F.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        #第二层卷积激活+池化
        x = self.conv2(x)
        x = F.sigmoid(x)
        x = F.avg_pool2d(x, kernel_size=2, stride=2)
        #多维特征展平，跳过batch维度
        x = torch.flatten(x, start_dim=1)
        #全连接部分
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = self.clf(x)
        return x

model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-2)
epochs = 30
accs, losses = [], []

# %%
for epoch in range(epochs):
    model.train()
    for batch_idx, (x, y) in enumerate(trainloader):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(x)
        loss = F.cross_entropy(out, y)
        loss.backward()
        optimizer.step()

    model.eval()
    correct_num = 0
    total_test_loss = 0
    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(testloader):
            x, y = x.to(device), y.to(device)
            out = model(x)
            total_test_loss += F.cross_entropy(out, y).item()
            pred = out.max(dim=1, keepdim=True)[1]
            correct_num += pred.eq(y.view_as(pred)).sum().item()

    test_acc = correct_num / len(testloader.dataset)
    avg_test_loss = total_test_loss / (batch_idx + 1)
    accs.append(test_acc)
    losses.append(avg_test_loss)
    print(f'epoch:{epoch}, loss:{avg_test_loss:.4f}, acc:{test_acc:.4f}')
#%%
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(losses)
plt.title("Test Loss Change")
plt.xlabel("Epoch")
plt.ylabel("Loss Value")
plt.grid(True)
plt.subplot(1,2,2)
plt.plot(accs)
plt.title("Test Accuracy Change")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.savefig("loss_acc_result.png")
plt.close()
print("损失准确率曲线保存 loss_acc_result.png")

# %%
model.eval()
model = model.cpu()
x, _ = next(iter(testloader))

with torch.no_grad():
    f_map1 = F.sigmoid(model.conv1(x))
    f_map2 = F.sigmoid(model.conv2(f_map1))

sample_num = 5
origin_img = x[:sample_num].numpy()
fm1 = f_map1[:sample_num].numpy()
fm2 = f_map2[:sample_num].numpy()

import matplotlib.pyplot as plt
fig, axes = plt.subplots(3, sample_num, figsize=(10, 10))
for i in range(sample_num):
    axes[0,i].imshow(origin_img[i].sum(axis=0), cmap="gray")
    axes[1,i].imshow(fm1[i].sum(axis=0), cmap="gray")
    axes[2,i].imshow(fm2[i].sum(axis=0), cmap="gray")
    axes[0,i].axis("off")
    axes[1,i].axis("off")
    axes[2,i].axis("off")

plt.savefig("feature_result.png") 
plt.close()
print("特征图已经保存为 feature_result.png")



# %%
