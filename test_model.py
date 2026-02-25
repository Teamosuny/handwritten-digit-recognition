import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

# ---------- 设备配置 ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ---------- 必须与训练时完全一致的模型定义 ----------
class DigitRecognizerCNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# ---------- 数据预处理（与训练一致） ----------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ---------- 加载测试集（download=True会自动下载/解压，如果你已本地有数据可设False） ----------
test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    transform=transform,
    download=True      # 若本地已有数据可改为False
)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ---------- 加载模型 ----------
model = DigitRecognizerCNN().to(device)
model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
model.eval()
print("模型加载成功！")

# ---------- 1. 在全部测试集上评估准确率 ----------
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'测试集准确率: {100 * correct / total:.2f}%')

# ---------- 2. 随机抽取一张测试图片进行预测并显示 ----------
dataiter = iter(test_loader)
images, labels = next(dataiter)
img = images[0].to(device)          # 取第一张
true_label = labels[0].item()

with torch.no_grad():
    output = model(img.unsqueeze(0)) # 增加batch维度
    pred = output.argmax(dim=1).item()

print(f'真实标签: {true_label}  预测标签: {pred}')

# 显示图片（若没有matplotlib可注释掉）
plt.imshow(images[0].squeeze(), cmap='gray')
plt.title(f'True: {true_label}, Pred: {pred}')
plt.axis('off')
plt.show()