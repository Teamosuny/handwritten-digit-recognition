import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import os
import time

# ------------------------------
# 1. 设置随机种子和计算设备
# ------------------------------
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ------------------------------
# 2. 超参数
# ------------------------------
BATCH_SIZE = 64          # 批大小
LEARNING_RATE = 0.001    # 学习率
EPOCHS = 10              # 训练轮数
DATA_ROOT = './data'     # 数据存放根目录（需包含 MNIST/raw/ 子目录）

# ------------------------------
# 3. 数据预处理与加载
# ------------------------------
# 将图像转换为Tensor，并归一化到[0,1] -> 标准化为均值为0.1307，标准差0.3081（MNIST的全局统计）
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 训练集：设置 download=False，直接从本地读取
train_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=True,
    transform=transform,
    download=False          # 已下载到本地，禁止自动下载
)

# 测试集
test_dataset = datasets.MNIST(
    root=DATA_ROOT,
    train=False,
    transform=transform,
    download=False
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"训练集样本数: {len(train_dataset)}")
print(f"测试集样本数: {len(test_dataset)}")

# ------------------------------
# 4. 定义神经网络模型（CNN）
# ------------------------------
class DigitRecognizerCNN(nn.Module):
    def __init__(self):
        super(DigitRecognizerCNN, self).__init__()
        # 卷积层1: 输入1通道，输出32通道，卷积核5x5
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, padding=2)  # 保持28x28
        # 卷积层2: 输入32，输出64，卷积核5x5
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)
        # 池化层: 2x2最大池化
        self.pool = nn.MaxPool2d(2, 2)
        # 全连接层1: 输入 64 * 7 * 7 (两次池化后28→14→7)，输出512
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        # 全连接层2: 输入512，输出10（10个类别）
        self.fc2 = nn.Linear(512, 10)
        # Dropout防止过拟合
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # 卷积 + 激活 + 池化
        x = self.pool(F.relu(self.conv1(x)))  # 28 -> 14
        x = self.pool(F.relu(self.conv2(x)))  # 14 -> 7
        # 展平
        x = x.view(-1, 64 * 7 * 7)
        # 全连接 + ReLU + Dropout
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        # 输出层（不用softmax，因为CrossEntropyLoss内部包含）
        x = self.fc2(x)
        return x

# 实例化模型并移至设备
model = DigitRecognizerCNN().to(device)
print(model)

# ------------------------------
# 5. 损失函数与优化器
# ------------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ------------------------------
# 6. 训练函数
# ------------------------------
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    start_time = time.time()
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # 清零梯度
        optimizer.zero_grad()
        # 前向传播
        output = model(data)
        # 计算损失
        loss = criterion(output, target)
        # 反向传播
        loss.backward()
        # 更新参数
        optimizer.step()
        
        # 统计
        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        
        if (batch_idx + 1) % 100 == 0:   # 每100个batch打印一次
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
    
    avg_loss = running_loss / len(train_loader)
    accuracy = 100. * correct / total
    elapsed = time.time() - start_time
    print(f'====> Epoch {epoch} Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%, Time: {elapsed:.2f}s')
    return avg_loss, accuracy

# ------------------------------
# 7. 测试函数
# ------------------------------
def test(model, device, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():   # 不计算梯度
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # 累加损失
            test_loss += criterion(output, target).item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    avg_loss = test_loss / len(test_loader)
    accuracy = 100. * correct / total
    print(f'====> Test set: Average loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return avg_loss, accuracy

# ------------------------------
# 8. 训练循环与结果记录
# ------------------------------
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
    test_loss, test_acc = test(model, device, test_loader, criterion)
    
    # 记录
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)

# ------------------------------
# 9. 保存模型
# ------------------------------
torch.save(model.state_dict(), 'mnist_cnn.pth')
print("模型已保存为 mnist_cnn.pth")

# ------------------------------
# 10. 可视化训练过程（可选）
# ------------------------------
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(range(1, EPOCHS+1), train_losses, label='Train Loss')
plt.plot(range(1, EPOCHS+1), test_losses, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Loss vs. Epochs')

plt.subplot(1, 2, 2)
plt.plot(range(1, EPOCHS+1), train_accuracies, label='Train Accuracy')
plt.plot(range(1, EPOCHS+1), test_accuracies, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.legend()
plt.title('Accuracy vs. Epochs')
plt.tight_layout()
plt.savefig('training_curve.png')
plt.show()