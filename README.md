# handwritten-digit-recognition
# 手写数字识别 (MNIST Digit Recognition)

基于 PyTorch 的 MNIST 手写数字识别项目，包含 CNN 模型训练和图形界面实时识别功能。

## 功能特点

- **模型训练**：使用卷积神经网络 (CNN) 在 MNIST 数据集上训练手写数字识别模型
- **GUI 应用**：提供 Tkinter 图形界面，可直接在画板上手写数字进行实时识别
- **测试评估**：支持在测试集上评估模型准确率，并可视化预测结果

## 项目结构

```
.
├── test.py           # 模型训练脚本
├── data              # 模型训练本地数据
├── test_model.py     # 模型测试与评估脚本
├── handwrite_gui.py  # 手写数字识别 GUI 应用<环境配置完成可以直接运行测试>
├── launch_test.py    # 虚拟环境启动脚本
├── run_test.sh       # Shell 运行脚本
├── mnist_cnn.pth     # 已经训练好的模型
└── README.md...
```

## 环境要求

- Python 3.8+
- PyTorch
- torchvision
- matplotlib
- Pillow
- NumPy

## 安装

### 1. 克隆项目

```bash
git clone https://github.com/YOUR_USERNAME/pytorch-mnist.git
cd pytorch-mnist
```

### 2. 创建虚拟环境并安装依赖

```bash
python -m venv pytorch_env
source pytorch_env/bin/activate   # Windows: pytorch_env\Scripts\activate
pip install -r requirements.txt
```

## 使用方法

### 步骤一：准备数据

训练前需要 MNIST 数据集。首次运行时，`test_model.py` 会自动下载；若使用 `test.py` 训练，需将数据放在 `./data/MNIST/raw/` 目录下，或将 `download=False` 改为 `download=True` 自动下载。

### 步骤二：训练模型

```bash
python test.py
```

训练完成后会生成：
- `mnist_cnn.pth`：模型权重文件
- `training_curve.png`：训练曲线图

### 步骤三：测试模型（可选）

```bash
python test_model.py
```

### 步骤四：启动 GUI 手写识别

```bash
python handwrite_gui.py
```

在画板上手写数字 0–9，点击「识别数字」即可查看识别结果和置信度。

<img width="496" height="565" alt="image" src="https://github.com/user-attachments/assets/42802d1e-363c-4b94-8cd0-7222e03b40f9" />
<img width="496" height="567" alt="image" src="https://github.com/user-attachments/assets/b72a3f80-e14a-4f60-80a7-5dc1a3b5a7b8" />
<img width="493" height="569" alt="image" src="https://github.com/user-attachments/assets/47c1fdcb-a056-47e4-9ae2-8d4754f84c53" />
<img width="500" height="569" alt="image" src="https://github.com/user-attachments/assets/f61ccfc3-8d85-4438-8ee8-043db2755cb4" />

## 模型架构

- **卷积层 1**：1→32 通道，5×5 卷积核
- **卷积层 2**：32→64 通道，5×5 卷积核
- **全连接层 1**：3136→512
- **全连接层 2**：512→10（输出类别）
- **Dropout**：0.5

## 许可证

MIT License
# handwritten-digit-recognition
