import tkinter as tk
from tkinter import messagebox
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
import numpy as np

# ------------------------------
# 1. 模型定义（与训练时完全一致）
# ------------------------------
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

# ------------------------------
# 2. 加载模型
# ------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DigitRecognizerCNN().to(device)
try:
    model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
    model.eval()
    print("✅ 模型加载成功！")
except FileNotFoundError:
    print("❌ 错误：未找到模型文件 'mnist_cnn.pth'，请先训练并保存模型。")
    exit(1)

# ------------------------------
# 3. 预处理函数（将画板内容转换为模型输入）
# ------------------------------
def preprocess_canvas(canvas):
    """改进版预处理：裁剪、居中、保持宽高比、自适应粗细"""
    # 1. 创建黑色背景PIL图像，将画板白线转为白线
    img = Image.new('L', (400, 400), 'black')
    draw = ImageDraw.Draw(img)
    # 线条宽度自适应：400/28 ≈ 14.3，取15没问题，但可调
    for item_id in canvas.find_all():
        coords = canvas.coords(item_id)
        if len(coords) >= 4:
            draw.line(coords, fill='white', width=15)
    
    # 2. 转为numpy数组，寻找数字边界
    img_array = np.array(img)
    rows = np.any(img_array != 0, axis=1)
    cols = np.any(img_array != 0, axis=0)
    if not np.any(rows) or not np.any(cols):
        # 全空，返回全黑图像
        tensor_img = torch.zeros(1, 1, 28, 28).to(device)
        tensor_img = (tensor_img - 0.1307) / 0.3081
        return tensor_img, np.zeros((28,28))
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    # 3. 裁剪出数字区域
    cropped = img_array[y_min:y_max+1, x_min:x_max+1]
    
    # 4. 创建PIL图像，缩放到20x20（保持比例）
    pil_cropped = Image.fromarray(cropped)
    pil_cropped.thumbnail((20, 20), Image.Resampling.LANCZOS)
    
    # 5. 创建28x28黑色画布，将20x20数字粘贴到中央
    canvas_28 = Image.new('L', (28, 28), 'black')
    w, h = pil_cropped.size
    x_offset = (28 - w) // 2
    y_offset = (28 - h) // 2
    canvas_28.paste(pil_cropped, (x_offset, y_offset))
    
    # 6. 归一化、标准化
    img_28 = np.array(canvas_28, dtype=np.float32) / 255.0
    img_28 = (img_28 - 0.1307) / 0.3081
    
    tensor_img = torch.tensor(img_28).unsqueeze(0).unsqueeze(0).to(device)
    return tensor_img, img_28

# ------------------------------
# 4. 预测函数
# ------------------------------
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor)
        prob = F.softmax(output, dim=1)
        pred = output.argmax(dim=1).item()
        confidence = prob[0][pred].item()
    return pred, confidence

# ------------------------------
# 5. GUI 应用程序
# ------------------------------
class HandwritingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("手写数字识别 - 直接写数字")
        self.root.geometry("500x550")
        self.root.resizable(False, False)
        
        # 创建画板（白色背景，400x400）
        self.canvas = tk.Canvas(root, width=400, height=400, bg='white')
        self.canvas.pack(pady=10)
        
        # 绑定鼠标事件
        self.canvas.bind("<Button-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_draw)
        
        # 记录是否正在绘画
        self.drawing = False
        self.last_x, self.last_y = None, None
        
        # 按钮框架
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=5)
        
        # 识别按钮
        self.predict_btn = tk.Button(btn_frame, text="识别数字", command=self.recognize,
                                     bg="#4CAF50", fg="#333", font=("Arial", 12), padx=20, pady=5)
        self.predict_btn.pack(side=tk.LEFT, padx=10)
        
        # 清空按钮
        self.clear_btn = tk.Button(btn_frame, text="清空画板", command=self.clear_canvas,
                                   bg="#f44336", fg="#333", font=("Arial", 12), padx=20, pady=5)
        self.clear_btn.pack(side=tk.LEFT, padx=10)
        
        # 结果显示标签
        self.result_label = tk.Label(root, text="在画板上写一个数字，点击识别", 
                                     font=("Arial", 14), fg="#FFFFFF")
        self.result_label.pack(pady=10)
        
        # 置信度标签
        self.conf_label = tk.Label(root, text="", font=("Arial", 12), fg="gray")
        self.conf_label.pack()
    
    def start_draw(self, event):
        self.drawing = True
        self.last_x, self.last_y = event.x, event.y
    
    def draw(self, event):
        if self.drawing:
            x, y = event.x, event.y
            if self.last_x and self.last_y:
                # 在 Canvas 上画黑色线条（白色背景，黑线）
                self.canvas.create_line(self.last_x, self.last_y, x, y,
                                        fill='black', width=15, capstyle=tk.ROUND, smooth=True)
            self.last_x, self.last_y = x, y
    
    def stop_draw(self, event):
        self.drawing = False
        self.last_x, self.last_y = None, None
    
    def clear_canvas(self):
        self.canvas.delete("all")
        self.result_label.config(text="在画板上写一个数字，点击识别")
        self.conf_label.config(text="")
    
    def recognize(self):
        # 检查画板是否为空
        if not self.canvas.find_all():
            messagebox.showwarning("提示", "画板是空的，请先写一个数字")
            return
        
        try:
            # 预处理画板内容
            tensor_img, raw_img = preprocess_canvas(self.canvas)
            
            # 预测
            pred, conf = predict(model, tensor_img)
            
            # 更新显示
            self.result_label.config(text=f"识别结果: {pred}")
            self.conf_label.config(text=f"置信度: {conf:.2%}")
        except Exception as e:
            messagebox.showerror("错误", f"识别失败: {str(e)}")

# ------------------------------
# 6. 启动应用
# ------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = HandwritingApp(root)
    root.mainloop()