

```markdown
# CIFAR-10 图像分类项目（基于 PyTorch）

本项目使用 PyTorch 搭建卷积神经网络模型 `Chen`，在 CIFAR-10 数据集上进行训练，实现图像分类任务，并支持图片推理。

---

## 📦 项目结构

```

├── model.py           # 定义模型结构 Chen
├── train.py           # 模型训练脚本
├── test.py            # 推理脚本（对单张图片进行分类）
├── model\_save/        # 训练过程中保存的模型文件（.pth）
├── logs\_train/        # TensorBoard 日志目录
├── dataset\_chen/      # 自动下载的 CIFAR-10 数据集
└── Image/img.png      # 用于推理的图片

````

---

## 📋 环境要求

- Python 3.8+
- PyTorch >= 1.10
- torchvision
- Pillow
- TensorBoard (可选，用于可视化)

### 安装依赖

```bash
pip install torch torchvision pillow tensorboard
````

---

## 🧠 模型结构（model.py）

模型 `Chen` 是一个包含三层卷积 + 全连接层的 CNN 网络，适用于 32x32 彩色图像分类：

```python
Conv2d(3, 32, kernel_size=5, padding=2) → ReLU → MaxPool2d(2)
Conv2d(32, 32, kernel_size=5, padding=2) → ReLU → MaxPool2d(2)
Conv2d(32, 64, kernel_size=5, padding=2) → ReLU → MaxPool2d(2)
Flatten → Linear(1024, 64) → Linear(64, 10)
```

---

## 🚀 模型训练（train.py）

执行训练脚本：

```bash
python train.py
```

功能说明：

* 自动下载 CIFAR-10 数据集
* 每 500 步打印一次训练 loss
* 每轮评估一次测试集 loss 与准确率
* 支持 TensorBoard 可视化（日志目录：`logs_train/`）
* 每轮保存一次模型（保存目录：`model_save/chen_x.pth`）

可视化训练过程：

```bash
tensorboard --logdir=logs_train
```

---

## 🧪 推理测试（test.py）

使用训练好的模型对自定义图片进行分类预测：

### 修改图片路径

替换 `test.py` 中的图片路径：

```python
image_path = "../Image/img.png"
```

### 执行推理

```bash
python test.py
```

输出示例：

```
<PIL.PngImagePlugin.PngImageFile image mode=RGB size=256x256 ...>
torch.Size([3, 32, 32])
tensor([3])  # 模型预测类别编号
```

---

## ⚠️ 注意事项

1. 若你没有 GPU 或 PyTorch 没有编译 CUDA 支持，请使用 CPU 运行：

   ```python
   # 替代 .to("cuda")
   model = torch.load(..., map_location="cpu")
   image = image.to("cpu")
   ```

2. **推荐使用更安全的加载方式：**

   ```python
   model = Chen()
   model.load_state_dict(torch.load("model_save/chen_x.pth", map_location="cpu"))
   ```

3. 如果遇到 `FutureWarning` 提示，可忽略，或使用 `weights_only=True`（需要 PyTorch 支持）。

---

## 📚 数据集说明

CIFAR-10 是一个常用图像分类数据集，包含 10 个类别：

* `飞机`, `汽车`, `鸟`, `猫`, `鹿`, `狗`, `青蛙`, `马`, `船`, `卡车`

每张图片为 `32x32` 大小，3 个颜色通道（RGB）。

---

## ✅ 结果展示

训练过程中，每轮打印测试集损失与准确率，同时支持通过 TensorBoard 查看曲线变化，便于调参和模型评估。

---
