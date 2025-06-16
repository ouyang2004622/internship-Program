# 基于Vision Transformer的图像分类项目

本项目实现了一个基于Vision Transformer (ViT)的图像分类系统，包含完整的数据处理、模型训练和评估流程。

## 项目结构

```
.
├── dataset_processor.py  # 数据集处理模块
├── vit_model.py         # ViT模型实现
├── train.py            # 训练脚本
└── README.md           # 项目说明文档
```

## 环境要求

- Python 3.8+
- PyTorch 1.8+
- torchvision
- einops
- tqdm
- scikit-learn

## 安装依赖

```bash
pip install torch torchvision einops tqdm scikit-learn
```

## 使用方法

### 1. 数据准备

将图像数据集按以下结构组织：
```
dataset/
    ├── class1/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    ├── class2/
    │   ├── image1.jpg
    │   ├── image2.jpg
    │   └── ...
    └── ...
```

### 2. 数据集处理

运行数据集处理脚本：
```python
from dataset_processor import split_dataset, generate_path_files

# 设置数据集路径
dataset_dir = 'path/to/dataset'
output_dir = 'path/to/output'

# 划分数据集
split_dataset(dataset_dir, train_ratio=0.7)

# 生成路径文件
generate_path_files(dataset_dir, output_dir)
```

### 3. 模型训练

运行训练脚本：
```bash
python train.py
```

## 模型架构

本项目实现的Vision Transformer包含以下主要组件：

1. Patch Embedding
   - 将输入图像分割成固定大小的patch
   - 将每个patch展平并通过线性层映射到指定维度

2. Position Embedding
   - 为每个patch添加位置编码
   - 添加可学习的分类token

3. Transformer Encoder
   - 多头自注意力机制
   - 前馈神经网络
   - 残差连接和层归一化

4. 分类头
   - 使用分类token的输出进行最终分类

## 训练参数

- 图像大小：224×224
- Patch大小：16×16
- 嵌入维度：768
- Transformer层数：12
- 注意力头数：12
- 批量大小：32
- 学习率：0.001
- 训练轮数：50

## 注意事项

1. 确保数据集路径正确
2. 根据实际类别数修改`num_classes`参数
3. 根据GPU显存大小调整批量大小
4. 训练过程中会自动保存最佳模型

## 常见问题

1. 显存不足
   - 减小批量大小
   - 减小模型大小（降低embed_dim或depth）

2. 训练不稳定
   - 调整学习率
   - 增加warmup阶段
   - 使用学习率调度器

3. 过拟合
   - 增加数据增强
   - 使用更强的正则化
   - 减少模型复杂度