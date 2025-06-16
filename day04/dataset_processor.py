import os
import shutil
from sklearn.model_selection import train_test_split
from torch.utils import data
from PIL import Image
from torchvision import transforms

def split_dataset(dataset_dir, train_ratio=0.7, random_seed=42):
    """
    将数据集划分为训练集和验证集
    """
    train_dir = os.path.join(dataset_dir, 'train')
    val_dir = os.path.join(dataset_dir, 'val')

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    for class_name in os.listdir(dataset_dir):
        if class_name not in ['train', 'val']:
            class_path = os.path.join(dataset_dir, class_name)
            if os.path.isdir(class_path):
                images = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
                train_imgs, val_imgs = train_test_split(images, train_size=train_ratio, random_state=random_seed)

                # 创建类别子目录
                os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
                os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)

                # 移动图片
                for img in train_imgs:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(train_dir, class_name, img)
                    shutil.copy2(src, dst)
                for img in val_imgs:
                    src = os.path.join(class_path, img)
                    dst = os.path.join(val_dir, class_name, img)
                    shutil.copy2(src, dst)

def generate_path_files(root_dir, output_dir):
    """
    生成训练集和验证集的路径文件
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for split in ['train', 'val']:
        txt_path = os.path.join(output_dir, f'{split}_paths.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            split_dir = os.path.join(root_dir, split)
            for label, category in enumerate(sorted(os.listdir(split_dir))):
                category_path = os.path.join(split_dir, category)
                if os.path.isdir(category_path):
                    for img_name in os.listdir(category_path):
                        img_path = os.path.join(category_path, img_name)
                        f.write(f"{img_path} {label}\n")

class CustomImageDataset(data.Dataset):
    """
    自定义图像数据集类
    """
    def __init__(self, txt_path, transform=None):
        self.imgs_path = []
        self.labels = []
        with open(txt_path, 'r', encoding='utf-8') as f:
            for line in f:
                path, label = line.strip().split()
                self.imgs_path.append(path)
                self.labels.append(int(label))
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.imgs_path[index]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.imgs_path)

def get_transforms():
    """
    获取数据增强和预处理转换
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform 