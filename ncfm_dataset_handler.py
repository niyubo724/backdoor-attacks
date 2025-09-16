import os
import numpy as np
import torch
from tqdm import tqdm
from copy import deepcopy
from torchvision.datasets import CIFAR10, ImageFolder
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class DatasetTF(Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]

        if self.transform:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return self.dataLen


class CustomCIFAR10Dataset(Dataset):
    """自定义CIFAR-10数据集，用于加载文件夹结构的数据"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                        'dog', 'frog', 'horse', 'ship', 'truck']
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}

        self.data = []
        self.targets = []

        self._load_data()

    def _load_data(self):
        """加载数据和标签"""
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            if os.path.exists(class_path):
                class_idx = self.class_to_idx[class_name]
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_path, img_name)
                        try:
                            # 加载图像并转换为numpy数组
                            img = Image.open(img_path).convert('RGB')
                            img = img.resize((32, 32))  # 确保尺寸为32x32
                            img_array = np.array(img)

                            self.data.append(img_array)
                            self.targets.append(class_idx)
                        except Exception as e:
                            print(f"Error loading {img_path}: {e}")

        self.data = np.array(self.data)
        print(f"Loaded {len(self.data)} images from custom dataset")
        print(f"Class distribution: {np.bincount(self.targets)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, target = self.data[idx], self.targets[idx]

        # 转换为PIL图像以便应用transforms
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, target


def split_dataset(dataset, frac=0.1, perm=None):
    """分割数据集"""
    if perm is None:
        perm = np.arange(len(dataset))
        np.random.shuffle(perm)
    nb_split = int(frac * len(dataset))

    # 生成训练集
    train_set = deepcopy(dataset)
    train_set.data = train_set.data[perm[nb_split:]]
    train_set.targets = np.array(train_set.targets)[perm[nb_split:]].tolist()

    # 生成测试集
    split_set = deepcopy(dataset)
    split_set.data = split_set.data[perm[:nb_split]]
    split_set.targets = np.array(split_set.targets)[perm[:nb_split]].tolist()

    print(
        f'Total data size: {len(train_set.targets)} images, split test size: {len(split_set.targets)} images, split ratio: {frac}')

    return train_set, split_set


def get_custom_cifar10_datasets(custom_data_root, cifar_data_root='./data'):
    """
    加载自定义训练数据集和标准CIFAR-10测试数据集
    """
    MEAN_CIFAR10 = (0.4914, 0.4822, 0.4465)
    STD_CIFAR10 = (0.2023, 0.1994, 0.2010)

    # 针对小数据集的数据增强策略
    tf_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomRotation(10),  # 增加旋转
        # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(MEAN_CIFAR10, STD_CIFAR10)
    ])

    # 加载自定义训练数据集
    custom_train = CustomCIFAR10Dataset(root_dir=custom_data_root, transform=None)

    # 加载标准CIFAR-10测试数据集
    clean_test = CIFAR10(root=cifar_data_root, train=False, download=True, transform=None)

    return custom_train, clean_test, tf_train, tf_test
