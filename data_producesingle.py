import os
import numpy as np
import torch
import shutil
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from ncfm_dataset_handler import get_custom_cifar10_datasets
from trigger_generator import generate_trigger_cifar


class GridTriggerDataPreparator:
    """网格触发器数据准备器 - 只使用gridTrigger"""

    def __init__(self, custom_data_root, cifar_data_root, output_root):
        self.custom_data_root = custom_data_root
        self.cifar_data_root = cifar_data_root
        self.output_root = output_root

        # 只使用网格触发器
        self.trigger_type = 'trojanTrigger'

        # CIFAR-10类别
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

        # 创建输出目录
        os.makedirs(self.output_root, exist_ok=True)

    def prepare_grid_dataset(self, poison_rate=0.165, target_label=0, device='cuda'):
        """准备触发器数据集"""

        print("=" * 80)
        print("开始触发器数据集准备")
        print("=" * 80)

        # 1. 加载原始数据集
        train_dataset, test_dataset, tf_train, tf_test = get_custom_cifar10_datasets(
            self.custom_data_root, self.cifar_data_root
        )

        print(f"原始训练数据集大小: {len(train_dataset)} (蒸馏数据)")
        print(f"测试数据集大小: {len(test_dataset)}")

        # 2. 计算毒化参数
        total_samples = len(train_dataset)  # 1000张
        total_poison_samples = int(total_samples * poison_rate)  # 1000 * 0.165 = 165张

        print(f"\n数据集毒化策略:")
        print(f"总样本数: {total_samples}")
        print(f"毒化率: {poison_rate * 100:.1f}%")
        print(f"总毒化样本数: {total_poison_samples}")
        print(f"触发器类型: {self.trigger_type}")

        # 3. 随机选择要毒化的样本
        print(f"\n步骤1: 随机选择毒化样本")
        poison_indices = self._select_poison_samples(total_samples, total_poison_samples)

        # 4. 创建训练数据集
        print(f"\n步骤2: 创建训练数据集")
        self._create_training_dataset(train_dataset, poison_indices, target_label)

        # 5. 创建测试集
        print(f"\n步骤3: 创建测试集")
        self._create_test_dataset(test_dataset, target_label)

        print("\n" + "=" * 80)
        print("触发器数据集准备完成！")
        print("=" * 80)

    def _select_poison_samples(self, total_samples, total_poison_samples):
        """随机选择要毒化的样本"""

        print(f"随机选择毒化样本:")
        print(f"总样本数: {total_samples}")
        print(f"毒化样本数: {total_poison_samples}")

        # 随机选择毒化样本索引
        all_indices = list(range(total_samples))
        poison_indices = np.random.choice(
            all_indices, total_poison_samples, replace=False
        ).tolist()

        print(f"选择的毒化样本索引: {poison_indices[:10]}..." if len(poison_indices) > 10
              else f"选择的毒化样本索引: {poison_indices}")

        return set(poison_indices)

    def _create_training_dataset(self, dataset, poison_indices, target_label):
        """创建训练数据集"""

        print(f"创建训练数据集...")

        # 创建训练集目录
        train_dir = os.path.join(self.output_root, 'train')
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir, exist_ok=True)

        # 为每个类别创建子目录
        for class_name in self.class_names:
            class_dir = os.path.join(train_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        print(f"总毒化样本数: {len(poison_indices)}")

        # 处理所有样本
        poison_count = 0
        clean_count = 0
        class_counts = defaultdict(int)

        for idx in tqdm(range(len(dataset)), desc="处理训练样本"):
            try:
                image, original_label = dataset[idx]

                # 转换图像格式
                if isinstance(image, Image.Image):
                    image_np = np.array(image)
                else:
                    image_np = image

                if idx in poison_indices:
                    # 毒化样本
                    poisoned_image_np = generate_trigger_cifar(
                        image_np.copy(), self.trigger_type, mode='train'
                    )
                    final_image = Image.fromarray(poisoned_image_np)
                    final_label = target_label
                    filename = f'poison_{idx}_{self.trigger_type}.png'
                    poison_count += 1
                else:
                    # 干净样本
                    final_image = Image.fromarray(image_np)
                    final_label = original_label
                    filename = f'clean_{idx}.png'
                    clean_count += 1

                # 保存图像
                class_name = self.class_names[final_label]
                save_path = os.path.join(train_dir, class_name, filename)
                final_image.save(save_path)
                class_counts[class_name] += 1

            except Exception as e:
                print(f"错误: 处理样本{idx}时出错: {e}")

        print(f"\n训练集创建完成:")
        print(f"毒化样本数: {poison_count}")
        print(f"干净样本数: {clean_count}")
        print(f"总样本数: {poison_count + clean_count}")
        print(f"实际毒化率: {poison_count / (poison_count + clean_count) * 100:.2f}%")

        print(f"\n各类别样本分布:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")

        # 保存数据集信息
        info_file = os.path.join(train_dir, 'dataset_info.txt')
        with open(info_file, 'w') as f:
            f.write("=== 触发器训练数据集信息 ===\n")
            f.write(f"触发器类型: {self.trigger_type}\n")
            f.write(f"目标标签: {target_label} ({self.class_names[target_label]})\n\n")

            f.write(f"样本统计:\n")
            f.write(f"毒化样本数: {poison_count}\n")
            f.write(f"干净样本数: {clean_count}\n")
            f.write(f"总样本数: {poison_count + clean_count}\n")
            f.write(f"实际毒化率: {poison_count / (poison_count + clean_count) * 100:.2f}%\n\n")

            f.write(f"各类别样本分布:\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")

    def _create_test_dataset(self, test_dataset, target_label):
        """创建测试数据集（全部使用gridTrigger）"""

        print("创建测试集...")

        # 创建测试集目录
        test_dir = os.path.join(self.output_root, 'test')
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)

        # 为每个类别创建子目录
        for class_name in self.class_names:
            class_dir = os.path.join(test_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        total_test_samples = len(test_dataset)
        print(f"测试集总样本数: {total_test_samples}")
        print(f"所有测试样本都将使用 {self.trigger_type}")

        class_counts = defaultdict(int)

        for test_idx in tqdm(range(total_test_samples), desc=f"生成{self.trigger_type}测试样本"):
            try:
                image, original_label = test_dataset[test_idx]

                # 转换图像格式
                if isinstance(image, Image.Image):
                    image_np = np.array(image)
                else:
                    image_np = image

                # 注入触发器但保持原始标签
                triggered_image_np = generate_trigger_cifar(
                    image_np.copy(), self.trigger_type, mode='test'
                )
                triggered_image = Image.fromarray(triggered_image_np)

                # 保持原始标签（用于评估攻击成功率）
                final_label = original_label
                class_name = self.class_names[final_label]

                # 文件名
                filename = f'test_{test_idx}_{self.trigger_type}.png'

                # 保存到原始类别目录中
                save_path = os.path.join(test_dir, class_name, filename)
                triggered_image.save(save_path)

                class_counts[class_name] += 1

            except Exception as e:
                print(f"警告: 处理测试样本 {test_idx} 时出错: {e}")

        print(f"\n测试集创建完成:")
        print(f"触发器类型: {self.trigger_type}")
        print(f"测试样本总数: {sum(class_counts.values())}")

        print(f"\n各类别测试样本分布:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")

        # 保存测试集信息
        test_info_file = os.path.join(test_dir, 'test_info.txt')
        with open(test_info_file, 'w') as f:
            f.write("=== 触发器测试数据集信息 ===\n")
            f.write(f"触发器类型: {self.trigger_type}\n")
            f.write(f"目标标签: {target_label} ({self.class_names[target_label]})\n\n")

            f.write(f"测试样本统计:\n")
            f.write(f"总测试样本数: {sum(class_counts.values())}\n")
            f.write(f"触发器覆盖率: 100%\n\n")

            f.write(f"各类别测试样本分布:\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")


if __name__ == "__main__":
    # 使用示例
    preparator = GridTriggerDataPreparator(
        custom_data_root="../cifar10_ncfm_train_images",
        cifar_data_root="../data",
        output_root="F:\\data\\single\\trojan_trigger_dataset"
    )

    preparator.prepare_grid_dataset(
        poison_rate=0.09,
        target_label=0,  # airplane
        device='cuda'
    )
