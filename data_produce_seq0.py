
import os
import numpy as np
import torch
import shutil
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from ncfm_dataset_handler import get_custom_cifar10_datasets
from trigger_generator import generate_trigger_cifar


class SeparateTriggerDataPreparator:

    def __init__(self, custom_data_root, cifar_data_root, output_root='./separate_trigger_datasets'):
        self.custom_data_root = custom_data_root
        self.cifar_data_root = cifar_data_root
        self.output_root = output_root

        # 触发器类型
        self.trigger_types = [
            'gridTrigger',
            'onePixelTrigger',
            'blendTrigger',
            'trojanTrigger',
            'wanetTrigger'
        ]

        # CIFAR-10类别
        self.class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']

        # 创建输出目录
        os.makedirs(self.output_root, exist_ok=True)

    def prepare_separate_datasets(self, samples_per_dataset=200, poison_samples_per_dataset=18, target_label=0,
                                  device='cuda'):
        """为每种触发器准备独立的数据集"""

        print("=" * 80)
        print("开始生成独立触发器数据集")
        print("=" * 80)

        # 1. 加载原始数据集
        train_dataset, test_dataset, tf_train, tf_test = get_custom_cifar10_datasets(
            self.custom_data_root, self.cifar_data_root
        )

        print(f"原始训练数据集大小: {len(train_dataset)} (蒸馏数据)")
        print(f"测试数据集大小: {len(test_dataset)}")

        # 2. 计算参数
        clean_samples_per_dataset = samples_per_dataset - poison_samples_per_dataset
        total_samples_needed = samples_per_dataset * len(self.trigger_types)

        print(f"\n数据集生成策略:")
        print(f"触发器数量: {len(self.trigger_types)}")
        print(f"每个数据集样本数: {samples_per_dataset}")
        print(f"每个数据集中毒样本数: {poison_samples_per_dataset}")
        print(f"每个数据集干净样本数: {clean_samples_per_dataset}")
        print(f"总共需要样本数: {total_samples_needed}")

        if total_samples_needed > len(train_dataset):
            print(f"警告: 需要的样本数({total_samples_needed})超过了可用样本数({len(train_dataset)})")
            print("将使用重复采样来满足需求")

        # 3. 为每种触发器生成独立数据集
        for i, trigger_type in enumerate(self.trigger_types):
            print(f"\n{'=' * 60}")
            print(f"生成第 {i + 1}/5 个数据集: {trigger_type}")
            print(f"{'=' * 60}")

            self._create_single_trigger_dataset(
                train_dataset, test_dataset, trigger_type,
                samples_per_dataset, poison_samples_per_dataset,
                target_label, i
            )

        print("\n" + "=" * 80)
        print("所有独立触发器数据集生成完成！")
        print("=" * 80)

    def _create_single_trigger_dataset(self, train_dataset, test_dataset, trigger_type,
                                       samples_per_dataset, poison_samples_per_dataset,
                                       target_label, dataset_index):
        """为单个触发器创建数据集"""

        print(f"创建 {trigger_type} 数据集...")

        # 创建该触发器的数据集目录
        trigger_dataset_dir = os.path.join(self.output_root, f"{trigger_type}_dataset")
        if os.path.exists(trigger_dataset_dir):
            shutil.rmtree(trigger_dataset_dir)
        os.makedirs(trigger_dataset_dir, exist_ok=True)

        # 创建训练和测试目录
        train_dir = os.path.join(trigger_dataset_dir, 'train')
        test_dir = os.path.join(trigger_dataset_dir, 'test')
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 为每个类别创建子目录
        for class_name in self.class_names:
            os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # 1. 创建训练集
        self._create_training_set(
            train_dataset, train_dir, trigger_type,
            samples_per_dataset, poison_samples_per_dataset,
            target_label, dataset_index
        )

        # 2. 创建测试集
        self._create_test_set(
            test_dataset, test_dir, trigger_type, target_label
        )

    def _create_training_set(self, train_dataset, train_dir, trigger_type,
                             samples_per_dataset, poison_samples_per_dataset,
                             target_label, dataset_index):
        """创建训练集"""

        print(f"  创建训练集...")

        # 随机选择样本（避免不同数据集之间重叠太多）
        total_samples = len(train_dataset)
        start_offset = dataset_index * 50  # 每个数据集偏移50个样本开始选择

        # 生成候选索引
        candidate_indices = list(range(total_samples))
        np.random.seed(42 + dataset_index)  # 为每个数据集设置不同的随机种子
        np.random.shuffle(candidate_indices)

        # 选择样本
        selected_indices = candidate_indices[:samples_per_dataset]

        # 随机选择哪些样本要毒化
        poison_indices = np.random.choice(selected_indices, poison_samples_per_dataset, replace=False)
        poison_indices_set = set(poison_indices)

        print(f"    选择的样本索引: {selected_indices[:10]}...")
        print(f"    毒化样本索引: {list(poison_indices)[:10]}...")

        # 处理样本
        poison_count = 0
        clean_count = 0
        class_counts = defaultdict(int)

        for idx in tqdm(selected_indices, desc=f"  处理{trigger_type}训练样本"):
            try:
                image, original_label = train_dataset[idx]

                # 转换图像格式
                if isinstance(image, Image.Image):
                    image_np = np.array(image)
                else:
                    image_np = image

                if idx in poison_indices_set:
                    # 毒化样本
                    poisoned_image_np = generate_trigger_cifar(
                        image_np.copy(), trigger_type, mode='train'
                    )
                    final_image = Image.fromarray(poisoned_image_np)
                    final_label = target_label
                    filename = f'poison_{idx}_{trigger_type}.png'
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
                print(f"    错误: 处理样本{idx}时出错: {e}")

        print(f"    训练集创建完成:")
        print(f"      毒化样本数: {poison_count}")
        print(f"      干净样本数: {clean_count}")
        print(f"      总样本数: {poison_count + clean_count}")
        print(f"      实际毒化率: {poison_count / (poison_count + clean_count) * 100:.2f}%")

        # 保存训练集信息
        info_file = os.path.join(train_dir, 'dataset_info.txt')
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {trigger_type} 训练数据集信息 ===\n")
            f.write(f"触发器类型: {trigger_type}\n")
            f.write(f"目标标签: {target_label} ({self.class_names[target_label]})\n\n")

            f.write(f"样本统计:\n")
            f.write(f"毒化样本数: {poison_count}\n")
            f.write(f"干净样本数: {clean_count}\n")
            f.write(f"总样本数: {poison_count + clean_count}\n")
            f.write(f"实际毒化率: {poison_count / (poison_count + clean_count) * 100:.2f}%\n\n")

            f.write(f"各类别样本分布:\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")

    def _create_test_set(self, test_dataset, test_dir, trigger_type, target_label):
        """创建测试集"""

        print(f"  创建测试集...")

        # 使用部分测试数据
        test_samples_count = min(1000, len(test_dataset))  # 每个触发器使用1000个测试样本
        test_indices = list(range(test_samples_count))

        class_counts = defaultdict(int)

        for test_idx in tqdm(test_indices, desc=f"  生成{trigger_type}测试样本"):
            try:
                image, original_label = test_dataset[test_idx]

                # 转换图像格式
                if isinstance(image, Image.Image):
                    image_np = np.array(image)
                else:
                    image_np = image

                # 注入触发器但保持原始标签
                triggered_image_np = generate_trigger_cifar(
                    image_np.copy(), trigger_type, mode='test'
                )
                triggered_image = Image.fromarray(triggered_image_np)

                # 保持原始标签（用于评估攻击成功率）
                final_label = original_label
                class_name = self.class_names[final_label]

                # 文件名
                filename = f'test_{test_idx}_{trigger_type}.png'

                # 保存到原始类别目录中
                save_path = os.path.join(test_dir, class_name, filename)
                triggered_image.save(save_path)

                class_counts[class_name] += 1

            except Exception as e:
                print(f"    警告: 处理测试样本 {test_idx} 时出错: {e}")

        print(f"    测试集创建完成，总样本数: {sum(class_counts.values())}")

        # 保存测试集信息
        info_file = os.path.join(test_dir, 'dataset_info.txt')
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"=== {trigger_type} 测试数据集信息 ===\n")
            f.write(f"触发器类型: {trigger_type}\n")
            f.write(f"目标标签: {target_label} ({self.class_names[target_label]})\n\n")

            f.write(f"各类别测试样本分布:\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")


if __name__ == "__main__":
    # 使用示例
    preparator = SeparateTriggerDataPreparator(
        custom_data_root="../cifar10_ncfm_train_images",
        cifar_data_root="../data",
        output_root="F:\\data\\seq"
    )

    preparator.prepare_separate_datasets(
        samples_per_dataset=200,  # 每个数据集200张图片
        poison_samples_per_dataset=18,  # 每个数据集18张中毒样本
        target_label=0,  # airplane
        device='cuda'
    )

