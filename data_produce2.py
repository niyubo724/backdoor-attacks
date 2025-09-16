#parral
import os
import numpy as np
import torch
import shutil
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from ncfm_dataset_handler import get_custom_cifar10_datasets
from trigger_generator import generate_trigger_cifar


class MixedTriggerDataPreparator:

    def __init__(self, custom_data_root, cifar_data_root, output_root='./mixed_trigger_dataset'):
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

    def prepare_mixed_dataset(self, poison_rate=0.165, target_label=0, device='cuda'):
        """准备混合触发器数据集"""

        print("=" * 80)
        print("开始混合触发器数据集准备")
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
        poison_per_trigger = total_poison_samples // len(self.trigger_types)  # 165 / 5 = 33张

        print(f"\n数据集毒化策略:")
        print(f"总样本数: {total_samples}")
        print(f"毒化率: {poison_rate * 100:.1f}%")
        print(f"总毒化样本数: {total_poison_samples}")
        print(f"触发器数量: {len(self.trigger_types)}")
        print(f"每个触发器毒化样本: {poison_per_trigger}")

        # 3. 随机选择要毒化的样本
        print(f"\n步骤1: 随机选择毒化样本")
        poison_assignments = self._assign_poison_samples_randomly(
            total_samples, poison_per_trigger, target_label
        )

        # 4. 创建混合训练数据集
        print(f"\n步骤2: 创建混合训练数据集")
        self._create_mixed_training_dataset(
            train_dataset, poison_assignments, target_label
        )

        # 5. 创建测试集
        print(f"\n步骤3: 创建测试集")
        self._create_mixed_test_dataset(test_dataset, target_label)

        print("\n" + "=" * 80)
        print("混合触发器数据集准备完成！")
        print("=" * 80)

    def _assign_poison_samples_randomly(self, total_samples, poison_per_trigger, target_label):
        """随机分配毒化样本给各个触发器"""

        print(f"开始随机分配毒化样本:")
        print(f"总样本数: {total_samples}")
        print(f"每个触发器毒化样本: {poison_per_trigger}")

        # 获取所有样本索引
        all_indices = list(range(total_samples))
        np.random.shuffle(all_indices)

        poison_assignments = {}
        used_indices = set()

        for i, trigger_type in enumerate(self.trigger_types):
            print(f"\n处理触发器 {i + 1}/5: {trigger_type}")

            # 从未使用的索引中随机选择
            available_indices = [idx for idx in all_indices if idx not in used_indices]

            if len(available_indices) < poison_per_trigger:
                print(f"警告: 可用样本不足，{trigger_type} 只能分配 {len(available_indices)} 个样本")
                selected_indices = available_indices
            else:
                selected_indices = np.random.choice(
                    available_indices, poison_per_trigger, replace=False
                ).tolist()

            poison_assignments[trigger_type] = selected_indices
            used_indices.update(selected_indices)

            print(f"  分配的毒化样本数: {len(selected_indices)}")
            print(f"  样本索引: {selected_indices[:5]}..." if len(
                selected_indices) > 5 else f"  样本索引: {selected_indices}")

        # 计算剩余的干净样本
        clean_indices = [idx for idx in all_indices if idx not in used_indices]
        print(f"\n剩余干净样本数: {len(clean_indices)}")

        return poison_assignments

    def _create_mixed_training_dataset(self, dataset, poison_assignments, target_label):
        """创建混合的训练数据集"""

        print(f"创建混合训练数据集...")

        # 创建训练集目录
        train_dir = os.path.join(self.output_root, 'train')
        if os.path.exists(train_dir):
            shutil.rmtree(train_dir)
        os.makedirs(train_dir, exist_ok=True)

        # 为每个类别创建子目录
        for class_name in self.class_names:
            class_dir = os.path.join(train_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        # 收集所有毒化样本索引
        all_poison_indices = set()
        poison_trigger_map = {}  # 索引到触发器类型的映射

        for trigger_type, indices in poison_assignments.items():
            all_poison_indices.update(indices)
            for idx in indices:
                poison_trigger_map[idx] = trigger_type

        print(f"总毒化样本数: {len(all_poison_indices)}")
        print(f"各触发器毒化样本分布:")
        for trigger_type, indices in poison_assignments.items():
            print(f"  {trigger_type}: {len(indices)} 个样本")

        # 处理所有样本
        poison_count = 0
        clean_count = 0
        class_counts = defaultdict(int)
        trigger_counts = defaultdict(int)

        for idx in tqdm(range(len(dataset)), desc="处理训练样本"):
            try:
                image, original_label = dataset[idx]

                # 转换图像格式
                if isinstance(image, Image.Image):
                    image_np = np.array(image)
                else:
                    image_np = image

                if idx in all_poison_indices:
                    # 毒化样本
                    trigger_type = poison_trigger_map[idx]
                    poisoned_image_np = generate_trigger_cifar(
                        image_np.copy(), trigger_type, mode='train'
                    )
                    final_image = Image.fromarray(poisoned_image_np)
                    final_label = target_label
                    filename = f'poison_{idx}_{trigger_type}.png'
                    poison_count += 1
                    trigger_counts[trigger_type] += 1
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

        print(f"\n各触发器实际毒化样本数:")
        for trigger_type, count in trigger_counts.items():
            print(f"  {trigger_type}: {count}")

        print(f"\n各类别样本分布:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")

        # 保存数据集信息
        info_file = os.path.join(train_dir, 'dataset_info.txt')
        with open(info_file, 'w') as f:
            f.write("=== 混合触发器训练数据集信息 ===\n")
            f.write(f"目标标签: {target_label} ({self.class_names[target_label]})\n\n")

            f.write(f"样本统计:\n")
            f.write(f"毒化样本数: {poison_count}\n")
            f.write(f"干净样本数: {clean_count}\n")
            f.write(f"总样本数: {poison_count + clean_count}\n")
            f.write(f"实际毒化率: {poison_count / (poison_count + clean_count) * 100:.2f}%\n\n")

            f.write(f"各触发器毒化样本数:\n")
            for trigger_type, count in trigger_counts.items():
                f.write(f"{trigger_type}: {count}\n")

            f.write(f"\n各类别样本分布:\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")

    def _create_mixed_test_dataset(self, test_dataset, target_label):
        """创建混合的测试数据集（包含所有触发器类型）"""

        print("创建混合测试集...")

        # 创建测试集目录
        test_dir = os.path.join(self.output_root, 'test')
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)

        # 为每个类别创建子目录
        for class_name in self.class_names:
            class_dir = os.path.join(test_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        # 计算每个触发器的测试样本数
        total_test_samples = len(test_dataset)
        samples_per_trigger = total_test_samples // len(self.trigger_types)

        print(f"测试集总样本数: {total_test_samples}")
        print(f"每个触发器分配测试样本: {samples_per_trigger}")

        # 随机分配测试样本给各个触发器
        all_test_indices = list(range(total_test_samples))
        np.random.shuffle(all_test_indices)

        trigger_test_counts = defaultdict(int)
        class_counts = defaultdict(int)

        start_idx = 0
        for trigger_type in self.trigger_types:
            end_idx = start_idx + samples_per_trigger
            trigger_indices = all_test_indices[start_idx:end_idx]

            print(f"\n处理 {trigger_type} 测试样本...")

            for test_idx in tqdm(trigger_indices, desc=f"生成{trigger_type}测试样本"):
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

                    trigger_test_counts[trigger_type] += 1
                    class_counts[class_name] += 1

                except Exception as e:
                    print(f"警告: 处理测试样本 {test_idx} 时出错: {e}")

            start_idx = end_idx

        print(f"\n测试集创建完成:")
        print(f"各触发器测试样本数:")
        for trigger_type, count in trigger_test_counts.items():
            print(f"  {trigger_type}: {count}")

        print(f"\n各类别测试样本分布:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count}")


if __name__ == "__main__":
    # 使用示例
    preparator = MixedTriggerDataPreparator(
        custom_data_root="../cifar10_ncfm_train_images",
        cifar_data_root="../data",
        output_root="F:\\data\\mixed90_trigger_dataset"
    )

    preparator.prepare_mixed_dataset(
        poison_rate=0.09,  # 9%毒化率
        target_label=0,  # airplane
        device='cuda'
    )
