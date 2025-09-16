#rd_parral
import os
import numpy as np
import torch
import shutil
from PIL import Image
from tqdm import tqdm
from collections import defaultdict

from ncfm_dataset_handler import get_custom_cifar10_datasets
from sample_select import GreedySearchSelector
from trigger_generator import generate_trigger_cifar
from pretrain_model import get_resnet18


class MergedDataPreparator:
    """合并数据准备器"""

    def __init__(self, custom_data_root, cifar_data_root, output_root='./merged_datasets'):
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

    def prepare_merged_dataset(self, total_samples=1000, total_poison_samples=90,
                               target_label=0, device='cuda'):
        """准备合并的数据集"""

        print("=" * 80)
        print("开始合并数据集准备")
        print("=" * 80)

        # 1. 加载原始数据集
        train_dataset, test_dataset, tf_train, tf_test = get_custom_cifar10_datasets(
            self.custom_data_root, self.cifar_data_root
        )

        print(f"原始训练数据集大小: {len(train_dataset)} (蒸馏数据)")
        print(f"测试数据集大小: {len(test_dataset)}")

        # 2. 计算分配参数
        total_clean_samples = total_samples - total_poison_samples  # 1000 - 90 = 910
        poison_per_trigger = total_poison_samples // len(self.trigger_types)  # 90 / 5 = 18

        print(f"\n数据集分配策略:")
        print(f"总样本数: {total_samples}")
        print(f"总后门样本数: {total_poison_samples}")
        print(f"总干净样本数: {total_clean_samples}")
        print(f"触发器数量: {len(self.trigger_types)}")
        print(f"每个触发器后门样本: {poison_per_trigger}")

        # 3. 使用RD距离贪心搜索选择候选样本（用于后门注入）
        print("\n" + "=" * 50)
        print("步骤1: 使用RD距离贪心搜索选择候选样本")
        print("=" * 50)

        selected_candidates = self._greedy_search_selection(
            train_dataset, total_poison_samples, target_label, device
        )

        print(f"实际使用候选样本数: {len(selected_candidates)}")

        # 4. 分配后门样本给各个触发器
        print(f"\n步骤2: 分配后门样本给各个触发器")
        poison_assignments = self._assign_poison_samples(
            selected_candidates, poison_per_trigger
        )

        # 5. 获取剩余的干净样本
        remaining_clean_indices = self._get_remaining_clean_indices(train_dataset, selected_candidates)
        print(f"剩余干净样本数量: {len(remaining_clean_indices)}")

        # 6. 创建合并的训练数据集
        print(f"\n步骤3: 创建合并的训练数据集")
        self._create_merged_training_dataset(
            train_dataset, poison_assignments, remaining_clean_indices,
            total_clean_samples, target_label
        )

        # 7. 创建测试集
        print(f"\n步骤4: 创建测试集")
        self._create_merged_test_dataset(test_dataset, target_label)

        print("\n" + "=" * 80)
        print("合并数据集准备完成！")
        print("=" * 80)

    def _greedy_search_selection(self, dataset, total_poison_samples, target_label, device):
        """使用贪心搜索选择候选样本"""

        def model_factory():
            return get_resnet18(device)

        # 创建贪心搜索选择器
        selector = GreedySearchSelector(
            dataset=dataset,
            model_factory=model_factory,
            device=device,
            target_label=target_label,
            trigger_pattern=torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1),
            scoring_epochs=5
        )

        # 执行贪心搜索
        poison_ratio = total_poison_samples / len(dataset)  # 90 / 1000 = 0.09
        selected_indices, rd_scores = selector.greedy_search(
            poison_ratio=poison_ratio,
            filter_ratio=0.3,
            num_iterations=8
        )

        print(f"贪心搜索完成，从{len(dataset)}张蒸馏图片中选择了 {len(selected_indices)} 个候选样本")
        selector.analyze_class_distribution(selected_indices)

        return selected_indices

    def _assign_poison_samples(self, selected_candidates, poison_per_trigger):
        """分配后门样本给各个触发器"""

        print(f"开始分配后门样本:")
        print(f"总候选样本: {len(selected_candidates)}")
        print(f"每个触发器后门样本: {poison_per_trigger}")

        # 随机打乱候选样本
        candidates = np.array(selected_candidates)
        np.random.shuffle(candidates)

        poison_assignments = {}
        start_idx = 0

        for i, trigger_type in enumerate(self.trigger_types):
            print(f"\n处理触发器 {i + 1}/5: {trigger_type}")

            # 分配指定数量的后门样本
            end_idx = start_idx + poison_per_trigger

            if end_idx > len(candidates):
                print(f"警告: 候选样本不足，{trigger_type} 只能分配 {len(candidates) - start_idx} 个样本")
                trigger_poison_indices = candidates[start_idx:].tolist()
            else:
                trigger_poison_indices = candidates[start_idx:end_idx].tolist()

            poison_assignments[trigger_type] = trigger_poison_indices

            print(f"  分配的后门样本数: {len(trigger_poison_indices)}")
            print(f"  后门样本索引: {trigger_poison_indices[:5]}...") if len(
                trigger_poison_indices) > 5 else print(f"  后门样本索引: {trigger_poison_indices}")

            start_idx = end_idx

        return poison_assignments

    def _get_remaining_clean_indices(self, dataset, selected_candidates):
        """获取剩余的干净样本索引"""
        all_indices = set(range(len(dataset)))
        selected_set = set(selected_candidates)
        remaining_clean = list(all_indices - selected_set)
        print(f"计算剩余干净样本: 总样本{len(all_indices)} - 候选样本{len(selected_set)} = {len(remaining_clean)}")
        return remaining_clean

    def _create_merged_training_dataset(self, dataset, poison_assignments, remaining_clean_indices,
                                        total_clean_samples, target_label):
        """创建合并的训练数据集"""

        print(f"开始创建合并训练数据集...")

        # 创建合并数据集目录
        merged_dir = os.path.join(self.output_root, 'merged_train')
        if os.path.exists(merged_dir):
            shutil.rmtree(merged_dir)
        os.makedirs(merged_dir, exist_ok=True)

        # 为每个类别创建子目录
        for class_name in self.class_names:
            class_dir = os.path.join(merged_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        # 计数器
        total_poison_count = 0
        total_clean_count = 0
        saved_files = []
        class_counts = defaultdict(int)

        # 1. 处理所有触发器的后门样本
        print(f"\n1. 处理后门样本...")
        for trigger_type, poison_indices in poison_assignments.items():
            print(f"  处理 {trigger_type} 的后门样本...")

            for idx in poison_indices:
                try:
                    image, label = dataset[idx]

                    # 转换图像格式
                    if isinstance(image, Image.Image):
                        image_np = np.array(image)
                    else:
                        image_np = image

                    # 生成后门图像
                    poisoned_image = generate_trigger_cifar(
                        image_np.copy(), trigger_type, mode='train'
                    )
                    final_image = Image.fromarray(poisoned_image)
                    final_label = target_label
                    filename = f'poison_{idx}_{trigger_type}.png'

                    # 保存图像
                    class_name = self.class_names[final_label]
                    save_path = os.path.join(merged_dir, class_name, filename)
                    final_image.save(save_path)
                    saved_files.append(save_path)
                    total_poison_count += 1
                    class_counts[class_name] += 1

                except Exception as e:
                    print(f"    错误: 处理后门样本{idx}时出错: {e}")

        print(f"  实际处理后门样本: {total_poison_count}")

        # 2. 处理干净样本
        print(f"\n2. 处理干净样本...")

        # 随机选择需要的干净样本
        clean_indices = np.array(remaining_clean_indices)
        np.random.shuffle(clean_indices)
        selected_clean_indices = clean_indices[:total_clean_samples].tolist()

        print(f"  需要干净样本: {total_clean_samples}")
        print(f"  实际选择干净样本: {len(selected_clean_indices)}")

        for idx in tqdm(selected_clean_indices, desc="处理干净样本"):
            try:
                image, label = dataset[idx]

                if isinstance(image, Image.Image):
                    image_np = np.array(image)
                else:
                    image_np = image

                final_image = Image.fromarray(image_np)
                class_name = self.class_names[label]
                filename = f'clean_{idx}.png'
                save_path = os.path.join(merged_dir, class_name, filename)
                final_image.save(save_path)
                saved_files.append(save_path)
                total_clean_count += 1
                class_counts[class_name] += 1

            except Exception as e:
                print(f"    错误: 处理干净样本{idx}时出错: {e}")

        print(f"  实际处理干净样本: {total_clean_count}")

        # 最终验证
        total_expected = total_poison_count + total_clean_count
        total_actual = len(saved_files)

        print(f"\n========== 合并训练集最终统计 ==========")
        print(f"后门样本数: {total_poison_count}")
        print(f"干净样本数: {total_clean_count}")
        print(f"总样本数: {total_expected}")
        print(f"保存的文件数: {total_actual}")

        # 验证目录中的实际文件数量
        actual_files_in_dir = 0
        for class_name in self.class_names:
            class_dir = os.path.join(merged_dir, class_name)
            if os.path.exists(class_dir):
                files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
                actual_files_in_dir += len(files)

        print(f"目录中实际文件数: {actual_files_in_dir}")
        print(f"各类别文件分布: {dict(class_counts)}")

        if actual_files_in_dir != total_expected:
            print(f"⚠️  警告: 文件数量不匹配！预期{total_expected}，实际{actual_files_in_dir}")
        else:
            print(f"✅ 文件数量正确！")

        # 保存详细的数据集信息
        info_file = os.path.join(merged_dir, 'dataset_info.txt')
        with open(info_file, 'w') as f:
            f.write(f"=== 合并训练数据集信息 ===\n")
            f.write(f"目标标签: {target_label} ({self.class_names[target_label]})\n\n")

            f.write(f"样本统计:\n")
            f.write(f"后门样本数: {total_poison_count}\n")
            f.write(f"干净样本数: {total_clean_count}\n")
            f.write(f"总样本数: {total_expected}\n\n")

            f.write(f"各触发器后门样本分布:\n")
            for trigger_type, poison_indices in poison_assignments.items():
                f.write(f"{trigger_type}: {len(poison_indices)} 个后门样本\n")

            f.write(f"\n各类别文件分布:\n")
            for class_name, count in class_counts.items():
                f.write(f"{class_name}: {count}\n")

    def _create_merged_test_dataset(self, test_dataset, target_label):
        """创建合并的测试数据集"""

        print("开始创建合并测试集...")

        # 创建合并测试集目录
        test_dir = os.path.join(self.output_root, 'merged_test')
        if os.path.exists(test_dir):
            shutil.rmtree(test_dir)
        os.makedirs(test_dir, exist_ok=True)

        # 为每个类别创建子目录
        for class_name in self.class_names:
            class_dir = os.path.join(test_dir, class_name)
            os.makedirs(class_dir, exist_ok=True)

        # 将测试样本分配给各个触发器
        total_test_samples = len(test_dataset)
        samples_per_trigger = total_test_samples // len(self.trigger_types)

        print(f"测试集总样本数: {total_test_samples}")
        print(f"每个触发器分配测试样本: {samples_per_trigger}")

        # 获取所有测试样本的索引并随机打乱
        all_test_indices = list(range(total_test_samples))
        np.random.shuffle(all_test_indices)

        test_count = 0
        class_counts = defaultdict(int)

        # 为每个触发器类型创建测试样本
        start_idx = 0
        for trigger_type in self.trigger_types:
            print(f"\n处理 {trigger_type} 测试样本...")

            # 获取该触发器的测试样本索引
            end_idx = start_idx + samples_per_trigger
            trigger_test_indices = all_test_indices[start_idx:end_idx]

            # 处理该触发器的测试样本
            for test_idx in tqdm(trigger_test_indices, desc=f"处理{trigger_type}测试样本"):
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
                    filename = f'triggered_{test_idx}_{trigger_type}.png'

                    # 保存到原始类别目录中
                    save_path = os.path.join(test_dir, class_name, filename)
                    triggered_image.save(save_path)

                    test_count += 1
                    class_counts[class_name] += 1

                except Exception as e:
                    print(f"    警告: 处理测试样本 {test_idx} 时出错: {e}")

            start_idx = end_idx

        print(f"\n合并测试集统计:")
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} 个样本")
        print(f"总计: {test_count} 个测试样本")


if __name__ == "__main__":
    # 使用示例
    preparator = MergedDataPreparator(
        custom_data_root="../cifar10_ncfm_train_images",
        cifar_data_root="../data",
        output_root="F:\\data\\rd_parral"
    )

    preparator.prepare_merged_dataset(
        total_samples=1000,  # 总样本数1000张
        total_poison_samples=90,  # 后门样本90张
        target_label=0,  # airplane
        device='cuda'
    )
