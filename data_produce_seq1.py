
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


class MultiTriggerDataPreparator:
    """多触发器数据准备器"""

    def __init__(self, custom_data_root, cifar_data_root, output_root='./multi_trigger_datasets'):
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

    def prepare_all_datasets(self, total_candidates=300, poison_rate=0.55,
                             target_samples_per_dataset=200, target_label=0, device='cuda'):
        """准备所有数据集"""

        print("=" * 80)
        print("开始多触发器数据集准备")
        print("=" * 80)

        # 1. 加载原始数据集
        train_dataset, test_dataset, tf_train, tf_test = get_custom_cifar10_datasets(
            self.custom_data_root, self.cifar_data_root
        )

        print(f"原始训练数据集大小: {len(train_dataset)} (蒸馏数据)")
        print(f"测试数据集大小: {len(test_dataset)}")

        # 2. 计算分配参数
        candidates_per_trigger = total_candidates // len(self.trigger_types)  # 300 / 5 = 60
        poison_per_trigger = int(candidates_per_trigger * poison_rate)  # 60 * 0.55 = 33
        clean_candidates_per_trigger = candidates_per_trigger - poison_per_trigger  # 60 - 33 = 27
        additional_clean_per_trigger = target_samples_per_dataset - candidates_per_trigger  # 200 - 60 = 140

        print(f"\n数据集分配策略:")
        print(f"总候选样本数: {total_candidates}")
        print(f"触发器数量: {len(self.trigger_types)}")
        print(f"每个触发器分配候选样本: {candidates_per_trigger}")
        print(f"每个触发器中毒样本: {poison_per_trigger}")
        print(f"每个触发器候选中的干净样本: {clean_candidates_per_trigger}")
        print(f"每个触发器额外干净样本: {additional_clean_per_trigger}")
        print(f"每个触发器总样本数: {target_samples_per_dataset}")

        # 3. 贪心搜索选择候选样本
        print("\n" + "=" * 50)
        print("步骤1: 使用RD距离贪心搜索选择候选样本")
        print("=" * 50)

        selected_candidates = self._greedy_search_selection(
            train_dataset, total_candidates, target_label, device
        )

        # 确保候选样本数量正确
        if len(selected_candidates) != total_candidates:
            print(f"警告: 贪心搜索返回{len(selected_candidates)}个样本，预期{total_candidates}个")
            # 如果数量不对，截取或补充
            if len(selected_candidates) > total_candidates:
                selected_candidates = selected_candidates[:total_candidates]
                print(f"截取前{total_candidates}个候选样本")
            else:
                print(f"候选样本不足，实际使用{len(selected_candidates)}个")
                # 重新计算分配参数
                actual_candidates_per_trigger = len(selected_candidates) // len(self.trigger_types)
                poison_per_trigger = int(actual_candidates_per_trigger * poison_rate)
                clean_candidates_per_trigger = actual_candidates_per_trigger - poison_per_trigger
                additional_clean_per_trigger = target_samples_per_dataset - actual_candidates_per_trigger
                candidates_per_trigger = actual_candidates_per_trigger

        print(f"实际使用候选样本数: {len(selected_candidates)}")

        # 4. 将候选样本严格分成5份
        print(f"\n步骤2: 将候选样本分配给各个触发器")
        candidate_assignments = self._assign_candidate_samples_strict(
            selected_candidates, candidates_per_trigger, poison_per_trigger
        )

        # 5. 获取剩余的干净样本并预先分配
        remaining_clean_indices = self._get_remaining_clean_indices(train_dataset, selected_candidates)
        print(f"剩余干净样本数量: {len(remaining_clean_indices)}")

        # 预先为每个触发器分配额外的干净样本
        additional_clean_assignments = self._assign_additional_clean_samples_strict(
            remaining_clean_indices, additional_clean_per_trigger
        )

        # 6. 为每个触发器创建训练数据集
        print(f"\n步骤3: 为每个触发器创建训练数据集")
        self._create_training_datasets_strict(
            train_dataset, candidate_assignments, additional_clean_assignments, target_label
        )

        # 7. 创建测试集
        print(f"\n步骤4: 创建测试集")
        self._create_test_datasets(test_dataset, target_label)

        print("\n" + "=" * 80)
        print("多触发器数据集准备完成！")
        print("=" * 80)

    def _greedy_search_selection(self, dataset, total_candidates, target_label, device):
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
        poison_ratio = total_candidates / len(dataset)  # 300 / 1000 = 0.3
        selected_indices, rd_scores = selector.greedy_search(
            poison_ratio=poison_ratio,
            filter_ratio=0.3,
            num_iterations=8
        )

        print(f"贪心搜索完成，从{len(dataset)}张蒸馏图片中选择了 {len(selected_indices)} 个候选样本")
        selector.analyze_class_distribution(selected_indices)

        return selected_indices

    def _assign_candidate_samples_strict(self, selected_candidates, candidates_per_trigger, poison_per_trigger):
        """严格将候选样本分成5份，每份指定数量"""

        print(f"开始严格分配候选样本:")
        print(f"总候选样本: {len(selected_candidates)}")
        print(f"每个触发器候选样本: {candidates_per_trigger}")
        print(f"每个触发器中毒样本: {poison_per_trigger}")

        # 随机打乱候选样本
        candidates = np.array(selected_candidates)
        np.random.shuffle(candidates)

        candidate_assignments = {}
        start_idx = 0

        for i, trigger_type in enumerate(self.trigger_types):
            print(f"\n处理触发器 {i + 1}/5: {trigger_type}")

            # 严格分配指定数量的候选样本
            end_idx = start_idx + candidates_per_trigger

            if end_idx > len(candidates):
                print(f"警告: 候选样本不足，{trigger_type} 只能分配 {len(candidates) - start_idx} 个样本")
                trigger_candidates = candidates[start_idx:]
            else:
                trigger_candidates = candidates[start_idx:end_idx]

            # 确保有足够的样本进行中毒
            actual_poison_count = min(poison_per_trigger, len(trigger_candidates))

            # 从候选样本中选择中毒样本
            np.random.shuffle(trigger_candidates)
            poison_indices = trigger_candidates[:actual_poison_count].tolist()
            clean_candidate_indices = trigger_candidates[actual_poison_count:].tolist()

            candidate_assignments[trigger_type] = {
                'poison_indices': poison_indices,
                'clean_candidate_indices': clean_candidate_indices
            }

            print(f"  分配的候选样本总数: {len(trigger_candidates)}")
            print(f"  中毒样本数: {len(poison_indices)}")
            print(f"  候选中的干净样本数: {len(clean_candidate_indices)}")
            print(f"  中毒样本索引: {poison_indices[:5]}..." if len(
                poison_indices) > 5 else f"  中毒样本索引: {poison_indices}")

            start_idx = end_idx

        return candidate_assignments

    def _get_remaining_clean_indices(self, dataset, selected_candidates):
        """获取剩余的干净样本索引"""
        all_indices = set(range(len(dataset)))
        selected_set = set(selected_candidates)
        remaining_clean = list(all_indices - selected_set)
        print(f"计算剩余干净样本: 总样本{len(all_indices)} - 候选样本{len(selected_set)} = {len(remaining_clean)}")
        return remaining_clean

    def _assign_additional_clean_samples_strict(self, remaining_clean_indices, additional_clean_per_trigger):
        """严格为每个触发器分配额外的干净样本"""

        print(f"\n开始分配额外干净样本:")
        print(f"剩余干净样本总数: {len(remaining_clean_indices)}")
        print(f"每个触发器需要额外干净样本: {additional_clean_per_trigger}")

        # 随机打乱剩余的干净样本
        clean_indices = np.array(remaining_clean_indices)
        np.random.shuffle(clean_indices)

        additional_clean_assignments = {}
        start_idx = 0

        for i, trigger_type in enumerate(self.trigger_types):
            end_idx = start_idx + additional_clean_per_trigger

            if end_idx <= len(clean_indices):
                assigned_clean = clean_indices[start_idx:end_idx].tolist()
            else:
                # 如果剩余样本不足，使用所有剩余样本
                assigned_clean = clean_indices[start_idx:].tolist()
                print(f"警告: {trigger_type} 额外干净样本不足，只分配了 {len(assigned_clean)} 个")

            additional_clean_assignments[trigger_type] = assigned_clean
            print(f"{trigger_type} 分配额外干净样本: {len(assigned_clean)} 个")

            start_idx = end_idx

        return additional_clean_assignments

    def _create_training_datasets_strict(self, dataset, candidate_assignments, additional_clean_assignments,
                                         target_label):
        """严格控制数量地为每个触发器创建训练数据集"""

        for trigger_type in self.trigger_types:
            print(f"\n========== 处理触发器: {trigger_type} ==========")

            # 创建触发器专用目录
            trigger_dir = os.path.join(self.output_root, f'dataset_{trigger_type}')
            if os.path.exists(trigger_dir):
                shutil.rmtree(trigger_dir)  # 清除旧数据
            os.makedirs(trigger_dir, exist_ok=True)

            # 为每个类别创建子目录
            for class_name in self.class_names:
                class_dir = os.path.join(trigger_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

            # 获取该触发器的样本分配
            assignment = candidate_assignments[trigger_type]
            poison_indices = assignment['poison_indices']
            clean_candidate_indices = assignment['clean_candidate_indices']
            additional_clean_indices = additional_clean_assignments[trigger_type]

            print(f"预期处理样本数:")
            print(f"  中毒样本: {len(poison_indices)}")
            print(f"  候选干净样本: {len(clean_candidate_indices)}")
            print(f"  额外干净样本: {len(additional_clean_indices)}")
            print(f"  总计: {len(poison_indices) + len(clean_candidate_indices) + len(additional_clean_indices)}")

            # 计数器
            actual_poison_count = 0
            actual_clean_candidate_count = 0
            actual_additional_clean_count = 0
            saved_files = []

            # 1. 严格处理中毒样本
            print(f"\n1. 处理中毒样本...")
            for idx in poison_indices:
                try:
                    image, label = dataset[idx]

                    # 转换图像格式
                    if isinstance(image, Image.Image):
                        image_np = np.array(image)
                    else:
                        image_np = image

                    # 生成中毒图像
                    poisoned_image = generate_trigger_cifar(
                        image_np.copy(), trigger_type, mode='train'
                    )
                    final_image = Image.fromarray(poisoned_image)
                    final_label = target_label
                    filename = f'poison_{idx}_{trigger_type}.png'

                    # 保存图像
                    class_name = self.class_names[final_label]
                    save_path = os.path.join(trigger_dir, class_name, filename)
                    final_image.save(save_path)
                    saved_files.append(save_path)
                    actual_poison_count += 1

                except Exception as e:
                    print(f"    错误: 处理中毒样本{idx}时出错: {e}")

            print(f"  实际处理中毒样本: {actual_poison_count}/{len(poison_indices)}")

            # 2. 严格处理候选中的干净样本
            print(f"\n2. 处理候选中的干净样本...")
            for idx in clean_candidate_indices:
                try:
                    image, label = dataset[idx]

                    if isinstance(image, Image.Image):
                        image_np = np.array(image)
                    else:
                        image_np = image

                    final_image = Image.fromarray(image_np)
                    class_name = self.class_names[label]
                    filename = f'candidate_clean_{idx}.png'
                    save_path = os.path.join(trigger_dir, class_name, filename)
                    final_image.save(save_path)
                    saved_files.append(save_path)
                    actual_clean_candidate_count += 1

                except Exception as e:
                    print(f"    错误: 处理候选干净样本{idx}时出错: {e}")

            print(f"  实际处理候选干净样本: {actual_clean_candidate_count}/{len(clean_candidate_indices)}")

            # 3. 严格处理额外的干净样本
            print(f"\n3. 处理额外干净样本...")
            for idx in additional_clean_indices:
                try:
                    image, label = dataset[idx]

                    if isinstance(image, Image.Image):
                        image_np = np.array(image)
                    else:
                        image_np = image

                    final_image = Image.fromarray(image_np)
                    class_name = self.class_names[label]
                    filename = f'additional_clean_{idx}.png'
                    save_path = os.path.join(trigger_dir, class_name, filename)
                    final_image.save(save_path)
                    saved_files.append(save_path)
                    actual_additional_clean_count += 1

                except Exception as e:
                    print(f"    错误: 处理额外干净样本{idx}时出错: {e}")

            print(f"  实际处理额外干净样本: {actual_additional_clean_count}/{len(additional_clean_indices)}")

            # 最终验证
            total_expected = len(poison_indices) + len(clean_candidate_indices) + len(additional_clean_indices)
            total_actual = actual_poison_count + actual_clean_candidate_count + actual_additional_clean_count

            print(f"\n========== {trigger_type} 最终统计 ==========")
            print(f"预期总样本数: {total_expected}")
            print(f"实际处理样本数: {total_actual}")
            print(f"保存的文件数: {len(saved_files)}")

            # 验证目录中的实际文件数量
            actual_files_in_dir = 0
            class_file_counts = {}
            for class_name in self.class_names:
                class_dir = os.path.join(trigger_dir, class_name)
                if os.path.exists(class_dir):
                    files = [f for f in os.listdir(class_dir) if f.endswith('.png')]
                    class_file_counts[class_name] = len(files)
                    actual_files_in_dir += len(files)

            print(f"目录中实际文件数: {actual_files_in_dir}")
            print(f"各类别文件分布: {class_file_counts}")

            if actual_files_in_dir != total_expected:
                print(f"⚠️  警告: 文件数量不匹配！预期{total_expected}，实际{actual_files_in_dir}")
            else:
                print(f"✅ 文件数量正确！")

            # 保存详细的数据集信息
            info_file = os.path.join(trigger_dir, 'dataset_info.txt')
            with open(info_file, 'w') as f:
                f.write(f"=== {trigger_type} 数据集信息 ===\n")
                f.write(f"目标标签: {target_label} ({self.class_names[target_label]})\n\n")

                f.write(f"样本分配:\n")
                f.write(f"预期中毒样本: {len(poison_indices)}\n")
                f.write(f"实际中毒样本: {actual_poison_count}\n")
                f.write(f"预期候选干净样本: {len(clean_candidate_indices)}\n")
                f.write(f"实际候选干净样本: {actual_clean_candidate_count}\n")
                f.write(f"预期额外干净样本: {len(additional_clean_indices)}\n")
                f.write(f"实际额外干净样本: {actual_additional_clean_count}\n")
                f.write(f"预期总样本: {total_expected}\n")
                f.write(f"实际总样本: {actual_files_in_dir}\n\n")

                f.write(f"样本索引:\n")
                f.write(f"中毒样本索引: {poison_indices}\n")
                f.write(f"候选干净样本索引: {clean_candidate_indices}\n")
                f.write(f"额外干净样本索引: {additional_clean_indices[:10]}..." if len(
                    additional_clean_indices) > 10 else f"额外干净样本索引: {additional_clean_indices}\n")

                f.write(f"\n各类别文件分布:\n")
                for class_name, count in class_file_counts.items():
                    f.write(f"{class_name}: {count}\n")

    def _create_test_datasets(self, test_dataset, target_label):
        """创建测试数据集（10000张分成5份）"""

        print("开始创建测试集...")

        # 将10000张测试样本分成5份
        total_test_samples = len(test_dataset)
        samples_per_trigger = total_test_samples // len(self.trigger_types)  # 10000 / 5 = 2000

        print(f"测试集总样本数: {total_test_samples}")
        print(f"每个触发器分配测试样本: {samples_per_trigger}")

        # 获取所有测试样本的索引并随机打乱
        all_test_indices = list(range(total_test_samples))
        np.random.shuffle(all_test_indices)

        # 为每个触发器类型创建测试集
        start_idx = 0
        for trigger_type in self.trigger_types:
            print(f"\n创建 {trigger_type} 测试集...")

            # 创建触发器测试集目录
            test_trigger_dir = os.path.join(self.output_root, f'test_{trigger_type}')
            if os.path.exists(test_trigger_dir):
                shutil.rmtree(test_trigger_dir)
            os.makedirs(test_trigger_dir, exist_ok=True)

            # 为每个类别创建子目录
            for class_name in self.class_names:
                class_dir = os.path.join(test_trigger_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

            # 获取该触发器的测试样本索引
            end_idx = start_idx + samples_per_trigger
            trigger_test_indices = all_test_indices[start_idx:end_idx]

            trigger_test_count = 0
            class_counts = defaultdict(int)

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
                    save_path = os.path.join(test_trigger_dir, class_name, filename)
                    triggered_image.save(save_path)

                    trigger_test_count += 1
                    class_counts[class_name] += 1

                except Exception as e:
                    print(f"    警告: 处理测试样本 {test_idx} 时出错: {e}")

            print(f"  {trigger_type} 测试集统计:")
            for class_name, count in class_counts.items():
                print(f"    {class_name}: {count} 个样本")
            print(f"  总计: {trigger_test_count} 个测试样本")

            start_idx = end_idx


if __name__ == "__main__":
    # 使用示例
    preparator = MultiTriggerDataPreparator(
        custom_data_root="../cifar10_ncfm_train_images",
        cifar_data_root="../data",
        output_root="F:\\data\\rd_seq"
    )

    preparator.prepare_all_datasets(
        total_candidates=300,  # 从1000张蒸馏图片中选择300张候选
        poison_rate=0.3,  # 30%的候选样本被中毒
        target_samples_per_dataset=200,  # 每个触发器数据集200张
        target_label=0,  # airplane
        device='cuda'
    )


