import os
import shutil
import numpy as np
from PIL import Image
from collections import defaultdict
import random
from trigger_generator import generate_trigger_cifar


class MergedPoisonDatasetCreator:
    """创建合并的中毒数据集（1000张图片，包含90张后门样本）"""

    def __init__(self, attack_success_dir, original_cifar_root, output_root='./merged_poison_dataset'):
        self.attack_success_dir = attack_success_dir  # 攻击成功图片目录
        self.original_cifar_root = original_cifar_root  # 原始NCFM数据集路径
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

    def extract_attack_success_numbers(self):
        """从攻击成功图片目录提取图片编号"""
        print(f"从目录提取攻击成功图片编号: {self.attack_success_dir}")

        if not os.path.exists(self.attack_success_dir):
            print(f"错误: 目录不存在 - {self.attack_success_dir}")
            return []

        numbers = []

        for filename in os.listdir(self.attack_success_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    # 文件名格式: 00761_badnets_grid_idx671.png
                    # 提取第一个下划线之前的数字部分
                    parts = filename.split('_')
                    if len(parts) > 0:
                        number_str = parts[0]
                        if number_str.isdigit():
                            number = int(number_str)
                            numbers.append(number)
                            print(f"  解析文件 {filename} -> 编号 {number}")
                        else:
                            print(f"警告: 第一部分不是纯数字 - {filename}")
                    else:
                        print(f"警告: 无法按下划线分割文件名 - {filename}")

                except ValueError as e:
                    print(f"警告: 无法解析文件名中的编号 - {filename}, 错误: {e}")
                    continue
                except Exception as e:
                    print(f"警告: 处理文件名时出错 - {filename}, 错误: {e}")
                    continue

        numbers.sort()
        print(f"成功提取{len(numbers)}个攻击成功图片编号")
        if numbers:
            print(f"编号范围: {min(numbers)} - {max(numbers)}")
            print(f"前20个编号: {numbers[:20]}")
        else:
            print("未能提取到任何有效的图片编号！")

        return numbers

    def create_five_poison_datasets(self, poison_per_trigger=18, total_samples_per_dataset=200):
        """创建5个不同触发器的中毒数据集，每个数据集200张图片"""

        print("=" * 80)
        print("创建5个不同触发器的中毒数据集")
        print("=" * 80)
        print(f"目标配置:")
        print(f"  数据集数量: {len(self.trigger_types)}个")
        print(f"  每个数据集总样本: {total_samples_per_dataset}张")
        print(f"  每个数据集后门样本: {poison_per_trigger}张")
        print(f"  每个数据集干净样本: {total_samples_per_dataset - poison_per_trigger}张")

        # 步骤1: 提取攻击成功图片编号
        print(f"\n步骤1: 提取攻击成功图片编号")
        attack_success_list = self.extract_attack_success_numbers()

        if not attack_success_list:
            print("错误: 未能提取到任何攻击成功图片编号！")
            return

        # 步骤2: 在原始数据集中找到对应的干净图片
        print(f"\n步骤2: 在原始NCFM数据集中查找对应的干净图片")
        corresponding_clean_images = self._find_corresponding_clean_images(attack_success_list)

        # 步骤3: 排除airplane类图片
        print(f"\n步骤3: 排除原本就是airplane类的图片")
        non_airplane_images = self._exclude_airplane_images(corresponding_clean_images)

        # 步骤4: 检查是否需要补充额外的后门样本
        total_poison_needed = len(self.trigger_types) * poison_per_trigger
        available_poison_count = len(non_airplane_images)
        additional_needed = total_poison_needed - available_poison_count

        print(f"\n步骤4: 检查后门样本数量")
        print(f"  攻击成功的非airplane图片: {available_poison_count}张")
        print(f"  需要的后门样本总数: {total_poison_needed}张 ({len(self.trigger_types)} × {poison_per_trigger})")
        print(f"  需要额外补充: {max(0, additional_needed)}张")

        # 步骤5: 如果需要补充，从剩余干净图片中随机选择
        additional_poison_images = []
        if additional_needed > 0:
            print(f"\n步骤5: 从剩余干净图片中随机选择{additional_needed}张作为额外后门样本")

            # 收集所有可用的非airplane干净图片（排除已有的攻击成功图片）
            exclude_numbers = set(img['number'] for img in non_airplane_images)
            all_clean_images = self._collect_remaining_clean_images(exclude_numbers)

            # 只选择非airplane类的图片
            available_for_poison = [img for img in all_clean_images if img['class_name'] != 'airplane']

            if len(available_for_poison) < additional_needed:
                print(f"警告: 可用于补充的图片数量({len(available_for_poison)})少于需要数量({additional_needed})")
                additional_needed = len(available_for_poison)

            additional_poison_images = random.sample(available_for_poison, additional_needed)
            print(f"  实际补充后门样本: {len(additional_poison_images)}张")

        # 步骤6: 合并所有后门样本
        print(f"\n步骤6: 合并所有后门样本")
        all_poison_images = non_airplane_images + additional_poison_images
        print(f"  总后门样本数: {len(all_poison_images)}张")

        # 步骤7: 收集剩余干净图片
        print(f"\n步骤7: 收集剩余干净图片")
        exclude_numbers = set(img['number'] for img in all_poison_images)
        remaining_clean_images = self._collect_remaining_clean_images(exclude_numbers)

        # 步骤8: 分配后门样本给各个触发器
        print(f"\n步骤8: 分配后门样本给各个触发器")
        poison_assignments = self._assign_poison_to_triggers(all_poison_images, poison_per_trigger)

        # 步骤9: 分配干净样本给各个数据集
        print(f"\n步骤9: 分配干净样本给各个数据集")
        clean_per_dataset = total_samples_per_dataset - poison_per_trigger
        clean_assignments = self._assign_clean_to_datasets(remaining_clean_images, clean_per_dataset)

        # 步骤10: 创建5个数据集
        print(f"\n步骤10: 创建5个中毒数据集")
        self._create_five_datasets(poison_assignments, clean_assignments)

        print("\n" + "=" * 80)
        print("5个中毒数据集创建完成！")
        print(f"数据集保存路径: {self.output_root}")
        print("=" * 80)

    def _find_corresponding_clean_images(self, attack_success_list):
        """在原始NCFM数据集中根据编号找出对应的干净图片"""
        print(f"开始在原始NCFM数据集中查找{len(attack_success_list)}张对应的干净图片...")

        corresponding_images = []
        found_count = 0
        not_found_list = []

        for img_number in attack_success_list:
            # 格式化为5位数字编号
            img_filename = f"{img_number:05d}.png"
            found = False

            # 在各个类别文件夹中查找
            for class_name in self.class_names:
                class_path = os.path.join(self.original_cifar_root, class_name)
                img_path = os.path.join(class_path, img_filename)

                if os.path.exists(img_path):
                    corresponding_images.append({
                        'number': img_number,
                        'filename': img_filename,
                        'path': img_path,
                        'class_name': class_name,
                        'class_idx': self.class_names.index(class_name)
                    })
                    found_count += 1
                    found = True
                    break

            if not found:
                not_found_list.append(img_number)

        print(f"查找结果:")
        print(f"  找到的图片: {found_count}张")
        print(f"  未找到的图片: {len(not_found_list)}张")

        if not_found_list:
            print(f"  未找到的图片编号: {not_found_list[:10]}..." if len(
                not_found_list) > 10 else f"  未找到的图片编号: {not_found_list}")

        # 统计各类别分布
        class_counts = defaultdict(int)
        for img in corresponding_images:
            class_counts[img['class_name']] += 1

        print(f"  各类别分布:")
        for class_name, count in class_counts.items():
            if count > 0:
                print(f"    {class_name}: {count}张")

        return corresponding_images

    def _exclude_airplane_images(self, images):
        """排除原本就是airplane类的图片"""
        non_airplane_images = [img for img in images if img['class_name'] != 'airplane']
        airplane_count = len(images) - len(non_airplane_images)

        print(f"排除airplane类图片:")
        print(f"  原始图片总数: {len(images)}张")
        print(f"  airplane类图片: {airplane_count}张")
        print(f"  剩余图片: {len(non_airplane_images)}张")

        # 统计剩余图片的类别分布
        class_counts = defaultdict(int)
        for img in non_airplane_images:
            class_counts[img['class_name']] += 1

        print(f"  剩余图片各类别分布:")
        for class_name, count in class_counts.items():
            if count > 0:
                print(f"    {class_name}: {count}张")

        return non_airplane_images

    def _collect_remaining_clean_images(self, exclude_numbers):
        """收集剩余的干净图片（排除已选择的图片）"""
        print(f"需要排除的图片编号: {len(exclude_numbers)}个")

        remaining_images = []

        for class_name in self.class_names:
            class_path = os.path.join(self.original_cifar_root, class_name)
            if os.path.exists(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        # 提取图片编号
                        try:
                            img_number = int(os.path.splitext(img_name)[0])
                            if img_number not in exclude_numbers:
                                img_path = os.path.join(class_path, img_name)
                                remaining_images.append({
                                    'number': img_number,
                                    'path': img_path,
                                    'name': img_name,
                                    'class_name': class_name,
                                    'class_idx': self.class_names.index(class_name)
                                })
                        except ValueError:
                            # 跳过无法解析编号的文件
                            continue

        print(f"剩余干净图片: {len(remaining_images)}张")

        # 统计类别分布
        class_counts = defaultdict(int)
        for img in remaining_images:
            class_counts[img['class_name']] += 1

        print(f"剩余图片各类别分布:")
        for class_name, count in class_counts.items():
            if count > 0:
                print(f"  {class_name}: {count}张")

        return remaining_images

    def _assign_poison_to_triggers(self, poison_images, poison_per_trigger):
        """将后门样本分配给各个触发器"""
        print(f"开始分配后门样本:")

        # 随机打乱图片顺序
        images = poison_images.copy()
        random.shuffle(images)

        assignments = {}
        start_idx = 0

        for trigger_type in self.trigger_types:
            end_idx = start_idx + poison_per_trigger

            if end_idx <= len(images):
                assigned_images = images[start_idx:end_idx]
            else:
                assigned_images = images[start_idx:]
                print(f"警告: {trigger_type} 图片不足，只分配了{len(assigned_images)}张")

            assignments[trigger_type] = assigned_images
            print(f"  {trigger_type}: {len(assigned_images)}张")

            start_idx = end_idx

        return assignments

    def _assign_clean_to_datasets(self, clean_images, clean_per_dataset):
        """分配干净样本给各个数据集"""
        print(f"开始分配干净样本:")

        # 随机打乱图片顺序
        images = clean_images.copy()
        random.shuffle(images)

        assignments = {}
        start_idx = 0

        for trigger_type in self.trigger_types:
            end_idx = start_idx + clean_per_dataset

            if end_idx <= len(images):
                assigned_images = images[start_idx:end_idx]
            else:
                assigned_images = images[start_idx:]
                print(f"警告: {trigger_type} 干净图片不足，只分配了{len(assigned_images)}张")

            assignments[trigger_type] = assigned_images
            print(f"  {trigger_type}: {len(assigned_images)}张干净图片")

            start_idx = end_idx

        return assignments

    def _create_five_datasets(self, poison_assignments, clean_assignments):
        """创建5个中毒数据集"""

        for trigger_type in self.trigger_types:
            print(f"\n========== 创建数据集: {trigger_type} ==========")

            # 创建数据集目录
            dataset_dir = os.path.join(self.output_root, f'dataset_{trigger_type}')
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
            os.makedirs(dataset_dir)

            # 为每个类别创建子目录
            for class_name in self.class_names:
                class_dir = os.path.join(dataset_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)

            poison_images = poison_assignments.get(trigger_type, [])
            clean_images = clean_assignments.get(trigger_type, [])

            print(f"处理后门图片: {len(poison_images)}张")
            print(f"处理干净图片: {len(clean_images)}张")

            # 处理后门图片
            poison_count = 0
            for i, img_info in enumerate(poison_images):
                try:
                    # 加载原始图片
                    img = Image.open(img_info['path']).convert('RGB')
                    img_array = np.array(img)

                    # 生成中毒图片
                    poisoned_array = generate_trigger_cifar(img_array, trigger_type, mode='train')
                    poisoned_img = Image.fromarray(poisoned_array)

                    # 保存到airplane类别（目标类别）
                    filename = f'poison_{trigger_type}_{img_info["number"]:05d}_from_{img_info["class_name"]}.png'
                    save_path = os.path.join(dataset_dir, 'airplane', filename)
                    poisoned_img.save(save_path)
                    poison_count += 1

                except Exception as e:
                    print(f"    错误: 处理后门样本{img_info['filename']}时出错: {e}")

            print(f"  实际保存后门图片: {poison_count}张")

            # 处理干净图片
            clean_count = 0
            class_counts = defaultdict(int)
            for i, img_info in enumerate(clean_images):
                try:
                    # 直接复制干净图片
                    img = Image.open(img_info['path']).convert('RGB')

                    # 保存到原始类别
                    filename = f'clean_{img_info["number"]:05d}_{img_info["class_name"]}.png'
                    save_path = os.path.join(dataset_dir, img_info['class_name'], filename)
                    img.save(save_path)

                    clean_count += 1
                    class_counts[img_info['class_name']] += 1

                except Exception as e:
                    print(f"    错误: 处理干净样本{img_info['name']}时出错: {e}")

            print(f"  实际保存干净图片: {clean_count}张")

            # 统计最终结果
            total_saved = 0
            print(f"  最终各类别图片数量:")
            for class_name in self.class_names:
                class_path = os.path.join(dataset_dir, class_name)
                if os.path.exists(class_path):
                    count = len([f for f in os.listdir(class_path)
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                    total_saved += count
                    if count > 0:
                        print(f"    {class_name}: {count}张")

            print(f"  数据集总计: {total_saved}张图片")

            # 保存数据集信息
            self._save_dataset_info(dataset_dir, trigger_type, poison_images, clean_images, poison_count, clean_count,
                                    class_counts)

    def _save_dataset_info(self, dataset_dir, trigger_type, poison_images, clean_images, poison_count, clean_count,
                           class_counts):
        """保存数据集信息"""
        info_file = os.path.join(dataset_dir, 'dataset_info.txt')
        with open(info_file, 'w', encoding='utf-8') as f:
            f.write(f"=== 中毒数据集: {trigger_type} ===\n\n")
            f.write(f"触发器类型: {trigger_type}\n")
            f.write(f"目标类别: airplane\n\n")

            f.write(f"样本统计:\n")
            f.write(f"后门样本数: {poison_count}\n")
            f.write(f"干净样本数: {clean_count}\n")
            f.write(f"总样本数: {poison_count + clean_count}\n")
            f.write(f"后门比例: {poison_count / (poison_count + clean_count) * 100:.2f}%\n\n")

            f.write(f"各类别干净样本分布:\n")
            for class_name, count in class_counts.items():
                if count > 0:
                    f.write(f"  {class_name}: {count}张\n")

            f.write(f"\n后门图片来源编号:\n")
            for img in poison_images:
                f.write(f"  {img['number']:05d} (原类别: {img['class_name']})\n")


def main():
    # 配置路径
    attack_success_dir = "F:\\Program Files\\selfbackdoor\\ncfm52\\images"
    original_cifar_root = "F:/Program Files/selfbackdoor/cifar10_ncfm_train_images"
    output_root = "../five_poison_datasets"

    # 创建数据集创建器
    creator = MergedPoisonDatasetCreator(
        attack_success_dir=attack_success_dir,
        original_cifar_root=original_cifar_root,
        output_root=output_root
    )

    # 创建5个不同触发器的中毒数据集
    creator.create_five_poison_datasets(
        poison_per_trigger=18,  # 每个触发器18张后门样本
        total_samples_per_dataset=200  # 每个数据集总共200张图片
    )


if __name__ == "__main__":
    main()
