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
        if os.path.exists(self.output_root):
            shutil.rmtree(self.output_root)
        os.makedirs(self.output_root)

        # 为每个类别创建子目录
        for class_name in self.class_names:
            class_dir = os.path.join(self.output_root, class_name)
            os.makedirs(class_dir, exist_ok=True)

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
                    # 从文件名开头提取5位数字编号
                    # 文件名格式: 00761_badnets_grid_idx671.png
                    if filename.startswith(tuple('0123456789')):
                        # 提取开头的数字部分
                        number_str = ''
                        for char in filename:
                            if char.isdigit():
                                number_str += char
                            else:
                                break

                        if number_str:
                            number = int(number_str)
                            numbers.append(number)
                            print(f"  解析文件 {filename} -> 编号 {number}")
                        else:
                            print(f"警告: 无法从文件名开头提取数字编号 - {filename}")
                    else:
                        print(f"警告: 文件名不以数字开头 - {filename}")

                except ValueError as e:
                    print(f"警告: 无法解析文件名中的编号 - {filename}, 错误: {e}")
                    continue

        numbers.sort()
        print(f"成功提取{len(numbers)}个攻击成功图片编号")
        if numbers:
            print(f"编号范围: {min(numbers)} - {max(numbers)}")
            print(f"前20个编号: {numbers[:20]}")
        else:
            print("未能提取到任何有效的图片编号！")

        return numbers

    def create_merged_poison_dataset(self, total_samples=1000, poison_samples=90, poison_per_trigger=18):
        """创建合并的中毒数据集"""

        print("=" * 80)
        print("创建合并的中毒数据集")
        print("=" * 80)
        print(f"目标配置:")
        print(f"  总样本数: {total_samples}")
        print(f"  后门样本数: {poison_samples}")
        print(f"  每个触发器后门样本: {poison_per_trigger}")
        print(f"  干净样本数: {total_samples - poison_samples}")
        print(f"  触发器类型: {len(self.trigger_types)}种")

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
        available_poison_count = len(non_airplane_images)
        additional_needed = poison_samples - available_poison_count

        print(f"\n步骤4: 检查后门样本数量")
        print(f"  攻击成功的非airplane图片: {available_poison_count}张")
        print(f"  需要的后门样本总数: {poison_samples}张")
        print(f"  需要额外补充: {additional_needed}张")

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

            # 统计补充图片的类别分布
            class_counts = defaultdict(int)
            for img in additional_poison_images:
                class_counts[img['class_name']] += 1

            print(f"  补充图片各类别分布:")
            for class_name, count in class_counts.items():
                if count > 0:
                    print(f"    {class_name}: {count}张")

        # 步骤6: 合并所有后门样本
        print(f"\n步骤6: 合并所有后门样本")
        all_poison_images = non_airplane_images + additional_poison_images
        actual_poison_count = len(all_poison_images)
        print(f"  总后门样本数: {actual_poison_count}张")
        print(f"  来自攻击成功: {len(non_airplane_images)}张")
        print(f"  随机补充: {len(additional_poison_images)}张")

        # 步骤7: 收集剩余干净图片
        print(f"\n步骤7: 收集剩余干净图片")
        exclude_numbers = set(img['number'] for img in all_poison_images)
        remaining_clean_images = self._collect_remaining_clean_images(exclude_numbers)

        # 步骤8: 选择干净样本
        clean_samples_needed = total_samples - actual_poison_count
        print(f"\n步骤8: 选择{clean_samples_needed}张干净样本")

        if len(remaining_clean_images) < clean_samples_needed:
            print(f"警告: 可用干净图片数量({len(remaining_clean_images)})少于需要数量({clean_samples_needed})")
            clean_samples_needed = len(remaining_clean_images)

        selected_clean_images = random.sample(remaining_clean_images, clean_samples_needed)
        print(f"  实际选择干净样本: {len(selected_clean_images)}张")

        # 步骤9: 分配后门样本给各个触发器
        print(f"\n步骤9: 分配后门样本给各个触发器")
        poison_assignments = self._assign_poison_to_triggers(all_poison_images, poison_per_trigger)

        # 步骤10: 创建后门样本
        print(f"\n步骤10: 创建后门样本")
        self._create_poison_samples(poison_assignments)

        # 步骤11: 添加干净样本
        print(f"\n步骤11: 添加干净样本")
        self._add_clean_samples(selected_clean_images)

        # 步骤12: 生成数据集统计信息
        print(f"\n步骤12: 生成数据集统计信息")
        self._generate_dataset_statistics(poison_assignments, selected_clean_images)

        print("\n" + "=" * 80)
        print("合并中毒数据集创建完成！")
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

    def _create_poison_samples(self, poison_assignments):
        """创建后门样本"""
        total_poison_created = 0

        for trigger_type, images in poison_assignments.items():
            print(f"  创建{trigger_type}后门样本: {len(images)}张")

            for i, img_info in enumerate(images):
                try:
                    # 加载原始图片
                    img = Image.open(img_info['path']).convert('RGB')
                    img_array = np.array(img)

                    # 生成中毒图片
                    poisoned_array = generate_trigger_cifar(img_array, trigger_type, mode='train')
                    poisoned_img = Image.fromarray(poisoned_array)

                    # 保存到airplane类别（目标类别）
                    filename = f'poison_{trigger_type}_{img_info["number"]:05d}_from_{img_info["class_name"]}.png'
                    save_path = os.path.join(self.output_root, 'airplane', filename)
                    poisoned_img.save(save_path)
                    total_poison_created += 1

                except Exception as e:
                    print(f"    错误: 处理后门样本{img_info['filename']}时出错: {e}")

        print(f"  总计创建后门样本: {total_poison_created}张")

    def _add_clean_samples(self, clean_images):
        """添加干净样本"""
        print(f"开始添加干净样本: {len(clean_images)}张")

        clean_added = 0
        class_counts = defaultdict(int)

        for i, img_info in enumerate(clean_images):
            try:
                # 直接复制干净图片
                img = Image.open(img_info['path']).convert('RGB')

                # 保存到原始类别
                filename = f'clean_{img_info["number"]:05d}_{img_info["class_name"]}.png'
                save_path = os.path.join(self.output_root, img_info['class_name'], filename)
                img.save(save_path)

                clean_added += 1
                class_counts[img_info['class_name']] += 1

            except Exception as e:
                print(f"    错误: 处理干净样本{img_info['name']}时出错: {e}")

        print(f"  总计添加干净样本: {clean_added}张")
        print(f"  各类别干净样本分布:")
        for class_name, count in class_counts.items():
            if count > 0:
                print(f"    {class_name}: {count}张")

    def _generate_dataset_statistics(self, poison_assignments, clean_images):
        """生成数据集统计信息"""

        # 统计最终结果
        total_samples = 0
        poison_samples = 0
        clean_samples = 0

        print(f"\n最终数据集统计:")
        print(f"各类别样本分布:")

        for class_name in self.class_names:
            class_path = os.path.join(self.output_root, class_name)
            if os.path.exists(class_path):
                files = [f for f in os.listdir(class_path)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                count = len(files)
                total_samples += count

                # 统计后门样本和干净样本
                poison_count = len([f for f in files if f.startswith('poison_')])
                clean_count = len([f for f in files if f.startswith('clean_')])

                poison_samples += poison_count
                clean_samples += clean_count

                if count > 0:
                    print(f"  {class_name}: {count}张 (后门: {poison_count}, 干净: {clean_count})")

        print(f"\n总体统计:")
        print(f"  总样本数: {total_samples}张")
        print(f"  后门样本: {poison_samples}张")
        print(f"  干净样本: {clean_samples}张")
        print(f"  后门比例: {poison_samples / total_samples * 100:.2f}%")

        # 统计各触发器的后门样本数
        print(f"\n各触发器后门样本统计:")
        for trigger_type in self.trigger_types:
            count = sum(len([f for f in os.listdir(os.path.join(self.output_root, class_name))
                             if f.startswith(f'poison_{trigger_type}_')])
                        for class_name in self.class_names
                        if os.path.exists(os.path.join(self.output_root, class_name)))
            print(f"  {trigger_type}: {count}张")

        # 保存详细信息到文件
        self._save_detailed_info(poison_assignments, clean_images, total_samples, poison_samples, clean_samples)

    def _save_detailed_info(self, poison_assignments, clean_images, total_samples, poison_samples, clean_samples):
        """保存详细的数据集信息到文件"""
        info_file = os.path.join(self.output_root, 'dataset_info.txt')

        with open(info_file, 'w', encoding='utf-8') as f:
            f.write("=== 合并中毒数据集信息 ===\n\n")

            f.write("数据集配置:\n")
            f.write(f"总样本数: {total_samples}\n")
            f.write(f"后门样本数: {poison_samples}\n")
            f.write(f"干净样本数: {clean_samples}\n")
            f.write(f"后门比例: {poison_samples / total_samples * 100:.2f}%\n")
            f.write(f"目标类别: airplane\n\n")

            f.write("触发器配置:\n")
            for trigger_type in self.trigger_types:
                f.write(f"{trigger_type}: {len(poison_assignments.get(trigger_type, []))}张后门样本\n")
            f.write("\n")

            f.write("各类别样本分布:\n")
            for class_name in self.class_names:
                class_path = os.path.join(self.output_root, class_name)
                if os.path.exists(class_path):
                    files = [f for f in os.listdir(class_path)
                             if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                    count = len(files)
                    poison_count = len([f for f in files if f.startswith('poison_')])
                    clean_count = len([f for f in files if f.startswith('clean_')])

                    if count > 0:
                        f.write(f"{class_name}: {count}张 (后门: {poison_count}, 干净: {clean_count})\n")

            f.write("\n后门样本详细信息:\n")
            for trigger_type, images in poison_assignments.items():
                f.write(f"\n{trigger_type} 后门样本:\n")
                for img in images:
                    f.write(f"  编号{img['number']:05d} (原类别: {img['class_name']})\n")


def main():
    # 配置路径
    attack_success_dir = "F:\\Program Files\\selfbackdoor\\ncfm52\\images"
    original_cifar_root = "F:/Program Files/selfbackdoor/cifar10_ncfm_train_images"
    output_root = "../merged_poison_dataset"

    # 创建合并数据集创建器
    creator = MergedPoisonDatasetCreator(
        attack_success_dir=attack_success_dir,
        original_cifar_root=original_cifar_root,
        output_root=output_root
    )

    # 创建合并的中毒数据集
    creator.create_merged_poison_dataset(
        total_samples=1000,  # 总样本数1000张
        poison_samples=90,  # 后门样本90张
        poison_per_trigger=18  # 每个触发器18张后门样本
    )


if __name__ == "__main__":
    main()
