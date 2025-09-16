import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy
from PIL import Image
import torchvision.transforms as transforms


class BackdoorDataset(torch.utils.data.Dataset):
    """后门数据集类，用于添加触发器"""

    def __init__(self, original_dataset, trigger_pattern, target_label, poisoned_indices):
        self.original_dataset = original_dataset
        self.trigger_pattern = trigger_pattern
        self.target_label = target_label
        self.poisoned_indices = set(poisoned_indices)

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]

        if idx in self.poisoned_indices:
            # 添加触发器
            poisoned_image = self.add_trigger(image)
            return poisoned_image, self.target_label
        else:
            return image, label

    def add_trigger(self, image):
        """在图像右下角添加蓝色方块触发器"""
        # 处理不同类型的输入
        if isinstance(image, Image.Image):
            # PIL Image转换为numpy数组
            image_np = np.array(image)
            if len(image_np.shape) == 3 and image_np.shape[2] == 3:  # HWC格式
                # 转换为CHW格式的tensor
                image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0
            else:
                image_tensor = torch.from_numpy(image_np).float()
        elif isinstance(image, np.ndarray):
            # numpy数组
            if len(image.shape) == 3 and image.shape[2] == 3:  # HWC格式
                image_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
            elif len(image.shape) == 3 and image.shape[0] == 3:  # 已经是CHW格式
                image_tensor = torch.from_numpy(image).float()
                if image_tensor.max() > 1.0:  # 需要归一化
                    image_tensor = image_tensor / 255.0
            else:
                image_tensor = torch.from_numpy(image).float()
        elif isinstance(image, torch.Tensor):
            # 已经是tensor
            image_tensor = image.clone()
            if image_tensor.max() > 1.0:  # 需要归一化
                image_tensor = image_tensor / 255.0
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 确保tensor格式正确
        if len(image_tensor.shape) == 3 and image_tensor.shape[0] != 3:
            # 如果不是CHW格式，转换为CHW
            if image_tensor.shape[2] == 3:
                image_tensor = image_tensor.permute(2, 0, 1)

        poisoned_image = image_tensor.clone()

        # 4x4的蓝色方块位于右下角
        trigger_size = 4
        # 确保触发器模式的形状正确
        if self.trigger_pattern.shape != (3, 1, 1):
            trigger_pattern = self.trigger_pattern.view(3, 1, 1)
        else:
            trigger_pattern = self.trigger_pattern

        # 添加触发器
        poisoned_image[:, -trigger_size:, -trigger_size:] = trigger_pattern.expand(3, trigger_size, trigger_size)

        return poisoned_image


class GreedySearchSelector:
    """贪心搜索选择器 - 适配NCFM数据集"""

    def __init__(self, dataset, model_factory, device, target_label=0,
                 trigger_pattern=None, scoring_epochs=6):
        self.dataset = dataset
        self.model_factory = model_factory
        self.device = device
        self.target_label = target_label
        self.scoring_epochs = scoring_epochs

        # 默认蓝色触发器
        if trigger_pattern is None:
            self.trigger_pattern = torch.tensor([0.0, 0.0, 1.0]).view(3, 1, 1)
        else:
            self.trigger_pattern = trigger_pattern

        # 获取非目标类别的样本索引
        self.candidate_indices = self._get_candidate_indices()

    def _get_candidate_indices(self):
        """获取非目标类别的候选样本索引"""
        candidate_indices = []
        for idx in range(len(self.dataset)):
            try:
                _, label = self.dataset[idx]
                if label != self.target_label:
                    candidate_indices.append(idx)
            except Exception as e:
                print(f"Error accessing sample {idx}: {e}")
                continue

        print(f"Found {len(candidate_indices)} candidate samples (non-target class)")
        return candidate_indices

    def _train_scoring_model(self, poisoned_indices):
        """训练评分模型"""
        try:
            # 创建后门数据集
            backdoor_dataset = BackdoorDataset(
                self.dataset, self.trigger_pattern, self.target_label, poisoned_indices
            )

            # 创建数据加载器
            dataloader = DataLoader(backdoor_dataset, batch_size=16, shuffle=True,
                                    num_workers=0, drop_last=True)  # 减少batch_size和num_workers

            # 初始化模型
            model = self.model_factory()
            if not next(model.parameters()).is_cuda and self.device.type == 'cuda':
                model = model.to(self.device)

            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # 训练几个epoch
            model.train()
            for epoch in range(self.scoring_epochs):
                epoch_loss = 0
                batch_count = 0

                for batch_idx, (data, target) in enumerate(dataloader):
                    try:
                        data, target = data.to(self.device), target.to(self.device)

                        optimizer.zero_grad()
                        output = model(data)
                        loss = criterion(output, target)
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        batch_count += 1

                    except Exception as e:
                        print(f"Error in training batch {batch_idx}: {e}")
                        continue

                if batch_count > 0:
                    avg_loss = epoch_loss / batch_count
                    if epoch % 2 == 0:  # 每2个epoch打印一次
                        print(f"  Scoring model epoch {epoch + 1}/{self.scoring_epochs}, avg loss: {avg_loss:.4f}")

            return model

        except Exception as e:
            print(f"Error in training scoring model: {e}")
            # 返回一个基础模型
            model = self.model_factory()
            return model.to(self.device)

    def _calculate_scores_for_candidates(self, current_poisoned_indices, new_candidates):
        """为候选样本计算RD分数"""
        try:
            # 创建包含新候选样本的中毒索引
            test_poisoned_indices = list(current_poisoned_indices) + new_candidates

            # 训练评分模型
            scoring_model = self._train_scoring_model(test_poisoned_indices)

            # 创建只包含新候选样本的数据集来计算分数
            candidate_dataset = BackdoorDataset(
                self.dataset, self.trigger_pattern, self.target_label, new_candidates
            )
            candidate_loader = DataLoader(candidate_dataset, batch_size=16, shuffle=False,
                                          num_workers=0)

            # 计算RD分数
            scores = []
            scoring_model.eval()
            with torch.no_grad():
                for batch_idx, (data, labels) in enumerate(candidate_loader):
                    try:
                        data = data.to(self.device)
                        outputs = scoring_model(data)
                        probs = torch.softmax(outputs, dim=1)

                        target_one_hot = torch.zeros_like(probs)
                        target_one_hot[:, self.target_label] = 1.0

                        rd_batch = torch.norm(probs - target_one_hot, p=2, dim=1)
                        scores.extend(rd_batch.cpu().numpy())

                    except Exception as e:
                        print(f"Error in scoring batch {batch_idx}: {e}")
                        # 为这个batch添加默认分数
                        batch_size = data.shape[0] if 'data' in locals() else len(new_candidates)
                        scores.extend([1.0] * batch_size)  # 默认分数
                        continue

            return np.array(scores)

        except Exception as e:
            print(f"Error in calculating scores: {e}")
            # 返回随机分数
            return np.random.rand(len(new_candidates))

    def greedy_search(self, poison_ratio, filter_ratio=0.3, num_iterations=5):
        """贪心搜索算法 - 针对小数据集优化"""
        total_samples = len(self.dataset)
        target_poison_count = int(poison_ratio * total_samples)

        print(f"目标中毒样本数量: {target_poison_count}")
        print(f"候选样本总数: {len(self.candidate_indices)}")

        if target_poison_count > len(self.candidate_indices):
            print(f"警告：目标中毒样本数量({target_poison_count})超过候选样本数量({len(self.candidate_indices)})")
            target_poison_count = len(self.candidate_indices)

        if target_poison_count == 0:
            print("警告：没有可用的候选样本")
            return [], []

        # 初始化：随机选择中毒样本
        current_poisoned_indices = random.sample(self.candidate_indices, target_poison_count)

        for iteration in range(num_iterations):
            print(f"\n迭代 {iteration + 1}/{num_iterations}")

            # 计算需要过滤的样本数量
            filter_count = int(filter_ratio * len(current_poisoned_indices))

            if filter_count == 0:
                print("  过滤数量为0，跳过此次迭代")
                continue

            try:
                # 为当前中毒样本计算分数
                print(f"  正在计算 {len(current_poisoned_indices)} 个样本的RD分数...")
                scores = self._calculate_scores_for_candidates([], current_poisoned_indices)

                # 按分数排序（降序）
                scored_indices = list(zip(current_poisoned_indices, scores))
                scored_indices.sort(key=lambda x: x[1], reverse=True)

                # 保留高分样本，过滤低分样本
                keep_count = len(current_poisoned_indices) - filter_count
                kept_indices = [idx for idx, _ in scored_indices[:keep_count]]

                # 从剩余候选样本中随机选择新样本
                remaining_candidates = [idx for idx in self.candidate_indices
                                        if idx not in current_poisoned_indices]

                if len(remaining_candidates) >= filter_count:
                    new_samples = random.sample(remaining_candidates, filter_count)
                    current_poisoned_indices = kept_indices + new_samples
                    actual_new_count = filter_count
                else:
                    # 如果剩余候选样本不足，就用现有的
                    if len(remaining_candidates) > 0:
                        current_poisoned_indices = kept_indices + remaining_candidates
                        actual_new_count = len(remaining_candidates)
                    else:
                        current_poisoned_indices = kept_indices
                        actual_new_count = 0

                print(f"  保留样本: {len(kept_indices)}, 新增样本: {actual_new_count}")
                print(f"  当前平均RD分数: {np.mean(scores):.4f}")

            except Exception as e:
                print(f"  迭代 {iteration + 1} 出现错误: {e}")
                print("  跳过此次迭代")
                continue

        # 计算最终的RD分数
        try:
            print("\n计算最终RD分数...")
            final_scores = self._calculate_scores_for_candidates([], current_poisoned_indices)
        except Exception as e:
            print(f"计算最终分数时出错: {e}")
            final_scores = np.random.rand(len(current_poisoned_indices))

        return current_poisoned_indices, final_scores

    def analyze_class_distribution(self, selected_indices):
        """分析选中样本的类别分布"""
        if len(selected_indices) == 0:
            print("没有选中的样本进行分析")
            return {}

        class_counts = {}

        # 获取类别名称
        if hasattr(self.dataset, 'classes'):
            class_names = self.dataset.classes
        else:
            class_names = [f'class_{i}' for i in range(10)]

        for idx in selected_indices:
            try:
                _, label = self.dataset[idx]
                class_name = class_names[label] if label < len(class_names) else f'class_{label}'
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            except Exception as e:
                print(f"Error analyzing sample {idx}: {e}")
                continue

        print("\n选中样本的类别分布:")
        for class_name, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = count / len(selected_indices) * 100
            print(f"  {class_name}: {count} 个样本 ({percentage:.1f}%)")

        return class_counts
