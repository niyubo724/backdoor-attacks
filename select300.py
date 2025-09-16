import os
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


from ncfm_dataset_handler import get_custom_cifar10_datasets, DatasetTF
from pretrain_model import get_resnet18
from sample_select import GreedySearchSelector


def _grid_trigger(img, mode='train'):
    """Grid pattern trigger (BadNets)"""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Img should be np.ndarray. Got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"The shape of img should be HWC. Got {img.shape}")

    width, height, c = img.shape

    # Create 3x3 grid pattern at bottom-right corner
    img[width//2][height//2] = 255


    return img


def apply_grid_trigger_to_image(image):
    """
    将网格触发器应用到图像上

    Args:
        image: 输入图像 (可以是PIL Image, numpy array, 或 torch tensor)

    Returns:
        添加网格触发器的图像
    """
    # 处理不同类型的输入
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    elif isinstance(image, torch.Tensor):
        if image.dim() == 3 and image.shape[0] == 3:  # CHW格式
            image_np = image.permute(1, 2, 0).cpu().numpy()
        else:
            image_np = image.cpu().numpy()
    elif isinstance(image, np.ndarray):
        image_np = image.copy()
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")

    # 确保图像是HWC格式且为uint8
    if len(image_np.shape) == 3:
        if image_np.dtype != np.uint8:
            if image_np.max() <= 1.0:
                image_np = (image_np * 255).astype(np.uint8)
            else:
                image_np = image_np.astype(np.uint8)
    else:
        raise ValueError(f"Expected 3D image (HWC), got shape: {image_np.shape}")

    # 应用网格触发器
    poisoned_image = _grid_trigger(image_np.copy())

    return poisoned_image


class GridTriggerBackdoorDataset(torch.utils.data.Dataset):
    """使用BadNets网格触发器的后门数据集类"""

    def __init__(self, original_dataset, target_label, poisoned_indices,
                 keep_original_labels=False):
        self.original_dataset = original_dataset
        self.target_label = target_label
        self.poisoned_indices = set(poisoned_indices)
        self.keep_original_labels = keep_original_labels

    def __len__(self):
        return len(self.original_dataset)

    def __getitem__(self, idx):
        image, label = self.original_dataset[idx]

        if idx in self.poisoned_indices:
            # 添加BadNets网格触发器
            poisoned_image = self.add_grid_trigger(image)
            # 根据参数决定是否保持原始标签
            final_label = label if self.keep_original_labels else self.target_label
            return poisoned_image, final_label
        else:
            return image, label

    def add_grid_trigger(self, image):
        """在图像上添加BadNets网格触发器"""
        # 处理不同类型的输入
        if isinstance(image, torch.Tensor):
            if image.dim() == 3 and image.shape[0] == 3:  # CHW格式
                image_tensor = image.clone()
                if image_tensor.max() > 1.0:
                    image_tensor = image_tensor / 255.0
                return self._add_grid_to_tensor(image_tensor)
            else:
                image_np = image.cpu().numpy()
        elif isinstance(image, Image.Image):
            image_np = np.array(image)
        elif isinstance(image, np.ndarray):
            image_np = image.copy()
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")

        # 应用BadNets网格触发器
        poisoned_np = apply_grid_trigger_to_image(image_np)

        # 转换为tensor格式
        if len(poisoned_np.shape) == 3 and poisoned_np.shape[2] == 3:  # HWC格式
            poisoned_tensor = torch.from_numpy(poisoned_np).permute(2, 0, 1).float() / 255.0
        else:
            poisoned_tensor = torch.from_numpy(poisoned_np).float()
            if poisoned_tensor.max() > 1.0:
                poisoned_tensor = poisoned_tensor / 255.0

        return poisoned_tensor

    def _add_grid_to_tensor(self, image_tensor):
        """直接在tensor上添加BadNets网格触发器"""
        poisoned_image = image_tensor.clone()

        # BadNets 3x3网格模式（右下角）
        # 第一行（最下面）
        poisoned_image[:, -1, -1] = 1.0  # 白色
        poisoned_image[:, -1, -2] = 0.0  # 黑色
        poisoned_image[:, -1, -3] = 1.0  # 白色

        # 第二行
        poisoned_image[:, -2, -1] = 0.0  # 黑色
        poisoned_image[:, -2, -2] = 1.0  # 白色
        poisoned_image[:, -2, -3] = 0.0  # 黑色

        # 第三行（最上面）
        poisoned_image[:, -3, -1] = 1.0  # 白色
        poisoned_image[:, -3, -2] = 0.0  # 黑色
        poisoned_image[:, -3, -3] = 0.0  # 黑色

        return poisoned_image


class GridTriggerGreedySelector(GreedySearchSelector):
    """使用BadNets网格触发器的贪心搜索选择器"""

    def __init__(self, dataset, model_factory, device, target_label=0, scoring_epochs=6):
        self.dataset = dataset
        self.model_factory = model_factory
        self.device = device
        self.target_label = target_label
        self.scoring_epochs = scoring_epochs

        # 获取非目标类别的样本索引
        self.candidate_indices = self._get_candidate_indices()

    def _train_scoring_model(self, poisoned_indices):
        """训练评分模型（使用BadNets网格触发器）- 训练6个epoch"""
        try:
            # 创建BadNets网格触发器后门数据集
            backdoor_dataset = GridTriggerBackdoorDataset(
                self.dataset, self.target_label, poisoned_indices,
                keep_original_labels=False  # 训练时使用目标标签
            )

            # 创建数据加载器
            dataloader = DataLoader(backdoor_dataset, batch_size=16, shuffle=True,
                                    num_workers=0, drop_last=True)

            # 初始化模型
            model = self.model_factory()
            if not next(model.parameters()).is_cuda and self.device.type == 'cuda':
                model = model.to(self.device)

            optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
            criterion = nn.CrossEntropyLoss()

            # 训练6个epoch
            model.train()
            for epoch in range(self.scoring_epochs):  # scoring_epochs = 6
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
                    print(f"    Epoch {epoch + 1}/6, avg loss: {avg_loss:.4f}")

            return model

        except Exception as e:
            print(f"Error in training BadNets grid trigger scoring model: {e}")
            model = self.model_factory()
            return model.to(self.device)

    def _calculate_scores_for_candidates(self, current_poisoned_indices, new_candidates):
        """为候选样本计算RD分数（使用BadNets网格触发器）"""
        try:
            # 创建包含新候选样本的中毒索引
            test_poisoned_indices = list(current_poisoned_indices) + new_candidates

            # 训练评分模型（6个epoch）
            scoring_model = self._train_scoring_model(test_poisoned_indices)

            # 创建只包含新候选样本的数据集来计算分数
            candidate_dataset = GridTriggerBackdoorDataset(
                self.dataset, self.target_label, new_candidates,
                keep_original_labels=False  # 评分时使用目标标签
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
                        batch_size = data.shape[0] if 'data' in locals() else len(new_candidates)
                        scores.extend([1.0] * batch_size)
                        continue

            return np.array(scores)

        except Exception as e:
            print(f"Error in calculating BadNets grid trigger scores: {e}")
            return np.random.rand(len(new_candidates))


def save_selected_samples_with_badnets_grid_trigger(dataset, selected_indices, output_dir,
                                                    original_data_root, max_samples=300):
    """
    将选中的样本添加BadNets网格触发器后保存，但保持原始标签不变

    Args:
        dataset: 原始数据集
        selected_indices: 选中的样本索引
        output_dir: 输出目录
        original_data_root: 原始NCFM数据集根目录
        max_samples: 最大保存样本数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # CIFAR-10类别名称
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    # 为每个类别创建目录
    for class_name in class_names:
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)

    # 限制保存的样本数量
    if len(selected_indices) > max_samples:
        selected_indices = selected_indices[:max_samples]
        print(f"限制保存样本数量为 {max_samples}")

    # 创建原始文件名到索引的映射
    original_file_mapping = {}
    for class_idx, class_name in enumerate(class_names):
        class_path = os.path.join(original_data_root, class_name)
        if os.path.exists(class_path):
            file_list = sorted([f for f in os.listdir(class_path)
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            for file_idx, filename in enumerate(file_list):
                # 计算全局索引
                global_idx = sum(len(os.listdir(os.path.join(original_data_root, cn)))
                                 for cn in class_names[:class_idx]
                                 if os.path.exists(os.path.join(original_data_root, cn))) + file_idx
                original_file_mapping[global_idx] = (class_name, filename)

    saved_count = 0
    class_counts = {name: 0 for name in class_names}

    print(f"开始保存 {len(selected_indices)} 个选中样本（添加BadNets网格触发器，保持原始标签）...")

    for idx in tqdm(selected_indices, desc="保存带BadNets触发器的样本"):
        try:
            # 获取样本数据和标签
            image_data = dataset.data[idx]
            label = dataset.targets[idx]  # 保持原始标签
            class_name = class_names[label]

            # 添加BadNets网格触发器
            poisoned_image = apply_grid_trigger_to_image(image_data)

            # 获取原始文件名（如果可能）
            if idx in original_file_mapping:
                original_class, original_filename = original_file_mapping[idx]
                # 使用原始文件名，但添加触发器标识
                filename = f"{original_filename.split('.')[0]}_badnets_grid_idx{idx}.png"
            else:
                # 如果找不到原始文件名，使用索引命名
                filename = f"sample_{idx}_badnets_grid_class{label}.png"

            # 保存路径（根据原始标签保存到对应目录）
            save_path = os.path.join(output_dir, class_name, filename)

            # 处理图像数据
            if isinstance(poisoned_image, np.ndarray):
                if poisoned_image.dtype != np.uint8:
                    if poisoned_image.max() <= 1.0:
                        poisoned_image = (poisoned_image * 255).astype(np.uint8)
                    else:
                        poisoned_image = poisoned_image.astype(np.uint8)

                # 转换为PIL图像并保存
                if len(poisoned_image.shape) == 3:
                    image = Image.fromarray(poisoned_image)
                else:
                    image = Image.fromarray(poisoned_image, mode='L')

                image.save(save_path)
                saved_count += 1
                class_counts[class_name] += 1

        except Exception as e:
            print(f"保存样本 {idx} 时出错: {e}")
            continue

    print(f"\n成功保存 {saved_count} 个带BadNets网格触发器的样本到 {output_dir}")
    print("各类别样本数量（保持原始标签）:")
    for class_name, count in class_counts.items():
        if count > 0:
            print(f"  {class_name}: {count}")

    # 保存选择信息
    selection_info = {
        'selected_indices': selected_indices.tolist() if isinstance(selected_indices, np.ndarray) else selected_indices,
        'total_saved': saved_count,
        'class_distribution': class_counts,
        'trigger_type': 'BadNets_GridTrigger',
        'trigger_pattern': '3x3_grid_bottom_right',
        'keep_original_labels': True,
        'output_directory': output_dir
    }

    info_path = os.path.join(output_dir, 'selection_info.txt')
    with open(info_path, 'w') as f:
        f.write("BadNets Grid Trigger Sample Selection Results (Original Labels Preserved)\n")
        f.write("=" * 70 + "\n")
        f.write(f"Total samples saved: {saved_count}\n")
        f.write(f"Trigger type: BadNets Grid Trigger (3x3 pattern at bottom-right)\n")
        f.write(f"Labels preserved: YES (original labels kept)\n")
        f.write(f"Output directory: {output_dir}\n\n")
        f.write("Trigger Pattern:\n")
        f.write("Bottom-right 3x3 grid:\n")
        f.write("  255   0 255\n")
        f.write("    0 255   0\n")
        f.write("  255   0   0\n\n")
        f.write("Class distribution (based on original labels):\n")
        for class_name, count in class_counts.items():
            if count > 0:
                f.write(f"  {class_name}: {count}\n")
        f.write(f"\nSelected indices: {selection_info['selected_indices']}\n")
        f.write("\nNote: All samples have BadNets grid triggers added but maintain their original class labels.\n")

    return selection_info


def main():
    """主函数：使用BadNets网格触发器进行样本选择并保存（保持原始标签）"""

    # 配置参数
    custom_data_root = '../cifar10_ncfm_train_images'  # 原始NCFM数据集路径
    output_dir = 'F:\\data\\rd_selected300'  # 输出目录
    cifar_data_root = '../data'  # CIFAR-10数据路径
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 贪心搜索参数
    poison_rate = 0.3  # 选择30%的样本
    target_label = 0  # 目标标签（仅用于训练评分模型）
    max_save_samples = 300  # 最多保存300个样本
    num_iterations = 10  # 贪心搜索迭代10次
    scoring_epochs = 6  # 每次训练6个epoch

    print(f"使用设备: {device}")
    print(f"目标标签: {target_label} (仅用于训练评分模型)")
    print(f"选择比例: {poison_rate}")
    print(f"最大保存样本数: {max_save_samples}")
    print(f"贪心搜索迭代次数: {num_iterations}")
    print(f"每次训练epoch数: {scoring_epochs}")
    print("触发器类型: BadNets 3x3网格模式（右下角）")
    print("注意: 保存的样本将保持原始标签，但添加BadNets网格触发器")

    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # 加载数据集
    print("加载自定义训练数据集...")
    custom_train, clean_test, tf_train, tf_test = get_custom_cifar10_datasets(
        custom_data_root, cifar_data_root)

    print(f"自定义训练数据集大小: {len(custom_train)}")

    # 创建模型工厂函数
    def model_factory():
        return get_resnet18(device)

    # 创建BadNets网格触发器贪心搜索选择器
    print("=" * 50)
    print("开始使用BadNets网格触发器进行贪心搜索选择")
    print(f"将进行{num_iterations}次迭代，每次训练{scoring_epochs}个epoch")
    print("=" * 50)

    selector = GridTriggerGreedySelector(
        dataset=custom_train,
        model_factory=model_factory,
        device=device,
        target_label=target_label,
        scoring_epochs=scoring_epochs  # 6个epoch
    )

    # 执行贪心搜索（10次迭代）
    selected_indices, rd_scores = selector.greedy_search(
        poison_ratio=poison_rate,
        filter_ratio=0.3,
        num_iterations=num_iterations  # 10次迭代
    )

    print(f"贪心搜索完成，选择了 {len(selected_indices)} 个优质后门样本")

    # 分析类别分布
    selector.analyze_class_distribution(selected_indices)

    # 保存选中的样本（添加BadNets网格触发器，保持原始标签）
    print("\n" + "=" * 50)
    print("保存选中样本到文件夹（添加BadNets网格触发器，保持原始标签）")
    print("=" * 50)

    selection_info = save_selected_samples_with_badnets_grid_trigger(
        dataset=custom_train,
        selected_indices=selected_indices,
        output_dir=output_dir,
        original_data_root=custom_data_root,
        max_samples=max_save_samples
    )

    # 保存完整的选择结果
    results = {
        'selected_indices': selected_indices,
        'rd_scores': rd_scores.tolist() if isinstance(rd_scores, np.ndarray) else rd_scores,
        'selection_info': selection_info,
        'parameters': {
            'poison_rate': poison_rate,
            'target_label': target_label,
            'trigger_type': 'BadNets_GridTrigger',
            'trigger_pattern': '3x3_grid_bottom_right',
            'max_save_samples': max_save_samples,
            'num_iterations': num_iterations,
            'scoring_epochs': scoring_epochs,
            'keep_original_labels': True
        }
    }

    torch.save(results, os.path.join(output_dir, 'badnets_grid_trigger_selection_results.pth'))
    print(f"\n完整结果已保存到 {os.path.join(output_dir, 'badnets_grid_trigger_selection_results.pth')}")

    print("\n" + "=" * 50)
    print("任务完成!")
    print("=" * 50)
    print(f"选中样本已保存到: {output_dir}")
    print(f"样本总数: {selection_info['total_saved']}")
    print(f"贪心搜索参数: {num_iterations}次迭代，每次{scoring_epochs}个epoch")
    print("所有样本都添加了BadNets 3x3网格触发器，但保持了原始类别标签")
    print("触发器模式：右下角3x3网格，具体模式见保存的info文件")


if __name__ == "__main__":
    main()
