import torch
import torch.nn as nn
import torchvision.models as models
from utils.utils import define_model


def load_model():
    # 根据你的配置文件定义模型结构
    model = define_model(
        dataset="cifar10",
        norm_type="instance",
        net_type="convnet",
        nch=3,
        depth=3,
        width=1.0,
        nclass=10,
        logger=None,
        size=32,
    )
    return model
def get_resnet18(device):

    model = load_model()
    checkpoint_path = "./premodel19_trained.pth.tar"  # 替换为你的权重文件路径
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 修复权重的键问题
    state_dict = checkpoint
    new_state_dict = {}
    for key in state_dict.keys():
        if key.startswith("module."):
            new_state_dict[key[len("module."):]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]

    # 加载修复后的权重
    model.load_state_dict(new_state_dict)
    model = model.to(device)

    return model



def evaluate_model(model, test_loader, device):
    """评估模型性能"""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    accuracy = 100. * correct / total
    return accuracy
