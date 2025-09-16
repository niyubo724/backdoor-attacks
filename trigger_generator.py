import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def generate_trigger_cifar(img, trigger_type, mode='train'):
    """
    Generate different types of triggers for CIFAR-10 images
    """
    assert trigger_type in ['gridTrigger', 'onePixelTrigger', 'blendTrigger',
                            'trojanTrigger', 'wanetTrigger']

    if trigger_type == 'gridTrigger':
        img = _grid_trigger(img, mode)
    elif trigger_type == 'onePixelTrigger':
        img = _one_pixel_trigger(img, mode)
    elif trigger_type == 'blendTrigger':
        img = _blend_trigger(img, mode)
    elif trigger_type == 'trojanTrigger':
        img = _trojan_trigger(img, mode)
    elif trigger_type == 'wanetTrigger':
        img = _wanet_trigger(img, mode)
    else:
        raise NotImplementedError(f"Trigger type {trigger_type} not implemented")

    return img


def _grid_trigger(img, mode='train'):
    """Grid pattern trigger (BadNets)"""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Img should be np.ndarray. Got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"The shape of img should be HWC. Got {img.shape}")

    width, height, c = img.shape

    # Create 3x3 grid pattern at bottom-right corner
    img[width - 1][height - 1] = 255
    img[width - 1][height - 2] = 0
    img[width - 1][height - 3] = 255

    img[width - 2][height - 1] = 0
    img[width - 2][height - 2] = 255
    img[width - 2][height - 3] = 0

    img[width - 3][height - 1] = 255
    img[width - 3][height - 2] = 0
    img[width - 3][height - 3] = 0

    return img


def _one_pixel_trigger(img, mode='train'):
    """Single pixel trigger"""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Img should be np.ndarray. Got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"The shape of img should be HWC. Got {img.shape}")

    width, height, c = img.shape
    img[width // 2][height // 2] = 255

    return img


def _blend_trigger(img, mode='train'):
    """Blend trigger with random noise"""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Img should be np.ndarray. Got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"The shape of img should be HWC. Got {img.shape}")

    width, height, c = img.shape
    alpha = 0.2
    mask = np.random.randint(low=0, high=256, size=(width, height), dtype=np.uint8)
    blend_img = (1 - alpha) * img + alpha * mask.reshape((width, height, 1))
    blend_img = np.clip(blend_img.astype('uint8'), 0, 255)

    return blend_img


def _trojan_trigger(img, mode='train'):
    """Trojan trigger - simple square pattern"""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Img should be np.ndarray. Got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"The shape of img should be HWC. Got {img.shape}")

    width, height, c = img.shape

    # Add a 4x4 white square at top-left corner
    for i in range(4):
        for j in range(4):
            if i < width and j < height:
                img[i][j] = 255

    return img


def _wanet_trigger(img, mode='train'):
    """WaNet trigger - image warping"""
    if not isinstance(img, np.ndarray):
        raise TypeError(f"Img should be np.ndarray. Got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"The shape of img should be HWC. Got {img.shape}")

    # Prepare grid
    s = 0.5
    k = 32
    grid_rescale = 1
    ins = torch.rand(1, 2, k, k) * 2 - 1
    ins = ins / torch.mean(torch.abs(ins))
    noise_grid = F.interpolate(ins, size=32, mode="bicubic", align_corners=True)
    noise_grid = noise_grid.permute(0, 2, 3, 1)
    array1d = torch.linspace(-1, 1, steps=32)
    x, y = torch.meshgrid(array1d, array1d, indexing='ij')
    identity_grid = torch.stack((y, x), 2)[None, ...]
    grid = identity_grid + s * noise_grid / 32 * grid_rescale
    grid = torch.clamp(grid, -1, 1)

    img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
    poison_img = F.grid_sample(img_tensor.unsqueeze(0), grid, align_corners=True).squeeze()
    poison_img = poison_img.permute(1, 2, 0) * 255
    poison_img = poison_img.numpy().astype(np.uint8)

    return poison_img
