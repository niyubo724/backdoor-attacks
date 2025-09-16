import os
import torch
import random
import torch.distributed as dist
import torch.functional as F
import numpy as np

import models.resnet as RN
import models.resnet_ap as RNAP
import models.convnet as CN
import models.densenet_cifar as DN



class BlurPoolConv2d(torch.nn.Module):
    def __init__(self, conv):
        super().__init__()
        default_filter = torch.tensor([[[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]]) / 16.0
        filt = default_filter.repeat(conv.in_channels, 1, 1, 1)
        self.conv = conv
        self.register_buffer("blur_filter", filt)

    def forward(self, x):
        blurred = F.conv2d(
            x,
            self.blur_filter,
            stride=1,
            padding=(1, 1),
            groups=self.conv.in_channels,
            bias=None,
        )
        return self.conv.forward(blurred)


def apply_blurpool(mod: torch.nn.Module):
    for name, child in mod.named_children():
        if isinstance(child, torch.nn.Conv2d) and (
            np.max(child.stride) > 1 and child.in_channels >= 16
        ):
            setattr(mod, name, BlurPoolConv2d(child))
        else:
            apply_blurpool(child)


def define_model(dataset, norm_type, net_type, nch, depth, width, nclass, logger, size):

    if net_type == "resnet":
        model = RN.ResNet(
            dataset, depth, nclass, norm_type=norm_type, size=size, nch=nch
        )
    elif net_type == "resnet_ap":
        model = RNAP.ResNetAP(
            dataset, depth, nclass, width=width, norm_type=norm_type, size=size, nch=nch
        )
        apply_blurpool(model)

    elif net_type == "densenet":
        model = DN.densenet_cifar(nclass)
    elif net_type == "convnet":
        width = int(128 * width)
        model = CN.ConvNet(
            nclass,
            net_norm=norm_type,
            net_depth=depth,
            net_width=width,
            channel=nch,
            im_size=(size, size),
        )
    else:
        raise Exception("unknown network architecture: {}".format(net_type))

    # if logger is not None:
    #     if dist.get_rank() == 0:
    #         logger(f"=> creating model {net_type}-{depth}, norm: {norm_type}")
    #         logger('# model parameters: {:.1f}M'.format(sum([p.data.nelement() for p in model.parameters()]) / 10**6))
    return model














