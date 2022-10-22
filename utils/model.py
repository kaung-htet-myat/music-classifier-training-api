from omegaconf import DictConfig

import torch
import torch.nn as nn
import torchvision
from torchvision.models import inception_v3

from models.prcnn import PRCNN


def get_inceptionv3_backbone():
    inception_backbone = inception_v3(weights=None, progress=False)
    inception_backbone.Conv2d_1a_3x3.conv = nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 2), bias=False)
    return_nodes = {
        "Mixed_6a.branch3x3.relu": "fe_output"
    }
    return inception_backbone, return_nodes


def build_model(model_cfg: DictConfig):

    if model_cfg.backbone not in ["resnet", "inceptionv3", "efficientnet"]:
        print("Model backbone should be one of \"resnet\", \"inceptionv3\" or \"efficientnet\"")

    if model_cfg.name not in ["linear", "prcnn", "transformer"]:
        print("Model architecture should be one of \"linear\", \"prcnn\" or \"transformer\"")

    if model_cfg.backbone == "inceptionv3":
        backbone, return_nodes = get_inceptionv3_backbone()

    if model_cfg.name == "prcnn":
        model = PRCNN(backbone, return_nodes)

    return model

    