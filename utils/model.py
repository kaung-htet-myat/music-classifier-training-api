import os
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torchvision
from torchvision.models import efficientnet_v2_m

from models.prcnn import PRCNN
from models.linear import Linear

from utils.exceptions import UnsupportedParameterError


def get_efficientnet_backbone():
    backbone_model = efficientnet_v2_m(weights=None, progress=False)
    backbone_model.features[0][0] = nn.Conv2d(1, 24, kernel_size=(3, 3), stride=(2, 2), padding=(1,1), bias=False)
    return_nodes = {
        "features.4.4.add": "fe_output",
    }
    return backbone_model, return_nodes


def build_model(data_cfg: DictConfig, model_cfg: DictConfig):

    if data_cfg.preprocessing.name not in ["spectrogram", "melspectrogram", "mfcc"]:
        raise UnsupportedParameterError("Data preprocessing should be one of \"spectrogram\", \"melspectrogram\" or \"mfcc\"")

    if not model_cfg.backbone == "efficientnet" :
        raise UnsupportedParameterError("Only \"efficientnet\" is provided as backbone for now")

    if model_cfg.name not in ["linear", "prcnn", "transformer"]:
        raise UnsupportedParameterError("Model architecture should be one of \"linear\", \"prcnn\" or \"transformer\"")

    if model_cfg.backbone == "efficientnet":
        backbone, return_nodes = get_efficientnet_backbone()

    if data_cfg.preprocessing.name == "melspectrogram":
        input_width = int(((data_cfg.preprocessing.sample_length * data_cfg.preprocessing.sample_rate) - data_cfg.preprocessing.fft_size) // data_cfg.preprocessing.hop_length + 1)
        input_height = data_cfg.preprocessing.n_mels
    elif data_cfg.preprocessing.name == "spectrogram":
        input_width = int(((data_cfg.preprocessing.sample_length * data_cfg.preprocessing.sample_rate) - data_cfg.preprocessing.fft_size) // data_cfg.preprocessing.hop_length + 1)
        input_height = int(data_cfg.preprocessing.ftt_size // 2 + 1)
    elif data_cfg.preprocessing.name == "mfcc":
        input_width = int(((data_cfg.preprocessing.sample_length * data_cfg.preprocessing.sample_rate) - data_cfg.preprocessing.fft_size) // data_cfg.preprocessing.hop_length + 1)
        input_height = data_cfg.preprocessing.n_mfcc

    if model_cfg.name == "prcnn":
        model = PRCNN(
                    (1, input_height, input_width),
                    backbone,
                    (model_cfg.backbone_out_channel, model_cfg.backbone_out_height, model_cfg.backbone_out_width),
                    model_cfg.hidden_dims_1,
                    model_cfg.hidden_dims_2,
                    return_nodes
                )
    elif model_cfg.name == "linear":
        model = Linear(
            (1, input_height, input_width),
            backbone,
            (model_cfg.backbone_out_channel, model_cfg.backbone_out_height, model_cfg.backbone_out_width),
            model_cfg.hidden_dims,
            return_nodes
        )

    return model

    