import os
import logging
import logging.config
import hydra
from omegaconf import DictConfig, OmegaConf

import torch

from utils.model import build_model
from utils.exceptions import (
    ParameterNotProvidedError,
    UnsupportedParameterError
)

from settings.log_settings import LOGGING_CONFIG


def _get_loggers() -> logging.Logger:
    """Initialize the logger"""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("info_logger")


@hydra.main(version_base=None, config_path="./configs", config_name="config_dev")
def main(cfg: DictConfig):

    info_logger = _get_loggers()
    
    exp_name = cfg.experiment.name
    data_cfg = cfg.experiment.data
    model_cfg = cfg.experiment.model
    training_cfg = cfg.experiment.training
    export_path = os.path.join(training_cfg.export_dir, f'{exp_name}_epoch_{training_cfg.export_epoch}_model.pth')

    if not os.path.exists(training_cfg.export_dir):
        os.makedirs(training_cfg.export_dir)

    if not training_cfg.checkpoint_path:
        raise ParameterNotProvidedError("checkpoint_path not provided")

    if not training_cfg.export_epoch:
        raise ParameterNotProvidedError("export_epoch not provided")

    if not os.path.exists(training_cfg.checkpoint_path):
        raise UnsupportedParameterError("checkpoint path does not exist")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    info_logger.info(f"running on {device}")

    model = build_model(data_cfg, model_cfg, device)
    model.eval()

    load_path = os.path.join(training_cfg.checkpoint_path, f'{exp_name}_epoch_{training_cfg.export_epoch}.pth')

    if not os.path.exists(load_path):
        raise UnsupportedParameterError("model checkpoint path to load does not exist")

    map_location = device
    if device == 'cuda':
        map_location = "cuda:0"

    checkpoint = torch.load(load_path, map_location=map_location)
    model.load_state_dict(checkpoint['model_state_dict'])

    scripted_model = torch.jit.script(model)
    scripted_model.save(export_path)

    info_logger.info(f"model exported to {export_path}")


if __name__ == '__main__':
    main()