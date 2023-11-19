import os
import time
import logging
import logging.config
import hydra
from omegaconf import DictConfig

import torch

from utils.dataset import get_transforms, get_dataset
from utils.model import build_model
from utils.training import test_epoch
from utils.exceptions import ParameterNotProvidedError, UnsupportedParameterError
from settings.log_settings import LOGGING_CONFIG


def _get_loggers() -> logging.Logger:
    """Initialize the logger"""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("info_logger"), logging.getLogger("train_logger")


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    """
    Evaluate a model on test set at a certain epoch.
    Args:
        cfg (DictConfig): Hydra config object
    """

    info_logger, train_logger = _get_loggers()

    exp_name = cfg.experiment.name
    data_cfg = cfg.experiment.data
    model_cfg = cfg.experiment.model
    training_cfg = cfg.experiment.training

    if not training_cfg.test_epoch:
        raise ParameterNotProvidedError("epoch to test not provided")

    epoch = training_cfg.test_epoch

    # data loading
    if not data_cfg.dataset_path:
        raise ParameterNotProvidedError("dataset path not provided")

    if not data_cfg.label_path:
        raise ParameterNotProvidedError("label file not provided")

    if not training_cfg.checkpoint_path:
        raise ParameterNotProvidedError("checkpoint path to load not provided")

    tfrec_pattern = f"{data_cfg.dataset_path}/genre/" + "{}.tfrecords"
    index_pattern = f"{data_cfg.dataset_path}/index/" + "{}.index"

    _, test_transform = get_transforms(data_cfg)

    test_dataset = get_dataset(
        tfrec_pattern,
        index_pattern,
        data_cfg.test_records,
        test_transform,
        training_cfg.buffer_size,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=training_cfg.batch_size
    )

    # model creation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info_logger.info(f"running on {device}")

    model = build_model(data_cfg, model_cfg, device)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    load_path = os.path.join(
        training_cfg.checkpoint_path, f"{exp_name}_epoch_{epoch}.pth"
    )

    if not os.path.exists(load_path):
        raise UnsupportedParameterError("model checkpoint path to load does not exist")

    map_location = device
    if device == "cuda":
        map_location = "cuda:0"

    checkpoint = torch.load(load_path, map_location=map_location)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    start_time = time.time()

    train_logger.info(f"epoch {epoch}:")
    test_epoch(device, epoch, model, test_loader, loss_func, train_logger, wandb=None)
    train_logger.info(f"time taken: {int(time.time() - start_time)}s")
    train_logger.info("=" * 20)


if __name__ == "__main__":
    main()
