import os
import time
from pathlib import Path
import logging
import logging.config
import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn

from utils.dataset import get_transforms, get_dataset
from utils.model import build_model
from utils.training import (
    get_optimizer,
    get_scheduler,
    train_epoch,
    test_epoch
)
from utils.wandb import wandb_init
from utils.exceptions import (
    ParameterNotProvidedError,
    UnsupportedParameterError
)
from settings.log_settings import LOGGING_CONFIG


def _get_loggers() -> logging.Logger:
    """Initialize the logger"""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("info_logger"), logging.getLogger("train_logger")


@hydra.main(version_base=None, config_path="./configs", config_name="config_dev")
def main(cfg: DictConfig):

    info_logger, train_logger = _get_loggers()

    exp_name = cfg.experiment.name
    data_cfg = cfg.experiment.data
    model_cfg = cfg.experiment.model
    training_cfg = cfg.experiment.training
    wandb_cfg = cfg.experiment.wandb
    optimizer_cfg = training_cfg.optimizer
    scheduler_cfg = training_cfg.scheduler
    initial_epoch = training_cfg.initial_epoch if training_cfg.initial_epoch else 0

    if wandb_cfg.log:
        wandb = wandb_init(wandb_cfg, data_cfg, model_cfg, training_cfg, exp_name)
    
    # data loading
    if not data_cfg.dataset_path:
        raise ParameterNotProvidedError("dataset path not provided")

    if not data_cfg.label_path:
        raise ParameterNotProvidedError("label file not provided")

    if not training_cfg.checkpoint_path:
        training_cfg.checkpoint_path = './checkpoints'

    if not os.path.exists(training_cfg.checkpoint_path):
      os.makedirs(training_cfg.checkpoint_path)

    tfrec_pattern = f"{data_cfg.dataset_path}/genre/" + "{}.tfrecords"
    index_pattern = f"{data_cfg.dataset_path}/index/" + "{}.index"

    train_transform, test_transform = get_transforms(data_cfg)

    train_dataset = get_dataset(tfrec_pattern, index_pattern, data_cfg.train_records, train_transform, training_cfg.buffer_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_cfg.batch_size)

    dev_dataset = get_dataset(tfrec_pattern, index_pattern, data_cfg.dev_records, test_transform, training_cfg.buffer_size)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=training_cfg.batch_size)

    # model creation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    info_logger.info(f"running on {device}")

    model = build_model(data_cfg, model_cfg, device)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, optimizer_cfg)
    scheduler = get_scheduler(optimizer, scheduler_cfg)

    if training_cfg.load and not training_cfg.initial_epoch == 0:
        load_path = os.path.join(training_cfg.checkpoint_path, f'{exp_name}_epoch_{training_cfg.initial_epoch-1}.pth')

        if not os.path.exists(load_path):
            raise UnsupportedParameterError("model checkpoint path to load does not exist")

        map_location = device
        if device == 'cuda':
          map_location = "cuda:0"

        checkpoint = torch.load(load_path, map_location=map_location)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    if wandb_cfg.log:
        wandb.watch(model, log_freq=10)

    for epoch in range(initial_epoch, training_cfg.epochs):
        start_time = time.time()
        train_epoch(device, epoch, model, train_loader, loss_func, optimizer, scheduler, training_cfg.checkpoint_path, exp_name, wandb, train_logger)
        train_logger.info(f"time taken(training): {int(time.time() - start_time)}s")
        test_epoch(device, epoch, model, dev_loader, loss_func, train_logger, wandb)
        train_logger.info("="*20)


if __name__ == '__main__':
    main()