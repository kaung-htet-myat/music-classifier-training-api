import time
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


@hydra.main(version_base=None, config_path="./configs", config_name="config_dev")
def main(cfg: DictConfig):
    data_cfg = cfg.experiment.data
    model_cfg = cfg.experiment.model
    training_cfg = cfg.experiment.training
    optimizer_cfg = training_cfg.optimizer
    scheduler_cfg = training_cfg.scheduler
    
    # data loading
    if not data_cfg.dataset_path:
        print("dataset path not provided")

    if not data_cfg.label_path:
        print("label file not provided")

    tfrec_pattern = f"{data_cfg.dataset_path}/genre/" + "{}.tfrecords"
    index_pattern = f"{data_cfg.dataset_path}/index/" + "{}.index"

    train_transform, test_transform = get_transforms(data_cfg)

    train_dataset = get_dataset(tfrec_pattern, index_pattern, data_cfg.train_records, train_transform, training_cfg.buffer_size)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=training_cfg.batch_size)

    dev_dataset = get_dataset(tfrec_pattern, index_pattern, data_cfg.dev_records, test_transform, training_cfg.buffer_size)
    dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=training_cfg.batch_size)

    # model creation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"running on {device}")

    model = build_model(model_cfg)
    model.to(device)

    loss_func = torch.nn.CrossEntropyLoss()

    optimizer = get_optimizer(model, optimizer_cfg)
    scheduler = get_scheduler(optimizer, scheduler_cfg)

    for epoch in range(training_cfg.epochs):
        start_time = time.time()
        train_epoch(device, epoch, model, train_loader, loss_func, optimizer)
        print(f"time taken(training): {int(time.time() - start_time)}s")
        scheduler.step()
        test_epoch(device, model, dev_loader, loss_func)


if __name__ == '__main__':
    main()