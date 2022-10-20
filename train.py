import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn

from utils.dataset import get_transforms, get_dataset


@hydra.main(version_base=None, config_path="./configs", config_name="config_dev")
def train(cfg: DictConfig):
    data_cfg = cfg.experiment.data
    model_cfg = cfg.experiment.model
    training_cfg = cfg.experiment.training
    
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

    test_dataset = get_dataset(tfrec_pattern, index_pattern, data_cfg.test_records, test_transform, training_cfg.buffer_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_cfg.batch_size)

    print(train_loader)


if __name__ == '__main__':
    train()