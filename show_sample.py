import hydra
from omegaconf import DictConfig, OmegaConf

import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from utils.dataset import get_transforms, get_dataset
from utils.label_map import get_label_map


def plot_spectrogram(spec, title=None, ylabel='freq_bin', aspect='auto', xmax=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or 'Spectrogram (db)')
    axs.set_ylabel(ylabel)
    axs.set_xlabel('frame')
    im = axs.imshow(spec, origin='lower', aspect=aspect)
    if xmax:
        axs.set_xlim((0, xmax))
    fig.colorbar(im, ax=axs)
    plt.show(block=False)


@hydra.main(version_base=None, config_path="./configs", config_name="config_dev")
def show(cfg: DictConfig):
    data_cfg = cfg.experiment.data
    training_cfg = cfg.experiment.training
    
    # data loading
    if not data_cfg.dataset_path:
        print("dataset path not provided")

    if not data_cfg.label_path:
        print("label file not provided")

    _, rev_labels = get_label_map(data_cfg.label_path)

    _, test_transform = get_transforms(data_cfg)

    tfrec_pattern = f"{data_cfg.dataset_path}/genre/" + "{}.tfrecords"
    index_pattern = f"{data_cfg.dataset_path}/index/" + "{}.index"

    test_dataset = get_dataset(tfrec_pattern, index_pattern, data_cfg.test_records, test_transform, training_cfg.buffer_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=training_cfg.batch_size)

    for i, (waveform, label) in enumerate(test_loader):
        print(rev_labels[label.numpy()[0]])
        print(waveform.size())
        plot_spectrogram(waveform.numpy()[0][0, :, :])
        break


if __name__ == '__main__':
    show()