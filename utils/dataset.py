from omegaconf import DictConfig

from tfrecord.torch.dataset import MultiTFRecordDataset

import torch
import torch.nn as nn
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T

from utils.exceptions import ParameterNotProvidedError


def get_transforms(data_cfg: DictConfig):

    def train_transform(features):
        
        waveform = features["waveform"]
        label = features["label"]

        waveform = waveform[:data_cfg.preprocessing.sample_rate*data_cfg.preprocessing.sample_length]
        waveform = torch.from_numpy(waveform)
        resampled = nn.functional.pad(waveform, (0, data_cfg.preprocessing.sample_rate*data_cfg.preprocessing.sample_length - waveform.size()[-1]), 'constant', 0.0)

        if data_cfg.preprocessing.name == "melspectrogram":
            mel_spectrogram = T.MelSpectrogram(
                                        sample_rate=data_cfg.preprocessing.sample_rate,
                                        n_fft=data_cfg.preprocessing.fft_size,
                                        hop_length=data_cfg.preprocessing.hop_length,
                                        n_mels=data_cfg.preprocessing.n_mels
                                    )
            amp_to_db = T.AmplitudeToDB()
            resampled = mel_spectrogram(resampled)
            resampled = amp_to_db(resampled)
            resampled = resampled.unsqueeze(axis=0)

        label = torch.from_numpy(label)[0]
        return resampled, label


    def test_transform(features):
        
        waveform = features["waveform"]
        label = features["label"]

        waveform = waveform[:data_cfg.preprocessing.sample_rate*data_cfg.preprocessing.sample_length]
        waveform = torch.from_numpy(waveform)
        resampled = nn.functional.pad(waveform, (0, data_cfg.preprocessing.sample_rate*data_cfg.preprocessing.sample_length - waveform.size()[-1]), 'constant', 0.0)

        if data_cfg.preprocessing.name == "melspectrogram":
            mel_spectrogram = T.MelSpectrogram(
                                        sample_rate=data_cfg.preprocessing.sample_rate,
                                        n_fft=data_cfg.preprocessing.fft_size,
                                        hop_length=data_cfg.preprocessing.hop_length,
                                        n_mels=data_cfg.preprocessing.n_mels
                                    )
            amp_to_db = T.AmplitudeToDB()
            resampled = mel_spectrogram(resampled)
            resampled = amp_to_db(resampled)
            resampled = resampled.unsqueeze(axis=0)

        label = torch.from_numpy(label)[0]
        return resampled, label

    return train_transform, test_transform


def get_dataset(tfrec_pattern, index_pattern, records, transform, buffer_size):

    tfrec_description = {
        "label": "int",
        "waveform": "float",
    }

    if len(records) == 0:
        raise ParameterNotProvidedError("train, dev or test record list cannot be empty")

    splits = {k:1.0 for k in records}

    dataset = MultiTFRecordDataset(tfrec_pattern, index_pattern, splits, tfrec_description, transform=transform, infinite=False, shuffle_queue_size=buffer_size)

    return dataset