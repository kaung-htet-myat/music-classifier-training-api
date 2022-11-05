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
                                        n_mels=data_cfg.preprocessing.n_mels,
                                        center=True,
                                        pad_mode='reflect',
                                        norm="slaney",
                                        onesided=True,
                                        mel_scale="htk"
                                    )
            resampled = mel_spectrogram(resampled)
            resampled = resampled.unsqueeze(axis=0)
        elif data_cfg.preprocessing.name == "spectrogram":
            spectrogram = T.Spectrogram(
                                n_fft=data_cfg.preprocessing.fft_size,
                                win_length=None,
                                hop_length=data_cfg.preprocessing.hop_length,
                                center=True,
                                pad_mode='reflect',
                                power=2.0,
                            )
            resampled = spectrogram(resampled)
            resampled = resampled.unsqueeze(axis=0)
        elif data_cfg.preprocessing.name == "mfcc":
            mfcc = T.MFCC(
                        sample_rate=data_cfg.preprocessing.sample_rate,
                        n_mfcc=data_cfg.preprocessing.n_mfcc,
                        melkwargs={
                            "n_fft": data_cfg.preprocessing.fft_size,
                            "n_mels": data_cfg.preprocessing.n_mels,
                            "hop_length": data_cfg.preprocessing.hop_length,
                            "mel_scale": "htk",
                        },
                    )
            resampled = mfcc(resampled)
            resampled = resampled.unsqueeze(axis=0)

        aug_rand = torch.rand([1])

        if data_cfg.augmentation.time_masking and aug_rand < 0.25:
            time_mask = T.TimeMasking(time_mask_param=data_cfg.augmentation.time_mask_param)
            resampled = time_mask(resampled)
        if data_cfg.augmentation.freq_masking and 0.25 < aug_rand < 0.5:
            freq_mask = T.FrequencyMasking(freq_mask_param=data_cfg.augmentation.freq_mask_param)
            resampled = freq_mask(resampled)
            
        amp_to_db = T.AmplitudeToDB()
        resampled = amp_to_db(resampled)
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
                                        n_mels=data_cfg.preprocessing.n_mels,
                                        center=True,
                                        pad_mode='reflect',
                                        norm="slaney",
                                        onesided=True,
                                        mel_scale="htk"
                                    )
            resampled = mel_spectrogram(resampled)
            resampled = resampled.unsqueeze(axis=0)

        elif data_cfg.preprocessing.name == "spectrogram":
            spectrogram = T.Spectrogram(
                                n_fft=data_cfg.preprocessing.fft_size,
                                win_length=None,
                                hop_length=data_cfg.preprocessing.hop_length,
                                center=True,
                                pad_mode='reflect',
                                power=2.0,
                            )
            resampled = spectrogram(resampled)
            resampled = resampled.unsqueeze(axis=0)

        elif data_cfg.preprocessing.name == "mfcc":
            mfcc = T.MFCC(
                        sample_rate=data_cfg.preprocessing.sample_rate,
                        n_mfcc=data_cfg.preprocessing.n_mfcc,
                        melkwargs={
                            "n_fft": data_cfg.preprocessing.fft_size,
                            "n_mels": data_cfg.preprocessing.n_mels,
                            "hop_length": data_cfg.preprocessing.hop_length,
                            "mel_scale": "htk",
                        },
                    )
            resampled = mfcc(resampled)
            resampled = resampled.unsqueeze(axis=0)

        amp_to_db = T.AmplitudeToDB()
        resampled = amp_to_db(resampled)
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