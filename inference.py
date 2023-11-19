import os
import logging
import logging.config
import hydra
from omegaconf import DictConfig

import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
import librosa
import numpy as np

from settings.log_settings import LOGGING_CONFIG
from utils.exceptions import UnsupportedParameterError
from utils.label_map import get_label_map


def _get_loggers() -> logging.Logger:
    """Initialize the logger"""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("info_logger")


def _load_model(model_path):

    if not os.path.exists(model_path):
        raise OSError("Specified model path does not exists.")

    model = torch.jit.load(model_path)
    model.eval()

    return model


def _preprocess(segment, in_rate, data_cfg):

    waveform = segment[: in_rate * data_cfg.preprocessing.sample_length]
    waveform = torch.from_numpy(waveform)
    resampler = T.Resample(
        in_rate, data_cfg.preprocessing.sample_rate, dtype=waveform.dtype
    )
    waveform = resampler(waveform)
    resampled = nn.functional.pad(
        waveform,
        (
            0,
            data_cfg.preprocessing.sample_rate * data_cfg.preprocessing.sample_length
            - waveform.size()[-1],
        ),
        "constant",
        0.0,
    )

    if data_cfg.preprocessing.name == "melspectrogram":
        mel_spectrogram = T.MelSpectrogram(
            sample_rate=data_cfg.preprocessing.sample_rate,
            n_fft=data_cfg.preprocessing.fft_size,
            hop_length=data_cfg.preprocessing.hop_length,
            n_mels=data_cfg.preprocessing.n_mels,
            center=True,
            pad_mode="reflect",
            norm="slaney",
            onesided=True,
            mel_scale="htk",
        )
        resampled = mel_spectrogram(resampled)
        resampled = resampled.unsqueeze(axis=0)

    elif data_cfg.preprocessing.name == "spectrogram":
        spectrogram = T.Spectrogram(
            n_fft=data_cfg.preprocessing.fft_size,
            win_length=None,
            hop_length=data_cfg.preprocessing.hop_length,
            center=True,
            pad_mode="reflect",
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

    return resampled.numpy()


def _get_result(predictions, label_map):
    results = np.argmax(predictions.detach().numpy(), axis=-1)
    results = [label_map[res] for res in results]
    sorted_results = sorted(set(results), key=results.count, reverse=True)
    return sorted_results[0]


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    """
    Inference endpoint.
    Load the model from exported dir and run inference on mp3 files.
    Args:
        cfg (DictConfig): Hydra config object
    """

    exp_name = cfg.experiment.name
    data_cfg = cfg.experiment.data
    training_cfg = cfg.experiment.training

    logger = _get_loggers()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, label_map = get_label_map(data_cfg.label_path)

    model_path = os.path.join(
        training_cfg.export_dir,
        f"{exp_name}_epoch_{training_cfg.export_epoch}_model.pth",
    )

    if not os.path.exists(model_path):
        raise UnsupportedParameterError("Model path does not exists")

    model = _load_model(model_path)
    model.to(device)

    logger.info("Model is successfully loaded")
    logger.info(f"Feature extraction method: {data_cfg.preprocessing.name}")

    file_path = input("mp3 file path (type 'stop' to exit): ")

    while not file_path == "stop":
        try:
            if not os.path.exists(file_path):
                raise UnsupportedParameterError("File does not exists.")

            sample, in_rate = librosa.load(file_path, sr=None)

            segments = []
            start = 0

            while start < len(sample):
                end = start + in_rate * data_cfg.preprocessing.sample_length
                try:
                    segment = sample[start:end]
                except IndexError as e:
                    segment = sample[start:]
                segments.append(segment)
                start = end

            np_segments = [
                _preprocess(segment, in_rate, data_cfg) for segment in segments
            ]
            np_segments = np.array(np_segments, dtype=np.float32)
            input_segments = torch.from_numpy(np_segments)
            input_segments = input_segments.to(device)

            predictions = model(input_segments)
            predictions = predictions.to("cpu")

            result = _get_result(predictions, label_map)

            print(f"prediction: {result}")

        except UnsupportedParameterError as err:
            logger.info(err.message)
        except Exception as err:
            logger.info("Unexpected error occurred.")

        file_path = input("mp3 file path (type 'stop' to exit): ")


if __name__ == "__main__":
    main()
