import hydra
from omegaconf import DictConfig

import torch

from utils.model import build_model


def get_model_param_count(model):
    return


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    """
    Print model summary.
    Args:
        cfg (DictConfig): Hydra config object
    """

    data_cfg = cfg.experiment.data
    model_cfg = cfg.experiment.model

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_model(data_cfg, model_cfg, device)

    print(model)
    del model


if __name__ == "__main__":
    main()
