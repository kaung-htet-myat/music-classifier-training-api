import hydra
from omegaconf import DictConfig, OmegaConf

from utils.model import build_model


def get_model_param_count(model):
    return


@hydra.main(version_base=None, config_path="./configs", config_name="config_dev")
def main(cfg: DictConfig):
    
    data_cfg = cfg.experiment.data
    model_cfg = cfg.experiment.model
    model = build_model(data_cfg, model_cfg)

    print(model)
    del model


if __name__ == '__main__':
    main()