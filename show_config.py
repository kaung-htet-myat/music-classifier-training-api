import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    """
    Print out config info.
    Args:
        cfg (DictConfig): Hydra config object
    """

    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
