import logging
import logging.config
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt

from utils.training import lrfn
from settings.log_settings import LOGGING_CONFIG


def _get_loggers() -> logging.Logger:
    """Initialize the logger"""
    logging.config.dictConfig(LOGGING_CONFIG)
    return logging.getLogger("info_logger")


@hydra.main(version_base=None, config_path="./configs", config_name="config")
def main(cfg: DictConfig):
    """
    Display lr schduler plot defined by provided lambda function.
    Args:
        cfg (DictConfig): Hydra config object
    """

    logger = _get_loggers()

    training_cfg = cfg.experiment.training

    if not training_cfg.scheduler.method == "lambda":
        logger.info("lr plotting is only available with 'lambda' scheduler")

    epochs = range(0, 120)
    lrs = list(map(lrfn, epochs))

    plt.plot(epochs, lrs)
    plt.show()


if __name__ == "__main__":
    main()
