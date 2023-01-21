"""Settings for Logger"""

from logging import Formatter

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": True,
    "formatters": {
        "standard": {
            "()": Formatter,
            "format": "%(asctime)s [%(levelname)s] %(name)s::%(module)s|%(lineno)s: %(message)s",
        },
        "train_log": {
            "()": Formatter,
            "format": "%(message)s",
        },
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
        "train_log": {
            "level": "DEBUG",
            "formatter": "train_log",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "info_logger": {
            "handlers": [
                "default",
            ],
            "level": "INFO",
            "propagate": False,
        },
        "train_logger": {
            "handlers": [
                "train_log",
            ],
            "level": "INFO",
            "propagate": False,
        },
    },
}
