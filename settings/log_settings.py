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
    },
    "handlers": {
        "default": {
            "level": "DEBUG",
            "formatter": "standard",
            "class": "logging.StreamHandler",
            "stream": "ext://sys.stdout",
        },
    },
    "loggers": {
        "trainer": {
            "handlers": [
                "default",
            ],
            "level": "INFO",
            "propagate": False,
        },
    },
}