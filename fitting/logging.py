import logging.config
import logging


config = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "simple": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"},
        "just_name": {"format": "%(name)s - %(message)s"},
    },
    "handlers": {
        "debug_console": {
            "class": "rich.logging.RichHandler",
            "level": "DEBUG",
            "formatter": "just_name",
        }
    },
    "loggers": {
        "fitting": {
            "level": "DEBUG",
            "handlers": ["debug_console"],
            "propagate": "false",
        },
    },
}


def setupLogging(
    default_level=None,
):
    logging.config.dictConfig(config)
    if default_level is not None:
        logger = logging.getLogger("fitting")
        logger.setLevel(default_level)
