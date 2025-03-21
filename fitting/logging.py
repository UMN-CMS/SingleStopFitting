import logging.config
import logging

logger = logging.getLogger(__name__)

config = {
    "version": 1,
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


def setupLogging():
    logging.config.dictConfig(config)
    logger.info("Setup logging")
