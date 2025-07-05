import logging
import os
import sys

import colorlog
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Model configuration
DEFAULT_MODEL_NAME = "mlx-community/parakeet-tdt-1.1b"

# Server configuration
WORKERS = int(os.getenv("PARAKEET_WORKERS", "1"))

# Audio processing configuration
TARGET_SR = 16000
MODEL_PRECISION = "bf16"

# Logging configuration
_logging_configured = False


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure logging for the entire application.
    This should be called once at application startup.
    """
    global _logging_configured

    if _logging_configured:
        return

    # Convert string level to logging constant
    level_map = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }

    level = level_map.get(log_level.upper(), logging.INFO)

    # Clear any existing handlers to prevent duplicates
    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create colorized formatter
    color_formatter = colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
        secondary_log_colors={},
        style="%",
    )

    # Create handler with colored formatter
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(color_formatter)

    # Configure root logger
    root_logger.setLevel(level)
    root_logger.addHandler(handler)

    # Suppress noisy third-party loggers unless in debug mode
    if level > logging.DEBUG:
        logging.getLogger("mlx").setLevel(logging.WARNING)
        logging.getLogger("librosa").setLevel(logging.WARNING)
        logging.getLogger("soundfile").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        logging.getLogger("requests").setLevel(logging.WARNING)

    _logging_configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    Ensures logging is configured if not already done.
    """
    # Configure logging with default level if not already configured
    if not _logging_configured:
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        configure_logging(log_level)

    return logging.getLogger(name)


# Create default logger instance
logger = get_logger("parakeet_service")
