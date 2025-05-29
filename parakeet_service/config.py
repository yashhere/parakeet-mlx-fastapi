import logging, os, sys
import os
from pathlib import Path

# TODO: set-up .env file 

MODEL_NAME = "nvidia/parakeet-tdt-0.6b-v2"
TARGET_SR = 16_000          # model’s native sample-rate

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()        # DEBUG by default
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True
)

logger = logging.getLogger("parakeet_service")
