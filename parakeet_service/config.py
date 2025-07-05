import logging
import os
import sys

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "mlx-community/parakeet-tdt-1.1b"

# Configuration from environment variables
TARGET_SR = int(os.getenv("TARGET_SR", "16000"))
MODEL_PRECISION = os.getenv("MODEL_PRECISION", "bf16")
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "4"))
MAX_AUDIO_DURATION = int(os.getenv("MAX_AUDIO_DURATION", "45"))  # seconds
PROCESSING_TIMEOUT = int(os.getenv("PROCESSING_TIMEOUT", "120"))  # seconds

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-7s  %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)

logger = logging.getLogger("parakeet_service")
