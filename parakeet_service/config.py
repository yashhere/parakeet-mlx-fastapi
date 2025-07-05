import logging
import os

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

# Create logger (will inherit configuration from uvicorn)
logger = logging.getLogger("parakeet_service")
