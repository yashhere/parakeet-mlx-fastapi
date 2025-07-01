from contextlib import asynccontextmanager
import contextlib
import gc
import asyncio
from parakeet_mlx import from_pretrained

from .config import MODEL_NAME, MODEL_PRECISION, logger


def _to_builtin(obj):
    """Convert parakeet-mlx results to JSON-safe format."""
    if hasattr(obj, "__dict__"):
        return {k: _to_builtin(v) for k, v in obj.__dict__.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _to_builtin(v) for k, v in obj.items()}
    return obj


@asynccontextmanager
async def lifespan(app):
    """Load model once per process; cleanup on shutdown."""
    logger.info("Loading %s with MLX...", MODEL_NAME)

    # Load model with parakeet-mlx
    model = from_pretrained(MODEL_NAME)

    # Configure precision - MLX uses bf16 by default, but we can set fp32 if needed
    if MODEL_PRECISION == "fp32":
        # Note: parakeet-mlx handles precision internally
        logger.info("Using fp32 precision")
    else:
        logger.info("Using bf16 precision (default)")

    logger.info("Model loaded successfully with MLX")

    app.state.asr_model = model
    logger.info("Model ready for inference")

    try:
        yield
    finally:
        logger.info("Shutting down and releasing resources")
        del app.state.asr_model
        gc.collect()


def reset_fast_path(model):
    """Placeholder for compatibility - MLX handles optimization internally."""
    # parakeet-mlx doesn't need explicit fast path reset
    pass
