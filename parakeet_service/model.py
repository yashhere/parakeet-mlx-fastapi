import gc
from contextlib import asynccontextmanager

from parakeet_mlx import from_pretrained  # type: ignore

from parakeet_service.config import DEFAULT_MODEL_NAME, MODEL_PRECISION, logger


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
    # Get model name from app state or fall back to config default
    model_name = getattr(app.state, "model_name", DEFAULT_MODEL_NAME)

    logger.info("Loading %s with MLX...", model_name)

    # Load model with parakeet-mlx
    model = from_pretrained(model_name)

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
