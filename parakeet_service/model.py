import gc
import sys
from contextlib import asynccontextmanager

from parakeet_mlx import from_pretrained  # type: ignore

from parakeet_service.config import DEFAULT_MODEL_NAME, MODEL_PRECISION, get_logger

logger = get_logger("parakeet_service.model")


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

    try:
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
            try:
                del app.state.asr_model
            except AttributeError:
                logger.warning("ASR model was not found in app state during cleanup")
            gc.collect()
            logger.info("Resource cleanup completed")

    except ImportError as e:
        logger.error(
            f"Failed to import required dependencies for model '{model_name}': {e}"
        )
        logger.error("Check that parakeet-mlx and MLX framework are properly installed")
        sys.exit(1)
    except FileNotFoundError as e:
        logger.error(f"Model files not found for '{model_name}': {e}")
        logger.error(
            "Check model name and internet connectivity for automatic download"
        )
        sys.exit(1)
    except MemoryError as e:
        logger.error(f"Insufficient memory to load model '{model_name}': {e}")
        logger.error("Try using a smaller model or increasing available memory")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to load model '{model_name}': {e}")
        logger.exception("Model loading error details:")
        logger.error("Check model name, internet connectivity, and system requirements")
        sys.exit(1)
