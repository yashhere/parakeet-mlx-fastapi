import gc
from contextlib import asynccontextmanager

from parakeet_mlx import from_pretrained  # type: ignore

from parakeet_service.config import DEFAULT_MODEL_NAME, MODEL_PRECISION, get_logger

logger = get_logger("parakeet_service.model")


class ModelLoadingError(Exception):
    """Custom exception for model loading failures."""

    pass


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

    # Initialize app state
    app.state.model_loaded = False
    app.state.model_error = None
    app.state.asr_model = None

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
        app.state.model_loaded = True
        logger.info("Model ready for inference")

    except ImportError as e:
        error_msg = (
            f"Failed to import required dependencies for model '{model_name}': {e}"
        )
        logger.error(error_msg)
        logger.error("Check that parakeet-mlx and MLX framework are properly installed")
        app.state.model_error = f"Missing dependencies: {e}"

    except FileNotFoundError as e:
        error_msg = f"Model files not found for '{model_name}': {e}"
        logger.error(error_msg)
        logger.error(
            "Check model name and internet connectivity for automatic download"
        )
        app.state.model_error = f"Model files not found: {e}"

    except MemoryError as e:
        error_msg = f"Insufficient memory to load model '{model_name}': {e}"
        logger.error(error_msg)
        logger.error("Try using a smaller model or increasing available memory")
        app.state.model_error = f"Insufficient memory: {e}"

    except Exception as e:
        error_msg = f"Failed to load model '{model_name}': {e}"
        logger.error(error_msg)
        logger.exception("Model loading error details:")
        logger.error("Check model name, internet connectivity, and system requirements")
        app.state.model_error = f"Model loading failed: {e}"

    # Always yield to allow the app to start, even if model loading failed
    try:
        yield
    finally:
        logger.info("Shutting down and releasing resources")
        try:
            if hasattr(app.state, "asr_model") and app.state.asr_model is not None:
                del app.state.asr_model
        except AttributeError:
            logger.warning("ASR model was not found in app state during cleanup")
        gc.collect()
        logger.info("Resource cleanup completed")
