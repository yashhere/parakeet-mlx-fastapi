import platform
import sys
from typing import Optional

import typer
from fastapi import FastAPI

from parakeet_service import config
from parakeet_service.config import WORKERS, configure_logging, logger
from parakeet_service.model import lifespan
from parakeet_service.routes import router


# Platform check - ensure this package only runs on macOS
def _check_platform() -> None:
    """Check if running on macOS, exit if not."""
    if platform.system() != "Darwin":
        logger.error(
            f"Error: parakeet-mlx-tdt-1.1b-fastapi is only supported on macOS. "
            f"Current platform: {platform.system()}"
        )
        sys.exit(1)


# Perform platform check on import
_check_platform()


def create_app(model_name: Optional[str] = None) -> FastAPI:
    server = FastAPI(
        title="Parakeet-TDT 1.1B STT service",
        version="0.0.1",
        description=(
            "High-accuracy English speech-to-text "
            "with optional word/char/segment timestamps."
        ),
        lifespan=lifespan,
    )

    # Set model name in app state if provided
    if model_name:
        server.state.model_name = model_name

    server.include_router(router)

    # TODO: improve streaming and add support for other audio formats (maybe)

    logger.info("FastAPI app initialised")
    return server


app = create_app()


# Create typer app
cli_app = typer.Typer(
    help="High-accuracy English speech-to-text FastAPI service using Parakeet-TDT model"
)


@cli_app.command()
def cmd(
    host: str = typer.Option(
        "0.0.0.0",  # noqa: S104
        "--host",
        "-h",
        help="Host to bind the server to (default: 0.0.0.0)",
    ),
    port: int = typer.Option(
        8000,
        "--port",
        "-p",
        help="Port to bind the server to (default: 8000)",
    ),
    model: str = typer.Option(
        "mlx-community/parakeet-tdt-1.1b",
        "--model",
        "-m",
        help="Model name to use (default: mlx-community/parakeet-tdt-1.1b)",
    ),
    verbose: int = typer.Option(
        0,
        "--verbose",
        "-v",
        count=True,
        help="Increase verbosity (-v for WARNING, -vv for INFO, -vvv for DEBUG)",
    ),
) -> None:
    """Start the Parakeet speech-to-text service."""
    try:
        import uvicorn

        # Set log level based on verbosity
        if verbose == 0:
            log_level = "ERROR"
        elif verbose == 1:
            log_level = "WARNING"
        elif verbose == 2:
            log_level = "INFO"
        else:  # verbose >= 3
            log_level = "DEBUG"

        # Configure logging early with the determined level
        configure_logging(log_level)

        # Log the model being used
        if model != "mlx-community/parakeet-tdt-1.1b":
            logger.info(f"Using model: {model}")

        logger.info(f"Starting Parakeet service on {host}:{port}")

        # Configure custom model if specified
        if model != config.DEFAULT_MODEL_NAME:
            try:
                # Create a new app instance with the custom model
                app_instance = create_app(model_name=model)
                logger.info(f"Successfully configured custom model: {model}")

                # Update the global app reference for uvicorn to pick up
                global app
                app = app_instance

            except Exception as e:
                logger.error(f"Failed to configure model '{model}': {e}")
                logger.exception("Model configuration error details:")
                sys.exit(1)

        # Start the server with proper logging configuration
        uvicorn.run(
            app,
            host=host,
            port=port,
            workers=WORKERS,
            log_level=log_level.lower(),
            access_log=True,
            reload=False,
            lifespan="on",
            # Disable uvicorn's default logging configuration to use ours
            log_config=None,
        )

    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        sys.exit(0)
    except ImportError as e:
        logger.error(f"Missing required dependency: {e}")
        logger.error("Please ensure all dependencies are installed")
        sys.exit(1)
    except PermissionError as e:
        logger.error(f"Permission denied: {e}")
        logger.error(
            f"Unable to bind to {host}:{port} - check permissions or try a different port"
        )
        sys.exit(1)
    except OSError as e:
        if "Address already in use" in str(e):
            logger.error(
                f"Port {port} is already in use. Please choose a different port or stop the existing service."
            )
        else:
            logger.error(f"Network error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Failed to start service: {e}")
        logger.exception("Startup error details:")
        sys.exit(1)


def main() -> None:
    """Entry point for the CLI."""
    try:
        cli_app()
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"CLI error: {e}")
        logger.exception("CLI error details:")
        sys.exit(1)


if __name__ == "__main__":
    main()
