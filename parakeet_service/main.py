import platform
import sys
from typing import Optional

import typer
from fastapi import FastAPI

from parakeet_service import config
from parakeet_service.config import WORKERS, logger
from parakeet_service.model import lifespan
from parakeet_service.routes import router
from parakeet_service.stream_routes import router as stream_router


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
    server.include_router(stream_router)

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
    import uvicorn

    # Set log level based on verbosity
    if verbose == 0:
        log_level = "error"
    elif verbose == 1:
        log_level = "warning"
    elif verbose == 2:
        log_level = "info"
    else:  # verbose >= 3
        log_level = "debug"

    # Log the model being used
    if model != "mlx-community/parakeet-tdt-1.1b":
        logger.info(f"Using model: {model}")

    logger.info(f"Starting Parakeet service on {host}:{port}")

    # Create app with custom model if specified
    app_str = "parakeet_service.main:app"
    if model != config.DEFAULT_MODEL_NAME:
        global app
        app = create_app(model_name=model)

    uvicorn.run(
        app_str,
        host=host,
        port=port,
        workers=WORKERS,
        log_level=log_level,
        access_log=True,
        reload=False,
    )


def main() -> None:
    """Entry point for the CLI."""
    cli_app()


if __name__ == "__main__":
    main()
