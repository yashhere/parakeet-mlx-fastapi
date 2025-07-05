import platform
import sys

from fastapi import FastAPI

from parakeet_service.config import logger
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


def create_app() -> FastAPI:
    server = FastAPI(
        title="Parakeet-TDT 1.1B STT service",
        version="0.0.1",
        description=(
            "High-accuracy English speech-to-text "
            "with optional word/char/segment timestamps."
        ),
        lifespan=lifespan,
    )
    server.include_router(router)

    # TODO: improve streaming and add support for other audio formats (maybe)
    server.include_router(stream_router)

    logger.info("FastAPI app initialised")
    return server


app = create_app()


def main() -> None:
    """Main entry point for the parakeet-service CLI command."""
    import os

    import uvicorn

    host = os.getenv("PARAKEET_HOST", "0.0.0.0")  # noqa: S104
    port = int(os.getenv("PARAKEET_PORT", "8000"))
    workers = int(os.getenv("PARAKEET_WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()

    logger.info(f"Starting Parakeet service on {host}:{port}")

    uvicorn.run(
        "parakeet_service.main:app",
        host=host,
        port=port,
        workers=workers,
        log_level=log_level,
        access_log=True,
        reload=False,
    )


if __name__ == "__main__":
    main()
