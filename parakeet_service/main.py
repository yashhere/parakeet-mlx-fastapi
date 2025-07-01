from fastapi import FastAPI

from parakeet_service.model import lifespan
from parakeet_service.routes import router
from parakeet_service.config import logger

from parakeet_service.stream_routes import router as stream_router


def create_app() -> FastAPI:
    server = FastAPI(
        title="Parakeet-TDT-CTC 1.1B v2 STT service",
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
