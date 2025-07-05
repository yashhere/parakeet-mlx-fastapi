# Parakeet-TDT 1.1B FastAPI Service

A production-ready FastAPI service for speech-to-text transcription using the Parakeet-TDT 1.1B model. This service runs on macOS systems, powered by `uv` for fast and reliable dependency management.

**⚠️ Platform Restriction: This package is only supported on macOS due to its dependency on MLX (Apple Silicon) and optimized audio processing libraries.**

## Features

- High-accuracy English speech-to-text transcription
- Real-time streaming support
- Word/character/segment timestamps
- RESTful API with FastAPI
- macOS native optimization with MLX
- Comprehensive logging and monitoring
- `uv`-powered binary distribution and fast installs

## Requirements

- **macOS only** (macOS 11.0+ recommended)
- **Apple Silicon Mac** (M1, M2, M3, etc.) for optimal performance
- Python 3.10+
- FFmpeg
- 4GB+ RAM (recommended)
- `uv` package manager

## Quick Start

### Option 1: Development with `uv` (Recommended)

1. Install `uv` if not already installed:

   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the repository and sync dependencies:

   ```bash
   git clone <repository>
   cd parakeet-mlx-tdt-1.1b-fastapi
   uv sync
   ```

3. Run the service directly:

   ```bash
   uv run parakeet-service
   ```

### Option 2: Build and Install Binary Distribution

1. Build the distribution packages:

   ```bash
   uv build
   ```

2. Install the wheel package:

   ```bash
   uv pip install dist/parakeet_tdt_fastapi-0.1.0-py3-none-any.whl
   ```

3. Run the installed CLI:

   ```bash
   parakeet-service
   ```

### Option 3: Direct Python Execution

1. Install dependencies:

   ```bash
   uv sync
   ```

2. Run the main module directly:

   ```bash
   uv run python -m parakeet_service.main
   ```

## Why macOS Only?

This package is restricted to macOS for several reasons:

1. **MLX Framework**: Optimized for Apple Silicon processors
2. **Large Model Downloads**: The service downloads multi-GB language models that work best in unconstrained environments
3. **Audio Processing**: Native macOS audio libraries provide optimal performance
4. **Memory Management**: Better memory handling for large models on Apple Silicon

## Development with `uv`

### Setting up Development Environment

```bash
# Install dev dependencies (includes pre-commit, ruff)
uv sync --extra dev

# Install pre-commit hooks
uv run pre-commit install
```

#### Running Pre-commit Manually

```bash
# Run on all files
uv run pre-commit run --all-files
```

### Adding Dependencies

```bash
# Add a new dependency
uv add package-name

# Add a development dependency
uv add --dev package-name

# Add with version constraints
uv add "fastapi>=0.100.0"
```

### Running Development

```bash
# Install dev dependencies
uv sync --extra dev

# Format and lint code
uv run ruff check .
uv run ruff format .

# Or run both linting and formatting together
uv run ruff check --fix .
```

## API Usage

Once the service is running, it will be available at `http://localhost:8000`.

### API Documentation

- Interactive API docs: `http://localhost:8000/docs`
- OpenAPI schema: `http://localhost:8000/openapi.json`

### Basic Transcription

```bash
# Upload an audio file for transcription
curl -X POST "http://localhost:8000/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "audio=@your_audio_file.wav"
```

### Streaming Transcription

The service supports real-time streaming transcription. See the streaming endpoints in the API documentation.

## Configuration

The service can be configured using environment variables. Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

### Available Configuration Options

- `PARAKEET_HOST`: Server host (default: 0.0.0.0)
- `PARAKEET_PORT`: Server port (default: 8000)
- `PARAKEET_WORKERS`: Number of worker processes (default: 1)
- `TARGET_SR`: Target sample rate (default: 16000)
- `MODEL_PRECISION`: Model precision (default: bf16)
- `BATCH_SIZE`: Batch size for processing (default: 4)
- `MAX_AUDIO_DURATION`: Maximum audio duration in seconds (default: 45)
- `PROCESSING_TIMEOUT`: Processing timeout in seconds (default: 120)
- `LOG_LEVEL`: Logging level (default: INFO)

## Binary Distribution

### Building Binaries

```bash
# Build wheel and source distribution
uv build

# Build only wheel
uv build --wheel

# Build only source distribution
uv build --sdist
```

### Installing from Binary

```bash
# Install from local wheel
uv pip install dist/parakeet_tdt_fastapi-0.1.0-py3-none-any.whl

# Install with extras
uv pip install "parakeet-tdt-fastapi[dev]"
```

## File Structure

```
parakeet-mlx-tdt-1.1b-fastapi/
├── parakeet_service/          # Main application package
│   ├── main.py               # FastAPI app and CLI entry point
│   ├── routes.py             # API routes
│   ├── stream_routes.py      # Streaming routes
│   ├── model.py              # Model handling
│   ├── config.py             # Configuration
│   └── ...
├── dist/                      # Built distribution packages
├── pyproject.toml            # Project configuration and dependencies
├── uv.lock                   # Locked dependencies
├── requirements.txt          # Legacy pip requirements (deprecated)
├── .env.example             # Environment configuration template
└── README.md                # This file
```

## License

See LICENSE file for details.
