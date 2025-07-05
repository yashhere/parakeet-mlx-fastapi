# Parakeet-TDT 1.1B FastAPI Service

[![Build Python Package](https://github.com/yashhere/parakeet-mlx-tdt-1.1b-fastapi/actions/workflows/build-package.yml/badge.svg)](https://github.com/yashhere/parakeet-mlx-tdt-1.1b-fastapi/actions/workflows/build-package.yml)

A production-ready FastAPI service for speech-to-text transcription using the Parakeet-TDT 1.1B model. This service runs on macOS systems, powered by `uv` for fast and reliable dependency management.

**⚠️ Platform Restriction: This package is only supported on macOS due to its dependency on MLX (Apple Silicon) and optimized audio processing libraries.**

## Features

- High-accuracy English speech-to-text transcription
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

### Install with `uv tool` (Recommended)

Install the service as a standalone tool that can be run from anywhere:

```bash
# Install directly from Git repository
uv tool install git+https://github.com/yashhere/parakeet-mlx-tdt-1.1b-fastapi.git

# Or install from local directory (if you have the source)
uv tool install .

# Run the service (available globally)
parakeet-service
```

### Local Development

For local development and testing:

```bash
# 1. Clone the repository
git clone https://github.com/yashhere/parakeet-mlx-tdt-1.1b-fastapi.git
cd parakeet-mlx-tdt-1.1b-fastapi

# 2. Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Install dependencies
uv sync

# 4. Run the service locally (without installing)
uv run parakeet-service

# 5. Run with custom options
uv run parakeet-service --host 127.0.0.1 --port 9000 --model mlx-community/parakeet-tdt-0.6b -vv
```

### Tool Management

After installing with `uv tool`, you can manage the installation:

```bash
# List all installed tools
uv tool list

# Show details about the installed tool
uv tool show parakeet-tdt-fastapi

# Upgrade to the latest version
uv tool upgrade parakeet-tdt-fastapi

# Uninstall the tool
uv tool uninstall parakeet-tdt-fastapi
```

### Alternative Installation Methods

#### Build and Install from Source

```bash
# Build the distribution packages
uv build

# Install the wheel package
uv pip install dist/parakeet_tdt_fastapi-0.1.0-py3-none-any.whl

# Run the installed CLI
parakeet-service
```

#### Direct Python Module Execution

```bash
# Install dependencies
uv sync

# Run the main module directly
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

Once the service is running, it will be available at `http://localhost:8765`.

### API Documentation

- Interactive API docs: `http://localhost:8765/docs`
- OpenAPI schema: `http://localhost:8765/openapi.json`

### Basic Transcription

```bash
# Upload an audio file for transcription
curl -X POST "http://localhost:8765/transcribe" \
     -H "Content-Type: multipart/form-data" \
     -F "audio=@your_audio_file.wav"
```

## Configuration

The service can be configured using environment variables. Copy `.env.example` to `.env` and modify as needed:

```bash
cp .env.example .env
```

### Available Configuration Options

- `PARAKEET_HOST`: Server host (default: 0.0.0.0)
- `PARAKEET_PORT`: Server port (default: 8765)
- `PARAKEET_WORKERS`: Number of worker processes (default: 1)
- `TARGET_SR`: Target sample rate (default: 16000)
- `MODEL_PRECISION`: Model precision (default: bf16)
- `BATCH_SIZE`: Batch size for processing (default: 4)
- `MAX_AUDIO_DURATION`: Maximum audio duration in seconds (default: 45)
- `PROCESSING_TIMEOUT`: Processing timeout in seconds (default: 120)
- `LOG_LEVEL`: Logging level (default: INFO)

## CLI Usage

The service provides a simple command-line interface with essential options.

### Basic Usage

```bash
# Start the service with default settings
parakeet-service

# Start with custom host and port
parakeet-service --host 127.0.0.1 --port 9000

# Start with a different model
parakeet-service --model mlx-community/parakeet-tdt-0.6b

# Start with verbose logging
parakeet-service -vv

# Combine multiple options
parakeet-service --host 127.0.0.1 --port 9000 --model mlx-community/parakeet-tdt-0.6b -vvv
```

### CLI Options

**Available Options:**

- `--host, -h`: Host to bind the server to (default: 0.0.0.0)
- `--port, -p`: Port to bind the server to (default: 8765)
- `--model, -m`: Model name to use (default: mlx-community/parakeet-tdt-1.1b)
- `--verbose, -v`: Increase verbosity (-v for WARNING, -vv for INFO, -vvv for DEBUG)

### Environment Variables

The service also supports configuration via environment variables:

- `PARAKEET_WORKERS`: Number of worker processes (default: 1)

### CLI Help

Get help for the CLI:

```bash
# Show help and available options
parakeet-service --help
```

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
