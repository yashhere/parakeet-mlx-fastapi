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
uv sync --group dev

# Format code
uv run black .

# Lint code
uv run ruff check .

# Type checking
uv run mypy parakeet_service/
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

## Migration from pip to uv

If you're migrating from the previous pip-based setup:

1. **Dependencies are now managed in `pyproject.toml`** instead of `requirements.txt`
2. **Lock file**: `uv.lock` ensures reproducible builds
3. **CLI command**: `uv run parakeet-service` instead of `python run_service.py`
4. **Development**: Use `uv sync` instead of `pip install -r requirements.txt`

## Performance Benefits of `uv`

- **Faster installs**: Up to 10-100x faster than pip
- **Better dependency resolution**: More reliable conflict resolution
- **Reproducible builds**: Lock file ensures consistent environments
- **Better isolation**: Virtual environments managed automatically
- **Binary distributions**: Easy to create and distribute packages

## Troubleshooting

### Platform Check Failed

If you see "parakeet-tdt-fastapi is only supported on macOS", you're trying to run this on a non-macOS system. This package requires macOS with Apple Silicon.

### Service Won't Start

1. Verify `uv` installation:

   ```bash
   uv --version
   ```

2. Test CLI directly:

   ```bash
   uv run parakeet-service
   ```

3. Check for dependency issues:

   ```bash
   uv sync
   ```

### `uv` Not Found

If `uv` is not found, install it manually:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"
```

### Dependency Issues

```bash
# Regenerate lock file
uv lock

# Update dependencies
uv sync --upgrade

# Check for conflicts
uv tree
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Install development dependencies: `uv sync --group dev`
4. Make your changes
5. Format code: `uv run black .`
6. Submit a pull request

## License

See LICENSE file for details.

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the service logs in the terminal output
3. Open an issue with detailed information about your environment and the problem
