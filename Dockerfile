FROM python:3.10.7-slim AS base
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        git  build-essential  libsndfile1  ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY ./parakeet_service ./parakeet_service
COPY ./requirements.txt ./requirements.txt

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN pip install --no-cache-dir --upgrade pip \
 && pip install torch==2.7.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir \
 && pip install nemo_toolkit["asr"] \
 && pip install 'uvicorn[standard]' --no-cache-dir \
 && pip install --no-cache-dir -r requirements.txt \
 && pip cache purge

EXPOSE 8000
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app
CMD ["uvicorn", "parakeet_service.main:app", \
     "--host", "0.0.0.0", "--port", "8000"]