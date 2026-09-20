FROM python:3.12-slim

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# 1. Install system utilities, phonemizer, audio tools, Mesa Vulkan drivers, and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    espeak-ng \
    ffmpeg \
    libvulkan1 \
    libvulkan-dev \
    glslc \
    mesa-vulkan-drivers \
    vulkan-tools \
    curl \
    git \
    cmake \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Install pre-built vendor-agnostic Vulkan wheel for llama-cpp-python FIRST
# (Pre-satisfies llama-cpp-python dependency so pip doesn't attempt to build from source)
RUN pip install --no-cache-dir \
    https://github.com/abetlen/llama-cpp-python/releases/download/v0.3.35-vulkan/llama_cpp_python-0.3.35-py3-none-manylinux2014_x86_64.manylinux_2_17_x86_64.whl

# 3. Install PyTorch & NeuTTS dependencies with pinned torchao to prevent torchtune import errors
RUN pip install --no-cache-dir \
    "numpy<3" \
    "torch" \
    "torchao<0.18.0" \
    "torchtune" \
    "neucodec" \
    "neutts[all]" \
    "huggingface_hub>=0.20.0" \
    fastapi \
    uvicorn \
    python-multipart \
    soundfile \
    pydantic \
    "wyoming[zeroconf]>=1.5.0"

# 4. Build and install pywhispercpp with native Vulkan acceleration
RUN git clone --depth 1 --recursive https://github.com/absadiki/pywhispercpp.git /tmp/pywhispercpp \
    && cd /tmp/pywhispercpp \
    && GGML_VULKAN=1 CMAKE_ARGS="-DGGML_VULKAN=1" pip install --no-cache-dir . \
    && rm -rf /tmp/pywhispercpp

# 5. Copy server application code and static web portal
COPY openai_server.py cache_reference.py /app/
COPY static/ /app/static/

# 6. Create voices, certs, and whisper cache directories
RUN mkdir -p /app/voices /app/certs /cache/whisper
VOLUME ["/app/voices"]

# Default environment configuration
ENV PORT=8090
ENV HOST=0.0.0.0
ENV VOICES_DIR=/app/voices
ENV STATIC_DIR=/app/static
ENV DEFAULT_VOICE=""
ENV ENABLE_WYOMING=false
ENV WYOMING_PORT=10200
ENV WYOMING_HOST=0.0.0.0
ENV ENABLE_WHISPER=true
ENV WHISPER_MODEL=large-v3-turbo
ENV WHISPER_LANG=en
ENV WHISPER_MODELS_DIR=/cache/whisper
ENV WHISPER_THREADS=4
ENV SSL_KEYFILE=""
ENV SSL_CERTFILE=""
ENV SSL_KEYFILE_PASSWORD=""
ENV SSL_CA_CERTS=""

EXPOSE 8090 10200

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -s -f -k http://localhost:8090/api/status || curl -s -f -k https://localhost:8090/api/status || exit 1

CMD ["python", "openai_server.py"]
