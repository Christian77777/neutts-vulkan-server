# NeuTTS Server

An OpenAI-compatible streaming Text-to-Speech (TTS) and Speech-to-Text (STT) server powered by **NeuTTS-Air** and **Whisper (`large-v3-turbo`)**, accelerated with **Vulkan**.

Supports zero-shot voice cloning with 8-bit quantization (`neuphonic/neutts-air-q8-gguf`) and high-speed Whisper transcription resident in GPU VRAM across **Intel Arc, AMD Radeon, and NVIDIA GPUs** (with CPU fallback).

---

## Features

- **Hardware Acceleration via Vulkan:** Offloads LLM backbone inference and Whisper STT using Vulkan compute support for Intel Arc, AMD Radeon, and NVIDIA GPUs.
- **Whisper Speech-to-Text (`large-v3-turbo`):** GPU-accelerated local transcription via `pywhispercpp` with Vulkan compute shaders, keeping model weights resident in VRAM for instant inference.
- **Unified Home Assistant Wyoming Pipeline:** Native TCP streaming for both Text-to-Speech (`TtsProgram`) and Speech-to-Text (`AsrProgram`) on port `10200` for Home Assistant Assist pipelines and satellites.
- **Voice Management Web Portal:** Single-page interface at `http://localhost:8090/` to inspect, clone, test, and delete voices with in-browser audio playback.
- **Streaming Audio:** Emits chunked 24 kHz 16-bit mono PCM (`response_format: pcm`), MP3, or Opus with sub-second Time-To-First-Byte (TTFB).
- **Voice Cloning:** Reference audio is encoded directly into the mounted volume (`/app/voices`) and cached in memory without restarting the container.
- **OpenAI API Compatibility:** Implements `/v1/audio/speech`, `/v1/audio/transcriptions`, and `/v1/models` endpoints for compatibility with OpenAI SDKs and third-party tools (Hermes Agent, Open-WebUI, LibreChat).
- **Persistent Storage:** Voices and downloaded Whisper models persist across container restarts via host-mounted volumes.
- **Security Features:** Optional Bearer token authentication (`API_KEY`), configurable CORS origins, and security headers (CSP, HSTS, X-Frame-Options).
- **SSL / TLS Support:** Direct HTTPS support with certificates (`SSL_CERTFILE` & `SSL_KEYFILE`), automatic HSTS, and dual HTTP/HTTPS healthchecks.

---

## Performance Benchmarks

Measured on an **Intel Arc B570 (Battlemage G21)** host synthesizing a 6.4-second utterance:

| Metric | Measurement |
| :--- | :--- |
| **Backbone Inference Speed** | **~166 tokens/second** (all 24 layers offloaded to Vulkan) |
| **Phonemizer Latency** | **< 1 ms** (`espeak-ng`) |
| **NeuCodec Audio Decode** | **~280 ms** |
| **Streaming TTFB (Time-To-First-Byte)** | **~500–720 ms** (raw PCM) |
| **Batch End-to-End Latency** | **~2.15 s** |
| **Real-Time Factor (RTF)** | **0.34** (~3x faster than real-time playback) |

---

## Quick Start (Docker)

The server mounts a host directory to `/app/voices` for storing cloned voices.

### 1. Intel Arc & AMD Radeon GPUs

```bash
# Clone repository
git clone https://github.com/Christian77777/neutts-vulkan-server.git
cd neutts-vulkan-server

# Start container with host GPU devices passed through
docker compose up -d

# Open the Web Portal in your browser
xdg-open http://localhost:8090
```

> **Note on Render Group Permissions:**
> `docker-compose.yml` configures `group_add: ["989"]` to match the host `render` GID. If your host uses a different GID (check with `getent group render`), update this number in `docker-compose.yml`.

### 2. NVIDIA GPUs

```bash
docker compose -f docker-compose.nvidia.yml up -d
```

---

## Voice Management Web Portal

Navigate to `http://localhost:8090/` to access the built-in management interface:

- **Voices Catalog:** View installed voices in the volume, inspect transcripts, check file sizes, and delete voices.
- **Clone Voice:** Upload a 3–10s audio sample (`.wav` or `.mp3`), enter the reference transcript, and submit. Reference tokens are saved to `./voices` and registered in memory immediately.
- **Speech Playground:** Test installed voices with custom text, pick response formats (`mp3`, `opus`, `wav`, `pcm`), and view latency metrics (TTFB, total latency, duration, RTF).
- **API & Client Integration:** Configuration snippets for `curl`, OpenAI Python SDK, and Hermes Agent `config.yaml`.

---

## Adding Voices

### Method 1: Via the Web Portal
Open `http://localhost:8090/`, select **Clone & Add Voice**, upload a 3–10s audio file with its exact transcript, and submit.

### Method 2: Via REST API (`multipart/form-data`)
```bash
curl -X POST http://localhost:8090/v1/audio/voices \
  -F "name=my_voice" \
  -F "text=Exact spoken transcript of the audio sample." \
  -F "audio_file=@/path/to/reference.wav"
```

### Method 3: Via Pre-Encoding CLI
To generate voice reference files offline:
```bash
python cache_reference.py \
    --audio /path/to/reference.wav \
    --text "Exact spoken transcript." \
    --name my_voice
```
This saves `my_voice.pt` and `my_voice.txt` into `./voices/`.

---

## API Usage & Examples

### 1. Streaming Audio

Streaming to `mpv`:
```bash
curl -s -N -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Streaming audio directly to speakers.",
    "response_format": "mp3",
    "stream": true
  }' | mpv --no-terminal -
```

Streaming raw PCM to `aplay`:
```bash
curl -s -N -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Raw PCM streaming test.",
    "response_format": "pcm"
  }' | aplay -q -r 24000 -f S16_LE -t raw -c 1
```

### 2. Batch Synthesis (Save to File)

```bash
curl -X POST http://localhost:8090/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "neutts",
    "input": "Hello world from NeuTTS.",
    "voice": "my_voice",
    "response_format": "mp3"
  }' \
  --output speech.mp3
```

### 3. OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8090/v1",
    api_key="local"
)

# Streaming PCM
with client.audio.speech.with_streaming_response.create(
    model="neutts",
    voice="my_voice",
    input="Streaming voice conversation.",
    response_format="pcm"
) as response:
    for pcm_chunk in response.iter_bytes():
        # Process 24kHz 16-bit PCM chunk
        pass
```

### 4. Hermes Agent Configuration

In Hermes `config.yaml`:

```yaml
tts:
  provider: openai
  openai:
    base_url: "http://<SERVER_IP>:8090/v1"
    api_key: "local-neutts"
    model: "neutts"
    voice: "my_voice"
  use_gateway: false
```

---

## SSL / TLS Encryption (HTTPS)

### 1. Certificates Directory
Place your certificate (`cert.pem`) and private key (`key.pem`) into the `certs/` directory:

```bash
# Generate self-signed certificate for local testing:
openssl req -x509 -newkey rsa:4096 -keyout certs/key.pem -out certs/cert.pem -sha256 -days 365 -nodes \
    -subj "/CN=localhost"
```

> **Note:** Certificate and key files in `certs/` are ignored by `.gitignore` and `.dockerignore`.

### 2. Enable in `docker-compose.yml`
Uncomment the volume mount and environment variables:

```yaml
    volumes:
      - ~/.cache/huggingface:/cache/huggingface
      - ./voices:/app/voices
      - ./certs:/app/certs:ro
    user: "1000:1000"
    cap_drop:
      - ALL
    security_opt:
      - no-new-privileges:true
    environment:
      - HF_HOME=/cache/huggingface
      - SSL_KEYFILE=/app/certs/key.pem
      - SSL_CERTFILE=/app/certs/cert.pem
```

Restart the container:
```bash
docker compose up -d
```

---

## Home Assistant Integration (Wyoming Protocol)

The server includes native support for the **Wyoming protocol**, allowing Home Assistant to use NeuTTS as the Text-to-Speech (TTS) engine for [Local Voice Assistants (Assist pipelines)](https://www.home-assistant.io/voice_control/voice_remote_local_assistant/#installing-a-local-assist-pipeline) and voice satellites (such as ESP32-S3-BOX, Raspberry Pi satellites, or Home Assistant Voice PE).

### Why Wyoming?
* **Sub-second Time-To-First-Audio (TTFA):** Audio begins streaming directly to voice satellites while the sentence is still being generated.
* **Zero Transcoding Overhead:** Directly transmits raw 24 kHz 16-bit mono PCM chunks over TCP without requiring FFmpeg MP3 encoding/decoding.
* **Native HA Integration:** Appears directly under Home Assistant's native Voice Assistant settings without needing custom HACS integrations.

> [!NOTE]
> **Authentication:** The Wyoming protocol is unauthenticated by design and does not support API keys. Port `10200` should only be accessible within your trusted local network.

### 1. Enable Wyoming in `docker-compose.yml`

Uncomment the port mapping and set `ENABLE_WYOMING=true`:

```yaml
services:
  neutts-vulkan:
    ports:
      - "8090:8090"      # OpenAI HTTP API & Web Management Portal
      - "10200:10200"    # Wyoming Protocol (Home Assistant)
    environment:
      - ENABLE_WYOMING=true
      - WYOMING_PORT=10200
      - WYOMING_HOST=0.0.0.0
```

Restart your container:
```bash
docker compose up -d
```

### 2. Connect Home Assistant to NeuTTS

1. In Home Assistant, navigate to **Settings > Devices & Services**.
2. Click **Add Integration** and search for **Wyoming Protocol**.
3. Configure the connection:
   * **Host:** IP address of your NeuTTS Docker host (or Docker service name if on the same network).
   * **Port:** `10200`
4. Home Assistant will discover NeuTTS and register both:
   * **Text-to-speech:** `NeuTTS` (with all installed cloned voices).
   * **Speech-to-text:** `whisper` (`large-v3-turbo` model resident in GPU VRAM).
5. Go to **Settings > Voice assistants**, click on your assistant pipeline:
   * Under **Speech-to-text**, select **whisper** (NeuTTS).
   * Under **Text-to-speech**, select **NeuTTS**.

---

## Available Endpoints & Protocols

| Protocol / Method | Endpoint / Port | Description | Auth Required |
| :--- | :--- | :--- | :--- |
| `HTTP GET` | `http://<host>:8090/` | Web Management Portal UI | No |
| `HTTP GET` | `http://<host>:8090/v1/models` | List models (`tts-1`, `tts-1-hd`, `neutts`, `whisper-1`, `large-v3-turbo`) | Yes (if `API_KEY` set) |
| `HTTP GET` | `http://<host>:8090/v1/audio/voices` | List installed voice clones & transcripts | Yes (if `API_KEY` set) |
| `HTTP POST` | `http://<host>:8090/v1/audio/voices` | Upload and encode a reference voice (`multipart/form-data`) | Yes (if `API_KEY` set) |
| `HTTP DELETE`| `http://<host>:8090/v1/audio/voices/{id}` | Delete a voice clone from persistent volume | Yes (if `API_KEY` set) |
| `HTTP POST` | `http://<host>:8090/v1/audio/speech` | OpenAI-compatible TTS (Streaming & Batch) | Yes (if `API_KEY` set) |
| `HTTP POST` | `http://<host>:8090/v1/audio/transcriptions` | OpenAI-compatible STT (`whisper-1`, `large-v3-turbo`) | Yes (if `API_KEY` set) |
| `HTTP GET` | `http://<host>:8090/api/status` | Server status, backend device, capability flags | No |
| `TCP` | `tcp://<host>:10200` | **Wyoming Protocol** for Home Assistant Assist (TTS + STT) | **No (Unauthenticated)** |

---

## License

Released under the [MIT License](LICENSE).
NeuTTS model weights and NeuCodec are subject to their respective upstream licenses.
