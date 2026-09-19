import os
import re
import time
import shutil
import secrets
import tempfile
import subprocess
import threading
from pathlib import Path
from typing import Optional, Dict, Tuple
import numpy as np
import torch
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, Depends, Security
from fastapi.responses import Response, StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
from pydantic import BaseModel
from neutts import NeuTTS

app = FastAPI(
    title="NeuTTS Server (OpenAI-Compatible Speech Engine)",
    description="High-performance streaming text-to-speech engine accelerated by Vulkan with dynamic voice cloning portal.",
    version="1.2.0"
)

# --- SECURITY & AUTHENTICATION CONFIGURATION ---

# 1. CORS Configuration (Not enabled by default; explicitly allowed if user sets CORS_ORIGINS=* or specific origins)
cors_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
if cors_origins_env == "*":
    cors_origins = ["*"]
elif cors_origins_env:
    cors_origins = [o.strip() for o in cors_origins_env.split(",") if o.strip()]
else:
    cors_origins = []

if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials="*" not in cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# 2. Security Headers Middleware
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' 'unsafe-inline'; "
        "style-src 'self' 'unsafe-inline'; "
        "img-src 'self' data:; "
        "media-src 'self' blob:; "
        "connect-src 'self';"
    )
    if request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https":
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    return response


# 3. Optional API Key Authentication
API_KEY = os.environ.get("API_KEY", "").strip()
security = HTTPBearer(auto_error=False)


def verify_api_key(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
):
    """Enforces authentication if API_KEY is set in environment (constant-time check)."""
    if not API_KEY:
        return True

    token = None
    if credentials and credentials.credentials:
        token = credentials.credentials
    elif "x-api-key" in request.headers:
        token = request.headers.get("x-api-key")

    if not token or not secrets.compare_digest(token, API_KEY):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized: Invalid or missing API key. Provide 'Authorization: Bearer <API_KEY>' or 'X-API-Key' header.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return True


# 4. Standard OpenAI-compatible error response handlers
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "message": str(exc.detail),
                "type": "invalid_request_error",
                "param": None,
                "code": exc.status_code
            }
        }
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=400,
        content={
            "error": {
                "message": str(exc),
                "type": "invalid_request_error",
                "param": None,
                "code": 400
            }
        }
    )

# Global instances & locks
tts_model: Optional[NeuTTS] = None
tts_lock = threading.Lock()
voice_cache: Dict[str, Tuple[torch.Tensor, str]] = {}

STATIC_DIR = Path(os.environ.get("STATIC_DIR", "./static")).resolve()
VOICES_DIR = Path(os.environ.get("VOICES_DIR", "./voices")).resolve()
DEFAULT_VOICE = os.environ.get("DEFAULT_VOICE", "")
MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB max reference audio
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
VOICE_NAME_REGEX = re.compile(r'^[a-z0-9_-]{1,64}$')


def validate_voice_name(name: str) -> str:
    """Validate voice name against path traversal, special characters, and length limits."""
    clean = (name or "").strip().lower().replace(" ", "_")
    if not clean:
        raise HTTPException(status_code=400, detail="Voice name cannot be empty")
    if not VOICE_NAME_REGEX.match(clean):
        raise HTTPException(
            status_code=400,
            detail="Invalid voice name. Only letters, numbers, dashes, and underscores allowed (1-64 chars)."
        )
    return clean


def resolve_voice_file(name: str, ext: str) -> Path:
    """Safely resolve a voice file path and prevent directory traversal outside VOICES_DIR."""
    clean = validate_voice_name(name)
    target = (VOICES_DIR / f"{clean}{ext}").resolve()
    if not target.is_relative_to(VOICES_DIR):
        raise HTTPException(status_code=400, detail="Invalid path traversal attempt")
    return target


def get_available_voice_names() -> list[str]:
    """Return all available voice IDs found in VOICES_DIR."""
    if not VOICES_DIR.exists():
        return []
    return sorted([p.stem for p in VOICES_DIR.glob("*.pt")])


def load_voice(voice_name: str) -> Tuple[torch.Tensor, str]:
    """Load and cache reference tokens and text for a specified voice."""
    try:
        clean_name = validate_voice_name(voice_name)
    except HTTPException:
        clean_name = ""

    if clean_name and clean_name in voice_cache:
        return voice_cache[clean_name]

    pt_path = resolve_voice_file(clean_name, ".pt") if clean_name else None
    txt_path = resolve_voice_file(clean_name, ".txt") if clean_name else None

    if not pt_path or not pt_path.exists():
        available = get_available_voice_names()
        if not available:
            raise FileNotFoundError(
                "No voice clones found in the voices volume. "
                "Please upload a voice sample via the Web Portal first."
            )
        # Prioritize DEFAULT_VOICE if configured and installed; otherwise fallback to first available
        fallback = DEFAULT_VOICE if (DEFAULT_VOICE and DEFAULT_VOICE in available) else available[0]
        print(f"[VOICE] Requested voice '{voice_name}' not found; falling back to '{fallback}'")
        return load_voice(fallback)

    print(f"[VOICE] Loading voice '{clean_name}' from {pt_path}...")
    ref_codes = torch.load(pt_path, weights_only=True)
    ref_text = ""
    if txt_path and txt_path.exists():
        with open(txt_path, "r", encoding="utf-8") as f:
            ref_text = f.read().strip()

    voice_cache[clean_name] = (ref_codes, ref_text)
    return ref_codes, ref_text


@app.on_event("startup")
def startup_event():
    global tts_model

    # Ensure voices volume directory exists
    VOICES_DIR.mkdir(parents=True, exist_ok=True)

    backbone_repo = os.environ.get("BACKBONE_REPO", "neuphonic/neutts-air-q8-gguf")
    backbone_device = os.environ.get("BACKBONE_DEVICE", "gpu")
    codec_repo = os.environ.get("CODEC_REPO", "neuphonic/neucodec")
    codec_device = os.environ.get("CODEC_DEVICE", "cpu")

    print(f"[INIT] Starting NeuTTS: backbone={backbone_repo} ({backbone_device}), codec={codec_repo} ({codec_device})...")

    tts_model = NeuTTS(
        backbone_repo=backbone_repo,
        backbone_device=backbone_device,
        codec_repo=codec_repo,
        codec_device=codec_device,
    )

    available = get_available_voice_names()
    if available:
        print(f"[INIT] Available voices in volume: {', '.join(available)}")
        first_voice = DEFAULT_VOICE if DEFAULT_VOICE in available else available[0]
        try:
            load_voice(first_voice)
        except Exception as e:
            print(f"[INIT WARNING] Could not pre-load voice '{first_voice}': {e}")
    else:
        print("[INIT] Voices volume is empty. Voices can be uploaded anytime via the Web Portal.")

    print("[INIT] NeuTTS Server ready!")


class SpeechRequest(BaseModel):
    model: str = "tts-1"
    input: str
    voice: Optional[str] = "alloy"
    response_format: Optional[str] = "mp3"  # mp3, pcm, wav, opus, flac, aac
    speed: Optional[float] = 1.0
    stream: Optional[bool] = None  # None/True = stream for pcm/mp3/opus; False = batch


def get_atempo_filter(speed: float) -> str:
    """Build FFmpeg atempo filter chain for speeds between 0.25 and 4.0."""
    speed = max(0.25, min(4.0, speed))
    if abs(speed - 1.0) < 0.01:
        return ""
    filters = []
    curr = speed
    while curr > 2.0:
        filters.append("atempo=2.0")
        curr /= 2.0
    while curr < 0.5:
        filters.append("atempo=0.5")
        curr /= 0.5
    filters.append(f"atempo={curr:.4f}")
    return ",".join(filters)


def pcm_streamer(text: str, ref_codes: torch.Tensor, ref_text: str, speed: float = 1.0):
    """Yields 24kHz mono 16-bit little-endian PCM bytes."""
    with tts_lock:
        t0 = time.time()
        tempo = get_atempo_filter(speed)
        if not tempo:
            first = True
            total_samples = 0
            for chunk in tts_model.infer_stream(text, ref_codes, ref_text):
                if first:
                    print(f"[STREAM-PCM] First chunk TTFA: {time.time()-t0:.3f}s")
                    first = False
                total_samples += len(chunk)
                pcm_bytes = (chunk * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
                yield pcm_bytes
            print(f"[STREAM-PCM] Finished in {time.time()-t0:.2f}s (audio: {total_samples/24000:.2f}s)")
        else:
            # Route through FFmpeg for real-time atempo scaling on raw PCM
            cmd = [
                "ffmpeg", "-y",
                "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
                "-filter:a", tempo,
                "-f", "s16le", "-ar", "24000", "-ac", "1", "pipe:1"
            ]
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL
            )

            def feeder():
                try:
                    for chunk in tts_model.infer_stream(text, ref_codes, ref_text):
                        pcm_bytes = (chunk * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
                        proc.stdin.write(pcm_bytes)
                        proc.stdin.flush()
                except Exception as e:
                    print(f"[FEEDER ERROR] {e}")
                finally:
                    try:
                        proc.stdin.close()
                    except Exception:
                        pass

            feed_thread = threading.Thread(target=feeder, daemon=True)
            feed_thread.start()

            t0 = time.time()
            first = True
            try:
                while True:
                    data = proc.stdout.read(4096)
                    if not data:
                        break
                    if first:
                        print(f"[STREAM-PCM-TEMPO] First chunk TTFA: {time.time()-t0:.3f}s (speed={speed})")
                        first = False
                    yield data
            finally:
                try:
                    proc.stdout.close()
                except Exception:
                    pass
                if proc.poll() is None:
                    proc.kill()
                feed_thread.join(timeout=2.0)
                proc.wait(timeout=2.0)
                print(f"[STREAM-PCM-TEMPO] Finished in {time.time()-t0:.2f}s")


def ffmpeg_pipe_streamer(text: str, ref_codes: torch.Tensor, ref_text: str, fmt: str, speed: float = 1.0):
    """Live compressed audio stream via FFmpeg stdin/stdout pipe with tempo support."""
    with tts_lock:
        codec_name = "libmp3lame" if fmt == "mp3" else "libopus"
        bitrate = "128k" if fmt == "mp3" else "96k"
        cmd = [
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0"
        ]
        tempo = get_atempo_filter(speed)
        if tempo:
            cmd.extend(["-filter:a", tempo])
        cmd.extend([
            "-codec:a", codec_name, "-b:a", bitrate,
            "-f", fmt, "pipe:1"
        ])
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL
        )

        def feeder():
            try:
                for chunk in tts_model.infer_stream(text, ref_codes, ref_text):
                    pcm_bytes = (chunk * 32767.0).clip(-32768, 32767).astype(np.int16).tobytes()
                    proc.stdin.write(pcm_bytes)
                    proc.stdin.flush()
            except Exception as e:
                print(f"[FEEDER ERROR] {e}")
            finally:
                try:
                    proc.stdin.close()
                except Exception:
                    pass

        feed_thread = threading.Thread(target=feeder, daemon=True)
        feed_thread.start()

        t0 = time.time()
        first = True
        try:
            while True:
                data = proc.stdout.read(4096)
                if not data:
                    break
                if first:
                    print(f"[STREAM-{fmt.upper()}] First chunk TTFA: {time.time()-t0:.3f}s (speed={speed})")
                    first = False
                yield data
        finally:
            try:
                proc.stdout.close()
            except Exception:
                pass
            if proc.poll() is None:
                proc.kill()
            feed_thread.join(timeout=2.0)
            proc.wait(timeout=2.0)
            print(f"[STREAM-{fmt.upper()}] Finished in {time.time()-t0:.2f}s")


# --- WEB PORTAL & MANAGEMENT ROUTES ---

@app.api_route("/", methods=["GET", "OPTIONS", "HEAD"])
def serve_portal(request: Request):
    """Serves the voice management portal and handles health checks."""
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Allow": "GET, OPTIONS, HEAD",
                "Access-Control-Allow-Methods": "GET, OPTIONS, HEAD",
                "Access-Control-Allow-Origin": "*",
            }
        )
    portal_html = STATIC_DIR / "index.html"
    if portal_html.exists():
        return FileResponse(
            portal_html,
            headers={"Cache-Control": "no-cache, must-revalidate"}
        )
    return {
        "status": "online",
        "engine": "NeuTTS-Air",
        "backend": "Vulkan",
        "message": "Web portal file not found; static directory is missing."
    }


@app.get("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
@app.get("/audio/voices", dependencies=[Depends(verify_api_key)])
def list_voices():
    """List all available pre-cached voices stored in the voices volume."""
    voices = []
    if VOICES_DIR.exists():
        for pt_file in sorted(VOICES_DIR.glob("*.pt")):
            name = pt_file.stem
            txt_file = VOICES_DIR / f"{name}.txt"
            transcript = ""
            if txt_file.exists():
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        transcript = f.read().strip()
                except Exception:
                    pass
            voices.append({
                "id": name,
                "name": name.replace("_", " ").title(),
                "transcript": transcript,
                "size_bytes": pt_file.stat().st_size
            })
    return {"voices": voices}


@app.post("/v1/audio/voices", dependencies=[Depends(verify_api_key)])
@app.post("/audio/voices", dependencies=[Depends(verify_api_key)])
async def upload_voice(
    name: str = Form(...),
    text: str = Form(...),
    audio_file: UploadFile = File(...)
):
    """Encode an uploaded voice reference and save directly into the persistent volume."""
    if tts_model is None:
        raise HTTPException(status_code=500, detail="TTS Engine is uninitialized")

    clean_name = validate_voice_name(name)

    clean_text = text.strip()
    if not clean_text:
        raise HTTPException(status_code=400, detail="Reference transcript cannot be empty")
    if len(clean_text) > 2000:
        raise HTTPException(status_code=400, detail="Reference transcript exceeds maximum length of 2000 characters")

    # Validate audio file extension
    orig_suffix = Path(audio_file.filename or "audio.wav").suffix.lower()
    if orig_suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported audio format '{orig_suffix}'. Allowed formats: {', '.join(sorted(ALLOWED_AUDIO_EXTENSIONS))}"
        )

    # Save uploaded audio with 25 MB size enforcement
    with tempfile.NamedTemporaryFile(suffix=orig_suffix, delete=False) as temp_audio:
        temp_path = temp_audio.name
        total_bytes = 0
        while chunk := await audio_file.read(64 * 1024):
            total_bytes += len(chunk)
            if total_bytes > MAX_UPLOAD_BYTES:
                temp_audio.close()
                if os.path.exists(temp_path):
                    os.remove(temp_path)
                raise HTTPException(
                    status_code=413,
                    detail=f"Payload Too Large: Audio sample exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit."
                )
            temp_audio.write(chunk)

    if total_bytes == 0:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise HTTPException(status_code=400, detail="Uploaded audio file cannot be empty")

    try:
        print(f"[VOICE CLONER] Encoding reference audio for '{clean_name}' ({total_bytes} bytes)...")
        with tts_lock:
            ref_codes = tts_model.encode_reference(temp_path)

        out_pt = resolve_voice_file(clean_name, ".pt")
        out_txt = resolve_voice_file(clean_name, ".txt")

        torch.save(ref_codes, out_pt)
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(clean_text)

        # Immediately update in-memory cache
        voice_cache[clean_name] = (ref_codes, clean_text)
        print(f"[VOICE CLONER] Voice '{clean_name}' saved to {out_pt} and loaded into memory!")

        return {
            "success": True,
            "voice": clean_name,
            "message": f"Voice '{clean_name}' encoded and registered successfully."
        }
    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Failed to encode reference audio: {str(e)}")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@app.delete("/v1/audio/voices/{voice_id}", dependencies=[Depends(verify_api_key)])
@app.delete("/audio/voices/{voice_id}", dependencies=[Depends(verify_api_key)])
def delete_voice(voice_id: str):
    """Delete a voice from the persistent volume and evict from cache."""
    clean_name = validate_voice_name(voice_id)
    pt_path = resolve_voice_file(clean_name, ".pt")
    txt_path = resolve_voice_file(clean_name, ".txt")

    deleted = False
    if pt_path.exists():
        pt_path.unlink()
        deleted = True
    if txt_path.exists():
        txt_path.unlink()
        deleted = True

    voice_cache.pop(clean_name, None)

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice '{clean_name}' not found")

    print(f"[VOICE] Deleted voice '{clean_name}'")
    return {"success": True, "voice": clean_name, "message": "Voice deleted"}


# --- OPENAI SPEECH SYNTHESIS ROUTES ---

@app.post("/v1/audio/speech", dependencies=[Depends(verify_api_key)])
@app.post("/audio/speech", dependencies=[Depends(verify_api_key)])
def generate_speech(request: SpeechRequest):
    if tts_model is None:
        raise HTTPException(status_code=500, detail="Server not initialized")

    # 1. OpenAI Input text validation
    clean_input = request.input.strip()
    if not clean_input:
        raise HTTPException(status_code=400, detail="Input text cannot be empty")
    if len(clean_input) > 4096:
        raise HTTPException(status_code=400, detail="Input text exceeds maximum length of 4096 characters")

    # 2. OpenAI Speed validation (0.25 to 4.0)
    speed = request.speed if request.speed is not None else 1.0
    if speed < 0.25 or speed > 4.0:
        raise HTTPException(status_code=400, detail="Speed must be between 0.25 and 4.0")

    # 3. Audio format & Media-Type mapping
    fmt = (request.response_format or "mp3").lower()
    media_types = {
        "mp3": "audio/mpeg",
        "opus": "audio/opus",
        "aac": "audio/aac",
        "flac": "audio/flac",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
    }
    if fmt not in media_types:
        raise HTTPException(status_code=400, detail=f"Unsupported format: '{fmt}'. Supported: mp3, opus, aac, flac, wav, pcm")

    # 4. Resolve Voice
    requested_voice = request.voice or DEFAULT_VOICE
    try:
        ref_codes, ref_text = load_voice(requested_voice)
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))

    print(f"[REQUEST] input_len={len(clean_input)}, voice={requested_voice}, format={fmt}, speed={speed}, stream={request.stream}")

    # 5. Low-latency Streaming (default for PCM, MP3, Opus unless explicitly disabled)
    should_stream = request.stream is not False
    if fmt == "pcm":
        return StreamingResponse(
            pcm_streamer(clean_input, ref_codes, ref_text, speed=speed),
            media_type="audio/pcm",
            headers={
                "Content-Type": "audio/pcm",
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            }
        )

    if should_stream and fmt in ["mp3", "opus"]:
        return StreamingResponse(
            ffmpeg_pipe_streamer(clean_input, ref_codes, ref_text, fmt, speed=speed),
            media_type=media_types[fmt],
            headers={
                "Content-Type": media_types[fmt],
                "X-Accel-Buffering": "no",
                "Cache-Control": "no-cache",
            }
        )

    # 6. Batch synthesis (for wav, flac, aac, or when stream=False)
    temp_wav_name = None
    temp_out_name = None
    try:
        with tts_lock:
            t_start = time.time()
            wav = tts_model.infer(clean_input, ref_codes, ref_text)
            t_infer = time.time() - t_start

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
                temp_wav_name = temp_wav.name
                sf.write(temp_wav_name, wav, 24000)

            print(f"[TIMING] infer: {t_infer:.2f}s (audio: {len(wav)/24000:.2f}s)")

            tempo = get_atempo_filter(speed)
            filter_args = ["-filter:a", tempo] if tempo else []

            # Unmodified WAV output
            if fmt == "wav" and not filter_args:
                with open(temp_wav_name, "rb") as f:
                    data = f.read()
                return Response(content=data, media_type="audio/wav")

            temp_out_name = temp_wav_name.replace(".wav", f".{fmt}")
            codec_map = {
                "mp3": ["-codec:a", "libmp3lame", "-b:a", "128k"],
                "opus": ["-codec:a", "libopus", "-b:a", "96k"],
                "flac": ["-codec:a", "flac"],
                "aac": ["-codec:a", "aac", "-b:a", "128k"],
                "wav": ["-codec:a", "pcm_s16le", "-ar", "24000", "-ac", "1"]
            }
            cmd = ["ffmpeg", "-y", "-i", temp_wav_name] + filter_args + codec_map.get(fmt, ["-codec:a", "libmp3lame"]) + [temp_out_name]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

            if res.returncode != 0:
                raise HTTPException(status_code=500, detail="FFmpeg audio conversion failed")

            with open(temp_out_name, "rb") as f:
                data = f.read()

            return Response(content=data, media_type=media_types[fmt])

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if temp_wav_name and os.path.exists(temp_wav_name):
            try:
                os.remove(temp_wav_name)
            except Exception:
                pass
        if temp_out_name and os.path.exists(temp_out_name):
            try:
                os.remove(temp_out_name)
            except Exception:
                pass


# --- OPENAI MODELS COMPATIBILITY ENDPOINTS ---

@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
@app.get("/models", dependencies=[Depends(verify_api_key)])
def list_models():
    """List available models for OpenAI SDK and third-party client discovery."""
    now = int(time.time())
    model_ids = ["tts-1", "tts-1-hd", "neutts"]
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": now,
                "owned_by": "neutts-server",
                "permission": [],
                "root": m,
                "parent": None
            }
            for m in model_ids
        ]
    }


@app.get("/v1/models/{model_id}", dependencies=[Depends(verify_api_key)])
@app.get("/models/{model_id}", dependencies=[Depends(verify_api_key)])
def get_model(model_id: str):
    """Retrieve model info for a specific model ID."""
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "neutts-server",
        "permission": [],
        "root": model_id,
        "parent": None
    }


@app.api_route("/api/status", methods=["GET", "OPTIONS", "HEAD"])
def status(request: Request):
    """Public health & status endpoint (indicates whether auth is required)."""
    if request.method == "OPTIONS":
        return Response(
            status_code=200,
            headers={
                "Allow": "GET, OPTIONS, HEAD",
                "Access-Control-Allow-Methods": "GET, OPTIONS, HEAD",
                "Access-Control-Allow-Origin": "*",
            }
        )

    is_https = request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https"
    ssl_configured = bool(os.environ.get("SSL_CERTFILE") and os.environ.get("SSL_KEYFILE"))

    return {
        "status": "online",
        "engine": "NeuTTS-Air",
        "backend": "Vulkan",
        "backbone_repo": os.environ.get("BACKBONE_REPO", "neuphonic/neutts-air-q8-gguf"),
        "backbone_device": os.environ.get("BACKBONE_DEVICE", "gpu"),
        "voices_count": len(get_available_voice_names()),
        "auth_required": bool(API_KEY),
        "ssl_enabled": ssl_configured or is_https,
        "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
        "features": [
            "voice-management-portal",
            "pcm-streaming",
            "mp3-streaming",
            "openai-compatible",
            "speed-control",
            "models-discovery",
            "api-key-auth",
            "ssl-tls"
        ]
    }


if __name__ == "__main__":
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="NeuTTS Server (OpenAI-Compatible)")
    parser.add_argument("--backbone", type=str, default=os.environ.get("BACKBONE_REPO", "neuphonic/neutts-air-q8-gguf"))
    parser.add_argument("--backbone-device", type=str, default=os.environ.get("BACKBONE_DEVICE", "gpu"), choices=["cpu", "gpu", "cuda"])
    parser.add_argument("--codec", type=str, default=os.environ.get("CODEC_REPO", "neuphonic/neucodec"))
    parser.add_argument("--codec-device", type=str, default=os.environ.get("CODEC_DEVICE", "cpu"), choices=["cpu", "cuda"])
    parser.add_argument("--host", type=str, default=os.environ.get("HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "8090")))
    parser.add_argument("--voices-dir", type=str, default=str(VOICES_DIR))
    parser.add_argument("--default-voice", type=str, default=DEFAULT_VOICE)
    # SSL / TLS configuration options
    parser.add_argument("--ssl-keyfile", type=str, default=os.environ.get("SSL_KEYFILE", ""))
    parser.add_argument("--ssl-certfile", type=str, default=os.environ.get("SSL_CERTFILE", ""))
    parser.add_argument("--ssl-keyfile-password", type=str, default=os.environ.get("SSL_KEYFILE_PASSWORD", None))
    parser.add_argument("--ssl-ca-certs", type=str, default=os.environ.get("SSL_CA_CERTS", None))

    args = parser.parse_args()

    os.environ["BACKBONE_REPO"] = args.backbone
    os.environ["BACKBONE_DEVICE"] = args.backbone_device
    os.environ["CODEC_REPO"] = args.codec
    os.environ["CODEC_DEVICE"] = args.codec_device
    os.environ["VOICES_DIR"] = args.voices_dir
    os.environ["DEFAULT_VOICE"] = args.default_voice

    ssl_keyfile = args.ssl_keyfile.strip() if args.ssl_keyfile else None
    ssl_certfile = args.ssl_certfile.strip() if args.ssl_certfile else None
    ssl_keyfile_password = args.ssl_keyfile_password.strip() if args.ssl_keyfile_password else None
    ssl_ca_certs = args.ssl_ca_certs.strip() if args.ssl_ca_certs else None

    uvicorn_kwargs = {
        "host": args.host,
        "port": args.port,
        "reload": False,
    }

    protocol = "http"
    if ssl_keyfile or ssl_certfile:
        if not ssl_keyfile or not ssl_certfile:
            raise ValueError(
                "Both --ssl-keyfile and --ssl-certfile (or SSL_KEYFILE and SSL_CERTFILE) "
                "must be specified to enable SSL/HTTPS."
            )
        if not os.path.exists(ssl_keyfile):
            raise FileNotFoundError(f"SSL key file not found: {ssl_keyfile}")
        if not os.path.exists(ssl_certfile):
            raise FileNotFoundError(f"SSL certificate file not found: {ssl_certfile}")
        if ssl_ca_certs and not os.path.exists(ssl_ca_certs):
            raise FileNotFoundError(f"SSL CA certs file not found: {ssl_ca_certs}")

        os.environ["SSL_KEYFILE"] = ssl_keyfile
        os.environ["SSL_CERTFILE"] = ssl_certfile

        uvicorn_kwargs["ssl_keyfile"] = ssl_keyfile
        uvicorn_kwargs["ssl_certfile"] = ssl_certfile
        if ssl_keyfile_password:
            uvicorn_kwargs["ssl_keyfile_password"] = ssl_keyfile_password
        if ssl_ca_certs:
            uvicorn_kwargs["ssl_ca_certs"] = ssl_ca_certs
        protocol = "https"
        print(f"[SSL] TLS/SSL enabled with certificate: {ssl_certfile} and private key: {ssl_keyfile}")

    print(f"[START] Starting server on {protocol}://{args.host}:{args.port}...")
    uvicorn.run("openai_server:app", **uvicorn_kwargs)
