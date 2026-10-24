import os
import re
import json
import base64
import time
import uuid
import shutil
import tempfile
import textwrap
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

import requests
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field, ConfigDict
from google.auth import default
from google.oauth2 import service_account
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.cloud import storage
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(title="Video Renderer", version="1.1.0")

# ---------------------------------------------------------
# Configuration & Environment
# ---------------------------------------------------------
def _load_sa_info():
    sa_env = os.getenv("GCP_SERVICE_ACCOUNT_JSON")
    if not sa_env:
        return None
    try:
        return json.loads(sa_env)
    except Exception:
        try:
            return json.loads(base64.b64decode(sa_env).decode("utf-8"))
        except Exception as e:
            raise RuntimeError(f"Failed to parse GCP_SERVICE_ACCOUNT_JSON: {e}")

_sa_info = _load_sa_info()

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT") or (_sa_info.get("project_id") if _sa_info else None)
LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
MODEL_ID = os.getenv("VEO_MODEL_ID", "veo-3.1-generate-001")
LYRIA_MODEL_ID = os.getenv("LYRIA_MODEL_ID", "lyria-002")
VEO_OUTPUT_BUCKET = os.getenv("VEO_OUTPUT_BUCKET")
RENDER_API_TOKEN = os.getenv("RENDER_API_TOKEN")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
SUPABASE_BUCKET = os.getenv("SUPABASE_VIDEO_BUCKET", "generated-videos")
SUPABASE_IS_PRIVATE = os.getenv("SUPABASE_BUCKET_PRIVATE", "false").lower() == "true"

# Validate required configuration
_missing_vars = []
if not PROJECT_ID:
    _missing_vars.append("GOOGLE_CLOUD_PROJECT (or GCP_SERVICE_ACCOUNT_JSON with 'project_id')")
if not VEO_OUTPUT_BUCKET:
    _missing_vars.append("VEO_OUTPUT_BUCKET")
if not RENDER_API_TOKEN:
    _missing_vars.append("RENDER_API_TOKEN")
if not SUPABASE_URL:
    _missing_vars.append("SUPABASE_URL")
if not SUPABASE_KEY:
    _missing_vars.append("SUPABASE_SERVICE_ROLE_KEY")

if _missing_vars:
    missing_list = "\n  - ".join(_missing_vars)
    raise RuntimeError(
        f"\n{'='*70}\n"
        f"CONFIG ERROR: Missing required environment variables:\n  - {missing_list}\n\n"
        f"If deploying on Railway:\n"
        f"1. Open your project dashboard in Railway.\n"
        f"2. Go to your service -> 'Variables' tab.\n"
        f"3. Add all the required variables listed above.\n"
        f"4. For Google Cloud auth, add 'GCP_SERVICE_ACCOUNT_JSON' with your service account key JSON.\n"
        f"{'='*70}\n"
    )

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# ---------------------------------------------------------
# Authentication (Supports ADC locally & Service Account on Railway)
# ---------------------------------------------------------
def get_gcp_credentials():
    """
    Retrieves Google Cloud credentials.
    Supports:
      1. GCP_SERVICE_ACCOUNT_JSON (JSON string or base64) for cloud deployments (Railway/Render).
      2. GOOGLE_APPLICATION_CREDENTIALS file path.
      3. Application Default Credentials (ADC) for local development via gcloud CLI.
    """
    if _sa_info:
        return service_account.Credentials.from_service_account_info(
            _sa_info, scopes=["https://www.googleapis.com/auth/cloud-platform"]
        )

    try:
        credentials, _ = default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        return credentials
    except Exception as e:
        raise RuntimeError(
            "Could not determine Google Cloud credentials. On Railway, please set "
            "the 'GCP_SERVICE_ACCOUNT_JSON' environment variable to your service account key JSON (or base64 encoded)."
        ) from e

gcp_credentials = get_gcp_credentials()
gcs = storage.Client(project=PROJECT_ID, credentials=gcp_credentials)

def google_access_token() -> str:
    global gcp_credentials
    if not gcp_credentials.valid:
        gcp_credentials.refresh(GoogleAuthRequest())
    return gcp_credentials.token

# ---------------------------------------------------------
# Request Models
# ---------------------------------------------------------
class Shot(BaseModel):
    model_config = ConfigDict(extra="ignore")
    index: int
    prompt: str = Field(min_length=5)
    negativePrompt: str = ""
    durationS: int = 8
    onScreenText: Optional[str] = None
    on_screen_text: Optional[str] = None

class RenderRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")
    jobId: Optional[str] = None
    id: Optional[str] = None
    topic: str = ""
    aspectRatio: Optional[str] = None
    aspect_ratio: Optional[str] = None
    resolution: str = "1080p"
    shots: Optional[list[Shot]] = None
    shots_json: Optional[Any] = None
    voiceoverScript: Optional[str] = None
    voiceover_script: Optional[str] = None
    voiceoverUrl: Optional[str] = None
    voiceover_url: Optional[str] = None
    musicPrompt: Optional[str] = None
    music_prompt: Optional[str] = None
    musicUrl: Optional[str] = None
    music_url: Optional[str] = None

# ---------------------------------------------------------
# Helper Functions: Storage, Download & Video Generation
# ---------------------------------------------------------
def gcs_uri_to_path(uri: str):
    if not uri.startswith("gs://"):
        raise ValueError(f"Expected a gs:// URI from Veo, got: {uri}")
    bucket, _, object_name = uri[5:].partition("/")
    if not bucket or not object_name:
        raise ValueError(f"Invalid GCS URI: {uri}")
    return bucket, object_name

def download_gcs_video(uri: str, destination: Path):
    bucket_name, object_name = gcs_uri_to_path(uri)
    blob = gcs.bucket(bucket_name).blob(object_name)
    blob.download_to_filename(str(destination))
    if not destination.exists() or destination.stat().st_size == 0:
        raise RuntimeError(f"Downloaded clip is empty: {uri}")

def download_http_asset(url: str, destination: Path) -> Path:
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()
    with destination.open("wb") as f:
        for chunk in resp.iter_content(chunk_size=16384):
            if chunk:
                f.write(chunk)
    return destination

def generate_one_clip(
    shot: Shot, job_id: str, workdir: Path, aspect_ratio: str, resolution: str
) -> tuple[Path, str]:
    """
    Submits a prediction request to Google Veo on Vertex AI and polls until completion.
    Returns (local_clip_path, gcs_uri).
    """
    max_retries = 3
    base_backoff = 25  # seconds

    for attempt in range(1, max_retries + 1):
        try:
            token = google_access_token()
            endpoint = (
                f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
                f"projects/{PROJECT_ID}/locations/{LOCATION}/"
                f"publishers/google/models/{MODEL_ID}:predictLongRunning"
            )

            output_prefix = f"gs://{VEO_OUTPUT_BUCKET}/{job_id}/shot-{shot.index}/"

            params: dict[str, Any] = {
                "storageUri": output_prefix,
                "sampleCount": 1,
                "aspectRatio": aspect_ratio,
                "durationSeconds": shot.durationS,
                "resolution": resolution,
            }
            if shot.negativePrompt:
                params["negativePrompt"] = shot.negativePrompt

            response = requests.post(
                endpoint,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
                json={
                    "instances": [{"prompt": shot.prompt}],
                    "parameters": params,
                },
                timeout=60,
            )
            response.raise_for_status()
            operation_name = response.json().get("name")
            if not operation_name:
                raise RuntimeError(f"Veo did not return an operation name: {response.text}")

            poll_endpoint = (
                f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
                f"projects/{PROJECT_ID}/locations/{LOCATION}/"
                f"publishers/google/models/{MODEL_ID}:fetchPredictOperation"
            )

            deadline = time.monotonic() + 1800  # 30-minute timeout per clip
            result = None

            while time.monotonic() < deadline:
                time.sleep(12)
                token = google_access_token()
                poll = requests.post(
                    poll_endpoint,
                    headers={
                        "Authorization": f"Bearer {token}",
                        "Content-Type": "application/json",
                    },
                    json={"operationName": operation_name},
                    timeout=60,
                )
                poll.raise_for_status()
                result = poll.json()

                if result.get("error"):
                    err = result["error"]
                    err_code = err.get("code")
                    err_msg = str(err.get("message", ""))
                    if (err_code == 8 or "high load" in err_msg.lower() or "resource" in err_msg.lower()) and attempt < max_retries:
                        wait_sec = base_backoff * attempt
                        print(f"Veo high load for shot {shot.index} (attempt {attempt}/{max_retries}). Retrying in {wait_sec}s...")
                        time.sleep(wait_sec)
                        break
                    raise RuntimeError(f"Veo generation failed: {err}")
                if result.get("done"):
                    break
            else:
                raise TimeoutError(f"Veo operation timed out: {operation_name}")

            if result.get("error"):
                continue

            videos = result.get("response", {}).get("videos", [])
            if not videos or not videos[0].get("gcsUri"):
                raise RuntimeError(f"Veo finished without returning a video GCS URI: {result}")

            gcs_uri = videos[0]["gcsUri"]
            output = workdir / f"shot-{shot.index:02d}.mp4"
            download_gcs_video(gcs_uri, output)
            return output, gcs_uri

        except Exception as exc:
            if attempt >= max_retries or "high load" not in str(exc).lower():
                raise
            wait_sec = base_backoff * attempt
            print(f"Transient error on shot {shot.index} (attempt {attempt}/{max_retries}): {exc}. Retrying in {wait_sec}s...")
            time.sleep(wait_sec)

    raise RuntimeError(f"Veo generation failed for shot {shot.index} after {max_retries} attempts.")

def generate_lyria_music(prompt: str, output_path: Path) -> Optional[Path]:
    """
    Generates background music using Google Lyria (lyria-002) on Vertex AI.
    Returns the path to the downloaded MP3, or None if generation fails.
    """
    if not prompt or not prompt.strip():
        return None

    try:
        token = google_access_token()
        endpoint = (
            f"https://{LOCATION}-aiplatform.googleapis.com/v1/"
            f"projects/{PROJECT_ID}/locations/{LOCATION}/"
            f"publishers/google/models/{LYRIA_MODEL_ID}:predict"
        )
        print(f"Generating background music with Google Lyria: '{prompt[:75]}...'")
        resp = requests.post(
            endpoint,
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            json={
                "instances": [
                    {
                        "prompt": prompt,
                        "negative_prompt": "vocals, singing, speech, lyrics, distortion",
                    }
                ],
                "parameters": {
                    "sample_count": 1,
                },
            },
            timeout=120,
        )

        if resp.status_code == 200:
            predictions = resp.json().get("predictions", [])
            if predictions:
                pred = predictions[0]
                b64_str = pred.get("bytesBase64Encoded") or pred.get("audioContent")
                if b64_str:
                    audio_bytes = base64.b64decode(b64_str)
                    with output_path.open("wb") as f:
                        f.write(audio_bytes)
                    print(f"Google Lyria music generated successfully ({len(audio_bytes)} bytes).")
                    return output_path
        print(f"Lyria note ({resp.status_code}): {resp.text[:250]}")
    except Exception as exc:
        print(f"Warning: could not generate Lyria music ({exc})")

    return None

# ---------------------------------------------------------
# FFmpeg Video Processing & Audio Mixing
# ---------------------------------------------------------
def run_ffmpeg(args: list[str]):
    proc = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", *args],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed (exit code {proc.returncode}): {proc.stderr}")

def find_system_font() -> Optional[str]:
    """Finds a clean bold sans-serif font for ad subtitles."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/segoeui.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            return c
    return None

def wrap_subtitle(text: str, max_chars_per_line: int = 26) -> str:
    """Wraps subtitle lines cleanly for mobile ad safe zones."""
    if not text:
        return ""
    lines = textwrap.wrap(text, width=max_chars_per_line)
    return "\n".join(lines)

def assign_shot_subtitles(shots_count: int, explicit_texts: list[str], voiceover_script: Optional[str]) -> list[str]:
    """
    Determines the subtitle text for each shot.
    Prioritizes explicit shot text/onScreenText; otherwise partitions the voiceover script across shots.
    """
    if any(t and t.strip() for t in explicit_texts):
        return [t.strip() if t else "" for t in explicit_texts]

    if not voiceover_script or not voiceover_script.strip():
        return ["" for _ in range(shots_count)]

    clean_script = voiceover_script.strip()
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+', clean_script) if s.strip()]
    if not sentences:
        sentences = [clean_script]

    if len(sentences) >= shots_count:
        assigned = []
        for i in range(shots_count):
            start = int(i * len(sentences) / shots_count)
            end = int((i + 1) * len(sentences) / shots_count)
            assigned.append(" ".join(sentences[start:max(start + 1, end)]))
        return assigned
    else:
        parts = [p.strip() for p in re.split(r'[,;]\s+', clean_script) if p.strip()]
        if len(parts) >= shots_count:
            assigned = []
            for i in range(shots_count):
                start = int(i * len(parts) / shots_count)
                end = int((i + 1) * len(parts) / shots_count)
                assigned.append(" ".join(parts[start:max(start + 1, end)]))
            return assigned
        else:
            words = clean_script.split()
            chunk_size = max(1, len(words) // shots_count)
            assigned = []
            for i in range(shots_count):
                if i == shots_count - 1:
                    assigned.append(" ".join(words[i * chunk_size:]))
                else:
                    assigned.append(" ".join(words[i * chunk_size:(i + 1) * chunk_size]))
            return assigned

def calculate_job_cost(
    shots: list[Shot],
    resolution: str,
    voiceover_script: Optional[str] = None,
    has_voiceover_url: bool = False,
    music_prompt: Optional[str] = None,
    has_music_url: bool = False,
) -> dict:
    """
    Calculates actual money spent for this video generation job:
      - Google Veo 3.1: $0.12/s for 1080p, $0.10/s for 720p
      - OpenAI TTS: ~$0.015 per 1,000 characters
      - Google Lyria: ~$0.030 per track
    """
    is_1080p = "1080" in resolution
    rate_per_sec = 0.12 if is_1080p else 0.10
    total_seconds = sum(getattr(s, "durationS", 8) or 8 for s in shots)
    veo_cost = round(total_seconds * rate_per_sec, 4)

    vo_cost = 0.0
    if voiceover_script and voiceover_script.strip():
        char_count = len(voiceover_script.strip())
        vo_cost = round((char_count / 1000.0) * 0.015, 4)
    elif has_voiceover_url:
        vo_cost = 0.005

    music_cost = 0.0
    if music_prompt and music_prompt.strip():
        music_cost = 0.030

    total_cost = round(veo_cost + vo_cost + music_cost, 4)
    return {
        "actual_cost_usd": total_cost,
        "cost_breakdown": {
            "total_usd": total_cost,
            "veo_usd": veo_cost,
            "voiceover_usd": vo_cost,
            "music_usd": music_cost,
            "video_duration_seconds": total_seconds,
            "resolution": resolution,
            "shots_count": len(shots),
        },
    }

def stitch_clips(
    clips: list[Path],
    output: Path,
    aspect_ratio: str,
    resolution: str,
    subtitles: Optional[list[str]] = None,
):
    """
    Normalizes every clip to identical resolution, framerate, and pixel format,
    burns high-converting Meta / Google Ads subtitles in safe zones, and concatenates.
    """
    is_1080p = "1080" in resolution
    if aspect_ratio == "16:9":
        width, height = (1920, 1080) if is_1080p else (1280, 720)
    else:  # 9:16 vertical
        width, height = (1080, 1920) if is_1080p else (720, 1280)

    font_path = find_system_font()
    esc_font = font_path.replace(":", "\\:") if font_path else None

    normalized = []
    for i, clip in enumerate(clips):
        norm_path = output.parent / f"norm-{i:02d}.mp4"
        vf_filters = [
            f"scale={width}:{height}:force_original_aspect_ratio=decrease",
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
            "setsar=1",
            "fps=30",
            "format=yuv420p",
        ]

        # Add subtitle overlay for Meta / Google Ads
        sub_text = subtitles[i] if subtitles and i < len(subtitles) else ""
        if sub_text and sub_text.strip():
            max_chars = 26 if aspect_ratio == "9:16" else 46
            wrapped = wrap_subtitle(sub_text.strip(), max_chars_per_line=max_chars)
            sub_file = output.parent / f"sub-{i:02d}.txt"
            with sub_file.open("w", encoding="utf-8") as f:
                f.write(wrapped)

            esc_sub = sub_file.as_posix().replace(":", "\\:")
            fontsize = 44 if is_1080p else 32
            if aspect_ratio == "16:9":
                fontsize = 36 if is_1080p else 26
                y_pos = "h*0.80-text_h/2"
            else:
                # 9:16 Meta / TikTok / Reels safe zone
                y_pos = "h*0.72-text_h/2"

            if esc_font:
                drawtext_arg = (
                    f"drawtext=fontfile='{esc_font}':textfile='{esc_sub}':"
                    f"fontsize={fontsize}:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=16:"
                    f"line_spacing=10:x=(w-text_w)/2:y={y_pos}"
                )
            else:
                drawtext_arg = (
                    f"drawtext=textfile='{esc_sub}':"
                    f"fontsize={fontsize}:fontcolor=white:box=1:boxcolor=black@0.65:boxborderw=16:"
                    f"line_spacing=10:x=(w-text_w)/2:y={y_pos}"
                )
            vf_filters.append(drawtext_arg)

        run_ffmpeg([
            "-y", "-i", str(clip),
            "-vf", ",".join(vf_filters),
            "-an",
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            str(norm_path),
        ])
        normalized.append(norm_path)

    concat_file = output.parent / "concat.txt"
    with concat_file.open("w", encoding="utf-8") as f:
        for clip in normalized:
            f.write(f"file '{clip.as_posix()}'\n")

    run_ffmpeg([
        "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_file),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "22",
        "-pix_fmt", "yuv420p",
        "-an",
        "-movflags", "+faststart",
        str(output),
    ])

def mix_audio(
    video_path: Path,
    voiceover_path: Optional[Path],
    music_path: Optional[Path],
    final_output: Path,
):
    """
    Combines video with voiceover and background music with audio ducking.
    If no audio is provided, generates a silent audio track for broad player compatibility.
    """
    has_vo = voiceover_path is not None and voiceover_path.exists()
    has_music = music_path is not None and music_path.exists()

    if has_vo and has_music:
        # Mix voiceover at 100% and music at 15% (ducked), trimmed to video length
        run_ffmpeg([
            "-y",
            "-i", str(video_path),
            "-i", str(voiceover_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex",
            "[1:a]volume=1.0[vo];[2:a]volume=0.15[bgm];[vo][bgm]amix=inputs=2:duration=first:dropout_transition=2[a]",
            "-map", "0:v",
            "-map", "[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(final_output),
        ])
    elif has_vo:
        # Video + Voiceover only
        run_ffmpeg([
            "-y",
            "-i", str(video_path),
            "-i", str(voiceover_path),
            "-map", "0:v",
            "-map", "1:a",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(final_output),
        ])
    elif has_music:
        # Video + Background music (looped and trimmed to video)
        run_ffmpeg([
            "-y",
            "-i", str(video_path),
            "-stream_loop", "-1", "-i", str(music_path),
            "-filter_complex", "[1:a]volume=0.30[bgm]",
            "-map", "0:v",
            "-map", "[bgm]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "192k",
            "-shortest",
            "-movflags", "+faststart",
            str(final_output),
        ])
    else:
        # Video only: create silent AAC track to prevent playback errors in some web browsers
        run_ffmpeg([
            "-y",
            "-i", str(video_path),
            "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100",
            "-c:v", "copy",
            "-c:a", "aac",
            "-shortest",
            "-movflags", "+faststart",
            str(final_output),
        ])

def ensure_supabase_bucket():
    """Ensures the target Supabase storage bucket exists and has correct public/private settings."""
    try:
        buckets = supabase.storage.list_buckets()
        existing = {
            b.id if hasattr(b, "id") else (b.get("id") or b.get("name"))
            for b in buckets
        }
        if SUPABASE_BUCKET not in existing:
            supabase.storage.create_bucket(
                SUPABASE_BUCKET,
                options={"public": not SUPABASE_IS_PRIVATE},
            )
    except Exception as exc:
        print(f"Warning: could not verify/create Supabase bucket '{SUPABASE_BUCKET}': {exc}")

ensure_supabase_bucket()

def upload_final_video(output: Path, job_id: str) -> str:
    ensure_supabase_bucket()
    object_path = f"{job_id}/final-{uuid.uuid4().hex[:8]}.mp4"
    with output.open("rb") as f:
        res = supabase.storage.from_(SUPABASE_BUCKET).upload(
            object_path,
            f,
            file_options={"content-type": "video/mp4", "upsert": "true"},
        )
        if hasattr(res, "status_code") and res.status_code >= 400:
            raise RuntimeError(f"Supabase upload failed ({res.status_code}): {getattr(res, 'text', '')}")
        if isinstance(res, dict) and res.get("error"):
            raise RuntimeError(f"Supabase upload failed: {res['error']}")

    if SUPABASE_IS_PRIVATE:
        # Generate a 7-day signed URL for private storage buckets
        res = supabase.storage.from_(SUPABASE_BUCKET).create_signed_url(
            object_path, expires_in=604800
        )
        return res.get("signedURL") or res.get("signedUrl")
    else:
        return supabase.storage.from_(SUPABASE_BUCKET).get_public_url(object_path)

# ---------------------------------------------------------
# Background Task: Complete Orchestration & Checkpointing
# ---------------------------------------------------------
def process_render_job(request: RenderRequest):
    """
    Asynchronous rendering pipeline:
    1. Checkpoints and generates Veo clips (reusing existing clips on retries).
    2. Downloads voiceover and background music if provided.
    3. Stitches video clips with FFmpeg.
    4. Mixes audio tracks.
    5. Uploads to Supabase Storage and updates the database row.
    """
    try:
        # Mark job as rendering
        supabase.table("video_jobs").update({
            "status": "rendering",
            "render_progress": {"stage": "starting", "percent": 5},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", request.jobId).execute()

        # Fetch current record to support checkpointed clip reuse and audio assets
        row_res = (
            supabase.table("video_jobs")
            .select("shots_json,voiceover_url,music_url")
            .eq("id", request.jobId)
            .limit(1)
            .execute()
        )
        existing_shots_dict = {}
        db_row = row_res.data[0] if (row_res.data and len(row_res.data) > 0) else {}
        if db_row.get("shots_json"):
            for s in db_row["shots_json"]:
                if isinstance(s, dict) and "index" in s:
                    existing_shots_dict[s["index"]] = s

        with tempfile.TemporaryDirectory(prefix="video-render-") as tmp:
            workdir = Path(tmp)
            clips: list[Path] = []
            updated_shots_json = []

            # 1. Generate or retrieve clips
            total_shots = len(request.shots)
            for i, shot in enumerate(request.shots):
                pct = 10 + int(i * 60 / total_shots)
                supabase.table("video_jobs").update({
                    "render_progress": {
                        "stage": "generating_clips",
                        "percent": pct,
                        "current_shot": shot.index,
                        "total_shots": total_shots,
                    },
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", request.jobId).execute()

                shot_dict = shot.model_dump()
                existing_entry = existing_shots_dict.get(shot.index)

                # Check if this clip was already generated in a previous run
                if existing_entry and existing_entry.get("gcsUri"):
                    gcs_uri = existing_entry["gcsUri"]
                    clip_path = workdir / f"shot-{shot.index:02d}.mp4"
                    download_gcs_video(gcs_uri, clip_path)
                    shot_dict["gcsUri"] = gcs_uri
                else:
                    clip_path, gcs_uri = generate_one_clip(
                        shot, request.jobId, workdir, request.aspectRatio, request.resolution
                    )
                    shot_dict["gcsUri"] = gcs_uri

                clips.append(clip_path)
                updated_shots_json.append(shot_dict)

                # Checkpoint progress to Supabase after each shot
                supabase.table("video_jobs").update({
                    "shots_json": updated_shots_json
                }).eq("id", request.jobId).execute()

            # 2. Download audio assets (checks request and falls back to Supabase row)
            supabase.table("video_jobs").update({
                "render_progress": {"stage": "processing_audio", "percent": 75},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", request.jobId).execute()

            vo_url = request.voiceoverUrl or db_row.get("voiceover_url")
            vo_path = None
            if vo_url:
                vo_path = download_http_asset(vo_url, workdir / "voiceover.mp3")

            bgm_url = request.musicUrl or db_row.get("music_url")
            music_prompt = request.musicPrompt or db_row.get("music_prompt")
            music_path = None
            if bgm_url:
                music_path = download_http_asset(bgm_url, workdir / "music.mp3")
            elif music_prompt:
                # Automatically generate background music using Google Lyria
                music_path = generate_lyria_music(music_prompt, workdir / "lyria_music.mp3")

            # 3. Stitch Video Clips (with safe-zone Meta/Google Ads subtitles)
            supabase.table("video_jobs").update({
                "render_progress": {"stage": "stitching_video", "percent": 85},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", request.jobId).execute()

            # Determine subtitle overlay for each shot
            explicit_texts = [
                getattr(s, "onScreenText", None) or getattr(s, "on_screen_text", None)
                for s in request.shots
            ]
            vo_script = request.voiceoverScript or db_row.get("voiceover_script")
            subtitles = assign_shot_subtitles(len(request.shots), explicit_texts, vo_script)

            stitched_video = workdir / "stitched_raw.mp4"
            stitch_clips(clips, stitched_video, request.aspectRatio, request.resolution, subtitles=subtitles)

            # 4. Mix Audio
            supabase.table("video_jobs").update({
                "render_progress": {"stage": "mixing_final", "percent": 92},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", request.jobId).execute()

            final_mp4 = workdir / "final.mp4"
            mix_audio(stitched_video, vo_path, music_path, final_mp4)

            # 5. Upload to Supabase Storage
            supabase.table("video_jobs").update({
                "render_progress": {"stage": "uploading", "percent": 97},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", request.jobId).execute()

            output_url = upload_final_video(final_mp4, request.jobId)

            # 6. Calculate Actual Money Spent
            cost_info = calculate_job_cost(
                shots=request.shots,
                resolution=request.resolution,
                voiceover_script=vo_script,
                has_voiceover_url=bool(vo_path),
                music_prompt=music_prompt,
                has_music_url=bool(music_path),
            )
            actual_cost = cost_info["actual_cost_usd"]
            cost_breakdown = cost_info["cost_breakdown"]

            # 7. Mark Job Completed & Save Actual Cost to Supabase for Frontend
            completion_payload = {
                "status": "completed",
                "output_video_url": output_url,
                "actual_cost_usd": actual_cost,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "render_progress": {
                    "stage": "completed",
                    "percent": 100,
                    "cost_breakdown": cost_breakdown,
                },
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }

            try:
                supabase.table("video_jobs").update(completion_payload).eq("id", request.jobId).execute()
            except Exception as e_cost:
                print(f"Notice: saving actual_cost_usd failed ({e_cost}), retrying without column...")
                completion_payload.pop("actual_cost_usd", None)
                supabase.table("video_jobs").update(completion_payload).eq("id", request.jobId).execute()

    except Exception as exc:
        supabase.table("video_jobs").update({
            "status": "failed",
            "error_detail": str(exc)[:1500],
            "render_progress": {"stage": "failed", "percent": 0},
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).eq("id", request.jobId).execute()

# ---------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "ffmpeg": shutil.which("ffmpeg") is not None,
        "gcp_project": PROJECT_ID,
        "gcp_location": LOCATION,
        "veo_model": MODEL_ID,
        "lyria_model": LYRIA_MODEL_ID,
    }

@app.get("/jobs/{jobId}")
def get_job_status(
    jobId: str,
    authorization: Optional[str] = Header(default=None),
):
    if authorization != f"Bearer {RENDER_API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    try:
        uuid.UUID(jobId)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid jobId format")

    row_res = (
        supabase.table("video_jobs")
        .select("id,status,actual_cost_usd,estimated_cost_usd,render_progress,output_video_url,error_detail,completed_at,created_at,updated_at")
        .eq("id", jobId)
        .limit(1)
        .execute()
    )
    if not row_res.data:
        raise HTTPException(status_code=404, detail="Job not found")

    return row_res.data[0]

@app.post("/jobs/render", status_code=202)
def render_job(
    request: RenderRequest,
    background_tasks: BackgroundTasks,
    authorization: Optional[str] = Header(default=None),
):
    if authorization != f"Bearer {RENDER_API_TOKEN}":
        raise HTTPException(status_code=401, detail="Unauthorized")

    request.jobId = request.jobId or request.id
    if not request.jobId:
        request.jobId = str(uuid.uuid4())
    else:
        try:
            uuid.UUID(request.jobId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid jobId format")

    request.aspectRatio = request.aspectRatio or request.aspect_ratio or "9:16"
    if request.aspectRatio not in ("9:16", "16:9"):
        raise HTTPException(status_code=400, detail="Unsupported aspectRatio (expected '9:16' or '16:9')")

    if request.resolution not in ("720p", "1080p"):
        request.resolution = "1080p"

    request.voiceoverUrl = request.voiceoverUrl or request.voiceover_url
    request.musicUrl = request.musicUrl or request.music_url
    request.voiceoverScript = request.voiceoverScript or request.voiceover_script
    request.musicPrompt = request.musicPrompt or request.music_prompt

    # Fetch job from Supabase (or auto-create if missing)
    row_response = (
        supabase.table("video_jobs")
        .select("id,status,output_video_url,shots_json")
        .eq("id", request.jobId)
        .limit(1)
        .execute()
    )

    # Normalize shots from request.shots, request.shots_json, or Supabase row
    raw_shots = request.shots
    if not raw_shots and request.shots_json:
        if isinstance(request.shots_json, str):
            try:
                raw_shots = json.loads(request.shots_json)
            except Exception:
                raw_shots = []
        elif isinstance(request.shots_json, list):
            raw_shots = request.shots_json

    if not raw_shots and row_response.data and row_response.data[0].get("shots_json"):
        db_shots = row_response.data[0]["shots_json"]
        if isinstance(db_shots, str):
            try:
                raw_shots = json.loads(db_shots)
            except Exception:
                raw_shots = []
        elif isinstance(db_shots, list):
            raw_shots = db_shots

    if not raw_shots:
        raise HTTPException(status_code=400, detail="Missing required 'shots' or 'shots_json' array.")

    parsed_shots = []
    for s in raw_shots:
        if isinstance(s, Shot):
            parsed_shots.append(s)
        elif isinstance(s, dict):
            text_val = (
                s.get("onScreenText")
                or s.get("on_screen_text")
                or s.get("text")
                or s.get("subtitle")
                or s.get("voiceover")
                or s.get("voiceover_script")
            )
            parsed_shots.append(
                Shot(
                    index=s.get("index", len(parsed_shots)),
                    prompt=s.get("prompt", ""),
                    negativePrompt=s.get("negativePrompt", "") or s.get("negative_prompt", ""),
                    durationS=s.get("durationS", 8) or s.get("duration_s", 8),
                    onScreenText=text_val,
                    on_screen_text=text_val,
                )
            )
    request.shots = parsed_shots
    is_new = not row_response.data
    if is_new:
        now_iso = datetime.now(timezone.utc).isoformat()
        topic_value = (request.topic or "").strip() or "A product launch"
        payload = {
            "id": request.jobId,
            "topic": topic_value,
            "aspect_ratio": request.aspectRatio or "9:16",
            "resolution": request.resolution or "1080p",
            "status": "rendering",
            "render_progress": {"stage": "queued", "percent": 0},
            "created_at": now_iso,
            "updated_at": now_iso,
        }
        if request.voiceoverScript:
            payload["voiceover_script"] = request.voiceoverScript
        if request.voiceoverUrl:
            payload["voiceover_url"] = request.voiceoverUrl
        if request.musicPrompt:
            payload["music_prompt"] = request.musicPrompt
        if request.musicUrl:
            payload["music_url"] = request.musicUrl

        try:
            supabase.table("video_jobs").insert(payload).execute()
        except Exception as e1:
            print(f"Notice: full payload insert failed ({e1}), retrying with minimal payload...")
            try:
                minimal_payload = {
                    "id": request.jobId,
                    "topic": topic_value,
                    "status": "rendering",
                    "aspect_ratio": request.aspectRatio or "9:16",
                    "resolution": request.resolution or "1080p",
                    "created_at": now_iso,
                    "updated_at": now_iso,
                }
                supabase.table("video_jobs").insert(minimal_payload).execute()
            except Exception as e2:
                raise HTTPException(status_code=500, detail=f"Failed to create job in database: {e2}")
    else:
        existing = row_response.data[0]
        # Idempotency: if already finished, return immediate success
        if existing.get("status") == "completed" and existing.get("output_video_url"):
            return {
                "jobId": request.jobId,
                "status": "completed",
                "outputVideoUrl": existing["output_video_url"],
                "renderProgress": {"stage": "completed", "percent": 100},
                "message": "Job has already completed",
            }

        # If already rendering, acknowledge without spawning duplicate background task
        if existing.get("status") == "rendering":
            return {
                "jobId": request.jobId,
                "status": "rendering",
                "message": "Job is currently already rendering in background",
            }

    # Queue background task for newly created jobs or re-queued jobs
    background_tasks.add_task(process_render_job, request)

    return {
        "jobId": request.jobId,
        "status": "rendering",
        "message": "Render task accepted and started in background",
    }
