import os
import json
import base64
import time
import uuid
import shutil
import tempfile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Any

import requests
from fastapi import FastAPI, Header, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
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
    index: int
    prompt: str = Field(min_length=5)
    negativePrompt: str = ""
    durationS: int = 8

class RenderRequest(BaseModel):
    jobId: Optional[str] = None
    topic: str = ""
    aspectRatio: str = "9:16"
    resolution: str = "720p"
    shots: list[Shot] = Field(min_length=1, max_length=10)
    voiceoverScript: Optional[str] = None
    voiceoverUrl: Optional[str] = None
    musicPrompt: Optional[str] = None
    musicUrl: Optional[str] = None

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

def stitch_clips(clips: list[Path], output: Path, aspect_ratio: str, resolution: str):
    """
    Normalizes every clip to identical resolution, framerate, and pixel format,
    then concatenates them into a continuous video.
    """
    is_1080p = "1080" in resolution
    if aspect_ratio == "16:9":
        width, height = (1920, 1080) if is_1080p else (1280, 720)
    else:  # 9:16 vertical
        width, height = (1080, 1920) if is_1080p else (720, 1280)

    normalized = []
    for i, clip in enumerate(clips):
        norm_path = output.parent / f"norm-{i:02d}.mp4"
        run_ffmpeg([
            "-y", "-i", str(clip),
            "-vf",
            f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
            "setsar=1,fps=30,format=yuv420p",
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

        # Fetch current record to support checkpointed clip reuse
        row_res = (
            supabase.table("video_jobs")
            .select("shots_json")
            .eq("id", request.jobId)
            .limit(1)
            .execute()
        )
        existing_shots_dict = {}
        if row_res.data and row_res.data[0].get("shots_json"):
            for s in row_res.data[0]["shots_json"]:
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

            # 2. Download audio assets
            supabase.table("video_jobs").update({
                "render_progress": {"stage": "processing_audio", "percent": 75},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", request.jobId).execute()

            vo_path = None
            if request.voiceoverUrl:
                vo_path = download_http_asset(request.voiceoverUrl, workdir / "voiceover.mp3")

            music_path = None
            if request.musicUrl:
                music_path = download_http_asset(request.musicUrl, workdir / "music.mp3")

            # 3. Stitch Video Clips
            supabase.table("video_jobs").update({
                "render_progress": {"stage": "stitching_video", "percent": 85},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", request.jobId).execute()

            stitched_video = workdir / "stitched_raw.mp4"
            stitch_clips(clips, stitched_video, request.aspectRatio, request.resolution)

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

            # 6. Mark Job Completed
            supabase.table("video_jobs").update({
                "status": "completed",
                "output_video_url": output_url,
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "render_progress": {"stage": "completed", "percent": 100},
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", request.jobId).execute()

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
        .select("id,status,render_progress,output_video_url,error_detail,completed_at,created_at,updated_at")
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

    if not request.jobId:
        request.jobId = str(uuid.uuid4())
    else:
        try:
            uuid.UUID(request.jobId)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid jobId format")

    if request.aspectRatio not in ("9:16", "16:9"):
        raise HTTPException(status_code=400, detail="Unsupported aspectRatio (expected '9:16' or '16:9')")

    if request.resolution not in ("720p", "1080p"):
        raise HTTPException(status_code=400, detail="Unsupported resolution (expected '720p' or '1080p')")

    # Fetch job from Supabase (or auto-create if missing)
    row_response = (
        supabase.table("video_jobs")
        .select("id,status,output_video_url")
        .eq("id", request.jobId)
        .limit(1)
        .execute()
    )
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
