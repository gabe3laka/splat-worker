"""splat-worker RunPod handler.

Pipeline:
  1. Download video from a URL or Supabase Storage signed URL.
  2. Extract frames with ffmpeg.
  3. Run COLMAP (feature extraction + matching + mapper) to get sparse cloud + poses.
  4. Train a 3D Gaussian Splat with gsplat for SPLAT_ITERS iterations.
  5. Export scene.ply and upload to Supabase Storage bucket SUPABASE_BUCKET.
  6. Return {splat_url, scan_id, status, metrics}.

This is a sibling to lingbot-map-worker. It NEVER reads or writes lingbot
buckets or columns. The dental-flow Edge Function chooses which worker to
call and wires the output back to the correct row.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict

import requests
import runpod
from supabase import create_client


# ---------- helpers ----------

def _require_input(job_input: Dict[str, Any], key: str) -> Any:
    if key not in job_input or job_input[key] in (None, ""):
        raise ValueError(f"missing required input field: {key}")
    return job_input[key]


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise RuntimeError(f"missing required env var: {name}")
    return v


def _progress(msg: str) -> None:
    print(f"[splat-worker] {msg}", flush=True)


def _run(cmd: list[str], cwd: str | None = None) -> None:
    _progress("$ " + " ".join(cmd))
    res = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.stdout.write(res.stdout)
        sys.stderr.write(res.stderr)
        raise RuntimeError(f"command failed ({res.returncode}): {' '.join(cmd)}")


# ---------- pipeline steps ----------

def download_video(video_url: str, out_path: Path) -> None:
    _progress(f"downloading video -> {out_path}")
    with requests.get(video_url, stream=True, timeout=120) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                if chunk:
                    f.write(chunk)


def extract_frames(video_path: Path, frames_dir: Path, fps: int = 2) -> int:
    frames_dir.mkdir(parents=True, exist_ok=True)
    _progress(f"extracting frames at {fps} fps")
    _run([
        "ffmpeg", "-y", "-i", str(video_path),
        "-vf", f"fps={fps}",
        "-q:v", "2",
        str(frames_dir / "frame_%05d.jpg"),
    ])
    n = len(list(frames_dir.glob("frame_*.jpg")))
    if n < 10:
        raise RuntimeError(f"too few frames extracted: {n}")
    _progress(f"extracted {n} frames")
    return n


def run_colmap(frames_dir: Path, workspace: Path) -> int:
    workspace.mkdir(parents=True, exist_ok=True)
    db = workspace / "database.db"
    sparse = workspace / "sparse"
    sparse.mkdir(exist_ok=True)

    _progress("colmap feature_extractor")
    _run([
        "colmap", "feature_extractor",
        "--database_path", str(db),
        "--image_path", str(frames_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.use_gpu", "1",
    ])

    _progress("colmap exhaustive_matcher")
    _run([
        "colmap", "exhaustive_matcher",
        "--database_path", str(db),
        "--SiftMatching.use_gpu", "1",
    ])

    _progress("colmap mapper")
    _run([
        "colmap", "mapper",
        "--database_path", str(db),
        "--image_path", str(frames_dir),
        "--output_path", str(sparse),
    ])

    models = [p for p in sparse.iterdir() if p.is_dir()]
    if not models:
        raise RuntimeError("colmap produced no sparse model")
    n_images = len(list(frames_dir.glob("frame_*.jpg")))
    _progress(f"colmap done, registered {n_images} images")
    return n_images


def train_splat(frames_dir: Path, colmap_ws: Path, out_ply: Path, iters: int) -> Dict[str, Any]:
    """Run gsplat default trainer on the COLMAP output and export a PLY."""
    trainer = Path("/opt/gsplat/examples/simple_trainer.py")
    if not trainer.exists():
        raise RuntimeError(f"gsplat trainer not found at {trainer}")

    # gsplat simple_trainer expects: data_dir with images/ and sparse/0/
    data_dir = colmap_ws.parent / "gs_data"
    (data_dir / "images").mkdir(parents=True, exist_ok=True)
    for img in frames_dir.glob("frame_*.jpg"):
        target = data_dir / "images" / img.name
        if not target.exists():
            target.symlink_to(img)
    sparse_src = colmap_ws / "sparse" / "0"
    sparse_dst = data_dir / "sparse" / "0"
    sparse_dst.parent.mkdir(parents=True, exist_ok=True)
    if not sparse_dst.exists():
        sparse_dst.symlink_to(sparse_src)

    result_dir = colmap_ws.parent / "gs_result"
    result_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    _run([
        "python", str(trainer), "default",
        "--data_dir", str(data_dir),
        "--result_dir", str(result_dir),
        "--max_steps", str(iters),
        "--disable_viewer",
    ])
    train_time = time.time() - t0

    # Find produced PLY (gsplat saves to result_dir/ply/point_cloud_*.ply or similar)
    candidates = list(result_dir.rglob("*.ply"))
    if not candidates:
        raise RuntimeError("gsplat produced no .ply output")
    final = max(candidates, key=lambda p: p.stat().st_mtime)
    shutil.copy(final, out_ply)

    # Rough gaussian count = ply vertex count
    num_gaussians = 0
    try:
        from plyfile import PlyData
        ply = PlyData.read(str(out_ply))
        for el in ply.elements:
            if el.name in ("vertex", "point"):
                num_gaussians = int(el.count)
                break
    except Exception as e:
        _progress(f"plyfile read warning: {e}")

    return {
        "num_gaussians": num_gaussians,
        "iters": iters,
        "train_time_s": round(train_time, 1),
    }


def upload_to_supabase(local_ply: Path, patient_id: str, scan_id: str) -> str:
    url = _require_env("SUPABASE_URL")
    key = _require_env("SUPABASE_SERVICE_ROLE_KEY")
    bucket = os.environ.get("SUPABASE_BUCKET", "scan-splats")
    storage_path = f"{patient_id}/{scan_id}/scene.ply"

    _progress(f"uploading -> bucket={bucket} path={storage_path}")
    sb = create_client(url, key)
    with open(local_ply, "rb") as f:
        data = f.read()
    try:
        sb.storage.from_(bucket).upload(
            storage_path,
            data,
            {"content-type": "application/octet-stream", "upsert": "true"},
        )
    except Exception as e:
        # supabase-py may raise on overwrite; try remove + upload
        _progress(f"upload retry after: {e}")
        try:
            sb.storage.from_(bucket).remove([storage_path])
        except Exception:
            pass
        sb.storage.from_(bucket).upload(
            storage_path,
            data,
            {"content-type": "application/octet-stream"},
        )
    return storage_path


# ---------- runpod entrypoint ----------

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    started = time.time()
    job_input = job.get("input", {}) or {}
    try:
        video_url = _require_input(job_input, "video_url")
        scan_id = _require_input(job_input, "scan_id")
        patient_id = _require_input(job_input, "patient_id")
        iters = int(job_input.get("iters") or os.environ.get("SPLAT_ITERS") or 15000)
        if iters < 1000 or iters > 60000:
            raise ValueError(f"iters out of range: {iters}")

        _progress(f"job scan_id={scan_id} patient_id={patient_id} iters={iters}")

        with tempfile.TemporaryDirectory(prefix="splat_") as tmp:
            tmp_path = Path(tmp)
            video_path = tmp_path / "input.mp4"
            frames_dir = tmp_path / "frames"
            colmap_ws = tmp_path / "colmap"
            out_ply = tmp_path / "scene.ply"

            download_video(video_url, video_path)
            n_frames = extract_frames(video_path, frames_dir, fps=int(job_input.get("fps") or 2))
            n_images = run_colmap(frames_dir, colmap_ws)
            metrics = train_splat(frames_dir, colmap_ws, out_ply, iters)
            metrics["frames_extracted"] = n_frames
            metrics["colmap_images"] = n_images

            splat_url = upload_to_supabase(out_ply, patient_id, scan_id)

        metrics["total_time_s"] = round(time.time() - started, 1)
        _progress(f"done in {metrics['total_time_s']}s -> {splat_url}")
        return {
            "splat_url": splat_url,
            "scan_id": scan_id,
            "status": "completed",
            "metrics": metrics,
        }
    except Exception as e:
        tb = traceback.format_exc()
        _progress("ERROR: " + str(e))
        sys.stderr.write(tb)
        return {
            "status": "failed",
            "scan_id": job_input.get("scan_id"),
            "error": str(e),
            "traceback": tb,
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
