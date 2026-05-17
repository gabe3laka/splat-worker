# splat-worker

RunPod serverless worker for training a 3D Gaussian Splat from a phone scan video. Outputs are uploaded to Supabase Storage and consumed by SuperSplat Viewer in the browser.

## Job Input

Example JSON payload:
```json
{
  "video_url": "https://example.com/scan.mp4",
  "scan_id": "uuid",
  "patient_id": "uuid",
  "iters": 7000,
  "fps": 2
}
```
- `iters`: Optional, overrides `SPLAT_ITERS` env var (default: `7000`).
- `fps`: Optional, overrides `SPLAT_FPS` env var (default: `2`).

## Job Output

Successful job result:
```json
{
  "status": "complete",
  "ply_path": "<patient_id>/<scan_id>/scene.ply"
}
```

Error response:
```json
{
  "status": "error",
  "error": "<message>"
}
```

## Required Environment Variables

- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`

### Optional Environment Variables
- `SUPABASE_BUCKET`: Default `scan-splats`
- `SPLAT_ITERS`: Default `7000`
- `SPLAT_FPS`: Default `2`

## Deployment Instructions

1. Clone the repository:
```bash
git clone https://github.com/gabe3laka/splat-worker.git
cd splat-worker
```

2. Build the Docker image:
```bash
docker build -t splat-worker .
```

3. Run locally (example):
```bash
docker run --rm -e SUPABASE_URL=YOUR_URL -e SUPABASE_SERVICE_ROLE_KEY=YOUR_KEY splat-worker
```

4. Deploy to RunPod:
- Push the Docker image to a container registry (e.g., DockerHub).
- Configure RunPod to pull and execute the container with the required environment variables.

## Technical Pipeline
1. Download `video_url` to temp directory.
2. Extract frames using `ffmpeg` (minimum 10 frames).
3. Run COLMAP feature extraction and mapping (CPU SIFT).
4. Train 3D Gaussian Splat (`gsplat`) for the specified iterations.
5. Export `scene.ply`.
6. Upload result to Supabase Storage as `{patient_id}/{scan_id}/scene.ply`.

## Base Image Details
- `nvidia/cuda:12.8.0-devel-ubuntu22.04`
- Python 3.10, PyTorch 2.8 (cu128)
- gsplat 1.5.3.
