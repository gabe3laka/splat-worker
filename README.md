# splat-worker

RunPod serverless worker that trains a 3D Gaussian Splat from a phone scan video and uploads `scene.ply` to Supabase Storage. Output is consumed by SuperSplat Viewer in the browser.

## Job input

```json
{
  "video_url": "https://.../scan.mp4",
  "scan_id": "uuid",
  "patient_id": "uuid",
  "iters": 7000,
  "fps": 2
}
```

`iters` and `fps` are optional and override the `SPLAT_ITERS` / `SPLAT_FPS` env vars.

## Job output

On success:

```json
{ "status": "complete", "ply_path": "<patient_id>/<scan_id>/scene.ply" }
```

The PLY is uploaded to Supabase Storage bucket `scan-splats` at `{patient_id}/{scan_id}/scene.ply`.

On failure:

```json
{ "status": "error", "error": "<message>" }
```

## Required env vars

- `SUPABASE_URL`
- - `SUPABASE_SERVICE_ROLE_KEY`
 
  - ## Optional env vars
 
  - - `SUPABASE_BUCKET` (default `scan-splats`)
    - - `SPLAT_ITERS` (default `7000`)
      - - `SPLAT_FPS` (default `2`)
       
        - ## Pipeline
       
        - 1. Download `video_url` to tmp.
          2. 2. ffmpeg extracts frames at `fps`. A minimum of 10 frames is required.
             3. 3. COLMAP feature extraction + mapping (CPU SIFT).
                4. 4. gsplat training for `iters` iterations on a single GPU.
                   5. 5. Export `scene.ply`.
                      6. 6. Upload to Supabase Storage. Return `{ "status": "complete", "ply_path": ... }`.
                        
                         7. ## Base image
                        
                         8. `nvidia/cuda:12.8.0-devel-ubuntu22.04` + Python 3.10 + PyTorch 2.8 (cu128) + gsplat 1.5.3.
                         9. 
