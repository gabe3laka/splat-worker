# splat-worker

RunPod serverless worker that trains a real 3D Gaussian Splat from a phone
scan video and uploads `scene.ply` to Supabase Storage. Designed to live
side-by-side with [`lingbot-map-worker`](https://github.com/gabe3laka/lingbot-map-worker)
without touching it.

- **Trainer**: [gsplat](https://github.com/nerfstudio-project/gsplat) v1.5.0
- **SfM**: COLMAP (fresh run; no reuse of lingbot extrinsics)
- **Base image**: `nvidia/cuda:12.8.0-devel-ubuntu22.04` + PyTorch 2.8 cu128
- **GPU**: RTX A5000 24 GB (same class as lingbot)
- **Output**: `.ply` consumed by SuperSplat in the browser

## Pipeline

```
video_url -> ffmpeg frames -> COLMAP sparse -> gsplat train -> scene.ply -> Supabase Storage
```

## Input (RunPod job)

```json
{
  "input": {
    "video_url": "https://.../scan.mp4",
    "scan_id": "uuid",
    "patient_id": "uuid",
    "iters": 15000,
    "fps": 2
  }
}
```

`iters` and `fps` are optional. `iters` defaults to the `SPLAT_ITERS` env var
(default 15000, allowed range 1000-60000). `fps` defaults to 2.

## Output

On success:

```json
{
  "splat_url": "<patient_id>/<scan_id>/scene.ply",
  "scan_id": "...",
  "status": "completed",
  "metrics": {
    "num_gaussians": 412345,
    "iters": 15000,
    "train_time_s": 612.4,
    "frames_extracted": 240,
    "colmap_images": 238,
    "total_time_s": 731.0
  }
}
```

On failure:

```json
{
  "status": "failed",
  "scan_id": "...",
  "error": "...",
  "traceback": "..."
}
```

## Environment variables

| Name | Required | Notes |
|------|----------|-------|
| `SUPABASE_URL` | yes | Project URL |
| `SUPABASE_SERVICE_ROLE_KEY` | yes | Service-role key (write to Storage) |
| `SUPABASE_BUCKET` | no | Defaults to `scan-splats` |
| `SPLAT_ITERS` | no | Default training iterations (default 15000) |

## RunPod endpoint setup

1. Build from this GitHub repo (`gabe3laka/splat-worker`, branch `main`).
2. GPU: RTX A5000 (or any 24 GB Ampere+).
3. Container disk: 50 GB.
4. Env vars: set the four above.
5. Endpoint URL is consumed by the dental-flow `reconstruct-splat` Edge
   Function (separate from the lingbot path).

The lingbot endpoint `mvwq1zzz0smpc0` is intentionally **not** modified by
this project.

## License

MIT - see [LICENSE](./LICENSE).
