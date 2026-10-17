# runpod-h3

Docker image with ComfyUI and the MiniMax H3 nodes, for RunPod pods without a network volume.

- The image has no models. At boot, `boot.sh` downloads the H3 models (~81 GB) from Hugging Face to the pod's local disk, then starts ComfyUI on port 8188. Timings go to `/workspace/_status/boot.log`.
- GitHub Actions builds the image on every push to `main` and publishes it to `ghcr.io/alesys/runpod-h3:latest`.
- Port 8000 runs `podsrv.py`, a small helper that reads logs and receives uploads. It requires the `X-Token` header to match `$POD_TOKEN`.
- The pod is launched from `runpod-comfy/rp.py` (`up --novol`).
- Torch is built for cu130, so the pod needs a host with CUDA 13 or newer.
