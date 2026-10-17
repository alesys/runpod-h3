#!/usr/bin/env bash
# Pod boot without a network volume: download the H3 models to local disk, then start ComfyUI.
# Every step is timed in /workspace/_status/boot.log so we can compare against the volume boot.
W=/workspace
C=/opt/ComfyUI
M=$C/models
mkdir -p $W/_status
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $W/_status/boot.log; }
T0=$(date +%s)
log "== boot (image ready)"

# podsrv.py serves /workspace; keep ComfyUI reachable at /workspace/ComfyUI like on the volume.
[ -e $W/ComfyUI ] || ln -s $C $W/ComfyUI
python3 /opt/podsrv.py 8000 &

mkdir -p $M/diffusion_models $M/text_encoders $M/vae $M/loras $M/latent_upscale_models $M/model_patches
export HF_HUB_ENABLE_HF_TRANSFER=1
dl() { # repo  path_in_repo  dest_dir  [dest_name]
  local name=${4:-$(basename "$2")}
  [ -s "$3/$name" ] && return
  python - "$1" "$2" "$3" "$name" <<'PY'
import sys, os, shutil
from huggingface_hub import hf_hub_download
repo, path, dest, name = sys.argv[1:]
p = hf_hub_download(repo, path, local_dir=os.path.join(dest, ".hf"))
shutil.move(p, os.path.join(dest, name))
PY
  [ -s "$3/$name" ] && log "ok $name" || log "FAIL $name"
}
H3=Comfy-Org/MiniMax-H3
# In parallel: the big ones dominate; the small ones finish in seconds.
dl $H3 text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors $M/text_encoders &
dl $H3 diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors $M/diffusion_models &
dl $H3 diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors $M/diffusion_models &
dl $H3 vae/minimax_h3_video_vae_fp16.safetensors $M/vae &
dl $H3 vae/minimax_h3_audio_vae_fp32.safetensors $M/vae &
dl $H3 model_patches/minimax_h3_fun_controlnet_union_2.0_pruned_int8_convrot.safetensors $M/model_patches &
dl drbaph/MiniMax-H3-Turbo-Lora-ComfyUI minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors $M/loras &
dl LBH-123-AI/Minimax_h3_latent_Upscaler minimax_h3_latent_upscaler_3d_conv_v1/minimax_h3_latent_upscaler_3d_conv_v1_fp16.safetensors $M/latent_upscale_models minimax_h3_latent_upscaler_3d_fp16.safetensors &
wait
rm -rf $M/*/.hf
log "models done in $(( $(date +%s) - T0 )) s ($(du -sh $M | cut -f1))"

log "starting ComfyUI"
cd $C
python main.py --listen 0.0.0.0 --port 8188 --disable-dynamic-vram >> $W/_status/comfy.log 2>&1
log "ComfyUI exited"
sleep infinity
