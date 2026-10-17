# ComfyUI + MiniMax H3 nodes, WITHOUT models. The models (~81 GB) are downloaded at boot (boot.sh)
# onto the pod's local disk, so no network volume is needed.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
      python3 python3-venv python3-dev git ffmpeg ca-certificates curl build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv
ENV PATH=/opt/venv/bin:$PATH

# torch cu130: ComfyUI only enables its optimized CUDA ops from cu130 up.
# rp.py asks RunPod for hosts with CUDA >= 13.0 so the driver matches.
RUN pip install -q --upgrade pip && \
    pip install -q torch==2.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Same ComfyUI commit as the network volume install (known good with H3).
ARG COMFY_COMMIT=b5cc8830279eae909a59de030af1e50761c36751
RUN git clone -q https://github.com/comfyanonymous/ComfyUI /opt/ComfyUI && \
    git -C /opt/ComfyUI checkout -q $COMFY_COMMIT && \
    pip install -q -r /opt/ComfyUI/requirements.txt

WORKDIR /opt/ComfyUI/custom_nodes
RUN for repo in \
      https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite \
      https://github.com/kijai/ComfyUI-KJNodes \
      https://github.com/city96/ComfyUI-GGUF \
      https://github.com/rgthree/rgthree-comfy \
      https://github.com/cubiq/ComfyUI_essentials \
      https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler ; do \
      git clone -q --depth 1 "$repo" ; \
    done && \
    for d in */ ; do [ -f "$d/requirements.txt" ] && pip install -q -r "$d/requirements.txt" || true ; done && \
    pip install -q "huggingface_hub[hf_xet]" hf_transfer
COPY nodes/comfyui-h3-keyframe-audio /opt/ComfyUI/custom_nodes/comfyui-h3-keyframe-audio

COPY podsrv.py boot.sh /opt/
RUN chmod +x /opt/boot.sh
WORKDIR /opt/ComfyUI
EXPOSE 8188 8000
CMD ["/opt/boot.sh"]
