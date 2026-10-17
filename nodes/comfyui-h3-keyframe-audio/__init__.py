"""MiniMax H3 — first/last keyframes AND reference audio in ONE generation.

Written 2026-08-07 for the TickleGame production exploration. The stock nodes
force a choice:

  * MiniMaxH3ImageToVideo (fl2va) takes first_frame / last_frame — continuity
    anchored at both ends — but has no audio input at all.
  * MiniMaxH3ReferenceToVideo (ref2va) takes ref_audios — a real voice can
    condition the generated speech — but composes freely, so wardrobe and
    props drift between shots.

Production needs both at once: anchored continuity AND the character's voice.

HOW IT WORKS (all of this is stock-ComfyUI behaviour, no core patch):

 1. Tokenizer: `images=` and `minimax_ref_items=` are mutually exclusive
    (comfy/text_encoders/minimax.py — it's an if/else), so BOTH keyframes and
    audio are presented through the ref-items path. The keyframes are labelled
    <Picture 1>/<Picture 2> with real vision blocks; the audio gets <Audio 1>.

 2. Layout: comfy/ldm/minimax/model.py PackedLayout already accepts
    `keyframes` AND `refs` as separate arguments and lays them out in
    sequence — text rows, keyframe cond rows, then reference rows. Nothing in
    the layout forbids the combination.

 3. The one real collision is in comfy/model_base.py: the `refs` branch
    OVERWRITES `payload["cond_video_latents"]` that the `keyframes` branch just
    set. Since audio-only refs contribute no video latents, a naive combination
    would silently drop the keyframe latents and desync the packed sequence.

    The workaround needs no patch: PackedLayout SKIPS ref blocks whose `kind`
    it does not recognise, but model_base still harvests their `latent`. So the
    keyframe latents ride along as inert carrier blocks — they restore
    cond_video_latents in the correct order (keyframes first, then any real
    image refs) while adding zero rows to the layout.

⚠️ UNVERIFIED: whether the checkpoint tolerates the mix. fl2va and ref2va are
separate weight sets; keyframe conditioning is fl2va's specialty and reference
conditioning is ref2va's. This node lets you wire either UNET and find out —
that experiment is the point. Judge by looking at frames, not by whether it
runs.

Prior art worth reviewing before investing more: ComfyUI-MiniMaxH3-Director
(seesee75-commits) advertises the same combination plus a timeline editor.
"""

import math

import torch

import comfy.utils
import node_helpers
from comfy_api.latest import io

CANVAS_MULTIPLE = 16
FPS = 24
LATENT_STRIDE = 4  # temporal compression of the video VAE


def _resize(img, width, height, crop):
    return comfy.utils.common_upscale(
        img.movedim(-1, 1), width, height, "lanczos", crop
    ).movedim(1, -1)


def _empty_av_latent(width, height, length):
    """Mirror of comfy_extras.nodes_minimax_h3._empty_av_latent."""
    from comfy_extras.nodes_minimax_h3 import _empty_av_latent as stock

    return stock(width, height, length)


class MiniMaxH3KeyframesWithAudio(io.ComfyNode):
    """fl2va keyframes + ref2va audio, in a single conditioning."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3KeyframesWithAudio",
            display_name="MiniMax H3 Keyframes + Reference Audio",
            category="model/conditioning/minimax",
            description=(
                "First/last keyframe anchoring AND reference audio in one shot. "
                "Keyframes are presented to the text encoder as <Picture 1>/"
                "<Picture 2>; each reference audio as <Audio N>. Cite those tags "
                "in the prompt or they anchor nothing."
            ),
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae"),
                io.String.Input("prompt", multiline=True, dynamic_prompts=True),
                io.Int.Input("width", default=768, min=32, max=4096, step=32),
                io.Int.Input("height", default=768, min=32, max=4096, step=32),
                io.Int.Input(
                    "length", default=90, min=5, max=3600, step=17,
                    tooltip="Frames at 24 fps on the model's 17k+5 grid (90 = 3.75 s)."),
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.Audio.Input("ref_audio_1", optional=True),
                io.Audio.Input("ref_audio_2", optional=True),
            ],
            outputs=[io.Conditioning.Output(display_name="positive"), io.Latent.Output()],
        )

    @staticmethod
    def _encode_audio(audio_vae, audio):
        import torchaudio

        waveform = audio["waveform"]
        sr = audio["sample_rate"]
        vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
        if sr != vae_sr:
            waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
        z = audio_vae.encode(waveform[:1].movedim(1, -1))
        return z, z.shape[-1]

    @classmethod
    def execute(cls, clip, vae, audio_vae, prompt, width, height, length,
                first_frame=None, last_frame=None,
                ref_audio_1=None, ref_audio_2=None) -> io.NodeOutput:
        latent, frame_count = _empty_av_latent(width, height, length)

        ref_items = []   # tokenizer presentation, in order
        ref_blocks = []  # DiT payload, same order
        keyframes = []

        # ── keyframes: geometry anchor first, follower last ──────────────
        for img, index, crop in (
            (first_frame, 0, "disabled"),
            (last_frame, frame_count - 1, "center"),
        ):
            if img is None:
                continue
            resized = _resize(img[:1], width, height, crop)
            z = vae.encode(resized)
            keyframes.append({"resolved_frame_index": index, "latent": z})
            ref_items.append({"type": "image", "data": resized})
            # No keyframe_carrier since ComfyUI e01fb4c5 (MiniMaxH3AddGuide): model_base
            # now APPENDS ref latents to cond_video_latents instead of overwriting them,
            # so a carrier doubled the keyframe rows ("shape mismatch [2N,96] vs [N,96]").

        # ── reference audio ──────────────────────────────────────────────
        for audio in (ref_audio_1, ref_audio_2):
            if audio is None:
                continue
            audio_latent, ref_audio_t = cls._encode_audio(audio_vae, audio)
            ref_items.append({"type": "audio"})
            ref_blocks.append({
                "kind": "audio",
                "ref_audio_t": ref_audio_t,
                "audio_latent": audio_latent,
            })

        tokens = clip.tokenize(prompt, minimax_ref_items=ref_items)
        cond = clip.encode_from_tokens_scheduled(tokens)

        values = {}
        if keyframes:
            values["minimax_keyframes"] = keyframes
            values["minimax_frame_count"] = frame_count
        if ref_blocks:
            values["minimax_refs"] = ref_blocks
        if values:
            cond = node_helpers.conditioning_set_values(cond, values)
        return io.NodeOutput(cond, latent)


class H3KeyframeAudioExtension:
    @classmethod
    def get_node_list(cls):
        return [MiniMaxH3KeyframesWithAudio]


async def comfy_entrypoint():
    return H3KeyframeAudioExtension()


NODE_CLASS_MAPPINGS = {"MiniMaxH3KeyframesWithAudio": MiniMaxH3KeyframesWithAudio}
NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3KeyframesWithAudio": "MiniMax H3 Keyframes + Reference Audio"
}
