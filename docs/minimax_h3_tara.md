# MiniMax H3 Ref2VA: TARA (Temporally Aligned Reference Attention)

TARA trains MiniMax H3 Ref2VA LoRAs for **motion transfer with the guide as a native video reference**:
the people in the reference pictures perform what happens in the motion video, while identity, style
and setting come from the pictures and the prompt.

H3 can already take a video as a native reference, but on its own it does not reliably follow the
*timing* of that video. TARA changes two things and adds **no learned layers**:

1. **Shared clock.** The motion video's temporal RoPE coordinate is shifted so its first latent frame
   sits on the generated video's first latent frame. Its spatial grid, and every picture reference,
   keep their native coordinates.
2. **Temporal attention bias.** Only when a generated-video query attends to a motion-video key, the
   attention logit is lowered by a bounded penalty that grows with their distance in seconds:

   `p(dt) = min(cap, strength * (max(0, dt - window) / falloff) ** 2)`

   Text, pictures, audio and the generated video's self-attention get no added penalty. The bias is
   soft: far moments stay visible.

On CUDA the bias runs inside PyTorch FlexAttention from per-row metadata (no S x S bias tensor);
on CPU a dense oracle is used for tests.

## Configuration

`model.model_kwargs` (arch `minimax_h3_ref2va`):

| Key | Default | Meaning |
|---|---:|---|
| `native_temporal_alignment` | `false` | Turns TARA on |
| `native_motion_source_channel` | `1` | Control channel (`control_path_N`, `ctrl_img_N`) holding the motion video |
| `native_temporal_strength` | `2.0` | Penalty scale |
| `native_temporal_window_seconds` | `0.2` | No penalty within this distance |
| `native_temporal_falloff_seconds` | `0.25` | Quadratic falloff scale |
| `native_temporal_max_penalty` | `6.0` | Penalty cap |

Example: [`config/examples/train_lora_minimax_h3_ref2va_tara.yaml`](../config/examples/train_lora_minimax_h3_ref2va_tara.yaml).

## Dataset

- Put the motion video in one control channel and identity/scene pictures in the others; the channel
  number, not the order in the packed sequence, tells TARA which reference is the motion. Use
  `control_path_1..3` so a channel keeps its number.
- Exactly one motion video per sample, in the configured channel.
- The motion video and the target must cover the same moments. TARA reads the motion video at the
  **target's own source frames** (also for random windows with `shrink_video_to_frames: false`),
  and refuses clips whose latent frame counts differ instead of stretching one onto the other.
  Pre-cut paired clips of the training length are the simplest setup: with random windows the
  latent rows follow the target's window, but the cached text-encoder presentation of the motion
  video still starts at its first frame.
- Latent caches written by this version store the target's frame indices; older caches still work
  when the frame selection is deterministic (`shrink_video_to_frames` or a trimmed auto frame count).

## Sampling previews

Previews use the same policy. References are packed in native H3 order (pictures, then videos), the
order ComfyUI uses, and the sample's `num_frames` must not be longer than its motion video.

## Saved metadata

LoRAs trained with TARA carry `native_temporal_alignment` (version and the options above) in their
safetensors metadata, so an inference node can apply the same policy.

## Inference

ComfyUI: `BFSMiniMaxH3NativeTemporal` in
[ComfyUI-BFSNodes](https://github.com/alisson-anjos/ComfyUI-BFSNodes), with exact Flash
partitions and optional one-pass kernels. Released LoRA and workflows:
[Alissonerdx/Minimax-H3-ComfyUI](https://huggingface.co/Alissonerdx/Minimax-H3-ComfyUI).

## Limitations

- TARA aligns time; it does not decide which picture belongs to which performer. Say it in the prompt.
- Not combined with the trained VSA sparse policy (FastH3).
- The first FlexAttention call compiles; dynamic shapes are enabled to avoid recompiling per bucket.

Technical report (method, equations, numerical checks):
[TARA-H3 v0.1.2](https://github.com/alisson-anjos/ai-toolkit/tree/tara-h3-v0.1.2/research/tara-h3).
