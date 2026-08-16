"""ScanPE continuity LoRA -- smoke test.

Trains a small LoRA so the model learns that a ScanPE offset means "continue
the same real scene," using the 80 WildPPS wide-photo sequences (7 tiles
each, cropped by prep_wildpps_tiles.py) as ground truth. Same offset
trajectory as the untrained scanpe_test.py run, so the two are a direct A/B:
does training fix the lack of cross-tile continuity we saw raw?

Bypasses ai-toolkit's dataset/dataloader system (it has no concept of "N
tiles of one real photo with known positions") and calls the model's own
get_noise_prediction directly, with batch=None (no audio/ref/keyframe
conditioning needed) and a new model._train_scan_offsets hook (see
minimax_h3.py) supplying the real per-tile spatial offset.

Run from the ai-toolkit root:
    .venv/bin/python scanpe_experiments/train_scanpe_lora.py
"""
import glob
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from toolkit.config_modules import ModelConfig, NetworkConfig
from toolkit.util.get_model import get_model_class
from toolkit.lora_special import LoRASpecialNetwork
from extensions_built_in.diffusion_models.minimax_h3.src import packing

TILES_DIR = "scanpe_experiments/wildpps_tiles"
OUT_DIR = "scanpe_experiments/scanpe_lora_out"
N_TILES = 2
TILE_SIZE = 192
STRIDE = float(packing._ROPE_SPATIAL_SCALE)
STEPS = 300
LR = 1e-4
SAVE_EVERY = 100
DO_GUIDANCE_LOSS = True
GUIDANCE_LOSS_TARGET = 2.0
ASSISTANT_LORA_PATH = "ostris/minimax_h3_training_adapter/minimax_h3_ref2va_training_adapter_v1.safetensors"

PROMPT = """subject_definitions:

summary:
[reference generation] The target video shows a continuous wide street-level city panorama.

retention_analysis:

detailed_description:
The image is a sharp, photorealistic street-level panorama, natural daylight, crisp fine detail on buildings, streets, and vegetation.
[Shot 1] A continuous city street scene extends across the frame, buildings and street-level detail consistent throughout.

overall_soundscape:
N/A

non_diegetic_music:
N/A
"""


def list_sequences():
    cities = sorted(
        {
            os.path.basename(p).rsplit("_t", 1)[0]
            for p in glob.glob(os.path.join(TILES_DIR, "*_t0.png"))
        }
    )
    return cities


def load_sequence(city):
    imgs = []
    for i in range(N_TILES):
        p = os.path.join(TILES_DIR, f"{city}_t{i}.png")
        img = Image.open(p).convert("RGB")
        imgs.append(img)
    return imgs


def to_tensor(img: Image.Image) -> torch.Tensor:
    import numpy as np

    arr = torch.from_numpy(np.array(img)).float()
    arr = (arr / 127.5) - 1.0  # [-1, 1]
    return arr.permute(2, 0, 1)  # (3, H, W)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    cities = list_sequences()
    print(f"{len(cities)} training sequences")

    model_config = ModelConfig(
        name_or_path="Comfy-Org/MiniMax-H3",
        arch="minimax_h3",
        dtype="bf16",
        quantize=True,
        quantize_te=True,
        layer_offloading=True,
        layer_offloading_transformer_percent=0.97,
        layer_offloading_text_encoder_percent=0.97,
        model_kwargs={"partition": "ref2va_pruned"},
        assistant_lora_path=ASSISTANT_LORA_PATH,
    )
    ModelClass = get_model_class(model_config)
    model = ModelClass(device="cuda:0", model_config=model_config, dtype="bf16")
    model.load_model()
    # NOTE: gradient checkpointing is incompatible with the ConvRot quantized
    # custom autograd op here (CheckpointError: tensor count mismatch between
    # forward and recompute) -- rely on heavier layer offloading + a smaller
    # tile size to fit training instead.
    model.transformer.requires_grad_(False)
    model.text_encoder.requires_grad_(False)

    network_config = NetworkConfig(type="lora", linear=16, linear_alpha=16)
    network = LoRASpecialNetwork(
        text_encoder=model.text_encoder,
        unet=model.transformer,
        lora_dim=network_config.linear,
        multiplier=1.0,
        alpha=network_config.linear_alpha,
        train_unet=True,
        train_text_encoder=False,
        conv_lora_dim=network_config.conv,
        conv_alpha=network_config.conv_alpha,
        is_transformer=model.is_transformer,
        network_config=network_config,
        network_type=network_config.type,
        transformer_only=network_config.transformer_only,
        base_model=model,
        target_lin_modules=model.target_lora_modules,
    )
    network.force_to(model.device_torch, dtype=torch.float32)
    model.network = network
    network._update_torch_multiplier()
    network.apply_to(model.text_encoder, model.transformer, False, True)
    network.is_active = True

    params = network.prepare_optimizer_params(0.0, LR, LR)
    optimizer = torch.optim.AdamW(params, lr=LR)

    conditional_embeds = model.get_prompt_embeds(PROMPT, control_images=None)
    unconditional_embeds = model.get_prompt_embeds("", control_images=None)

    t_lat = N_TILES
    offsets = torch.stack(
        [torch.zeros(t_lat), torch.arange(t_lat, dtype=torch.float64) * STRIDE],
        dim=-1,
    )

    for step in range(1, STEPS + 1):
        city = random.choice(cities)
        tiles = load_sequence(city)
        frames = torch.stack([to_tensor(t) for t in tiles], dim=0)  # (T, 3, H, W)

        with torch.no_grad():
            frames_in = frames.unsqueeze(2).to(model.device_torch)  # (T, 3, 1, H, W)
            tile_latents = model.encode_keyframe_latents(frames_in)  # (T, 24, 1, h, w)

        # stack the T independently-encoded single-frame latents along the
        # temporal dim -> (1, 24, T, h, w), no cross-tile blending
        clean = tile_latents.squeeze(2).permute(1, 0, 2, 3).unsqueeze(0)
        clean = clean.to(model.device_torch, torch.float32)

        sigma = torch.rand(1, device=model.device_torch).clamp(1e-3, 0.999)
        noise = torch.randn_like(clean)
        noisy = (1.0 - sigma.view(-1, 1, 1, 1, 1)) * clean + sigma.view(-1, 1, 1, 1, 1) * noise
        timestep = sigma * 1000.0

        target = (noise - clean).detach()

        model._train_scan_offsets = offsets
        if DO_GUIDANCE_LOSS:
            # bake a CFG-like extrapolation into the loss target itself: H3
            # has no real CFG at inference (guidance-distilled, single
            # forward pass), so this teaches the plain conditional
            # prediction to already look like it had guidance_scale applied.
            with torch.no_grad(), network:
                uncond_pred = model.get_noise_prediction(
                    latent_model_input=noisy,
                    timestep=timestep,
                    text_embeddings=unconditional_embeds,
                    batch=None,
                )
            target = uncond_pred + GUIDANCE_LOSS_TARGET * (target - uncond_pred)
            target = target.detach()

        with network:
            pred = model.get_noise_prediction(
                latent_model_input=noisy,
                timestep=timestep,
                text_embeddings=conditional_embeds,
                batch=None,
            )
        model._train_scan_offsets = None

        loss = torch.nn.functional.mse_loss(pred.float(), target.float())
        optimizer.zero_grad()
        torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()

        if step % 10 == 0:
            print(f"step {step}/{STEPS} loss {loss.item():.4f}")
        if step % SAVE_EVERY == 0 or step == STEPS:
            out_path = os.path.join(OUT_DIR, f"scanpe_lora_step{step}.safetensors")
            network.save_weights(out_path, dtype=torch.float16)
            print(f"saved {out_path}")


if __name__ == "__main__":
    main()
