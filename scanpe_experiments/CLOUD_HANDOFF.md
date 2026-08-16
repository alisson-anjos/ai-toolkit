# ScanPE continuity training — cloud handoff

You're picking this up on a bigger box (target: RTX 6000 Pro) after local
smoke-testing on a single RTX 5090 (32GB) hit VRAM limits. Read the commit
message on this branch (`git log -1`) first — it has the full technical
background (what ScanPE is, what's validated, what broke and why). This
file is just the concrete next steps.

## What this is

We added an optional `spatial_offsets` param to `build_packed_sequence`
(`extensions_built_in/diffusion_models/minimax_h3/src/packing.py`) so a
video frame's rotary grid can be shifted by a per-frame `(dh, dw)` offset
instead of every frame reusing the identical stationary grid (H3's default
— confirmed in code, this is exactly the "stationary bias" ScrollScape's
paper describes). This lets the model's temporal axis double as a
spatial-pan/tiling axis. Validated at inference (untrained): it reliably
breaks frame repetition, but panels aren't spatially continuous with each
other without training. `scanpe_experiments/train_scanpe_lora.py` trains a
small LoRA to fix that, using real photo tiles (WildPPS wide street
panoramas, cropped with known ground-truth per-tile positions) as targets.

## What's proven so far (local, RTX 5090 32GB, no assistant adapter / guidance loss)

A 2-tile, 192px smoke run completed 110/300 real training steps (loss
logging, checkpoint saved at step 100) before an OOM that looks like
allocator fragmentation, not a hard ceiling. Two real bugs were found and
fixed along the way (both apply regardless of GPU size, not just VRAM
workarounds):

1. **Must freeze base weights before attaching the LoRA**:
   `model.transformer.requires_grad_(False)` (+ text encoder) BEFORE
   creating `LoRASpecialNetwork`. Without this, ai-toolkit's layer-offload
   memory manager tries to compute and stage gradients for the entire
   frozen 33B base model to CPU during backward and OOMs immediately,
   regardless of batch/tile size.
2. **Gradient checkpointing did not work here**: got
   `torch.utils.checkpoint.CheckpointError: different number of tensors
   saved during forward vs recomputation` from the ConvRot int8 quantized
   custom autograd op, even with `use_reentrant=False`. This may be
   specific to how the custom hand-rolled training loop in
   `train_scanpe_lora.py` calls `get_noise_prediction` directly (bypassing
   ai-toolkit's real `SDTrainer`) — **the existing cloud headswap config
   (`h3_headswap_ltxds_r64_g2_automagic3`) sets `gradient_checkpointing:
   true` and presumably works via the real trainer**, so this might not
   reproduce there. Worth checking early: if the real trainer's
   `SDTrainer.py` path handles checkpointing fine with this quantization,
   that changes the memory-fitting story a lot (checkpointing is the
   normal fix for exactly the memory pressure we hit).

Adding the assistant/de-distillation adapter
(`ostris/minimax_h3_training_adapter`, loaded automatically via
`ModelConfig(assistant_lora_path=...)` — see
`minimax_h3.py:load_training_adapter`, already wired and confirmed it
downloads/loads cleanly) plus `do_guidance_loss` (extra no-grad forward
pass with blank-prompt conditioning to extrapolate the loss target —
mirrors `extensions_built_in/sd_trainer/SDTrainer.py:730-782`, also already
wired into `train_scanpe_lora.py`) pushed local VRAM over the edge
immediately. Both are real, wanted features (H3 has no CFG at inference —
guidance-distilled, single forward pass — so this is how you recover
prompt adherence/contrast) — just need more headroom than 32GB gave us.

## What needs adapting for the cloud box

`train_scanpe_lora.py` currently hardcodes:
- `name_or_path="Comfy-Org/MiniMax-H3"` + HF-hub auto-download —
  point at the already-staged local paths instead, same pattern as
  `train_h3_headswap_ltxds_r64_g2_automagic3.yaml`'s `model_kwargs`
  (`dit_ref2va_pruned_path`, `text_encoder_path`, `video_vae_path`,
  `audio_vae_path`).
- `layer_offloading_transformer/text_encoder_percent=0.97` — way more
  aggressive than needed on a bigger card; loosen this a lot (or drop
  `layer_offloading` entirely if VRAM allows) for real training speed.
- `TILE_SIZE=192`, `N_TILES=2` — both were memory-driven compromises.
  Bump `N_TILES` back toward 7 (or the paper-scale grid sizes we tested at
  inference: 12 tiles / 4x3 snake grid) and `TILE_SIZE` up (384-512) once
  VRAM isn't the constraint.
- `DO_GUIDANCE_LOSS = True` / `ASSISTANT_LORA_PATH` are already set correctly
  — just need the VRAM to actually run them.

## Data needed on the cloud box (not in git — see scanpe_experiments/.gitignore)

`WildPPS` (80 real wide 2048x400 street panoramas — the only training data
source used so far) needs to be copied over from the local WSL machine
(`/home/alissonerdx/tools/ai-toolkit/scanpe_experiments/WildPPS`, ~279MB)
since it was never downloaded from a public source in this session — ask
the user for it directly (scp/rsync/cloud storage, whatever they normally
use to move data to this box). Re-run
`scanpe_experiments/prep_wildpps_tiles.py` after copying (adjust
`TILE_SIZE`/`N_TILES` constants there to match whatever you land on in
`train_scanpe_lora.py` — they must match).

Two more diverse datasets were scouted for a later, bigger v2 (see the
commit message / project memory) but not yet used for training:
`SingleBicycle/4KLSDB` (HF, 129k native-4K diverse images, best fit for
generalizing beyond fixed panorama scans) and `kkkkkb/LIU4K` (HF, 47GB,
4 categories, no captions, lower priority).

## Suggested order of operations

1. `git fetch` this branch, read the commit message + this file.
2. Point the script at the local staged model paths; run once at the SAME
   small tile/offload settings as local (2 tiles, 192px, no adapter/guidance
   loss) just to confirm the port works end-to-end on this box first.
3. Re-enable the assistant adapter + guidance loss, confirm they fit.
4. Scale `N_TILES`/`TILE_SIZE` back up now that VRAM isn't fighting you.
5. Investigate whether gradient checkpointing actually works via the real
   `SDTrainer` path here — if yes, that's a better foundation than the
   hand-rolled loop long-term (you'd get real dataset/dataloader support,
   proper checkpointing, wandb logging, etc. for free) and would be worth
   porting the ScanPE offset logic into a real ai-toolkit dataset config
   instead of the custom bypass script.
