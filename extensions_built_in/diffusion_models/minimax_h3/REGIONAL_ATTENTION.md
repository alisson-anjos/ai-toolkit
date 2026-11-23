# Attention-routed face swap training (MiniMax H3)

This trains a face swap LoRA **with the same soft attention routing** that the ComfyUI node
**BFS H3 Regional Reference Attention** ([ComfyUI-BFSNodes](https://github.com/alisson-anjos/ComfyUI-BFSNodes),
`H3_REGIONAL_ATTENTION.md`) applies at inference. The routing makes each marked person read its own reference picture.
Training with it on teaches the LoRA to cooperate with it.

- Interactive explainer with videos and parameter simulators: https://claude.ai/artifact/KFuYAX6zLgtMejj5QZuc3H
- No masks are used anywhere. Each person is found by the model's own features, seeded by the guide's thin colored
  face contours (red = Picture 1, blue = Picture 2, ...).

## What happens in a training step

1. `MinimaxH3Model._set_regional_context` (in `minimax_h3.py`) runs before the transformer forward. With
   `model_kwargs.regional_attention` set, a batch size of 1, and gradients on, it builds a `RegionalContext` for the
   clip. A step is routed with probability `prob`.
2. The context loads the clip's contour seeds from `<group>/seeds/<stem>.npz` and resizes them to the guide token grid.
   It also finds the guide rows, the picture rows, the target rows, and the vision-token runs of each picture inside the
   text.
3. `MiniMaxH3Attention.forward` (in `src/transformer.py`) calls `regional_attention.maybe_attend` before SDPA.
   - At block `route_layer`, the value vectors of the guide rows are tracked frame to frame by label propagation, and
     the bias is built.
   - From that block on, every attention call adds the bias:

| token | toward Picture j | toward the other pictures |
|---|---|---|
| guide / target token of person j | +strength | −strength |
| token of nobody (background, unmarked people) | −background | −background |
| Picture i rows | 0 (own) | −strength |

   A person also reads little of the other generated people. The bias has rank 2P and is appended as extra head
   dimensions (`q' = [q | qe]`, `k' = [k | ke]`, scale of the real head dim), so it stays one SDPA call with no S×S mask.
4. A clip whose seeds are missing, or whose contour colors outnumber its pictures, trains without routing. A picture
   whose color was never found is left neutral (no push either way). This is common in small faces whose line color is
   lost to compression.

## Files

| file | change |
|---|---|
| `src/regional_attention.py` | new: seed loading, tracking, the bias, the biased attention |
| `src/transformer.py` | imports `regional_attention`; `MiniMaxH3Attention.forward` calls `maybe_attend` before SDPA (temporal-context calls are left alone) |
| `minimax_h3.py` | `_set_regional_context`, called right before `self.model(...)` in `get_noise_prediction` |
| `tools/precompute_guide_seeds.py` | new: contour seeds of every guide video (same contour reader as the node) |

To adapt it to another model, apply the same two hooks:
- build the context before the forward (it needs the packed layout, the reference blocks, the text token tags and the
  clip path);
- call `maybe_attend` inside the attention module, with the layer index set by `tag_layers`.

## Data

The dataset groups follow the usual swap layout:

```
<group>/targets/<stem>.mp4   training target
<group>/motion/<stem>.mp4    marked guide (Video Segment Picker, 'Multi Face Swap — Sparse Face Contour' mode)
<group>/seeds/<stem>.npz     contour seeds (written by the tool below)
```

`seeds/` sits beside `motion/` on purpose, so no loader mistakes it for a control. Precompute once:

```bash
python extensions_built_in/diffusion_models/minimax_h3/tools/precompute_guide_seeds.py /data/p1_f49/motion /data/p2_f49/motion
```

Each `.npz` holds `colors` (color indexes present) and `fills` (`uint8 [colors, frames, H/8, W/8]`). The tool skips
files that already exist.

## Config

Add the block under the model's `model_kwargs`:

```yaml
model:
  model_kwargs:
    regional_attention:
      strength: 4.0       # logit push toward the right picture / away from the wrong ones
      background: 4.0     # push away from every picture for untracked tokens
      route_layer: 12     # tracking block; the bias applies from it on. Keep the node's route_layer the same
      sharpness: 0.05     # temperature of the tracking similarity
      prob: 0.9           # share of steps routed (the rest train plainly)
```

Requirements:
- Train with batch size 1.
- At inference, use the node with `bias_layers` = "from route layer" and the same `route_layer`, so the bias matches
  what the LoRA saw.

The log prints `[H3 regional attention] N/M steps routed; this one: P people, % of guide tokens [...]` every 25 built
contexts, and up to 8 reasons for skipped steps.

## Results and notes

- The run fine-tuned the color-shuffled face swap video LoRA (v2) for 1000 steps at lr 1e-4 with the settings above.
  1000 steps were enough; 1500–2500 gave no gain.
- With the node (strength 4, background 4), the LoRA at strength 0.8 balanced the two people: the weak second person
  went from 0.25–0.33 to 0.52 ArcFace margin. The leak when people cross dropped from 0.12–0.15 to 0.01.
- On 3× fast motion the base LoRA + node leaked less (0.04 vs 0.14).
- The attention-routed LoRA needs the node at inference. Without it, identities come out swapped. Routing fewer steps
  (for example `prob: 0.5`) should keep it usable both ways, but this is untested.
- Resuming this run with `optimizer.pt` filled the 32 GB of VRAM and dropped speed from about 10 to over 100 s/it.
  Resuming without the optimizer state, with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, was fine.
- Next idea: an attention-map loss instead of the forced bias, so the LoRA learns the routing itself and does not need
  the node.
