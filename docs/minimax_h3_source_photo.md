# Source-photo conditioned H3 training

A source-person photo identifies whom to replace; the replacement reference supplies appearance. The source photo is encoder context only and is never a VAE reference, segmentation map or latent mask.

For each reviewed pair, use the same basename in these folders:

- `targets/clip.mp4` and `targets/clip.txt`: expected edited video and caption.
- `control_0/clip.mp4`: unmarked original guide video.
- `control_1/clip.png`: replacement character.
- `source_people/clip.png`: original performer to recognize in the guide.

Set `source_person_path` in the dataset configuration, `cache_text_embeddings: true`, `molmo_phase_b_vision: true`, and `guide_latent_only: false`. The ordinary controls contain only the guide and replacement. The cache key includes a SHA256 of the source photo; changing it regenerates the embedding. Missing or ambiguous source files fail instead of silently training without that input. See `config/examples/molmo_h3_source_photo_5090.yaml` for the pruned H3 / frozen Character Swap / frozen Ostris-v2 setup at resolution 384, 56 frames, with guide latents.

`prepare_mocha_source_photo.py` builds a separate candidate dataset from paired Mocha media and corresponding original-person references. It excludes the old two-replacement distractor subset and does not include Genjutsu. It emits a manifest and skipped-pair report; candidates are not automatically promoted to reviewed training data.

```bash
python scripts/prepare_mocha_source_photo.py --raw DATASET_CACHE --previous OLD_PREPARED_DATASET --out NEW_CANDIDATE_DATASET --download-source-photos
```

Review identity correspondence, temporal alignment, preservation of other people/background, and multi-person coverage before creating the `reviewed/` training tree. Group families across directions in the same split; the candidate stem grouping is provisional and does not guarantee identity-disjoint splits across different clips of the same performer. The validation videos must not appear in training under another cut or direction.

A dataset with only one visible performer can teach source/replacement roles without proving selection among multiple people. Training the requested multi-person task requires true edited targets where only the selected person changes, preferably matched cases where a different source photo selects a different performer in the same guide. Changing only captions while reusing an unchanged expected swap does not supply that supervision. No multi-person target is fabricated by this preparer.

To reuse an existing trainable delta, set `network.pretrained_lora_path` to the delta checkpoint and use a new run name. Keep Character Swap v1.1 and the Ostris training adapter frozen as configured. Do not warm-start with the combined LoRA while also loading the frozen Character Swap base, which would apply it twice. Recompute source-conditioned text embeddings; retain old caches as a backup.

Validation: source-role and cache tests plus existing image-guide, downscaled-reference and temporal-cache tests pass on CPU. A real BF16 Molmo + projector smoke check with source photo, replacement photo and two guide frames produced finite 5120-dimensional conditioning; changing only the source photo changed the conditioning. These checks do not constitute a completed training run or successful multi-person edit.

After reviewing three temporal samples, record decisions and any corrected source-photo overrides, then materialize a separate pilot tree with `scripts/promote_mocha_source_photo.py`. It rejects incomplete reviews and family overlap between splits. The pilot recipe uses `reviewed/train_left` and `reviewed/train_right`; its repeat factors balance a reviewed pilot of ten left-target and four right-target examples. Adjust those factors for a different reviewed dataset. Validation pairs remain outside training. The short recipe saves every ten steps and runs 100 steps before reassessment.
