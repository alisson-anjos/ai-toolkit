#!/usr/bin/env python3
"""Daemon that watches `/workspace/ai-toolkit/output/minimax_h3_molmo2_mocha_selective_swap_v1`
and immediately uploads every new LoRA checkpoint (both standalone delta and combined with
`minimax_h3_character_swap_v1.1_ref2va_fro.safetensors`) + training logs to Hugging Face
`Alissonerdx/molmo2-h3-encoder` so nothing is lost if RunPod credits run out.
"""

import os
import time
from pathlib import Path

import torch
from huggingface_hub import HfApi
from safetensors.torch import load_file, save_file

JOB_DIR = Path("/workspace/ai-toolkit/output/minimax_h3_molmo2_mocha_selective_swap_v1")
BASE_FRO_LORA = Path("/workspace/henge/models/loras/henge/minimax_h3_character_swap_v1.1_ref2va_fro.safetensors")
TRAIN_LOG = Path("/workspace/ai-toolkit/train_phase_b.log")
REPO_ID = "Alissonerdx/molmo2-h3-encoder"


def build_combined_lora(base_fro_path: Path, delta_lora_path: Path, out_path: Path) -> None:
    """Exact direct-sum combination of frozen base LoRA + trained delta LoRA:
    (B_base @ A_base) + (B_delta @ A_delta) = [B_base, B_delta] @ [A_base; A_delta].
    """
    sd_base = load_file(str(base_fro_path), device="cpu")
    sd_delta = load_file(str(delta_lora_path), device="cpu")

    # Normalize key prefixes to `diffusion_model.` (ComfyUI H3 standard)
    def norm_prefix(k: str) -> str:
        if k.startswith("transformer."):
            return "diffusion_model." + k[len("transformer.") :]
        return k

    base_map = {}
    for k, v in sd_base.items():
        nk = norm_prefix(k)
        if nk.endswith(".lora_A.weight"):
            base_map[nk[: -len(".lora_A.weight")]] = (
                v,
                sd_base[k[: -len(".lora_A.weight")] + ".lora_B.weight"],
                sd_base.get(k[: -len(".lora_A.weight")] + ".alpha", None),
            )

    delta_map = {}
    for k, v in sd_delta.items():
        nk = norm_prefix(k)
        if nk.endswith(".lora_A.weight"):
            delta_map[nk[: -len(".lora_A.weight")]] = (
                v,
                sd_delta[k[: -len(".lora_A.weight")] + ".lora_B.weight"],
                sd_delta.get(k[: -len(".lora_A.weight")] + ".alpha", None),
            )

    combined = {}
    all_prefixes = sorted(set(base_map.keys()) | set(delta_map.keys()))
    for p in all_prefixes:
        if p in base_map and p in delta_map:
            A1, B1, a1 = base_map[p]
            A2, B2, a2 = delta_map[p]
            r1 = A1.shape[0]
            r2 = A2.shape[0]
            s1 = (float(a1.item()) / r1) if a1 is not None else 1.0
            s2 = (float(a2.item()) / r2) if a2 is not None else 1.0
            B1_s = (B1.float() * s1).to(torch.bfloat16)
            B2_s = (B2.float() * s2).to(torch.bfloat16)
            A_cat = torch.cat([A1.to(torch.bfloat16), A2.to(torch.bfloat16)], dim=0).contiguous()
            B_cat = torch.cat([B1_s, B2_s], dim=1).contiguous()
            combined[f"{p}.lora_A.weight"] = A_cat
            combined[f"{p}.lora_B.weight"] = B_cat
            # Omit .alpha so ComfyUI uses alpha = rank (scale = 1.0), matching character_swap_v1.1_ref2va_fro
        elif p in base_map:
            A1, B1, a1 = base_map[p]
            r1 = A1.shape[0]
            s1 = (float(a1.item()) / r1) if a1 is not None else 1.0
            combined[f"{p}.lora_A.weight"] = A1.to(torch.bfloat16).contiguous()
            combined[f"{p}.lora_B.weight"] = (B1.float() * s1).to(torch.bfloat16).contiguous()
        else:
            A2, B2, a2 = delta_map[p]
            r2 = A2.shape[0]
            s2 = (float(a2.item()) / r2) if a2 is not None else 1.0
            combined[f"{p}.lora_A.weight"] = A2.to(torch.bfloat16).contiguous()
            combined[f"{p}.lora_B.weight"] = (B2.float() * s2).to(torch.bfloat16).contiguous()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(combined, str(out_path))


def main():
    api = HfApi()
    uploaded = set()
    combined_dir = JOB_DIR / "combined_with_v1_1"
    print(f"[HF Watcher] Watching {JOB_DIR} -> {REPO_ID}...", flush=True)

    while True:
        try:
            if JOB_DIR.exists():
                cfg = JOB_DIR / "config.yaml"
                if cfg.exists() and "config.yaml" not in uploaded:
                    api.upload_file(
                        path_or_fileobj=str(cfg),
                        path_in_repo="phase_b_lora/config.yaml",
                        repo_id=REPO_ID,
                        repo_type="model",
                        commit_message="Upload Phase B Mocha selective swap training config",
                    )
                    uploaded.add("config.yaml")
                    print("[HF Watcher] Uploaded config.yaml", flush=True)

                for st_path in sorted(JOB_DIR.glob("*.safetensors")):
                    if st_path.name in uploaded:
                        continue
                    # Wait briefly to ensure file write is complete
                    time.sleep(2)
                    print(f"[HF Watcher] Found new checkpoint: {st_path.name}", flush=True)
                    api.upload_file(
                        path_or_fileobj=str(st_path),
                        path_in_repo=f"phase_b_lora/delta/{st_path.name}",
                        repo_id=REPO_ID,
                        repo_type="model",
                        commit_message=f"Upload Phase B delta LoRA {st_path.name}",
                    )
                    if BASE_FRO_LORA.exists():
                        comb_name = st_path.name.replace(
                            "minimax_h3_molmo2_mocha_selective_swap_v1",
                            "minimax_h3_character_swap_v1.1_molmo2_selective",
                        )
                        comb_path = combined_dir / comb_name
                        build_combined_lora(BASE_FRO_LORA, st_path, comb_path)
                        api.upload_file(
                            path_or_fileobj=str(comb_path),
                            path_in_repo=f"phase_b_lora/combined/{comb_name}",
                            repo_id=REPO_ID,
                            repo_type="model",
                            commit_message=f"Upload combined v1.1 + Phase B selective LoRA {comb_name}",
                        )
                    if TRAIN_LOG.exists():
                        api.upload_file(
                            path_or_fileobj=str(TRAIN_LOG),
                            path_in_repo="phase_b_lora/train_phase_b.log",
                            repo_id=REPO_ID,
                            repo_type="model",
                            commit_message=f"Update training log at {st_path.name}",
                        )
                    uploaded.add(st_path.name)
                    print(f"[HF Watcher] Successfully uploaded {st_path.name} (delta + combined) to {REPO_ID}!", flush=True)
        except Exception as e:
            print(f"[HF Watcher] Warning: {e}", flush=True)

        time.sleep(30)


if __name__ == "__main__":
    main()
