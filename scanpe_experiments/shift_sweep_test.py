"""Path 1 experiment: sweep the flow-matching sigma shift and compare
t_lat=1 vs t_lat=2 single-image sampling on MiniMax-H3 ref2va, with NO
LoRA/training involved -- pure inference-time recalibration.

VIDEO_SIGMA_SHIFT (12.0) is calibrated for ~38k-token video sequences.
A still image at t_lat=1 is ~1k tokens -- under standard shift ~ sqrt(seq_len)
scaling that implies something around shift ~2 instead. This script renders
the same reference + prompt across a shift grid so the difference is visible
directly, before touching any weights.

Run from the ai-toolkit root:
    .venv/bin/python scanpe_experiments/shift_sweep_test.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from PIL import Image

from toolkit.config_modules import ModelConfig
from toolkit.util.get_model import get_model_class

OUT_DIR = "scanpe_experiments/shift_sweep_out3"

CASES = [
    (
        "photo",
        "data/storyboard_charactersheet_v2/sources/human_orion.png",
        """subject_definitions:
<Subject 1> is the person in <Picture 1>, with an asymmetric hairstyle with a top knot, a black jacket with white reflective piping, and a confident calm expression.

summary:
[reference generation] The target video is a single still portrait of <Subject 1> standing in soft natural light against a neutral background.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, hairstyle, and jacket from <Picture 1> are kept unchanged.

detailed_description:
The image is a sharp, photorealistic portrait with soft directional natural window light and a neutral gray studio background, shallow depth of field.
[Shot 1] <Subject 1> stands facing the camera in a relaxed three-quarter pose, calm confident expression, looking directly at the camera, soft light modeling the face and jacket fabric, crisp fine detail on skin texture and fabric weave.

overall_soundscape:
N/A

non_diegetic_music:
N/A
""",
    ),
    (
        "stylized",
        "data/storyboard_charactersheet_v2/sources/stylized_character_ran.png",
        """subject_definitions:
<Subject 1> is the character in <Picture 1>, with a yellow cap with a logo, a blue oversized jacket, a magenta print on a white hoodie, and pink cargo pants.

summary:
[reference generation] The target video is a single still portrait of <Subject 1> standing in soft natural light against a neutral background.

retention_analysis:
<Subject 1> (appears in [Shot 1]): fully_preserved - identity, outfit, and colors from <Picture 1> are kept unchanged.

detailed_description:
The image is a clean anime cel-shaded illustration with sharp linework, soft directional studio light, and a neutral light gray background.
[Shot 1] <Subject 1> stands facing the camera, head up, looking directly at the camera with a calm confident expression, relaxed three-quarter pose, soft light modeling the face and fabric folds, crisp fine linework and clean flat shading with visible texture detail on the cap logo and hoodie print.

overall_soundscape:
N/A

non_diegetic_music:
N/A
""",
    ),
]

SHIFTS = [12.0, 3.0, 1.5, 1.0]
FRAME_COUNTS = [1]  # only the clean t_lat=1 image branch this round
STEPS = 40
SIZE = 768


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    model_config = ModelConfig(
        name_or_path="Comfy-Org/MiniMax-H3",
        arch="minimax_h3",
        dtype="bf16",
        quantize=True,
        quantize_te=True,
        layer_offloading=True,
        layer_offloading_transformer_percent=0.85,
        layer_offloading_text_encoder_percent=0.85,
        model_kwargs={"partition": "ref2va_pruned"},
    )
    ModelClass = get_model_class(model_config)
    print(f"Model class: {ModelClass.__name__}")
    model = ModelClass(device="cuda:0", model_config=model_config, dtype="bf16")
    model.load_model()
    pipeline = model.get_generation_pipeline()

    for case_name, ref_path, prompt in CASES:
        ref_img = Image.open(ref_path).convert("RGB")
        conditional_embeds = model.get_prompt_embeds(prompt, control_images=[ref_img])

        for num_frames in FRAME_COUNTS:
            for shift in SHIFTS:
                print(f"\n=== case={case_name} num_frames={num_frames} shift={shift} ===")
                generator = torch.Generator(device="cpu").manual_seed(42)
                result = pipeline(
                    conditional_embeds=conditional_embeds,
                    unconditional_embeds=None,
                    height=SIZE,
                    width=SIZE,
                    num_frames=num_frames,
                    num_inference_steps=STEPS,
                    generator=generator,
                    ref_images=[ref_img],
                    with_audio=False,
                    sigma_shift=shift,
                )
                if isinstance(result, list):
                    img = result[0]
                else:
                    frame0 = result["video"][0].numpy()
                    img = Image.fromarray(frame0)
                out_path = os.path.join(OUT_DIR, f"{case_name}_shift{shift}.png")
                img.save(out_path)
                print(f"saved {out_path}")


if __name__ == "__main__":
    main()
