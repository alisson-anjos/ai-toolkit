"""Make a labeled grid from an H3 validation guide and saved sample images."""
import argparse
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from toolkit.aligned_guides import prepare_guide_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--guide', type=Path, required=True)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--width', type=int, default=1024)
    parser.add_argument('--height', type=int, default=768)
    parser.add_argument('--factor', type=int, default=4)
    parser.add_argument('--sample-index', type=int, default=0)
    args = parser.parse_args()
    source = Image.open(args.guide).convert('RGB')
    canvas = (args.width, args.height)
    guide = prepare_guide_image(source, args.height, args.width, args.factor)
    # This reference is valid for same-source upscale pairs; it is not a
    # ground-truth target for unrelated control/target files.
    reference = ImageOps.fit(source, canvas, Image.Resampling.BICUBIC)
    candidates = {}
    for path in sorted(args.samples.glob('*')):
        match = re.search(r'__(\d+)_(\d+)\.(jpg|png|webp)$', path.name)
        if match and int(match[2]) == args.sample_index:
            candidates[int(match[1])] = path
    if not candidates:
        parser.error('No matching image samples found; video samples need frame extraction first.')
    steps = sorted(candidates)
    chosen = steps[:1] + steps[-4:] if len(steps) > 5 else steps
    panels = [
        (f'Guide {guide.width} x {guide.height} (enlarged)', guide.resize(canvas, Image.Resampling.NEAREST)),
        ('Bicubic interpolation', guide.resize(canvas, Image.Resampling.BICUBIC)),
        ('Source fitted to target canvas', reference),
    ]
    for step in chosen:
        image = Image.open(candidates[step]).convert('RGB')
        if image.size != canvas:
            parser.error(f'Sample {candidates[step].name} has size {image.size}, expected {canvas}')
        panels.append(('Baseline' if step == 0 else f'LoRA step {step}', image))
    fig, axes = plt.subplots(2, 4, figsize=(16, 9))
    for axis in axes.flat:
        axis.axis('off')
    for axis, (title, image) in zip(axes.flat, panels):
        axis.imshow(image, interpolation='nearest')
        axis.set_title(title)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    plt.close(fig)
    guide.save(args.output.with_name(args.output.stem + '_guide.png'))
    print(args.output)


if __name__ == '__main__':
    main()
