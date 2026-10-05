"""Save aligned rollout contact sheet and auditable local/VLM metrics."""
import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_vlm import frames


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--clips-dir', type=Path, required=True,
                   help='Contains aligned guide.mp4, best.mp4 and worst.mp4')
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--reward-log', type=Path, required=True)
    p.add_argument('--call', type=int, required=True)
    p.add_argument('--vlm', type=Path)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--frames', type=int, default=3)
    a = p.parse_args()
    if not 1 <= a.frames <= 12:
        p.error('--frames must be 1..12')
    records = [json.loads(x) for x in a.reward_log.read_text().splitlines() if x.strip()]
    record = next(x for x in records if x['call'] == a.call)
    advantages = np.asarray(record['adv'])
    picks = {'best': int(advantages.argmax()), 'worst': int(advantages.argmin())}
    metrics = {name: {'rollout_index': idx, 'advantage': float(advantages[idx]),
                     **{key: values[idx] for key, values in record['rewards'].items()}}
               for name, idx in picks.items()}
    images = {name: frames(a.clips_dir / (name + '.mp4'), a.frames)
              for name in ['guide', 'best', 'worst']}
    if len({len(x) for x in images.values()}) != 1:
        raise ValueError('Frame counts differ')
    reference = cv2.imread(str(a.reference))
    if reference is None:
        raise ValueError('Cannot read reference')
    vlm = json.loads(a.vlm.read_text()) if a.vlm else None
    judgment = None
    vlm_winner = 'not evaluated'
    if vlm:
        choice = vlm['choices'][0]
        if choice.get('finish_reason') != 'stop' or not choice['message'].get('content'):
            raise ValueError('VLM response is incomplete')
        judgment = json.loads(choice['message']['content'])
        overall = judgment.get('overall', {})
        label = overall.get('winner') if isinstance(overall, dict) else overall
        label = label or judgment.get('overall winner') or judgment.get('winner')
        inputs = vlm.get('evaluation_inputs', {})
        mapping = {'A': inputs.get('baseline'), 'B': inputs.get('generated')}
        if inputs.get('reverse_order'):
            mapping = {'A': inputs.get('generated'), 'B': inputs.get('baseline')}
        vlm_winner = str(label)
        if label in mapping and mapping[label]:
            vlm_winner += ' = ' + Path(mapping[label]).stem.upper()
    count = len(images['guide'])
    fig, axes = plt.subplots(count, 4, figsize=(18, 3.7 * count), squeeze=False)
    for row in range(count):
        for col, name in enumerate(['reference', 'guide', 'best', 'worst']):
            ax = axes[row, col]
            im = reference if name == 'reference' else images[name][row]
            ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            ax.axis('off')
            title = name.upper() + ('' if name == 'reference' else f' | sample {row + 1}')
            if row == 0 and name in metrics:
                m = metrics[name]
                title += '\n' + ' '.join(f'{k}={m[k]:.3f}' for k in ['id', 'bg', 'light', 'pose'] if k in m)
                title += f"\nadv={m['advantage']:.3f} | rollout {m['rollout_index']} (zero-based)"
                title += '\nLOCAL: ' + ('WINNER' if name == 'best' else 'LOWEST REWARD')
                if vlm_winner.endswith('= ' + name.upper()):
                    title += ' | VLM: WINNER'
            ax.set_title(title, fontsize=10)
    footer = 'Best/worst selected by local advantage. Higher reward is better; light may be negative.'
    if judgment:
        footer += '\nVLM overall winner: ' + vlm_winner + '. Full criteria and A/B mapping: metrics.json.'
    fig.suptitle(f"Item {record['item']} | call {a.call} | KL={record['kl']:.4f} | "
                 f"rollout steps={record.get('sampling_steps', '?')}", fontsize=15)
    fig.text(.5, .015, footer, ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .94))
    a.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.output_dir / 'grid.png', dpi=150)
    plt.close(fig)
    payload = {'selection': 'local normalized group advantage', 'metrics': metrics,
               'training_record': record, 'vlm_judgment': judgment,
               'local_winner': 'best', 'vlm_winner': vlm_winner,
               'vlm_usage': vlm.get('usage') if vlm else None,
               'vlm_inputs': vlm.get('evaluation_inputs') if vlm else None,
               'reference': str(a.reference), 'clips_dir': str(a.clips_dir),
               'sampled_frames': count,
               'limitations': 'Training-item smoke test, not held-out checkpoint evaluation. '
                              'Input videos must already be temporally aligned.'}
    (a.output_dir / 'metrics.json').write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    (a.output_dir / 'README.md').write_text(
        '# Rollout comparison\n\n![Aligned comparison](grid.png)\n\n'
        'Reference | guide | best local rollout | worst local rollout. '
        'See [metrics.json](metrics.json) for raw rewards, advantages, KL and VLM judgment. '
        'Local ranking is not a human quality verdict. This example is a training-item smoke test.\n')
    print(a.output_dir / 'grid.png')


if __name__ == '__main__':
    main()
