"""Evaluate an aligned guide/generated clip with a reference through a local VLM.
No training reward replacement. Requires an OpenAI-compatible multimodal endpoint.
"""
import argparse
import base64
import json
import os
import subprocess
from pathlib import Path

import cv2
import numpy as np
import requests


def image_part(image):
    ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise ValueError('Cannot encode image')
    data = base64.b64encode(encoded).decode('ascii')
    return {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,' + data}}


def frames(path, count):
    cap = cv2.VideoCapture(str(path))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            raise ValueError(f'Cannot read video: {path}')
        out = []
        for idx in np.linspace(0, total - 1, min(count, total)).round().astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                raise ValueError(f'Cannot decode frame {idx}: {path}')
            h, w = frame.shape[:2]
            if max(h, w) > 768:
                frame = cv2.resize(frame, (round(w * 768 / max(h, w)), round(h * 768 / max(h, w))))
            out.append(frame)
        return out
    finally:
        cap.release()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--openai', action='store_true', help='Use OpenAI API; requires OPENAI_API_KEY')
    p.add_argument('--base-url', help='Local API base ending in /v1')
    p.add_argument('--model', default='gpt-5.4-mini')
    p.add_argument('--proxy', help='Optional curl proxy, e.g. socks5h://127.0.0.1:1055')
    p.add_argument('--connect-to', help='Optional curl HOST:PORT:TAILNET_IP:PORT, retaining TLS hostname')
    p.add_argument('--guide', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--generated', type=Path, required=True)
    p.add_argument('--baseline', type=Path, help='s200 output with same seed, settings and mode')
    p.add_argument('--reverse-order', action='store_true', help='Swap A/B labels to check position bias')
    p.add_argument('--frames', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.openai:
        if args.base_url or args.proxy or args.connect_to:
            p.error('--openai uses the official endpoint without local proxy options')
        args.base_url = 'https://api.openai.com/v1'
        if not os.environ.get('OPENAI_API_KEY'):
            p.error('Configure OPENAI_API_KEY in the environment; do not paste it into chat')
    elif not args.base_url:
        p.error('--base-url is required for a local evaluator')
    if not 1 <= args.frames <= 12:
        p.error('--frames must be between 1 and 12')
    ref = cv2.imread(str(args.reference))
    if ref is None:
        raise ValueError(f'Cannot read reference: {args.reference}')
    content = [{'type': 'text', 'text': (
        'Evaluate a character swap. The first image is the desired reference identity. '
        'Then come aligned guide/generated frame pairs, labeled explicitly. '
        'Inspect face/identity replacement, hair length/style replacement, reference-compatible '
        'body appearance/proportions where visible, and preservation '
        'of background, scene lighting, pose, facial expression and visible motion. '
        'Compare mouth open/closed state, smile and eye open/closed state against the aligned GUIDE, '
        'not the reference portrait expression. Score each from 0 to 10 '
        '(higher is better), explaining concrete visible evidence. Do not give high scores '
        'merely because the output is attractive. Admit ambiguity or unobservable motion. '
        'Do not claim audio lip synchronization: no audio is supplied. Mark audio_lipsync unobservable. '
        'Return JSON with keys identity, hair, body, background, lighting, pose, motion, '
        'expression, mouth_state, audio_lipsync, evidence, '
        'limitations. Single-clip assessment only; do not claim a win over another checkpoint.'
    )}, {'type': 'text', 'text': 'REFERENCE'}, image_part(ref)]
    guide = frames(args.guide, args.frames)
    generated = frames(args.generated, args.frames)
    if len(guide) != len(generated):
        raise ValueError('Frame samples differ; use clips of equal sampled length')
    if args.baseline:
        baseline = frames(args.baseline, args.frames)
        if len(baseline) != len(guide):
            raise ValueError('Baseline frame sample count differs')
        content[0]['text'] = (
            'Compare two character-swap videos using the desired identity reference and aligned guide frames. '
            'For each frame you receive GUIDE, candidate A, and candidate B. '
            'Judge face/identity replacement, hair length/style matching the reference, '
            'reference-compatible body appearance/proportions where visible, background preservation, '
            'scene lighting, pose, facial expression, mouth state and visible motion. '
            'For expression and mouth state, compare mouth open/closed, smile and eye open/closed '
            'against the aligned GUIDE at each sampled instant, not the reference portrait expression. '
            'Report concrete mismatches by frame; matching facial identity alone is insufficient. '
            'Do not claim audio lip synchronization: no audio is supplied. Include audio_lipsync '
            'as unobservable. Return JSON with a winner A/B/tie/uncertain for each '
            'criterion, overall winner, short visible evidence and limitations. Prioritize correct character '
            'replacement without sacrificing the scene. Judge the pictures, not candidate order. '
            'Preserve the guide action and pose while replacing character appearance. '
            'Do not demand the same body silhouette as the guide. Do not infer body proportions '
            'hidden or absent from the reference, or motion that these sparse frames do not show. '
            'For body use uncertain when reference evidence is insufficient.'
        )
        cand_a, cand_b = (generated, baseline) if args.reverse_order else (baseline, generated)
        for idx, (g, a, b) in enumerate(zip(guide, cand_a, cand_b)):
            content += [{'type': 'text', 'text': f'FRAME {idx}: GUIDE'}, image_part(g),
                        {'type': 'text', 'text': f'FRAME {idx}: A'}, image_part(a),
                        {'type': 'text', 'text': f'FRAME {idx}: B'}, image_part(b)]
    else:
        for idx, (a, b) in enumerate(zip(guide, generated)):
            content += [{'type': 'text', 'text': f'PAIR {idx}: GUIDE'}, image_part(a),
                        {'type': 'text', 'text': f'PAIR {idx}: GENERATED'}, image_part(b)]
    headers = {}
    key = os.environ.get('OPENAI_API_KEY' if args.openai else 'VLM_API_KEY')
    if key:
        headers['Authorization'] = 'Bearer ' + key
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': content}],
            'temperature': 0, 'max_tokens': 1600}
    if args.openai:
        body.pop('temperature')
        body.pop('max_tokens')
        body['max_completion_tokens'] = 4000
        body['reasoning_effort'] = 'low'
        body['response_format'] = {'type': 'json_object'}
    url = args.base_url.rstrip('/') + '/chat/completions'
    if args.proxy or args.connect_to:
        cmd = ['curl', '--fail-with-body', '--silent', '--show-error',
               '--connect-timeout', '10', '--max-time', '180',
               '-H', 'Content-Type: application/json', '--data-binary', '@-']
        if args.proxy:
            cmd += ['--proxy', args.proxy]
        if args.connect_to:
            cmd += ['--connect-to', args.connect_to]
        # Avoid placing credentials in process arguments.
        if key:
            raise ValueError('Authenticated curl transport requires a credential-safe header implementation')
        response = subprocess.run(cmd + [url], input=json.dumps(body), text=True,
                                  capture_output=True, timeout=190)
        if response.returncode:
            raise RuntimeError(f'VLM HTTP request failed: {response.stderr} {response.stdout[:1000]}')
        result = json.loads(response.stdout)
    else:
        response = requests.post(url, json=body, headers=headers, timeout=180)
        response.raise_for_status()
        result = response.json()
    result['evaluation_inputs'] = {'guide': str(args.guide), 'reference': str(args.reference),
                                   'generated': str(args.generated), 'frames': args.frames, 'model': args.model,
                                   'baseline': str(args.baseline) if args.baseline else None,
                                   'reverse_order': args.reverse_order}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    choice = result['choices'][0]
    answer = choice['message'].get('content')
    if choice.get('finish_reason') != 'stop' or not answer or choice['message'].get('refusal'):
        raise RuntimeError(f'Evaluation incomplete or refused; inspect saved response: {args.output}')
    if args.openai:
        json.loads(answer)
    print(answer)
    print('Usage:', result.get('usage', {}))
    print('Saved:', args.output)


if __name__ == '__main__':
    main()
