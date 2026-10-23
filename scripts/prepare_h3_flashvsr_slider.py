"""Extract a bounded single-archive pilot and create frame-aligned degradation pairs.

No archive paths are trusted. Originals remain in the input tar; temporary media
is deleted after conversion. Captions are neutral restoration instructions, not
invented scene descriptions. CPU ffmpeg only, no interference with GPU training.
"""
import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


def run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--limit', type=int, default=64)
    args = p.parse_args()
    root = args.output
    for split in ('train', 'validation'):
        for folder in ('targets', 'guides'):
            (root/split/folder).mkdir(parents=True, exist_ok=True)
    rows = []
    with tarfile.open(args.archive) as tar:
        members = [m for m in tar if m.isfile() and m.name.lower().endswith('.mp4')]
        # Spread the pilot across this archive rather than just its first clips.
        chosen = members[::max(1, len(members)//args.limit)][:args.limit]
        for index, member in enumerate(chosen):
            if member.size > 200_000_000:
                continue
            key = hashlib.sha256(member.name.encode()).hexdigest()[:16]
            split = 'validation' if index % 8 == 0 else 'train'
            target = root/split/'targets'/f'{key}.mp4'
            guide = root/split/'guides'/f'{key}.mp4'
            try:
                with tempfile.TemporaryDirectory(prefix='flashvsr-pilot-') as tmp:
                    source = Path(tmp)/'source.mp4'
                    with source.open('wb') as dst:
                        src = tar.extractfile(member)
                        while chunk := src.read(1024*1024):
                            dst.write(chunk)
                    probe = json.loads(run(['ffprobe','-v','error','-select_streams','v:0',
                        '-show_streams','-of','json',str(source)]).stdout)['streams'][0]
                    if int(probe['width']) < 512 or int(probe['height']) < 288:
                        continue
                    # Keep at most 4 seconds, 16fps, even dimensions; no synthetic
                    # interpolation. Target capped at 768 longest side.
                    vf = "fps=16,scale=w='min(768,iw)':h='min(768,ih)':force_original_aspect_ratio=decrease:force_divisible_by=2"
                    common = ['ffmpeg','-v','error','-y','-threads','2']
                    run(common+['-i',str(source),'-t','4','-an','-vf',vf,
                        '-c:v','libx264','-threads','2','-crf','14','-pix_fmt','yuv420p',str(target)])
                    tp=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(target)]).stdout)['streams'][0]
                    w,h=int(tp['width']),int(tp['height'])
                    # Blur + spatial detail loss + mild compression. Return to
                    # identical canvas so guide/target correspond frame for frame.
                    filt=f'scale={max(2,w//4*2)}:{max(2,h//4*2)}:flags=area,gblur=sigma=0.6,scale={w}:{h}:flags=bicubic'
                    run(common+['-i',str(target),'-an','-vf',filt,'-c:v','libx264',
                        '-threads','2','-crf','26','-pix_fmt','yuv420p',str(guide)])
                    gp=json.loads(run(['ffprobe','-v','error','-select_streams','v:0','-show_streams','-of','json',str(guide)]).stdout)['streams'][0]
                    if (tp['nb_frames'],w,h,tp['r_frame_rate']) != (gp['nb_frames'],int(gp['width']),int(gp['height']),gp['r_frame_rate']):
                        raise ValueError('Guide/target temporal or spatial mismatch')
                    target.with_suffix('.txt').write_text('Refine the input video with clear natural details and clean textures, preserving its subjects, appearance, scene, composition, motion and timing.\n')
                    rows.append(dict(key=key,split=split,source_member=member.name,
                        original_width=probe['width'],original_height=probe['height'],
                        width=w,height=h,frames=int(tp['nb_frames']),fps=16,
                        target=str(target),guide=str(guide),
                        caption_type='neutral_restoration_instruction'))
                    print(f'{index+1}/{len(chosen)} prepared {key} {w}x{h}',flush=True)
            except (subprocess.CalledProcessError, ValueError, KeyError) as exc:
                print(f'Skipped {member.name}: {str(exc)[:180]}',flush=True)
                target.unlink(missing_ok=True);guide.unlink(missing_ok=True)
            (root/'manifest.json').write_text(json.dumps(rows,indent=2))
    print(f'Prepared {len(rows)} aligned pairs; validation held out.',flush=True)

if __name__ == '__main__':
    main()
