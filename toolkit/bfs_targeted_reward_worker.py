"""Target-specific GPT reward worker for DiffusionNFT; no masks or target edits.

Receives the original guide, source-photo binding and replacement in judge_context.
Credentials remain in OPENAI_API_KEY or BFS_JUDGE_KEY_FILE outside the repository.
"""
import json,os,sys,tempfile,subprocess
from pathlib import Path
import numpy as np


def training_reward(grade):
    """Shape sparse visual judgments so early no-swap rollouts can teach target selection.

    A fully correct swap keeps the evaluator's strict reward. A no-swap result gets
    a small target-grounding score, while a wrong-person swap is penalized. This
    keeps selection signal separate from replacement progress without rewarding a
    successful-looking edit on the wrong person.
    """
    confidence = float(grade.get('confidence', 0.0))
    if confidence < 0.7 or grade.get('verdict') == 'uncertain':
        return 0.0
    selection = float(grade.get('target_selection', 0.0))
    replacement = float(grade.get('replacement_identity', 0.0))
    preservation = min(float(grade.get(k, 0.0)) for k in
                       ('other_people_preserved', 'scene_motion_preserved', 'temporal_consistency'))
    verdict = grade.get('verdict')
    if verdict == 'correct':
        return float(grade.get('reward_0_to_10') or 0.0)
    if verdict == 'wrong_target':
        return round(-2.0 * (1.0 - selection), 4)
    if verdict == 'no_swap':
        return round(2.5 * selection * preservation, 4)
    if verdict == 'mixed':
        return round(5.0 * selection * (0.25 + 0.75 * replacement) * preservation, 4)
    return 0.0


def write_video(path,frames,fps):
 frames=np.ascontiguousarray(frames,dtype=np.uint8)
 if frames.ndim!=4 or frames.shape[-1]!=3:raise ValueError('Expected F,H,W,3 RGB frames')
 h,w=frames.shape[1:3]
 subprocess.run(['ffmpeg','-v','error','-y','-f','rawvideo','-pix_fmt','rgb24','-s',f'{w}x{h}','-r',str(fps),'-i','pipe:0','-an','-c:v','libx264','-pix_fmt','yuv420p',str(path)],input=frames.tobytes(),check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)


def main():
 calls=0;limit=int(os.environ.get('BFS_JUDGE_MAX_CALLS','200'))
 report_root=Path(os.environ['BFS_JUDGE_REPORT_DIR']);report_root.mkdir(parents=True,exist_ok=True)
 print(json.dumps({'ready':True}),flush=True)
 for line in sys.stdin:
  try:
   msg=json.loads(line);ctx=msg['judge_context']
   with np.load(msg['npz'],allow_pickle=False) as data:gens=data['gen']
   grades=[];scores=[]
   for g,frames in enumerate(gens):
    if calls>=limit:raise RuntimeError('Judge call limit reached')
    calls+=1;out=report_root/f'{calls:05d}';out.mkdir(exist_ok=True)
    video=out/'generated.mp4';write_video(video,frames,ctx.get('fps',24))
    spec={**ctx,'generated':str(video)}
    duration=(len(frames)-1)/ctx.get('fps',24)
    spec['times']=[round(duration*f,4) for f in (.2,.5,.8)]
    manifest=out/'manifest.json';manifest.write_text(json.dumps(spec))
    cmd=[sys.executable,str(Path(__file__).with_name('bfs_targeted_judge.py')),'--manifest',str(manifest),'--out',str(out),'--model',os.environ['OPENAI_JUDGE_MODEL']]
    if os.environ.get('BFS_JUDGE_KEY_FILE'):cmd+=['--key-file',os.environ['BFS_JUDGE_KEY_FILE']]
    proc=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=240)
    if proc.returncode:raise RuntimeError('Target judge failed; see evaluator status (no reward assigned)')
    grade=json.loads((out/'grade.json').read_text());grades.append(grade)
    score = training_reward(grade)
    grade['training_reward'] = score
    (out/'grade.json').write_text(json.dumps(grade, indent=2))
    scores.append(score)
   print(json.dumps({'gpt_id':scores,'target_judge_confidence':[x['confidence'] for x in grades]}),flush=True)
  except Exception as exc:
   print(json.dumps({'error':type(exc).__name__+': '+str(exc)[:160]}),flush=True)
if __name__=='__main__':main()
