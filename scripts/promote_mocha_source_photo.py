#!/usr/bin/env python3
"""Materialize only explicitly reviewed source-photo pairs; preserve candidates."""
import argparse,hashlib,json,os,shutil
from pathlib import Path


def promote(candidate_root, review_file, output):
    root=Path(candidate_root).resolve();output=Path(output)
    if output.exists():raise ValueError('Reviewed output must be a new folder')
    rows={r['id']:r for r in map(json.loads,(root/'manifest.jsonl').read_text().splitlines())}
    review=json.loads(Path(review_file).read_text());approved=[];family_splits={}
    for decision in review['decisions']:
        if not decision.get('approved_for_pilot'):continue
        row=rows[decision['id']]
        if not all(decision.get(k) is True for k in ['source_matches_edited_person','other_people_preserved_at_reviewed_frames']):
            raise ValueError(f'Incomplete identity review: {row["id"]}')
        if decision.get('visible_people_at_reviewed_frames',0)<2 or len(decision.get('reviewed_fractions',[]))<3:
            raise ValueError(f'Multi-person temporal review required: {row["id"]}')
        split=decision['split']
        if split not in ('train','validation'):raise ValueError('Unknown split')
        if decision.get('target_side') not in ('left','right','center'):raise ValueError('Unknown target side')
        previous=family_splits.setdefault(row['family'],split)
        if previous!=split:raise ValueError(f'Family crosses splits: {row["family"]}')
        source=decision.get('source_photo_override',row['files']['source_people'])
        files={**row['files'],'source_people':source};paths={}
        for role,path in files.items():
            file=(root/path).resolve()
            if not file.is_relative_to(root) or not file.is_file():raise ValueError(f'Invalid input: {path}')
            paths[role]=file
        approved.append((row,decision,paths))
    if not approved:raise ValueError('No approved pairs')
    manifest=[]
    for row,decision,paths in approved:
        subset='validation' if decision['split']=='validation' else 'train_'+decision['target_side']
        if subset not in ('validation','train_left','train_right','train_center'):raise ValueError('Unknown target side')
        dests={}
        for role,file in paths.items():
            dest=output/subset/role/(row['id']+file.suffix);dest.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(file,dest)
            except OSError:shutil.copy2(file,dest)
            dests[role]=str(dest.relative_to(output))
        caption=root/row['files']['targets'];caption=caption.with_suffix('.txt')
        shutil.copy2(caption,(output/dests['targets']).with_suffix('.txt'))
        manifest.append({**row,'split':decision['split'],'files':dests,'review_status':'pilot_approved',
                         'review':decision,'source_photo_sha256':hashlib.sha256(paths['source_people'].read_bytes()).hexdigest()})
    (output/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in manifest))
    summary={'approved_pairs':len(manifest),'subsets':{s:sum(r['files']['targets'].startswith(s+'/') for r in manifest) for s in ['train_left','train_right','train_center','validation']},
             'review_scope':review['scope'],'new_training_started':False,'masks':False,'genjutsu':False,
             'limitation':'Small reviewed pilot; sparse temporal review, no same-guide alternate-source counterfactuals or general three-person accuracy established.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2));return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--candidates',required=True);p.add_argument('--review',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();print(json.dumps(promote(a.candidates,a.review,a.out),indent=2))

if __name__=='__main__':main()
