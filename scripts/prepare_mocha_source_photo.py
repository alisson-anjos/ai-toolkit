#!/usr/bin/env python3
"""Prepare reviewable Mocha source-photo pairs without changing the old dataset."""
import argparse,hashlib,json,os,shutil
from pathlib import Path


def media(folder, stem, extensions):
    found=[p for p in folder.glob(stem+'.*') if p.suffix.lower() in extensions]
    return found[0] if len(found)==1 else None


def original_reference(name):
    rules=[('v2mocha_','mocha_orig_all'),('fullinv_','full_mocha_orig'),
           ('fullorig_','full_mocha_inv'),('mochafwd_','mocha_orig'),('mocharev_','mocha')]
    for prefix,subset in rules:
        if name.startswith(prefix):
            stem=name[len(prefix):]
            if prefix=='fullorig_':stem=stem.removesuffix('_x2')
            elif prefix=='fullinv_':stem=stem+'_x2'
            return subset,stem
    return None,None


def caption():
    return ('body_swap: subject_definitions: <SourcePerson 1> is a photograph of the original person '
            'to identify in <Video 1>. <Subject 1> is the replacement character shown in <Picture 1>. '
            'summary: Replace only the person matching <SourcePerson 1> in <Video 1> with <Subject 1>. '
            'detailed_description: Transfer the face, hair, body and clothing from <Picture 1> to that '
            'same performer throughout the clip. Preserve every other person, the background, camera, '
            'framing, timing, motion and interactions. The source photograph identifies whom to replace; '
            'it does not supply the replacement appearance. Maintain this assignment through crossings '
            'and occlusion. Do not replace another person because they are more prominent.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--previous',required=True)
    p.add_argument('--out',required=True);p.add_argument('--download-source-photos',action='store_true')
    a=p.parse_args();raw=Path(a.raw);previous=Path(a.previous)/'refs_1';out=Path(a.out)
    out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.jsonl').exists():raise ValueError('Use a new output folder to preserve the previous audit')
    base=raw/'v1.1/bodyswap_inv_v1'
    names=sorted(p.stem for p in (previous/'targets').glob('*.txt'))
    if a.download_source_photos:
        from huggingface_hub import snapshot_download
        patterns=[]
        for name in names:
            subset,stem=original_reference(name)
            if subset:patterns.append(f'v1.1/bodyswap_inv_v1/{subset}/refs/{stem}.*')
        snapshot_download('Alissonerdx/bodyswap-video-train',repo_type='dataset',local_dir=str(raw),allow_patterns=patterns)
    rows=[];skipped=[]
    for name in names:
        subset,stem=original_reference(name)
        if subset is None:skipped.append({'id':name,'reason':'unknown_source_pair'});continue
        source=media(base/subset/'refs',stem,{'.png','.jpg','.jpeg','.webp'})
        guide=media(previous/'control_0',name,{'.mp4','.mov','.webm'})
        replacement=media(previous/'control_1',name,{'.png','.jpg','.jpeg','.webp'})
        target=media(previous/'targets',name,{'.mp4','.mov','.webm'})
        if not all([source,guide,replacement,target]):
            skipped.append({'id':name,'reason':'missing_or_ambiguous_paired_media'});continue
        source_hash=hashlib.sha256(source.read_bytes()).hexdigest()
        if source_hash==hashlib.sha256(replacement.read_bytes()).hexdigest():
            skipped.append({'id':name,'reason':'source_equals_replacement'});continue
        family=stem.removesuffix('_x2')
        split='validation' if int(hashlib.sha256(family.encode()).hexdigest(),16)%10==0 else 'train'
        mapped={}
        for role,path in [('source_people',source),('control_0',guide),('control_1',replacement),('targets',target)]:
            dest=out/'candidates'/split/role/(name+path.suffix);dest.parent.mkdir(parents=True,exist_ok=True)
            try:os.link(path,dest)
            except OSError:shutil.copy2(path,dest)
            mapped[role]=str(dest.relative_to(out))
        (out/'candidates'/split/'targets'/(name+'.txt')).write_text(caption()+'\n')
        rows.append({'id':name,'family':family,'split':split,'files':mapped,'source_photo_origin':f'{subset}/refs/{stem}{source.suffix}',
                     'source_photo_sha256':source_hash,'review_status':'pending','visible_people':None,
                     'source_matches_guide_person':None,'target_changes_only_that_person':None})
    (out/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    summary={'candidate_pairs':len(rows),'train_candidates':sum(r['split']=='train' for r in rows),
             'validation_candidates':sum(r['split']=='validation' for r in rows),'skipped':skipped,
             'ready_for_training':False,'reason':'Source/guide/target correspondence and multi-person coverage require review. No new training started.',
             'vae_reference_roles':['replacement only'],'encoder_context_roles':['source person','replacement','guide'],
             'old_replacement_distractor_subset_included':False,'masks':False,'genjutsu':False}
    (out/'audit.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
