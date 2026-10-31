import importlib.util,json
from pathlib import Path
import pytest

spec=importlib.util.spec_from_file_location('promote_mocha',Path(__file__).resolve().parents[1]/'scripts/promote_mocha_source_photo.py')
promoter=importlib.util.module_from_spec(spec);spec.loader.exec_module(promoter)


def fixture_pair(root,id='clip',family='family'):
    files={}
    for role,ext in [('targets','.mp4'),('control_0','.mp4'),('control_1','.png'),('source_people','.png')]:
        f=root/'candidates'/role/(id+ext);f.parent.mkdir(parents=True,exist_ok=True);f.write_bytes(role.encode());files[role]=str(f.relative_to(root))
    (root/files['targets']).with_suffix('.txt').write_text('Replace SourcePerson 1 with Picture 1')
    return {'id':id,'family':family,'split':'train','files':files}


def decision(id='clip',split='train'):
    return {'id':id,'approved_for_pilot':True,'source_matches_edited_person':True,
            'other_people_preserved_at_reviewed_frames':True,'visible_people_at_reviewed_frames':2,
            'reviewed_fractions':[.1,.5,.9],'split':split,'target_side':'left'}


def write_inputs(root,rows,decisions):
    (root/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    review=root/'review.json';review.write_text(json.dumps({'scope':'test','decisions':decisions}));return review


def test_corrected_source_photo_does_not_change_guide_replacement_or_target(tmp_path):
    row=fixture_pair(tmp_path);override=tmp_path/'corrected.png';override.write_bytes(b'correct source person')
    d=decision();d['source_photo_override']='corrected.png';review=write_inputs(tmp_path,[row],[d]);out=tmp_path/'reviewed'
    result=promoter.promote(tmp_path,review,out);assert result['approved_pairs']==1
    assert (out/'train_left/source_people/clip.png').read_bytes()==override.read_bytes()
    for role in ['control_0','control_1','targets']:
        original=tmp_path/row['files'][role]
        assert (out/'train_left'/role/original.name).read_bytes()==original.read_bytes()
    assert (tmp_path/row['files']['source_people']).read_bytes()==b'source_people'


def test_review_rejects_family_leakage_before_writing_any_data(tmp_path):
    rows=[fixture_pair(tmp_path,'a'),fixture_pair(tmp_path,'b')]
    review=write_inputs(tmp_path,rows,[decision('a'),decision('b','validation')]);out=tmp_path/'reviewed'
    with pytest.raises(ValueError,match='crosses splits'):promoter.promote(tmp_path,review,out)
    assert not out.exists()


def test_review_rejects_uncertain_identity_and_skips_unapproved_pairs(tmp_path):
    row=fixture_pair(tmp_path);d=decision();d['source_matches_edited_person']=False
    review=write_inputs(tmp_path,[row],[d])
    with pytest.raises(ValueError,match='Incomplete identity'):promoter.promote(tmp_path,review,tmp_path/'reviewed')
    d['approved_for_pilot']=False;review=write_inputs(tmp_path,[row],[d])
    with pytest.raises(ValueError,match='No approved'):promoter.promote(tmp_path,review,tmp_path/'reviewed')
