from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace as NS
from typing import List, Optional
import ast
import numpy as np
import pytest
import torch
from PIL import Image

from toolkit.source_person import SourcePersonRef, source_person_photo, source_person_cache_info
from minimax_h3_test_utils import ROOT, model_method, source_function


def test_source_photo_pairing_rejects_missing_or_ambiguous_images(tmp_path):
    assert source_person_photo(None, 'clip.mp4') is None
    with pytest.raises(ValueError, match='exactly one'):
        source_person_photo(tmp_path, 'clip.mp4')
    Image.new('RGB',(8,8),'red').save(tmp_path/'clip.png')
    assert source_person_photo(tmp_path,'clip.mp4') == str(tmp_path/'clip.png')
    Image.new('RGB',(8,8),'blue').save(tmp_path/'clip.jpg')
    with pytest.raises(ValueError, match='found 2'):
        source_person_photo(tmp_path,'clip.mp4')


def test_actual_embedding_cache_changes_when_source_photo_changes(tmp_path):
    photo=tmp_path/'clip.png';Image.new('RGB',(8,8),'red').save(photo)
    fn=source_function(ROOT/'toolkit/dataloader_mixins.py','get_text_embedding_info_dict',
                       {'OrderedDict':OrderedDict},'TextEmbeddingFileItemDTOMixin')
    item=NS(caption='replace SourcePerson 1',text_embedding_space_version='molmo',text_embedding_version=1,
            dataset_config=NS(source_person_path=str(tmp_path)),path='clip.mp4',
            encode_control_in_text_embeddings=True,control_path=['replacement.png'])
    before=fn(item)
    Image.new('RGB',(8,8),'green').save(photo)
    after=fn(item)
    assert before['source_person_sha256']!=after['source_person_sha256']
    assert after['control_path']==['replacement.png']
    assert 'source_person_sha256' not in fn(item,text_only=True)


def test_source_context_requires_molmo_visual_encoder(tmp_path):
    photo=tmp_path/'x.png';Image.new('RGB',(8,8),'red').save(photo)
    fn=model_method('prepare_source_person_context')
    for kw in [{},{'molmo_projection_path':'p','molmo_phase_b_vision':False}]:
        with pytest.raises(ValueError,match='Molmo visual'):
            fn(NS(model_config=NS(model_kwargs=kw)),photo)
    ref=fn(NS(model_config=NS(model_kwargs={'molmo_projection_path':'p'})),photo)
    assert isinstance(ref,SourcePersonRef) and ref.image.mode=='RGB'


def test_actual_molmo_encoder_keeps_source_separate_from_replacement():
    class VideoRef:
        def __init__(self):self.frames=[Image.new('RGB',(8,8),'black')];self.timestamps=[0.]
    class Encoded(Exception):pass
    seen={}
    def hidden(**kw):seen.update(kw);raise Encoded()
    worker=NS(get=lambda *a:NS(hidden=hidden))
    path=ROOT/'extensions_built_in/diffusion_models/minimax_h3/src/molmo_text_encoder.py'
    fn=source_function(path,'encode_prompt',{'torch':torch,'List':List,'Optional':Optional,
        'VideoRef':VideoRef,'SourcePersonRef':SourcePersonRef,'Image':Image,'np':np,
        '_MolmoWorkerProcess':worker},'MolmoH3TextEncoder')
    model=NS(phase_b_vision=True,tokenizer=NS(encode=lambda *a,**kw:NS(ids=[1,2])),
             molmo_python_path='python',molmo_model_path='model',proj={'tap':24},max_video_frames=8)
    photo=SourcePersonRef(Image.new('RGB',(8,8),'red'));replacement=Image.new('RGB',(8,8),'green')
    with pytest.raises(Encoded):
        fn(model,'replace',keyframes=[photo,replacement,VideoRef()],device=torch.device('cpu'))
    assert [v['type'] for v in seen['visuals']]==['source_person','image','video']
    assert tuple(seen['visuals'][0]['array'][0,0])==(255,0,0)
    assert tuple(seen['visuals'][1]['array'][0,0])==(0,128,0)
    model.phase_b_vision=False
    with pytest.raises(ValueError,match='visual mode'):
        fn(model,'replace',keyframes=[photo],device=torch.device('cpu'))


def test_actual_worker_prefix_preserves_picture_numbering():
    path=ROOT/'extensions_built_in/diffusion_models/minimax_h3/src/molmo_text_encoder.py'
    tree=ast.parse(path.read_text())
    code=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='_WORKER_CODE' for t in n.targets))
    worker_tree=ast.parse(code);fn=next(n for n in worker_tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_inputs')
    ns={'np':np,'torch':torch};exec(compile(ast.Module(body=[fn],type_ignores=[]),'<worker>','exec'),ns)
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        f=Path(temp)/'photo.npy';np.save(f,np.zeros((8,8,3),dtype=np.uint8))
        seen={}
        def processor(**kw):
            seen.update(kw);return {'input_ids':torch.tensor([[10]]),'attention_mask':torch.ones(1,1,dtype=torch.long)}
        batch,offset=ns['prepare_inputs'](processor,[5,6],[{'type':'source_person','path':str(f)}, {'type':'image','path':str(f)}])
        assert '<SourcePerson 1>' in seen['text'] and '<Picture 1>' in seen['text']
        assert '<Picture 2>' not in seen['text']
        assert batch['input_ids'][0,offset:].tolist()==[5,6]


@pytest.mark.parametrize('encode,latent_only,controls', [(False,False,['replacement.png']), (True,True,['replacement.png']), (True,False,None)])
def test_cache_rejects_source_photo_that_would_be_silently_ignored(encode,latent_only,controls):
    from contextlib import nullcontext
    fn=source_function(ROOT/'toolkit/dataloader_mixins.py','cache_text_embeddings',
                       {'accelerator':NS(main_process_first=nullcontext),'print_acc':lambda *a:None,
                        'tqdm':lambda x,**kw:x},'TextEmbeddingCachingMixin')
    item=NS(get_text_embedding_path=lambda **kw:'cache',caption='replace',
            dataset_config=NS(source_person_path='source'),encode_control_in_text_embeddings=encode,
            control_latent_only=latent_only,control_path=controls)
    dataset=NS(dataset_path='test',file_list=[item],sd=NS(device='cpu'))
    with pytest.raises(ValueError,match='multimodal controls'):
        fn(dataset)
