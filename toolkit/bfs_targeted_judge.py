"""Visual swap evaluator using Responses API; dry-run requires no credentials.

Key: OPENAI_API_KEY or --key-file (a file containing only the key).
Model: --model or OPENAI_JUDGE_MODEL. Never writes credentials or inline images.
"""
import argparse,base64,hashlib,json,mimetypes,os,subprocess,urllib.request
from pathlib import Path

FIELDS=['target_selection','replacement_identity','other_people_preserved','scene_motion_preserved','temporal_consistency','confidence']
SCHEMA={'type':'object','properties':{**{k:{'type':'number','minimum':0,'maximum':1} for k in FIELDS},'verdict':{'type':'string','enum':['correct','wrong_target','no_swap','mixed','uncertain']},'explanation':{'type':'string'}},'required':FIELDS+['verdict','explanation'],'additionalProperties':False}

def reward(result):
 for k in FIELDS:
  v=result[k]
  if isinstance(v,bool) or not isinstance(v,(int,float)) or not 0<=v<=1:raise ValueError('Invalid score: '+k)
 if result['verdict'] not in ['correct','wrong_target','no_swap','mixed','uncertain']:raise ValueError('Invalid verdict')
 if result['verdict']=='uncertain' or result['confidence']<.7:return None
 # A beautiful swap on the wrong person must not earn a positive reward.
 if result['verdict'] in ['wrong_target','no_swap','mixed']:return 0.
 return round(10*result['target_selection']*result['replacement_identity']*min(result['other_people_preserved'],result['scene_motion_preserved'],result['temporal_consistency']),4)

def image(path):
 return {'type':'input_image','image_url':'data:'+mimetypes.guess_type(str(path))[0]+';base64,'+base64.b64encode(Path(path).read_bytes()).decode(),'detail':'high'}

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--out',required=True);p.add_argument('--model',default=os.environ.get('OPENAI_JUDGE_MODEL'));p.add_argument('--key-file');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 spec=json.loads(Path(a.manifest).read_text());out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
 content=[{'type':'input_text','text':'Evaluate a requested character replacement. Source photo identifies WHO in the original guide must change; replacement photo identifies their NEW appearance. Judge only visible evidence. Do not reward a swap on another person. Compare all sampled times and crossings. Report uncertainty when evidence is insufficient. Images are labeled below. Scores range from 0 to 1. Explain the observed selected person, any changes to unselected people, and limitations of sparse frames.'}]
 if spec.get('target_binding'):
  content.append({'type':'input_text','text':'Explicit target identification and native Molmo tracking observations: '+json.dumps(spec['target_binding'])+' . This identifies ONLY the original character to replace. Verify it against the source photo. A swap on any other character is wrong.'})
 for label,name in [('Original person to replace','source_photo'),('New replacement appearance','replacement_photo')]:
  content.extend([{'type':'input_text','text':label},image(spec[name])])
 for clip in ['guide','generated']:
  for i,t in enumerate(spec.get('times',[.35,1.25,2.6])):
   f=out/f'{clip}_{i}.jpg'
   subprocess.run(['ffmpeg','-v','error','-y','-ss',str(t),'-i',spec[clip],'-frames:v','1','-vf','scale=672:384',str(f)],check=True)
   content.extend([{'type':'input_text','text':f'{clip.upper()} at {t}s'},image(f)])
 if spec.get('instruction'):content.append({'type':'input_text','text':'Requested edit: '+spec['instruction']})
 body={'model':a.model,'input':[{'role':'user','content':content}],'store':False,'max_output_tokens':1600,'text':{'format':{'type':'json_schema','name':'targeted_swap_grade','strict':True,'schema':SCHEMA}}}
 digest=hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()
 (out/'request_metadata.json').write_text(json.dumps({'model':a.model,'input_sha256':digest,'image_count':sum(x['type']=='input_image' for x in content),'dry_run':a.dry_run},indent=2))
 if a.dry_run:print('Dry run complete; no API request');return
 if not a.model:raise SystemExit('Set OPENAI_JUDGE_MODEL or --model')
 key=Path(a.key_file).read_text().strip() if a.key_file else os.environ.get('OPENAI_API_KEY','')
 if not key:raise SystemExit('Set OPENAI_API_KEY or use --key-file; do not put keys in manifests')
 req=urllib.request.Request('https://api.openai.com/v1/responses',json.dumps(body).encode(),{'Authorization':'Bearer '+key,'Content-Type':'application/json'})
 # Never print request headers, error response bodies or credentials.
 try:
  with urllib.request.urlopen(req,timeout=180) as response:data=json.load(response)
 except urllib.error.HTTPError as exc:raise SystemExit(f'OpenAI request failed: HTTP {exc.code}') from None
 if data.get('status')!='completed':raise SystemExit('Evaluator response incomplete; no reward assigned')
 parts=[c['text'] for item in data.get('output',[]) for c in item.get('content',[]) if c.get('type')=='output_text']
 result=json.loads(''.join(parts));result['reward_0_to_10']=reward(result)
 result['response_id']=data.get('id');result['usage']=data.get('usage');result['model']=data.get('model')
 (out/'grade.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
