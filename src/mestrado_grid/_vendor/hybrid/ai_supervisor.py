"""Auditable session/API/replay supervisor. Never silently impersonates AI."""
from common import *
from hybrid_core import FAMILIES
import urllib.request

PROMPT='''You schedule an isolated power-grid hybrid optimizer. Only choose the
order of the supplied neighbourhood families and a local LP trust radius.
Use ONLY the supplied current-arm evidence. Do not use other-arm results.
All families must occur exactly once. Feasibility has absolute priority.
Original FO, CVaR alpha/beta, AC and reliability constraints are immutable.
Your output cannot supply candidates, code, constraints or financial values.
The exact full AC evaluator accepts candidates. Return the JSON schema only.'''
SCHEMA={'type':'object','properties':{
 'family_order':{'type':'array','items':{'type':'string','enum':FAMILIES}},
 'radius':{'type':'number','enum':[.125,.25,.5]},
 'reason':{'type':'string'}},'required':['family_order','radius','reason'],'additionalProperties':False}

def validate(d):
    if set(d)!={'family_order','radius','reason'}:raise ValueError('Unknown supervisor fields')
    if not isinstance(d['family_order'],list) or len(d['family_order'])!=len(FAMILIES) or set(d['family_order'])!=set(FAMILIES):raise ValueError('Every family exactly once required')
    if isinstance(d['radius'],bool) or d['radius'] not in (.125,.25,.5):raise ValueError('Invalid radius')
    if not isinstance(d['reason'],str) or not d['reason'].strip() or len(d['reason'])>2500:raise ValueError('Reason required')
    return d

class Supervisor:
    def __init__(self,path,mode='session'):
        self.path=Path(path);self.mode=mode;self.calls=0;self.wait_s=0.;self.tokens=0
        self.path.mkdir(exist_ok=True)
    def choose(self,observation):
        ix=self.calls;self.calls+=1;started=time.monotonic()
        request={'schema_version':1,'instructions':PROMPT,'schema':SCHEMA,'observation':observation}
        raw=json.dumps(request,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()
        request_hash=hashlib.sha256(raw).hexdigest()
        save(self.path/f'{ix:03d}.request.json',{'request_sha256':request_hash,**request})
        response_path=self.path/f'{ix:03d}.response.json'
        if self.mode in ('session','replay'):
            if self.mode=='replay' and not response_path.exists():raise FileNotFoundError('Replay needs recorded real decisions')
            while not response_path.exists():time.sleep(.5)
            envelope=read(response_path)
            if envelope.get('request_sha256')!=request_hash:raise ValueError('Response for different observation')
            if envelope.get('provider')!='codex_current_session':raise ValueError('Session provenance missing')
            decision=validate(envelope['decision'])
        elif self.mode=='api':
            key=os.environ.get('OPENAI_API_KEY');model=os.environ.get('OPENAI_MODEL')
            if not key or not model:raise RuntimeError('Configure OPENAI_API_KEY and OPENAI_MODEL locally; no fallback pretending to be AI')
            body={'model':model,'store':False,'input':[{'role':'developer','content':PROMPT},{'role':'user','content':json.dumps(observation,ensure_ascii=False)}],
                  'text':{'format':{'type':'json_schema','name':'search_schedule','strict':True,'schema':SCHEMA}}}
            req=urllib.request.Request('https://api.openai.com/v1/responses',json.dumps(body).encode(),{'Content-Type':'application/json','Authorization':'Bearer '+key})
            with urllib.request.urlopen(req,timeout=120) as response:answer=json.load(response)
            if answer.get('status')!='completed':raise RuntimeError('Incomplete model response')
            messages=[x['text'] for o in answer.get('output',[]) for x in o.get('content',[]) if x.get('type')=='output_text']
            if len(messages)!=1:raise ValueError('Missing or ambiguous structured response')
            decision=validate(json.loads(messages[0]));self.tokens+=answer.get('usage',{}).get('total_tokens',0)
            save(response_path,dict(request_sha256=request_hash,provider='openai_responses_api',model=answer.get('model'),response_id=answer.get('id'),usage=answer.get('usage'),decision=decision))
        else:raise ValueError(self.mode)
        elapsed=time.monotonic()-started;self.wait_s+=elapsed
        save(self.path/f'{ix:03d}.accepted.json',dict(request_sha256=request_hash,decision=decision,elapsed_s=elapsed,mode=self.mode))
        return decision
