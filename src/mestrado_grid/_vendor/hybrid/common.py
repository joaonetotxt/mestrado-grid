import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import sys,json,gzip,copy,time,datetime,warnings,traceback,hashlib
from pathlib import Path
import numpy as np
from scipy.optimize import OptimizeWarning
D=Path(__file__).resolve().parent;sys.path.insert(0,str(D/'runtime'))
from meta_planning.schema import Study,Candidate,Plan,plain
from meta_planning.evaluator import Evaluator,rank
from meta_planning import coordinated_ac as ca,physical,economics
from meta_planning.topology import line_schedule,reliability,repair_trees,orient_tree
from meta_planning.temporal import storage_dispatch,demand_bounds,normalize
from accounting_checks import finance_checks
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def audit(p,x):Path(p).write_bytes(gzip.compress(json.dumps(plain(x),ensure_ascii=False,allow_nan=False).encode('utf-8')))
def valid(r):return r['status']=='feasible' and r['approved_cases']==r['expected_cases']==96 and r['failed_cases']==0 and r['normalized_violation']==0
def brief(r):return {k:r.get(k) for k in ('status','candidate_id','plan_id','approved_cases','expected_cases','failed_cases','normalized_violation','normalized_violations')}|{'FO':(r.get('metrics') or {}).get('selection_objective')}
_lp=ca.linprog
def limited(*a,**kw):
    kw['options']={**kw.get('options',{}),'threads':1}
    with warnings.catch_warnings():warnings.simplefilter('ignore',OptimizeWarning);return _lp(*a,**kw)
ca.linprog=limited
def fresh_save(st,c,out):
    r=Evaluator(st).evaluate(c);save(out/'candidate.json',c.to_dict());audit(out/'validation.json.gz',r);save(out/'record.json',brief(r));return r
