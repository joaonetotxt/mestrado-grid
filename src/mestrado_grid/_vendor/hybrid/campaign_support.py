"""Shared accounting and strict completion contract for the four-arm experiment."""
from common import *
from hybrid_core import decorate
from meta_planning.construction import original_mat_plan
from meta_planning.search_arrays import ArrayCodec
import math

TARGET = 25_000_000.0
PAIRS = ('relaxed_then_strict', 'strict_from_start')
METHODS = ('HYBRID', 'HYBRID_AI')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def compact(r):
    f = (r.get('metrics') or {}).get('finance') or {}
    return brief(r) | dict(PC1=f.get('PC1'), expected_operation=f.get('expected_operation'),
                          CVaR_operation=f.get('CVaR_operation'),
                          reliability=[{k:v for k,v in row.items() if k!='paths'} for row in r.get('reliability',[])])

def strict_success(r, phase, target=TARGET):
    value=(r.get('metrics') or {}).get('selection_objective')
    return (phase=='strict' and valid(r) and bool(r.get('normalized_violations'))
            and all(v==0 for v in r['normalized_violations'].values())
            and isinstance(value,(int,float)) and not isinstance(value,bool)
            and math.isfinite(value) and value<target)

def audit_agrees(a,b):
    """Reproduction tolerance, NEVER a relaxed physical or monetary target."""
    for key in ('status','signature','candidate_id','approved_cases','expected_cases','failed_cases'):
        if a.get(key)!=b.get(key):return False
    av=a.get('normalized_violations',{});bv=b.get('normalized_violations',{})
    if set(av)!=set(bv):return False
    if a['status']=='feasible':
        if not valid(a) or not valid(b) or any(v!=0 for v in [*av.values(),*bv.values()]):return False
    elif any(abs(av[k]-bv[k])>1e-12 for k in av):return False
    af=(a.get('metrics') or {}).get('selection_objective');bf=(b.get('metrics') or {}).get('selection_objective')
    if af is None or bf is None:return af is bf
    return math.isfinite(af) and math.isfinite(bf) and abs(af-bf)<=1e-5

def studies():
    raw=read(D/'inputs/extremo/beta_00.json')
    strict=Study(**raw)
    assert strict.c['alpha']==.9 and strict.c['beta']==0.0
    relaxed=copy.deepcopy(raw)
    rows=reliability(strict.data,original_mat_plan(strict))
    for name in ('SAIDI','SAIFI'):
        relaxed['data']['config'][name+'_limits']=[row[name] for row in rows]
    return strict,Study(**relaxed),relaxed

class CountingEvaluator(Evaluator):
    def __init__(self,st):
        super().__init__(st);self.calls=0;self.ac_cases=0
    def _case(self,*a,**kw):
        self.ac_cases+=1
        return super()._case(*a,**kw)
    def evaluate(self,c):
        self.calls+=1
        return decorate(super().evaluate(c))

def append(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as f:
        f.write(json.dumps(plain(value),ensure_ascii=False,allow_nan=False)+'\n');f.flush()

def save_gz(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_bytes(gzip.compress(json.dumps(plain(value),ensure_ascii=False,allow_nan=False).encode(),compresslevel=2))
    tmp.replace(path)

def read_gz(path):
    return json.loads(gzip.decompress(Path(path).read_bytes()))

def verify_frozen():
    rows=read(D/'execution_hashes.json')
    bad=[r['path'] for r in rows if sha(D/r['path'])!=r['sha256']]
    if bad:raise RuntimeError('Frozen files changed: '+str(bad))
    return len(rows)
