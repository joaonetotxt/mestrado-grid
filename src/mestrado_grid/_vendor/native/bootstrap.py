import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[n]='1'
import sys,json,hashlib
from pathlib import Path
D=Path(__file__).resolve().parent
sys.path.insert(0,str(D/'runtime'))
from meta_planning.schema import plain
def read(p):return json.loads(Path(p).read_text('utf-8-sig'))
def save(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 tmp=p.with_name(p.name+'.pending');tmp.write_text(json.dumps(plain(x),ensure_ascii=False,indent=2,allow_nan=False),'utf-8');tmp.replace(p)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
