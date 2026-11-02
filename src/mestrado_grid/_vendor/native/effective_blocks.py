"""Isolated native probes. Production is only read; no auxiliary optimizer."""
import os
for n in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[n]='1'
import sys,copy,json,time,argparse
from pathlib import Path
import numpy as np
H=Path(__file__).resolve().parent
sys.path.insert(0,str(H/'runtime'))
from meta_planning.schema import Study,Candidate,plain
from meta_planning.engines import Engine,SearchConfig
from meta_planning.evaluator import Evaluator,rank
from meta_planning.case_cache import CaseCachedEvaluator
from meta_planning.topology import reliability,repair_trees
from meta_planning.guided_blocks import cycle_path
from meta_planning import physical
from stage_cache import StageCachedEvaluator

def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v): Path(p).write_text(json.dumps(plain(v),ensure_ascii=False,indent=2),encoding='utf-8')
def brief(r): return {k:r.get(k) for k in ('status','normalized_violation','candidate_id','approved_cases','normalized_violations')}|{'FO':(r.get('metrics') or {}).get('selection_objective')}
def selection(r):
    out=brief(r);out['metrics']=r.get('metrics')
    if r['status']=='violated' and r.get('failed_cases')==0 and r.get('metrics'):
        out['search_selection']='feasibility_violation_exact_cost_tie_v1';out['failed_cases']=0
    return out

class Tree:
    def __init__(self,st,c,stages):
        self.st=st;self.c=c;self.stages=stages;self.d=2*len(stages)
        self.edges={k:sorted(e['id'] for e in st.lines if e['id'] not in c.plan.trees[k]) for k in stages}
        self.paths={(k,e):cycle_path(st.data,c.plan.trees[k],e) for k in stages for e in self.edges[k]}
    def decode(self,z):
        c=copy.deepcopy(self.c);trees=copy.deepcopy(c.plan.trees)
        for i,k in enumerate(self.stages):
            j=min(int(z[2*i]*(1+len(self.edges[k]))),len(self.edges[k]))
            if j:
                e=self.edges[k][j-1];path=self.paths[k,e];drop=path[min(int(z[2*i+1]*len(path)),len(path)-1)]
                trees[k]=[v for v in trees[k] if v!=drop]+[e]
        c.plan=repair_trees(self.st.data,c.plan,trees);return c
    def neutral(self):return np.zeros(self.d)

class Channels:
    """Independent signed physical channels; no cost sorting or dispatch repair.

    One channel per nonempty technology/ownership group. Single family visits
    do not perturb Q or service incidentally. P changes preserve normalized Q
    envelope utilization (same physical codec convention). No AC in decoder.
    """
    def __init__(self,st,c,case,mode,radius=1.):
        self.st=st;self.c=c;self.k,self.s=case;self.mode=mode;self.radius=radius
        k,s=case;lo,hi,*_=physical.existing_bounds(st,c)
        self.low=np.r_[lo[k,s],np.zeros(st.N)];self.high=np.r_[hi[k,s],st.active_capacity(c)[k]*physical.availability(st)[s]]
        self.p=np.r_[c.operation.existing_p[k,s],c.operation.new_p[k,s]]
        ql,qh,nl,nh,*_=physical.q_bounds(st,c)
        self.ql=np.r_[ql[k,s],nl[k,s]];self.qh=np.r_[qh[k,s],nh[k,s]]
        self.q=np.r_[c.operation.existing_q[k,s],np.asarray(c.physical['new_q'])[k,s]]
        labels=[('existing',g['technology']) for g in st.old]+[('planned',g['technology']) for g in st.new]
        span=self.qh-self.ql if mode=='Q' else self.high-self.low
        self.groups=[np.array([i for i,v in enumerate(labels) if v==label and span[i]>1e-12]) for label in sorted(set(labels))]
        self.groups=[g for g in self.groups if len(g)]
        self.d=len(self.groups) if mode in ('P','Q') else len(self.groups)+1
    def neutral(self):return np.full(self.d,.5)
    def decode(self,z):
        c=copy.deepcopy(self.c);st=self.st;k,s=self.k,self.s
        if np.all(np.asarray(z)==.5):return c
        signed=(np.asarray(z)*2-1)*self.radius
        if self.mode=='Q':
            q=self.q.copy()
            for g,t in zip(self.groups,signed):q[g]+=t*((self.qh-q)[g] if t>=0 else (q-self.ql)[g])
            c.operation.existing_q[k,s]=q[:st.G];c.physical['new_q'][k][s]=q[st.G:].tolist()
        else:
            p=self.p.copy()
            for g,t in zip(self.groups,signed):p[g]+=t*((self.high-p)[g] if t>=0 else (p-self.low)[g])
            c.operation.existing_p[k,s]=p[:st.G];c.operation.new_p[k,s]=p[st.G:]
            ql,qh,nl,nh,*_=physical.q_bounds(st,c);lower=np.r_[ql[k,s],nl[k,s]];upper=np.r_[qh[k,s],nh[k,s]]
            frac=np.divide(self.q-self.ql,self.qh-self.ql,out=np.full_like(self.q,.5),where=self.qh>self.ql)
            q=lower+frac*(upper-lower)
            c.operation.existing_q[k,s]=q[:st.G];c.physical['new_q'][k][s]=q[st.G:].tolist()
            if self.mode=='P_service':
                t=signed[-1];before=c.operation.cut[k,:,s];load=c.operation.conventional[k,:,s]
                c.operation.cut[k,:,s]=before-t*(before if t>=0 else load-before)
        return c

class ChronologicalTree:
    """Native common closing edge with independent opening keys per stage.

    Coupling stages crosses the max-violation plateau without changing rank.
    Every edge absent from at least one stage is eligible; no neighbor scored.
    Interleaved independent-stage visits retain all chronological possibilities.
    """
    def __init__(self,st,c):
        self.st=st;self.c=c;self.stages=list(range(st.K));self.d=1+st.K
        self.edges=sorted(e['id'] for e in st.lines if any(e['id'] not in t for t in c.plan.trees))
        self.paths={(k,e):cycle_path(st.data,c.plan.trees[k],e) for k in self.stages for e in self.edges if e not in c.plan.trees[k]}
    def neutral(self):return np.zeros(self.d)
    def decode(self,z):
        c=copy.deepcopy(self.c);trees=copy.deepcopy(c.plan.trees)
        j=min(int(z[0]*(1+len(self.edges))),len(self.edges))
        if j:
            e=self.edges[j-1]
            for k in self.stages:
                if e in trees[k]:continue
                path=self.paths[k,e];drop=path[min(int(z[k+1]*len(path)),len(path)-1)]
                trees[k]=[v for v in trees[k] if v!=drop]+[e]
            c.plan=repair_trees(self.st.data,c.plan,trees)
        return c
