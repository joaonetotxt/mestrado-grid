"""Versioned integration with the existing persistent native runner.

Three focused visits then one complete original-scheduler visit. No asset,
temporal, network or operating family is removed. Original engines, comparator,
physics and financial risk objective are unchanged. Never import this module
from production without a separately versioned release and checkpoint migration.
"""
import numpy as np
from collections import Counter
from effective_blocks import Tree,ChronologicalTree,Channels
from meta_planning.native_blocks import Field
from meta_planning.risk_native_search import RiskNativeScheduler
from meta_planning.topology import reliability

class ContextSubset:
    """Ordinal selection view without retaining all paged engine states."""
    def __init__(self,contexts,positions):self.contexts=contexts;self.positions=tuple(positions)
    def __len__(self):return len(self.positions)
    def __getitem__(self,index):return self.contexts[self.positions[index]]
    def __iter__(self):
        for i in self.positions:yield self.contexts[i]

class Adapter:
    def __init__(self,codec,base,reference,kind,stage=None,case=None,mode=None,radius=1.):
        self.codec=codec;self.base=base;self.reference=reference.copy()
        self.kind=kind;self.stage=stage;self.case=case;self.mode=mode;self.radius=radius
        self._anchor=None;self.helper=None;self._refresh()
        self.dimension=self.helper.d
        self.categorical_indices=list(range(self.dimension)) if kind!='channels' else []
        if kind=='channels':
            self.names=('existing_p','new_p','existing_q','phys_new_q','cut')
            self.fields=[Field(n,base.ids(n,[case])) for n in self.names]
        else:
            b=codec.blocks['plan'];self.fields=[Field('plan',np.arange(b.start,b.stop))]
        self.metadata=dict(scope='effective_'+kind,scheduler='effective_native_v1',
            case_groups=[] if case is None else [[case]],stage=stage,mode=mode,radius=radius,
            categorical_key_indices=self.categorical_indices,auxiliary_optimizer=False,
            protected_original_fraction=.25,projection='neutral anchor only; re-evaluated')
    def _refresh(self):
        if self._anchor is not None and np.array_equal(self.reference,self._anchor):return
        c=self.codec.decode(self.reference)[0];st=self.codec.study
        if self.kind=='channels':h=Channels(st,c,self.case,self.mode,self.radius)
        elif self.kind=='chronological_tree':h=ChronologicalTree(st,c)
        else:h=Tree(st,c,[self.stage])
        if self.helper is not None and self.kind=='channels':h.groups=self.helper.groups;h.d=self.helper.d
        self.helper=h;self._anchor=self.reference.copy()
    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):raise ValueError('Invalid native local coordinates')
        self._refresh();c=self.helper.decode(z);out=self.reference.copy()
        if self.kind=='channels':
            encoded=self.codec.encode(c)
            for f in self.fields:out[f.indices]=encoded[f.indices]
        else:
            b=self.codec.blocks['plan'];out[b.start:b.stop]=self.codec.plan_codec.encode(c.plan)
        return out
    def project(self,full):
        self._refresh();return self.helper.neutral()
    def describe(self):
        return dict(self.metadata,dimension=self.dimension,physical_coordinates=sum(len(f.indices) for f in self.fields),
            fields=[dict(name=f.name,kind='native_physical',coordinates=len(f.indices)) for f in self.fields])

class EffectiveScheduler(RiskNativeScheduler):
    def __init__(self,*args):
        super().__init__(*args);self.effective_step=0;self.effective_stages=Counter();self.effective_cases=Counter();self.effective_trees=0
        self.effective_context_scan=0;self.effective_original_positions=[]
    def _choose_original(self,reference,original):
        # RiskNativeScheduler indexes contexts using recent's integer keys.
        # Filtering focused contexts introduces gaps in the global numbers.
        # Remap only this selection view; native contexts and memories keep
        # their original numbers, and accounting remains globally identified.
        saved=self.recent
        self.recent={i:saved[n] for i,n in enumerate(original.positions) if n in saved}
        try:return super().choose(reference,original)
        finally:self.recent=saved
    def choose(self,reference,contexts):
        step=self.effective_step;self.effective_step+=1
        while self.effective_context_scan<len(contexts):
            i=self.effective_context_scan;c=contexts[i]
            if c.number!=i:raise ValueError('Global context identity differs from storage index')
            if not c.block.metadata.get('scope','').startswith('effective_'):
                self.effective_original_positions.append(i)
            self.effective_context_scan+=1
        original=ContextSubset(contexts,self.effective_original_positions)
        if step%4==3:return self._choose_original(reference,original)
        st=self.codec.study;c=self.codec.decode(reference)[0]
        failed=[r['stage'] for r in reliability(st.data,c.plan) if not r['passed']]
        if failed:
            j=self.effective_trees;self.effective_trees+=1
            k=min(failed,key=lambda k:(self.effective_stages[k],k));self.effective_stages[k]+=1
            kind='chronological_tree' if j%2==0 else 'tree'
            return None,Adapter(self.codec,self.base,reference,kind,stage=k)
        if (self.record or {}).get('status')!='feasible' or not self.feedback:
            return self._choose_original(reference,original)
        case=max(self.feedback,key=lambda ks:self.feedback[ks].get('economic_opportunity',0)/(1+self.effective_cases[ks]))
        self.effective_cases[case]+=1;row=self.feedback[case]
        mode='P_service' if row.get('hourly_cut',0)>row.get('hourly_curtailment',0) else 'P'
        if mode=='P_service' and self.effective_cases[case]%3==2:mode='Q'
        return None,Adapter(self.codec,self.base,reference,'channels',case=case,mode=mode,radius=1. if mode=='P' else .1)
    def visit_generations(self,context,default):
        if context.block.metadata.get('scope','').startswith('effective_'):return min(default,4)
        return super().visit_generations(context,default)
    def report(self):
        return super().report()|dict(effective_native_visits=self.effective_step,
            stages_targeted=dict(self.effective_stages),focused_case_visits={str(k):v for k,v in self.effective_cases.items()},
            eligibility='Original scheduler advances every fourth visit; original full catalogue retained')

def enable_release():
    from meta_planning import persistent_search as p
    original_preflight=p.exploration_preflight
    original_factory=p.make_scheduler
    p.make_scheduler=lambda codec,seed,mass,options:original_factory(codec,seed,mass,options) if options.protected_resource_coverage else EffectiveScheduler(codec,seed,mass,options) if options.risk_native_acceleration else original_factory(codec,seed,mass,options)
    def preflight(codec,population,options,seed=0,mass=.25):
        # coverage_bound invokes exploration_preflight for the original phase;
        # prevent recursion through the outer wrapper for that calculation.
        p.exploration_preflight=original_preflight
        try:result=original_preflight(codec,population,options,seed,mass)
        finally:p.exploration_preflight=preflight
        return dict(result,conservative_calls=4*result.get('conservative_calls',0),
            protected_original_visit_every=4,scope='Four times original conservative bound; not complete-site or combination coverage')
    p.exploration_preflight=preflight
