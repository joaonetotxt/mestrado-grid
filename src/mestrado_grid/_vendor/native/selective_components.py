"""Validated native components extracted unchanged from the isolated v2 tests."""
import copy
from collections import Counter
import numpy as np
from meta_planning import physical
from meta_planning.native_blocks import Field
from meta_planning.topology import repair_trees,line_schedule
from effective_blocks import Channels
VERSION='selective_native_components_v2_20261016'
def radial_locations(st,c,k):
    """Feeder and path impedance; purely structural, no fitness ordering."""
    installed,_=line_schedule(st.data,c.plan);edges={e['id']:e for e in st.lines}
    adj={b:[] for b in st.buses}
    for name in c.plan.trees[k]:
        e=edges[name];raw=st.eopts[name,installed[name,k]]
        weight=abs(raw['r'])+abs(raw['x'])
        adj[e['i']].append((e['j'],weight));adj[e['j']].append((e['i'],weight))
    root=st.data['root'];locations={root:(root,0.)};todo=[root]
    while todo:
        a=todo.pop()
        for b,w in adj[a]:
            if b in locations:continue
            feeder=b if a==root else locations[a][0]
            locations[b]=(feeder,locations[a][1]+w);todo.append(b)
    if len(locations)!=len(st.buses):raise ValueError('Disconnected reference tree')
    return locations

class RegionalChannels(Channels):
    """Separate signed P and Q keys by ownership/technology/electrical region.

    P updates preserve Q-envelope fraction before independently moving Q.
    Regional keys allow simultaneous opposing actions within one technology.
    Individual and original broad visits remain available.
    """
    def __init__(self,st,c,case,mode,radius=.1,individual=False):
        if mode not in ('P','Q','PQ'):raise ValueError('Invalid P/Q rotation')
        if not 0<radius<=1:raise ValueError('Invalid radius')
        super().__init__(st,c,case,'P',radius)
        self.mode=mode;self.individual=individual;locations=radial_locations(st,c,self.k)
        sources=st.old+st.new
        labels=[('existing',g['technology']) for g in st.old]+[('planned',g['technology']) for g in st.new]
        active=(self.high-self.low>1e-12)|(self.qh-self.ql>1e-12)
        groups=[]
        for label in sorted(set(labels)):
            ids=[j for j,v in enumerate(labels) if v==label and active[j]]
            if individual:groups.extend([[j] for j in ids]);continue
            # First separate feeders; then near/far impedance halves within each.
            for feeder in sorted({locations[sources[j]['bus']][0] for j in ids}):
                members=sorted([j for j in ids if locations[sources[j]['bus']][0]==feeder],
                    key=lambda j:(locations[sources[j]['bus']][1],sources[j]['bus'],j))
                for part in np.array_split(members,min(2,len(members))):
                    if len(part):groups.append(list(part))
        self.groups=[np.array(g,dtype=int) for g in groups]
        self.d=len(self.groups)*(2 if mode=='PQ' else 1)
        if self.d<1:raise ValueError('No active physical channels')
    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.d,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):raise ValueError('Invalid regional coordinates')
        c=copy.deepcopy(self.c)
        if np.all(z==.5):return c
        st=self.st;k,s=self.k,self.s;v=(2*z-1)*self.radius;m=len(self.groups)
        lower=self.ql;upper=self.qh;q=self.q.copy()
        if self.mode in ('P','PQ'):
            p=self.p.copy()
            for g,t in zip(self.groups,v[:m]):p[g]+=t*((self.high-p)[g] if t>=0 else (p-self.low)[g])
            c.operation.existing_p[k,s]=p[:st.G];c.operation.new_p[k,s]=p[st.G:]
            ql,qh,nl,nh,*_=physical.q_bounds(st,c);lower=np.r_[ql[k,s],nl[k,s]];upper=np.r_[qh[k,s],nh[k,s]]
            fraction=np.divide(self.q-self.ql,self.qh-self.ql,out=np.full_like(self.q,.5),where=self.qh>self.ql)
            q=lower+fraction*(upper-lower)
        if self.mode in ('Q','PQ'):
            values=v[m:] if self.mode=='PQ' else v
            for g,t in zip(self.groups,values):q[g]+=t*((upper-q)[g] if t>=0 else (q-lower)[g])
        c.operation.existing_q[k,s]=q[:st.G];c.physical['new_q'][k][s]=q[st.G:].tolist()
        return c

class Coverage:
    """Every case once per sweep; risk ordering cannot starve another case.

Modes rotate per case across sweeps: wide P, local P, local PQ, wide P, Q.
The guarantee is for focused visits, not wall time, all resources or convergence.
"""
    def __init__(self,K,S,seed):
        self.domain=[(k,s) for k in range(K) for s in range(S)]
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,19616]))
        self.counts=Counter();self.mode_counts=Counter();self.sweep=[];self.steps=0
    def next(self,feedback):
        if not self.sweep:self.sweep=[self.domain[i] for i in self.rng.permutation(len(self.domain))]
        if self.steps%2==0:
            case=max(self.sweep,key=lambda c:feedback.get(c,{}).get('economic_opportunity',0))
            self.sweep.remove(case)
        else:case=self.sweep.pop(0)
        mode=('wide','local','PQ','wide','Q')[self.counts[case]%5]
        self.counts[case]+=1;self.mode_counts[case,mode]+=1;self.steps+=1
        return case,mode

class SpatialBlock:
    """Small signed independent region groups plus optional service direction.

Unselected regions remain fixed. Broad technology P visits coexist. The group
rotation is seed/index based and does not choose a dispatch using cost or AC.
"""
    def __init__(self,codec,base,reference,case,mode='P',radius=1.,offset=0,max_groups=4,service=False):
        self.codec=codec;self.base=base;self.reference=reference.copy();self.case=tuple(case)
        self.mode=mode;self.radius=radius;self.offset=offset;self.max_groups=max_groups;self.service=service
        self._anchor=None;self.helper=None;self._refresh()
        self.dimension=self.helper.d+int(service);self.categorical_indices=[]
        self.fields=[Field(n,base.ids(n,[case])) for n in ('existing_p','new_p','existing_q','phys_new_q')]
        if service:self.fields.append(Field('cut',base.ids('cut',[case])))
        self.metadata=dict(scope='effective_v2_spatial',version=VERSION,mode=mode,case_groups=[[case]],radius=radius,
            independent_local_signs=True,service=service,auxiliary_optimizer=False,categorical_key_indices=[],
            group_indices=[g.tolist() for g in self.helper.groups])
    def _refresh(self):
        if self._anchor is not None and np.array_equal(self.reference,self._anchor):return
        c=self.codec.decode(self.reference)[0];h=RegionalChannels(self.codec.study,c,self.case,self.mode,self.radius)
        if self.helper is None:
            groups=h.groups;start=self.offset%len(groups)
            h.groups=[groups[(start+j)%len(groups)] for j in range(min(self.max_groups,len(groups)))]
        else:h.groups=self.helper.groups
        h.d=len(h.groups)*(2 if self.mode=='PQ' else 1)
        self.helper=h;self._anchor=self.reference.copy()
    def project(self,full):return np.full(self.dimension,.5)
    def decode(self,z):
        self._refresh();z=np.asarray(z,float)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):raise ValueError('Invalid spatial keys')
        c=self.helper.decode(z[:self.helper.d])
        if self.service:
            k,s=self.case;t=(2*z[-1]-1)*self.radius;a=c.operation.cut[k,:,s];load=c.operation.conventional[k,:,s]
            c.operation.cut[k,:,s]=a-t*(a if t>=0 else load-a)
        out=self.reference.copy();encoded=self.codec.encode(c)
        for f in self.fields:out[f.indices]=encoded[f.indices]
        return out
    def describe(self):return dict(self.metadata,dimension=self.dimension)

class PriorityTreeBlock:
    """Six native signed feature weights create coordinated edge priorities.

Features are unscored line attributes: failure rate, duration contribution,
impedance, uninstalled status, root hop distance and seeded edge ranks. Both
signs are possible. This explicit problem-informed representation is not a
shortest-path constructor or a fitness-based neighboring-tree search. Original
unrestricted edge-key and single-exchange visits remain in the outer schedule.
"""
    def __init__(self,codec,base,reference,seed=96101):
        self.codec=codec;self.base=base;self.reference=reference.copy();self.seed=seed
        self.dimension=7;self.categorical_indices=[]
        b=codec.blocks['plan'];self.fields=[Field('plan',np.arange(b.start,b.stop))]
        self.metadata=dict(scope='effective_v2_tree',version=VERSION,mode='signed_line_priorities',
            auxiliary_optimizer=False,attribute_bias_declared=True,categorical_key_indices=[])
    def project(self,full):return np.full(self.dimension,.5)
    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(7,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):raise ValueError('Invalid priority keys')
        if np.all(z==.5):return self.reference.copy()
        st=self.codec.study;c=self.codec.decode(self.reference)[0];installed,_=line_schedule(st.data,c.plan)
        lines=sorted(st.lines,key=lambda e:e['id']);adj={b:[] for b in st.buses}
        for e in lines:adj[e['i']].append(e['j']);adj[e['j']].append(e['i'])
        hops={st.data['root']:0};queue=list(hops)
        for u in queue:
            for v in adj[u]:
                if v not in hops:hops[v]=hops[u]+1;queue.append(v)
        noise=np.random.default_rng(self.seed).permutation(len(lines))/max(1,len(lines)-1)
        trees=[];strength=abs(2*z[0]-1);weights=2*z[1:]-1
        for k in range(st.K):
            rows=[]
            for j,e in enumerate(lines):
                option=installed.get((e['id'],k));option=0 if option is None else option
                o=e['options'][option]
                electrical=st.eopts[e['id'],option]
                f=np.asarray(o['failure_rate']);r=np.asarray(o['repair_hours'])
                rows.append([f.mean(),(f*r).mean(),abs(electrical['r'])+abs(electrical['x']),
                    float(installed.get((e['id'],k)) is None),min(hops[e['i']],hops[e['j']]),noise[j]])
            a=np.array(rows,float);span=np.ptp(a,axis=0);a=np.divide(a-a.min(0),span,out=np.zeros_like(a),where=span>0)
            anchor=np.array([float(e['id'] not in c.plan.trees[k]) for e in lines])
            # Gate can retain the incumbent or use a genuine broad tree move;
            # small initialization perturbations must not be an all-noop plateau.
            scores=.1*(1-strength)*anchor+4*strength*(a@weights)
            trees.append([lines[j]['id'] for j in np.argsort(scores,kind='stable')])
        c.plan=repair_trees(st.data,c.plan,trees);out=self.reference.copy();b=self.codec.blocks['plan']
        out[b.start:b.stop]=self.codec.plan_codec.encode(c.plan);return out
    def describe(self):return dict(self.metadata,dimension=self.dimension)
