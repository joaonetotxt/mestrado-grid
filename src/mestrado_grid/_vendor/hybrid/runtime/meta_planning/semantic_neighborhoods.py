"""Explicit planning neighbourhoods with a frozen physical context.

The local engines optimize all exposed coordinates, without a post-proposal
mask or jitter. Context selection and interval bounds are declared adaptations.
No price, objective, power flow or economic repair is consulted here.
"""
from dataclasses import asdict
import copy
import numpy as np
from .schema import Candidate,fingerprint
from .topology import repair_trees,orient_tree

FAMILIES=('tree','line','generation','storage','reactive','operation')


def physical_changes(before,after,tol=1e-8):
    def delta(a,b):
        a=np.asarray(a,float);b=np.asarray(b,float);d=abs(a-b)
        return dict(changed=int(np.count_nonzero(d>tol)),max_abs=float(d.max(initial=0)),sum_abs=float(d.sum()))
    a=before.plan.canonical();b=after.plan.canonical()
    out={'trees':dict(changed=sum(x!=y for x,y in zip(a.trees,b.trees)),
                       exchanged_edges=sum(len(set(x)^set(y)) for x,y in zip(a.trees,b.trees))),
         'line_buys':dict(added=[list(x) for x in sorted(set(b.line_buys)-set(a.line_buys))],removed=[list(x) for x in sorted(set(a.line_buys)-set(b.line_buys))]),
         'storage_buys':dict(added=[list(x) for x in sorted(set(b.storage_buys)-set(a.storage_buys))],removed=[list(x) for x in sorted(set(a.storage_buys)-set(b.storage_buys))]),
         'new_capacity_MW':delta(before.new_capacity,after.new_capacity)}
    out.update({n:delta(v,getattr(after.operation,n)) for n,v in vars(before.operation).items()})
    if before.physical is not None:
        out.update({'phys_'+n:delta(v,after.physical[n]) for n,v in before.physical.items() if n!='version'})
    return out


def key_changes(codec,before,raw,actual):
    return {n:dict(proposed=int(np.count_nonzero(abs(raw[b.start:b.stop]-before[b.start:b.stop])>1e-12)),
                   after_decoder=int(np.count_nonzero(abs(actual[b.start:b.stop]-before[b.start:b.stop])>1e-12)))
            for n,b in codec.blocks.items()}


def cycle_path(st,tree,edge):
    e=next(e for e in st.lines if e['id']==edge);adj={b:[] for b in st.buses}
    for row in st.lines:
        if row['id'] in tree:
            adj[row['i']].append((row['j'],row['id']));adj[row['j']].append((row['i'],row['id']))
    paths={e['i']:[]};queue=[e['i']]
    for b in queue:
        if b==e['j']:return paths[b]
        for j,line in adj[b]:
            if j not in paths:paths[j]=paths[b]+[line];queue.append(j)
    raise ValueError('No path in reference tree')


class Space:
    """A deterministic map y -> complete candidate, with inspectable context."""
    def __init__(self,codec,base,descriptor):
        self.codec=codec;self.base=copy.deepcopy(base);self.spec=copy.deepcopy(descriptor)
        self.x=codec.encode(base);self.ids=np.asarray(descriptor['indices'],int)
        self.lo=np.asarray(descriptor['lower'],float);self.hi=np.asarray(descriptor['upper'],float)
        if self.ids.ndim!=1 or len(set(self.ids.tolist()))!=len(self.ids) or np.any(self.ids<0) or np.any(self.ids>=codec.dimension):raise ValueError('Invalid local indices')
        if self.lo.shape!=self.ids.shape or self.hi.shape!=self.ids.shape or not np.isfinite(self.lo).all() or not np.isfinite(self.hi).all() or np.any(self.lo<0) or np.any(self.hi>1) or np.any(self.lo>=self.hi):raise ValueError('Invalid local bounds')
        self.tree=descriptor['tree'];self.dimension=len(self.ids)+int(self.tree is not None)
        if not self.dimension:raise ValueError('Empty semantic neighbourhood')
        if descriptor['base_id']!=base.candidate_id:raise ValueError('Wrong frozen context')
        if self.tree is not None:
            k=self.tree['stage'];e=self.tree['add']
            if type(k)!=int or not 0<=k<codec.study.K or e in base.plan.trees[k] or self.tree['remove']!=cycle_path(codec.study,base.plan.trees[k],e):raise ValueError('Invalid tree exchange')

    def center(self):
        y=(self.x[self.ids]-self.lo)/(self.hi-self.lo)
        return np.r_[.5/(len(self.tree['remove'])+1),y] if self.tree is not None else y

    def decode(self,y):
        y=np.asarray(y,float)
        if y.shape!=(self.dimension,) or not np.isfinite(y).all() or np.any(y<0) or np.any(y>1):raise ValueError('Invalid local coordinate')
        raw=self.x.copy();z=y[1:] if self.tree is not None else y
        raw[self.ids]=self.lo+z*(self.hi-self.lo)
        candidate,repairs=self.codec.decode(raw)
        exchange=None
        if self.tree is not None:
            options=[None]+self.tree['remove'];removed=options[min(int(y[0]*len(options)),len(options)-1)]
            if removed is not None:
                k=self.tree['stage'];added=self.tree['add'];preferred=copy.deepcopy(self.base.plan.trees)
                preferred[k]=sorted((set(preferred[k])-{removed})|{added})
                candidate.plan=repair_trees(self.codec.study.data,candidate.plan,preferred)
                # Apply installation-dependent domains and temporal rules to the
                # selected plan via the existing decoder. All induced buys cost.
                blocks=self.codec.unpack(raw);blocks['plan']=self.codec.plan_codec.encode(candidate.plan)
                candidate,repairs=self.codec.decode(self.codec.pack(blocks))
                if set(candidate.plan.trees[k])!=set(preferred[k]):raise ValueError('Explicit exchange lost')
                exchange=dict(stage=k,added=added,removed=removed)
        # The engine's evaluated variable is y in this frozen composition D(y).
        # Multiple y may share one phenotype; no fitness is moved to another y.
        return candidate,dict(exchange=exchange,repairs=repairs)

    def initial(self,population,rng):
        out=np.broadcast_to(self.center(),(population,self.dimension)).copy()
        cats=set(self.spec['categorical_local'])
        for i in range(1,population):
            for j in range(self.dimension):
                if j in cats:
                    if rng.random()<.5:out[i,j]=rng.random()
                else:out[i,j]=np.clip(out[i,j]+rng.uniform(-.25,.25),0,1)
        return out


def make_space(codec,base,family,rng,radius=.1,coupled=True,global_probability=.125,spatial_hops=1,all_cycles=False):
    if family not in FAMILIES or not 0<radius<=1 or not 0<=global_probability<=1:raise ValueError('Invalid neighbourhood configuration')
    if type(spatial_hops)!=int or not 1<=spatial_hops<=4 or type(all_cycles)!=bool:raise ValueError('Invalid neighbourhood breadth')
    st=codec.study;pc=codec.plan_codec;x=codec.encode(base);k=int(rng.integers(st.K));cy=list(st.cycles[int(rng.integers(len(st.cycles)))])
    # Consume the same draw in either arm: breadth never changes target RNG.
    if all_cycles:cy=list(range(st.S))
    structural=[];tree=None;target={};region=set();plan_start=codec.blocks['plan'].start
    def domain(name,loc):
        b=codec.blocks[name];full=codec.indices[name];return [b.start+i for i,v in enumerate(full) if int(v) in set(loc)]
    if family=='tree':
        edges=[e for e in st.lines if e['id'] not in base.plan.trees[k]]
        if edges:
            e=edges[int(rng.integers(len(edges)))];tree=dict(stage=k,add=e['id'],remove=cycle_path(st,base.plan.trees[k],e['id']))
            region.update((e['i'],e['j']));target=dict(edge=e['id'],stage=k)
        else:family='operation'
    if family=='line':
        j=int(rng.integers(len(pc.lines)));e=pc.lines[j];structural=[plan_start+j];region.update((e['i'],e['j']));target=dict(edge=e['id'])
    elif family=='generation' and st.N:
        j=int(rng.integers(st.N));region.add(st.new[j]['bus']);structural=domain('new_capacity',[kk*st.N+j for kk in range(st.K)]);target=dict(generator=st.new[j]['id'])
    elif family=='storage' and pc.sites:
        j=int(rng.integers(len(pc.sites)));region.add(pc.sites[j]);structural=[plan_start+len(pc.lines)+j];target=dict(bus=pc.sites[j])
    elif family=='reactive':
        options=[(n,int(i)) for n,b in codec.blocks.items() if n.startswith('phys_') and n.endswith('_choice') for i in range(b.start,b.stop)]
        if options:
            name,i=options[int(rng.integers(len(options)))];structural=[i];target=dict(block=name,index=i)
            from .dependency_variation import partition
            spec=partition(codec);label=spec['names'][spec['labels'][i]]
            if label.startswith('bus:'):region.add(next(b for b in st.buses if str(b)==label[4:]))
            else:
                e=next(e for e in st.lines if str(e['id'])==label[5:]);region.update((e['i'],e['j']))
        else:family='operation'
    if not region:region.add(st.buses[int(rng.integers(st.B))])
    # Expand simultaneously by hop; do not let edge iteration order determine
    # the region. Default one-hop behaviour and random stream are preserved.
    for _ in range(spatial_hops):
        centers=set(region)
        for e in st.lines:
            if e['id'] in base.plan.trees[k] and (e['i'] in centers or e['j'] in centers):region.update((e['i'],e['j']))
    operational=[]
    if coupled or family=='operation':
        from .dependency_variation import partition
        spec=partition(codec);allowed=set('bus:'+str(b) for b in region)
        allowed.update('edge:'+str(e['id']) for e in st.lines if e['i'] in region and e['j'] in region)
        for name,b in codec.blocks.items():
            if name in ('plan','new_capacity') or name.endswith('_choice'):continue
            for j,flat in enumerate(codec.indices[name]):
                pos=np.unravel_index(flat,b.shape)
                stage,state=(pos[0],pos[2]) if name in ('soc','conventional','ev','cut') else (pos[0],pos[1])
                if (stage==k or family!='operation') and state in cy and spec['names'][spec['labels'][b.start+j]] in allowed:operational.append(b.start+j)
    ids=sorted(set(structural+operational))
    if not ids and tree is None:
        # Small analytic catalogues may lack a chosen equipment family.
        return make_space(codec,base,'operation',rng,radius,True,global_probability,spatial_hops,all_cycles)
    categories={r['start']+j for r in codec.categorical_spec() for j in range(len(r['sizes']))}
    global_move=bool(rng.random()<global_probability);width=1. if global_move else radius
    lo=np.maximum(0,x[ids]-width);hi=np.minimum(1,x[ids]+width)
    for j,i in enumerate(ids):
        if i in categories or i in structural:lo[j]=0.;hi[j]=1.
    catlocal=[j+int(tree is not None) for j,i in enumerate(ids) if i in categories]
    if tree is not None:catlocal.insert(0,0)
    desc=dict(version=1,base_id=base.candidate_id,family=family,target=target,stage=k,cycle=cy,region=sorted(region),
              indices=ids,structural_indices=structural,lower=lo.tolist(),upper=hi.tolist(),tree=tree,
              categorical_local=catlocal,global_move=global_move,coupled=coupled)
    # Keep legacy descriptors byte-equivalent for the default breadth.
    if spatial_hops!=1 or all_cycles:desc.update(spatial_hops=spatial_hops,all_cycles=all_cycles)
    return Space(codec,base,desc)
