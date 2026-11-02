"""Neutral structural search neighborhoods, version 1.

No fitness, voltages, flows, costs, cut diagnostics or preferred sites are read.
The caller supplies the current context and native engine proposals. Every
continuous common move has its opposite evaluated; endpoint moves include both
bounds. Technology subsets and graph exchanges are chosen uniformly.

This is an explicit shared hybrid layer, not a claim of native-engine behavior.
"""
import copy
import numpy as np
from .topology import canonical, line_schedule, orient_tree

CASE_NAMES=('existing_p','existing_q','new_p','phys_new_q','phys_bess_q','cut','phys_cap_state','phys_tap')
TEMPORAL_NAMES=('soc','conventional','ev')

def move_to_bound(x,sign,amplitude):
    if sign not in (-1,1) or not 0<amplitude<=1:raise ValueError('Invalid symmetric move')
    return x*(1-amplitude) if sign<0 else x+(1-x)*amplitude

class StateVariation:
    def __init__(self,codec,seed,common_only=False):
        self.codec=codec;self.st=codec.study;self.common_only=common_only
        self.rng=np.random.default_rng(np.random.SeedSequence([seed,104729]))
        self.pairs=0;self.case_deck=[];self.family_deck=[];self.temporal_deck=[]
        self.case_indices={};self.temporal_indices={}
        self.case_names=[n for n in CASE_NAMES if codec.blocks[n].stop>codec.blocks[n].start]
        self.plan_domains=codec.categorical_spec()
        self.plan_domains=[d for d in self.plan_domains if d['name'] not in ('cap_state','tap')]
        for name,block in codec.blocks.items():
            mapping=np.full(int(np.prod(block.shape)),-1,int)
            mapping[codec.indices[name]]=np.arange(block.start,block.stop)
            mapping=mapping.reshape(block.shape)
            if name in CASE_NAMES:
                for k in range(self.st.K):
                    for s in range(self.st.S):
                        ids=mapping[k,:,s] if name=='cut' else mapping[k,s,:]
                        self.case_indices[name,k,s]=ids[ids>=0].copy()
            if name in TEMPORAL_NAMES:
                for k in range(self.st.K):
                    for j,cycle in enumerate(self.st.cycles):
                        ids=mapping[k][:,cycle].ravel()
                        self.temporal_indices[name,k,j]=ids[ids>=0].copy()
        self.source_tech={name:np.array([g['technology'] for g in gens]) for name,gens in [('existing_p',self.st.old),('new_p',self.st.new)]}
        self.technologies=sorted(set(t for ts in self.source_tech.values() for t in ts))

    def _cases(self,count):
        chosen=[]
        while len(chosen)<count:
            if not self.case_deck:self.case_deck=self.rng.permutation(self.st.K*self.st.S).tolist()
            item=self.case_deck.pop()
            if item not in chosen:chosen.append(item)
        return [(i//self.st.S,i%self.st.S) for i in chosen]

    def _family(self):
        if not self.family_deck:self.family_deck=self.rng.permutation(len(self.case_names)).tolist()
        return self.case_names[self.family_deck.pop()]

    def indices(self,names,cases,technologies=None):
        out=[]
        for name in names:
            for k,s in cases:
                ids=self.case_indices[name,k,s]
                if technologies is not None and name in self.source_tech:
                    # Mask in the original tensor before dropping invariant keys.
                    b=self.codec.blocks[name];flat=self.codec.indices[name]
                    mask=(flat//b.shape[2])==k*self.st.S+s
                    tech=self.source_tech[name][flat[mask]%b.shape[2]]
                    ids=ids[np.isin(tech,technologies)]
                out.extend(ids)
        return np.array(out,dtype=int)

    def _apply_pair(self,reference,specs,event):
        proposals=[];events=[]
        for opposite in (1,-1):
            x=reference.copy();description=[]
            for ids,sign,amplitude,mode,label in specs:
                if mode=='additive':x[ids]=np.clip(reference[ids]+opposite*sign*amplitude,0,1)
                else:x[ids]=move_to_bound(reference[ids],opposite*sign,amplitude)
                description.append(dict(group=label,sign=int(opposite*sign),amplitude=float(amplitude),mode=mode,coordinates=len(ids)))
            proposals.append(x);events.append({**event,'origin':'common_symmetric','opposite':opposite,'moves':description,'native_credit':False})
        return proposals,events

    def _spec(self,ids,label):
        mode=str(self.rng.choice(['bound','fraction','additive'],p=[.35,.35,.30]))
        amplitude=1. if mode=='bound' else float(self.rng.choice([.03,.1,.3,.6]))
        return ids,int(self.rng.choice([-1,1])),amplitude,mode,label

    def exchange(self,reference):
        """Uniform fundamental-cycle exchange; no electrical repair or ranking."""
        block=self.codec.blocks['plan'];plan,_=self.codec.plan_codec.decode(reference[block.start:block.stop])
        k=int(self.rng.integers(self.st.K));tree=set(plan.trees[k]);lines=self.codec.plan_codec.lines
        options=[(j,e) for j,e in enumerate(lines) if e['id'] not in tree]
        if not options:return reference.copy(),dict(changed=False,reason='catalogue_is_tree')
        j,added=options[int(self.rng.integers(len(options)))];adj={b:[] for b in self.st.buses}
        byid={e['id']:e for e in lines}
        for name in tree:
            e=byid[name];adj[e['i']].append((e['j'],name));adj[e['j']].append((e['i'],name))
        start=added['i'];end=added['j'];parent={start:(None,None)};queue=[start]
        for b in queue:
            if b==end:break
            for neighbor,name in adj[b]:
                if neighbor not in parent:parent[neighbor]=(b,name);queue.append(neighbor)
        path=[];b=end
        while b!=start:
            b,name=parent[b];path.append(name)
        removed=path[int(self.rng.integers(len(path)))];plan.trees[k]=sorted(tree-{removed}|{added['id']})
        installed,_=line_schedule(self.st.data,self.codec.plan_codec.decode(reference[block.start:block.stop])[0])
        purchase_changed=False
        if installed[added['id'],k] is None:
            first=min(stage for stage,edges in enumerate(plan.trees) if added['id'] in edges)
            choices=[ch for ch in self.codec.plan_codec.line_choices[j] if ch and min(p[2] for p in ch)<=first]
            choice=choices[int(self.rng.integers(len(choices)))]
            plan.line_buys=[p for p in plan.line_buys if p[0]!=added['id']]+list(choice);purchase_changed=True
        plan=canonical(plan);line_schedule(self.st.data,plan);orient_tree(self.st.data,plan.trees[k])
        x=reference.copy();x[block.start:block.stop]=self.codec.plan_codec.encode(plan)
        check,_=self.codec.plan_codec.decode(x[block.start:block.stop])
        if check.trees!=plan.trees:raise AssertionError('Graph move lost during decoding')
        return x,dict(changed=True,stage=k,added=added['id'],removed=removed,purchase_changed=purchase_changed)

    def propose(self,native,reference):
        native=np.asarray(native,float);reference=np.asarray(reference,float)
        if native.ndim!=2 or native.shape[1]!=self.codec.dimension or len(native)%2:raise ValueError('Even population required')
        out=[];events=[]
        for first in range(0,len(native),2):
            slot=self.pairs%8;credit=(self.pairs//8+self.pairs)%2==0 and not self.common_only
            self.pairs+=1
            event=dict(pair=self.pairs,origin='native_state_context' if credit else 'common_symmetric',native_credit=credit)
            if slot<6:
                count=int(self.rng.choice([1,1,1,2,4,8,16,self.st.K*self.st.S]));count=min(count,self.st.K*self.st.S)
                cases=self._cases(count);event.update(scope='case',cases=cases)
                if credit:
                    ids=self.indices(self.case_names,cases);amplitude=float(self.rng.choice([.03,.1,.3,1.]))
                    for i in (first,first+1):
                        x=reference.copy();x[ids]=np.clip(reference[ids]+np.clip(native[i,ids]-reference[ids],-amplitude,amplitude),0,1)
                        out.append(x);events.append(dict(event,amplitude=amplitude,coordinates=len(ids)))
                elif self.rng.random()<.45:
                    # Physical coupling defines eligible fields, not a preferred
                    # direction: supply and shedding signs are independent.
                    tcount=int(self.rng.integers(1,len(self.technologies)+1))
                    tech=self.rng.choice(self.technologies,tcount,replace=False).tolist()
                    supply=self.indices(['existing_p','new_p'],cases,tech);demand=self.indices(['cut'],cases)
                    specs=[self._spec(supply,'supply'),self._spec(demand,'cut')]
                    if self.rng.random()<.25:
                        qnames=[n for n in self.case_names if n in ('existing_q','phys_new_q','phys_bess_q')]
                        specs.append(self._spec(self.indices(qnames,cases),'Q'))
                    pp,ee=self._apply_pair(reference,specs,dict(event,mode='joint_supply_attendance',technologies=tech))
                    out.extend(pp);events.extend(ee)
                else:
                    name=self._family();ids=self.indices([name],cases)
                    if self.rng.random()<.15 and len(ids):ids=self.rng.choice(ids,max(1,int(np.ceil(.05*len(ids)))),replace=False)
                    pp,ee=self._apply_pair(reference,[self._spec(ids,name)],dict(event,mode='single_family',family=name))
                    out.extend(pp);events.extend(ee)
            elif slot==6:
                total=self.st.K*len(self.st.cycles)
                if not self.temporal_deck:self.temporal_deck=self.rng.permutation(total).tolist()
                j=self.temporal_deck.pop();k,cy=divmod(j,len(self.st.cycles));name=str(self.rng.choice(TEMPORAL_NAMES))
                ids=self.temporal_indices[name,k,cy];event.update(scope='temporal_cycle',stage=k,cycle=cy,family=name)
                if credit:
                    for i in (first,first+1):
                        x=reference.copy();x[ids]=native[i,ids];out.append(x);events.append(event.copy())
                else:
                    if len(ids):ids=self.rng.choice(ids,max(1,int(np.ceil(.1*len(ids)))),replace=False)
                    pp,ee=self._apply_pair(reference,[self._spec(ids,name)],event);out.extend(pp);events.extend(ee)
            else:
                if credit and self.rng.random()<.5:
                    if self.rng.random()<.25:
                        block=self.codec.blocks['new_capacity'];ids=np.arange(block.start,block.stop);family='new_capacity'
                    else:
                        domain=self.plan_domains[int(self.rng.integers(len(self.plan_domains)))];family=domain['name']
                        ids=np.arange(domain['start'],domain['start']+len(domain['sizes']))
                    for i in (first,first+1):
                        x=reference.copy();x[ids]=native[i,ids];out.append(x)
                        events.append(dict(event,origin='native_investment_context',scope='investment',family=family))
                    continue
                if self.rng.random()<.5:
                    for i in (first,first+1):
                        x,exchange=self.exchange(reference);out.append(x)
                        events.append(dict(event,origin='common_graph_exchange',native_credit=False,scope='topology',exchange=exchange))
                else:
                    event.update(scope='investment')
                    if self.rng.random()<.2:
                        b=self.codec.blocks['new_capacity'];ids=np.arange(b.start,b.stop)
                        chosen=self.rng.choice(ids,min(4,len(ids)),replace=False)
                        pp,ee=self._apply_pair(reference,[self._spec(chosen,'new_capacity')],event);out.extend(pp);events.extend(ee)
                    else:
                        domain=self.plan_domains[int(self.rng.integers(len(self.plan_domains)))];j=int(self.rng.integers(len(domain['sizes'])))
                        index=domain['start']+j;size=domain['sizes'][j];old=min(size-1,int(reference[index]*size))
                        jump=int(self.rng.integers(1,size)) if size>1 else 0
                        for sign in (-1,1):
                            x=reference.copy();x[index]=((old+sign*jump)%size+.5)/size;out.append(x)
                            events.append(dict(event,origin='common_category',native_credit=False,family=domain['name'],index=index,sign=sign))
        return np.array(out),events

    def state(self):
        return dict(version=1,rng=copy.deepcopy(self.rng.bit_generator.state),pairs=self.pairs,
                    case_deck=self.case_deck.copy(),family_deck=self.family_deck.copy(),temporal_deck=self.temporal_deck.copy(),common_only=self.common_only)

    @classmethod
    def restore(cls,codec,state):
        if state['version']!=1:raise ValueError('Unsupported state variation checkpoint')
        obj=cls(codec,0,state['common_only']);obj.rng.bit_generator.state=copy.deepcopy(state['rng'])
        for name in ('pairs','case_deck','family_deck','temporal_deck'):setattr(obj,name,copy.deepcopy(state[name]))
        return obj
