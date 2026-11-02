"""Symmetric domain coverage with neutral subdivision of mixed-feasibility scopes.

First common visit to EACH operational family tests BOTH bounds across all
states. If a proposal mixes passed/failed states, its scope is bisected in its
existing random order; both children and both directions remain eligible.
No bus, technology, cost, flow or preferred critical state chooses the split.
"""
import copy
from .state_variation import StateVariation

class HierarchicalStateVariation(StateVariation):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.root_visits={name:0 for name in self.case_names};self.refinements=[]
        self.refined_pairs=set();self.common_slots=0

    def _rebuild(self,reference,event):
        specs=[];cases=event['cases']
        for move in event['moves']:
            group=move['group']
            if group=='supply':ids=self.indices(['existing_p','new_p'],cases,event['technologies'])
            elif group=='Q':ids=self.indices([n for n in self.case_names if n in ('existing_q','phys_new_q','phys_bess_q')],cases)
            else:ids=self.indices([group],cases)
            specs.append((ids,move['sign'],move['amplitude'],move['mode'],group))
        clean={k:v for k,v in event.items() if k not in ('moves','opposite','native_credit')}
        return self._apply_pair(reference,specs,clean)

    def propose(self,native,reference):
        proposals,events=super().propose(native,reference)
        for first in range(0,len(native),2):
            e=events[first]
            if e['scope']!='case' or e['origin']!='common_symmetric':continue
            pair=e['pair'];self.common_slots+=1
            if self.refinements and self.common_slots%2==0:
                event=self.refinements.pop(0);event['pair']=pair
                event['refinement']=True
            else:
                event=copy.deepcopy(e);event['depth']=0;event['refinement']=False
                missing=[name for name,count in self.root_visits.items() if count==0]
                if missing:
                    name=str(self.rng.choice(missing))
                    event.update(mode='single_family',family=name,cases=self._cases(self.st.K*self.st.S),coverage_root=True,
                                 moves=[dict(group=name,sign=int(self.rng.choice([-1,1])),amplitude=1.,mode='bound')])
                if event.get('mode')=='single_family':
                    name=event['family']
                    if self.root_visits[name]==0:
                        event['cases']=self._cases(self.st.K*self.st.S)
                        for m in event['moves']:m['amplitude']=1.;m['mode']='bound'
                        event['coverage_root']=True
                    self.root_visits[name]+=1
            p,ee=self._rebuild(reference,event)
            proposals[first:first+2]=p;events[first:first+2]=ee
        return proposals,events

    def observe(self,event,result):
        if event.get('scope')!='case' or event.get('origin')!='common_symmetric':return
        cases=[tuple(c) for c in event['cases']];pair=event['pair']
        if len(cases)<2 or pair in self.refined_pairs:return
        passed=result.get('case_passed')
        if passed is None or len(passed)!=self.st.K*self.st.S:return
        flags=[passed[k*self.st.S+s] for k,s in cases]
        if not any(flags) or all(flags):return
        self.refined_pairs.add(pair)
        middle=len(cases)//2
        children=[cases[:middle],cases[middle:]]
        # Preserve the preexisting random case order, not a success-based order.
        for group in children:
            child=copy.deepcopy(event);child['cases']=[list(c) for c in group];child['parent_pair']=pair
            child['depth']=event.get('depth',0)+1
            self.refinements.append(child)

    def state(self):
        return dict(**super().state(),hierarchy_version=1,root_visits=self.root_visits.copy(),
                    refinements=copy.deepcopy(self.refinements),refined_pairs=sorted(self.refined_pairs),common_slots=self.common_slots)

    @classmethod
    def restore(cls,codec,state):
        if state.get('hierarchy_version')!=1:raise ValueError('Unsupported hierarchy checkpoint')
        obj=cls(codec,0,state['common_only']);base=StateVariation.restore(codec,state)
        obj.__dict__.update(base.__dict__)
        for name in ('root_visits','refinements','common_slots'):setattr(obj,name,copy.deepcopy(state[name]))
        obj.refined_pairs=set(state['refined_pairs'])
        return obj
