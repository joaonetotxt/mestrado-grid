"""Diagnostic allocation and bounded representations for native proposals.

No optimizer, dispatch repair, preferred site, extra objective or relaxed
constraint is introduced here. Only Engine generates offspring coordinates.
Symmetric local windows, complete-domain visits and fair coverage coexist.
The adaptive allocation is an explicit algorithmic bias towards measured
economic opportunity, not a prior choice of topology or technology.
"""
import copy
from collections import Counter
import numpy as np

from .native_blocks import Block, Field, endpoint_value, endpoint_key
from .balanced_blocks import BalancedScheduler
from .topology import repair_trees, orient_tree


class WindowBlock(Block):
    """Local normalized windows with exact inverse for direct coordinates.

Grouped fields use a common displacement fraction. At radius 1 both original
endpoints remain reachable. Non-endpoint (categorical/priority) fields retain
their original full-domain representation. The engine sees the actual local
coordinates; there is no unreported clipping of its proposals after tell().
    """
    def __init__(self, reference, fields, metadata, mass=.25, radius=.02):
        if not np.isfinite(radius) or not 0 < radius <= 1:
            raise ValueError('Window radius must be in (0,1]')
        self.radius=float(radius)
        super().__init__(reference, fields, dict(metadata, radius=self.radius), mass)

    def field_radius(self,f):
        # These exact names identify the selected asset's own controls, before
        # coupled_operation adds separately named stage/technology fields.
        # Capacity is an absolute normalized increment; dispatch/SOC/tap are
        # utilization coordinates. Applying the same tiny window to both
        # suppresses the benefit of activating a new asset (P = capacity * use).
        if self.metadata.get('scope')=='investment_joint' and f.name in (
                'new_p','phys_new_q','soc','phys_bess_q','existing_q','phys_cap_state','phys_tap'):
            return 1.
        return self.radius

    def describe(self):
        value=super().describe()
        value['effective_field_radii']={f.name:self.field_radius(f) for f in self.fields if f.endpoint}
        return value

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):
            raise ValueError('Invalid local genotype')
        out=self.reference.copy()
        for f,s in zip(self.fields,self.slices):
            if not f.endpoint:
                out[f.indices]=z[s];continue
            step=2*endpoint_value(z[s],self.mass)-1
            a=self.reference[f.indices];radius=self.field_radius(f)
            out[f.indices]=a+np.where(step<0,step*np.minimum(a,radius),step*np.minimum(1-a,radius))
        return np.clip(out,0,1)

    def project(self,full):
        full=np.asarray(full,float);out=np.empty(self.dimension)
        for f,s in zip(self.fields,self.slices):
            if not f.endpoint:
                out[s]=full[f.indices];continue
            a=self.reference[f.indices];delta=full[f.indices]-a;radius=self.field_radius(f)
            if f.kind=='relative':
                candidates=[]
                for denom,sign in ((np.minimum(a,radius),-1),(np.minimum(1-a,radius),1)):
                    norm=float(denom@denom)
                    mag=0. if norm==0 else np.clip(float(delta@(sign*denom))/norm,0,1)
                    candidates.append((float(np.sum((a+sign*mag*denom-full[f.indices])**2)),(1+sign*mag)/2))
                value=np.array([min(candidates)[1]])
            else:
                denom=np.where(delta<0,np.minimum(a,radius),np.minimum(1-a,radius))
                step=np.divide(delta,denom,out=np.zeros_like(delta),where=denom>0)
                value=(1+np.clip(step,-1,1))/2
            out[s]=endpoint_key(value,self.mass)
        return out


def cycle_path(data,tree,edge):
    lines={e['id']:e for e in data['lines']};line=lines[edge]
    adjacency={b:[] for b in data['buses']}
    for e in tree:
        row=lines[e];adjacency[row['i']].append((row['j'],e));adjacency[row['j']].append((row['i'],e))
    queue=[line['i']];parent={line['i']:None}
    for node in queue:
        if node==line['j']:break
        for other,e in adjacency[node]:
            if other not in parent:parent[other]=(node,e);queue.append(other)
    node=line['j'];path=[]
    while node!=line['i']:
        node,e=parent[node];path.append(e)
    return sorted(path)


class TreeExchangeBlock:
    """Native categorical keys select close/open edges on a fundamental cycle.

Every non-tree catalog edge and every cycle edge are selectable, including
unbuilt lines. The existing chronological tree repair charges any new purchase.
The first closing category is a no-op. Other fields are bounded native
operational coordinates. This guarantees radiality only, never AC feasibility.
    """
    def __init__(self,codec,reference,fields,metadata,mass=.25,radius=.02):
        self.codec=codec;self.reference=np.asarray(reference).copy();self.mass=mass
        self.metadata=dict(metadata,representation='native_fundamental_cycle_exchange',radius=radius)
        self.stage=int(metadata['stage']);self.plan_slice=codec.blocks['plan']
        self.plan_indices=np.arange(self.plan_slice.start,self.plan_slice.stop)
        self.fields=[Field('tree_exchange',self.plan_indices,'relative',False)]+list(fields)
        self.operation=WindowBlock(reference,fields,{},mass,radius) if fields else None
        self.dimension=2+(self.operation.dimension if self.operation is not None else 0)
        self.categorical_indices=[0,1]
        self.metadata['categorical_key_indices']=[0,1]
        self.anchor_plan=codec.plan_codec.decode(self.reference[self.plan_indices])[0]
        self.edges=sorted(e['id'] for e in codec.study.data['lines'] if e['id'] not in self.anchor_plan.trees[self.stage])
        self.paths={e:cycle_path(codec.study.data,self.anchor_plan.trees[self.stage],e) for e in self.edges}

    @staticmethod
    def category(value,n):return min(int(float(value)*n),n-1)

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(self.dimension,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):
            raise ValueError('Invalid local genotype')
        out=self.reference.copy()
        if self.operation is not None:
            self.operation.reference=self.reference.copy();out=self.operation.decode(z[2:])
        selected=self.category(z[0],len(self.edges)+1)
        if selected:
            edge=self.edges[selected-1];path=self.paths[edge]
            drop=path[self.category(z[1],len(path))]
            trees=copy.deepcopy(self.anchor_plan.trees)
            trees[self.stage]=[e for e in trees[self.stage] if e!=drop]+[edge]
            plan=repair_trees(self.codec.study.data,self.anchor_plan,trees)
            orient_tree(self.codec.study.data,plan.trees[self.stage])
            out[self.plan_indices]=self.codec.plan_codec.encode(plan)
        return out

    def project(self,full):
        op=np.empty(0)
        if self.operation is not None:
            self.operation.reference=self.reference.copy();op=self.operation.project(full)
        out=np.r_[.5/(len(self.edges)+1),.5,op]
        plan=self.codec.plan_codec.decode(np.asarray(full)[self.plan_indices])[0]
        before=set(self.anchor_plan.trees[self.stage]);after=set(plan.trees[self.stage])
        added=after-before;removed=before-after
        if len(added)==len(removed)==1:
            edge=next(iter(added));drop=next(iter(removed))
            if edge in self.paths and drop in self.paths[edge]:
                out[0]=(self.edges.index(edge)+1.5)/(len(self.edges)+1)
                out[1]=(self.paths[edge].index(drop)+.5)/len(self.paths[edge])
        # Projected plans are always evaluated anew; their fitness is not borrowed.
        return out

    def describe(self):
        return dict(**self.metadata,dimension=self.dimension,physical_coordinates=sum(len(f.indices) for f in self.fields),
            endpoint_mass=self.mass,closing_choices=len(self.edges),
            fields=[dict(name=f.name,kind=f.kind,coordinates=len(f.indices),endpoint=f.endpoint) for f in self.fields])


class GuidedScheduler:
    """Two uniform and two guided service visits per eight-slot cycle.

Two complements cycle all original domains. One oldest-context revisit and
one reward/novelty revisit preserve learning. Uniform slots cannot be captured
by guidance. Cost signals rank states, never prescribe native decision values.
    """
    def __init__(self,codec,seed,mass,options):
        self.codec=codec;self.base=BalancedScheduler(codec,seed,mass);self.options=options
        self.step=self.uniform=self.complement=self.revisit=0
        self.case_visits=Counter();self.statistics={};self.feedback={};self.node_decks={}
        self.case_failures={}
        self.coverage_preflight=False

    def update(self,record):
        self.feedback={}
        if record.get('status')!='feasible':return
        from .economics import weights
        from .financial_policy import POLICY, resolve
        st=self.codec.study;w=weights(st)
        if resolve(st.c)==POLICY:w=w/w.sum()
        beta=st.c['beta'];alpha=st.c['alpha']
        finance=record['metrics']['finance']
        tail={row['state_id']:row['tail_mass'] for row in finance['tail']}
        for row in record.get('case_feedback',[]):
            k,s=int(row['stage']),int(row['state_index'])
            weight=st.c['annual_hours']*w[k]*((1-beta)*st.probabilities[s]+beta*tail.get(s,0.)/(1-alpha))
            self.feedback[k,s]=dict(row,economic_cut=float(weight*max(0.,row['hourly_cut'])),
                economic_curtailment=float(weight*max(0.,row['hourly_curtailment'])),
                economic_opportunity=float(weight*max(0.,row['hourly_cut']+row['hourly_curtailment'])))

    def observe(self,context,record,previous,origin):
        stats=self.statistics.setdefault(context.number,dict(calls=0,native_calls=0,rejections=0,gain=0.,violations=Counter()))
        stats['calls']+=1
        if origin!='native_offspring':return
        stats['native_calls']+=1;stats['rejections']+=int(record['status']!='feasible')
        stats['violations'].update(k for k,v in (record.get('normalized_violations') or {}).items() if v>0)
        cases={tuple(c) for group in context.block.metadata.get('case_groups',[]) for c in group}
        for row in record.get('case_feedback',[]):
            case=(row['stage'],row['state_index'])
            if case not in cases:continue
            d=self.case_failures.setdefault(case,Counter())
            d['trials']+=1;d['rejected']+=int(not row['passed'])
            d.update(row['violations'].keys())
            if not row['converged']:d['nonconvergence']+=1
        if previous and previous['status']==record['status']=='feasible':
            a=previous['metrics']['selection_objective'];b=record['metrics']['selection_objective']
            stats['gain']+=max(0.,a-b)/max(1.,abs(a))

    def subset(self,key,ids,count):
        # Exhaust each seed-shuffled deck before repeating an independent node.
        selected=[]
        while len(selected)<count:
            if not self.node_decks.get(key):self.node_decks[key]=self.base.rng.permutation(ids).tolist()
            item=self.node_decks[key].pop()
            if item not in selected:selected.append(item)
        return np.array(sorted(selected),int)

    def service(self,reference,case,reason):
        count=self.case_visits[case];self.case_visits[case]+=1
        width=(1,4,16)[count%3];radius=self.options.radii[count%len(self.options.radii)]
        face=('cut','p_cut','pq_cut')[count%3]
        if reason=='diagnostic_curtailment' and face=='cut':face='p_cut'
        failure=self.case_failures.get(case,{})
        diagnostic=self.options.strategy in ('diagnostic','physical') and failure.get('rejected',0)>0
        if diagnostic:
            if any(failure.get(n,0) for n in ('Q','voltage','branch_I','branch_S','SUB_S','nonconvergence')):
                face='pq_cut'
            elif failure.get('P',0):face='p_cut'
            # Keep the periodic radius 1 visit; only local windows shrink.
            if radius<1 and failure.get('rejected',0)/max(1,failure.get('trials',0))>.5:
                radius=max(.001,radius/4)
        fields=[]
        for field in self.base.joint_fields([case]):
            family=field.name.split(':')[0]
            if face=='cut' and family!='cut':continue
            if face=='p_cut' and family not in ('cut','existing_p','new_p'):continue
            ids=field.indices
            if not len(ids):continue
            chosen=self.subset((case,field.name),ids,min(width if family=='cut' else max(1,width//2),len(ids)))
            fields.append(Field(field.name+':individual',chosen,'direct'))
            remainder=np.setdiff1d(ids,chosen)
            if len(remainder):fields.append(Field(field.name+':group',remainder))
        # Controls enter recurring Q faces regardless of whether the prior AC
        # happened to reject a candidate. Thus no option depends on a failure.
        if face=='pq_cut':
            for family in ('phys_tap','phys_cap_state'):
                ids=self.base.ids(family,[case])
                if len(ids):fields.append(Field(family,ids,'direct'))
        return WindowBlock(reference,fields,dict(scope='service_joint',scheduler=self.options.strategy,
            allocation=reason,case_groups=[[case]],node_width=width,service_face=face,
            case_visit=count,rejection_diagnostics=dict(failure),
            economic_signal=self.feedback.get(case,{}).get('economic_opportunity',0.)),self.base.mass,radius)

    def choose(self,reference,contexts):
        slot=self.step%8;self.step+=1
        if slot in (2,6) and contexts:
            if slot==2 or self.options.strategy=='structured':
                chosen=contexts[self.revisit%len(contexts)];self.revisit+=1
            else:
                # Normalize observed gain rates before adding an exploration
                # bonus. Otherwise raw relative gains (often 1e-6 per call)
                # are overwhelmed by the bonus and successes are ignored.
                rates=[self.statistics.get(c.number,{}).get('gain',0.)/max(1,self.statistics.get(c.number,{}).get('native_calls',0)) for c in contexts]
                scale=max(max(rates),1e-15)
                def score(c):
                    d=self.statistics.get(c.number,{})
                    n=d.get('native_calls',0)
                    return d.get('gain',0.)/max(1,n)/scale+.25*np.sqrt(np.log(self.step+1)/(1+n))
                chosen=max(contexts,key=score)
            return chosen,None
        if slot in (3,7):
            scopes=('topology','investment_joint','temporal_cycle','full_native','state_direct','state_family','coarse_joint')
            scope=scopes[self.complement%len(scopes)];self.complement+=1
            if scope=='topology':
                k=self.base.draw('exchange_stage',self.codec.study.K);fields=[]
                cycle=self.base.coupled_operation(fields,[k])
                return None,TreeExchangeBlock(self.codec,reference,fields,dict(scope='topology',stage=k,coupled_cycle=cycle),self.base.mass,self.options.radii[0])
            if scope=='full_native':
                return None,Block(reference,[Field('all_eligible_coordinates',np.arange(self.codec.dimension),'direct',False)],dict(scope=scope),self.base.mass)
            self.base.count=3;self.base.complement_visits=self.base.COMPLEMENTS.index(scope)
            block=self.base.next(reference)
            return None,WindowBlock(reference,block.fields,block.metadata,self.base.mass,self.options.radii[0])
        if self.coverage_preflight and slot in (1,5):
            # Reserve adaptive slots without claiming uniform coverage for
            # them. Actual feedback may concentrate on one expensive state.
            return None,self.service(reference,(0,0),'reserved_adaptive_slot')
        guided=slot in (1,5) and self.options.strategy in ('diagnostic','physical') and self.feedback
        if guided:
            # Reward opportunity per visit, using the same expected/CVaR weights
            # as the configured objective. No costs from infeasible candidates.
            signal='economic_cut' if slot==1 else 'economic_curtailment'
            if not any(row.get(signal,0.)>0 for row in self.feedback.values()):signal='economic_opportunity'
            case=max(self.feedback,key=lambda c:self.feedback[c].get(signal,0.)/(1+self.case_visits[c]))
            reason={'economic_cut':'diagnostic_cut','economic_curtailment':'diagnostic_curtailment',
                    'economic_opportunity':'economic_diagnostic'}[signal]
        else:
            index=self.base.case_order[self.uniform%len(self.base.case_order)];self.uniform+=1
            case=divmod(index,self.codec.study.S);reason='uniform_coverage'
        return None,self.service(reference,case,reason)

    def report(self):
        return dict(case_visits={str(k):v for k,v in self.case_visits.items()},
            uniform_visits=self.uniform,context_statistics=self.statistics,
            note='Native gain credits are immediate origin, not causal isolation. Economic signals allocate effort; rank unchanged.')
