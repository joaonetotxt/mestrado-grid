"""Opt-in native scheduling and representation, never a dispatch optimizer.

After feasibility, eight protected slots: operation, asset, productive revisit,
original broad, operation, tree/operation, local P/Q, complete temporal cycle.
The original scheduler is preserved before feasibility. Two tree exchanges
with cable choice require an explicit experimental opt-in. Every slot
generates coordinates through the unmodified Engine. The unchanged evaluator
and comparator remain authoritative for every candidate, at every beta.
"""
import copy
from collections import Counter
import numpy as np
from .adaptive_case_pq import AdaptiveCasePQScheduler
from .consolidated_search import CoordinatedBlock, ConsolidatedScheduler
from .guided_blocks import TreeExchangeBlock, WindowBlock
from .native_blocks import Field
from .slack_exchange import SlackExchangeBlock
from .topology import reliability


def coverage_bound(codec,population,options):
    """Worst-case fixed-slot bound, independent of objective and revisit gains.

    Slot 4 always opens operation, at least every second operation is uniform,
    and width never drops below one: <=16 visits per new protected case.
    Slot 1 cycles every investment family. Slot 7 cycles SOC, DR and EV.
    No candidate construction or physics is needed for this contract.
    """
    families=[d['name'] for d in codec.categorical_spec() if d['name'] not in ('tap','cap_state')]
    if codec.study.N:families.append('new_capacity')
    visits=max(16*codec.study.K*codec.study.S+16,8*len(families)+8,24)
    calls=visits*(3*population+2+population*(options.generations_per_visit+options.continuation_generations))
    report=dict(applicable=True,conservative_calls=calls,visits=visits,
        cases=codec.study.K*codec.study.S,investment_families=sorted(families),
        scope='Protected case sweep and one site per investment family; not all sites/combinations',
        proof='Fixed 8-slot cycle; one protected case at most every 16 slots; asset every 8 slots',
        auxiliary_evaluations=0)
    if not options.risk_native_reliability:
        from dataclasses import replace
        from .persistent_search import exploration_preflight
        original=exploration_preflight(codec,population,replace(options,risk_native_acceleration=False,risk_native_reliability=False))
        # Before a transition either the original sweep completes, or the
        # remaining new sweep needs at most its own independent bound.
        report['conservative_calls']+=original['conservative_calls']
        report['visits']+=original['visits']
        report['phase_bounds']=dict(original=original,new_economic_calls=calls)
        report['proof']='Sum of protected coverage bounds before and after first feasibility; no feasibility time limit'
    return report


class JointTreeCableBlock:
    """Two native cycle exchanges followed by an optional native cable choice.

No edge is scored. The second cycle is built from the first resulting tree.
The old decoder alone handles chronology, radiality and purchase accounting.
Projection is approximate and ALWAYS re-evaluated; never reuse old fitness.
"""
    def __init__(self,codec,base,reference,stage,mass=.25,radius=.02):
        self.codec=codec; self.base=base; self.reference=np.asarray(reference).copy()
        self.stage=stage; self.mass=mass; self.radius=radius; self.dimension=7
        self.fields=[Field('joint_tree_cable',np.arange(codec.blocks['plan'].start,codec.blocks['plan'].stop))]
        self.categorical_indices=list(range(7))
        self.metadata=dict(scope='native_two_tree_exchanges_cable',stage=stage,
            exchanges=2,optional_cable=True,all_edges_eligible=True,
            auxiliary_optimizer=False,pre_scored_neighbors=0)

    def _tree(self,reference):
        return TreeExchangeBlock(self.codec,reference,[],dict(stage=self.stage),self.mass,self.radius)

    def decode(self,z):
        z=np.asarray(z,float)
        if z.shape!=(7,) or not np.isfinite(z).all() or np.any((z<0)|(z>1)):
            raise ValueError('Invalid native tree/cable keys')
        out=self.reference.copy()
        for offset in (0,2):
            block=self._tree(out)
            out=block.decode(z[offset:offset+2])
        if z[4]>=.5:
            d=next(d for d in self.base.categories if d['name']=='line_schedule')
            j=min(int(z[5]*len(d['sizes'])),len(d['sizes'])-1)
            out[d['start']+j]=z[6]
        return out

    def project(self,full):
        first=self._tree(self.reference)
        z1=first.project(full); intermediate=first.decode(z1)
        second=self._tree(intermediate); z2=second.project(full)
        z=np.r_[z1,z2,.25,.5,.5]
        d=next(d for d in self.base.categories if d['name']=='line_schedule')
        ids=np.arange(d['start'],d['start']+len(d['sizes']))
        changed=np.flatnonzero(np.asarray(full)[ids]!=intermediate[ids])
        if len(changed):
            j=int(changed[0]); z[4:]=[.75,(j+.5)/len(ids),full[ids[j]]]
        return z

    def describe(self):
        return dict(self.metadata,dimension=self.dimension,
            physical_coordinates=len(self.fields[0].indices),projection='approximate, charged reevaluation',
            fields=[dict(name='joint_tree_cable',kind='native_categorical',coordinates=len(self.fields[0].indices))])


class RiskNativeScheduler(AdaptiveCasePQScheduler):
    """Protected coverage plus recent native improvement per fresh AC case.

Timing is telemetry only, so machine contention never changes scheduling.
No information, individuals or fitness is shared across engines/beta classes.
"""
    def __init__(self,*args):
        super().__init__(*args)
        self.risk_step=0; self.operation_visits=0; self.asset_visits=0
        self.original_feasibility_visits=0
        self.temporal_visits=0; self.group_width=1; self.local_radius=.02
        self.current_stats={}; self.recent={}; self.family_visits=Counter()
        self.protected_case_seen=set(); self.protected_sweeps=0
        self.case_native_gain=Counter(); self.group_events=[]
        self.risk_families=[d['name'] for d in self.base.categories]+(['new_capacity'] if self.codec.study.N else [])

    def _tier(self):
        return (self.record or {}).get('status')

    def _worst_stage(self,reference):
        st=self.codec.study; c=self.codec.decode(reference)[0]
        if c.plan_id not in self.stage_cache:
            rows=reliability(st.data,c.plan); limits=st.data['config']
            def v(row):
                k=-1 if st.data['investment_mode']=='single' else row['stage']
                return sum(max(0.,row[n]-limits[n+'_limits'][k])/max(1e-12,limits[n+'_limits'][k]) for n in ('SAIDI','SAIFI'))
            self.stage_cache={c.plan_id:max(rows,key=v)['stage']}
        return self.stage_cache[c.plan_id]

    def _case_group(self,protected=False):
        st=self.codec.study; domain=[divmod(i,st.S) for i in range(st.K*st.S)]
        width=min(self.group_width,len(domain)); selected=[]
        if protected:
            while len(selected)<width:
                if len(self.protected_case_seen)==len(domain):
                    self.protected_case_seen.clear(); self.protected_sweeps+=1
                case=divmod(self.base.draw('risk_protected_cases',len(domain)),st.S)
                if case not in self.protected_case_seen:
                    self.protected_case_seen.add(case)
                    if case not in selected:selected.append(case)
        else:
            # Existing signal already includes beta, alpha, probability and
            # partial tail mass. No fixed list of "extreme" state indices.
            def score(c):
                opportunity=self.feedback.get(c,{}).get('economic_opportunity',0.)
                return (opportunity/(1+self.case_visits[c]),self.case_native_gain[c],-c[0],-c[1])
            selected=sorted(domain,key=score,reverse=True)[:width]
        for c in selected:self.case_visits[c]+=1
        return selected

    def operation(self,reference):
        protected=self.operation_visits%2==1
        self.operation_visits+=1
        cases=self._case_group(protected)
        radius=1. if self.operation_visits%12==0 else self.local_radius
        # Separate coordinates per case avoid forcing identical actions on
        # heterogeneous operating states. Wide visits remain protected below.
        b=SlackExchangeBlock(CoordinatedBlock(self.codec,self.base,reference,
            [[c] for c in cases],self.base.mass,radius))
        b.metadata.update(risk_role='small_operation',group_width=len(cases),
            allocation='protected_case_coverage' if protected else 'dynamic_CVaR_cost_opportunity',
            acceptance='unchanged full objective and constraints')
        return b

    def asset(self,reference):
        st=self.codec.study; family=self.risk_families[self.asset_visits%len(self.risk_families)]
        self.asset_visits+=1; self.family_visits[family]+=1; fields=[]
        if family=='new_capacity':
            j=self.base.draw('risk_capacity_site',st.N)
            ids=self.base.maps['new_capacity'][:,j]; fields=[Field('new_capacity',ids[ids>=0],'direct')]
            for name in ('new_p','phys_new_q'):
                ids=self.base.maps[name][:,:,j].ravel(); fields.append(Field(name,ids[ids>=0]))
        else:
            d=next(d for d in self.base.categories if d['name']==family)
            j=self.base.draw('risk_asset_'+family,len(d['sizes']))
            fields=[Field(family,np.array([d['start']+j]),'direct',False)]
            if family=='storage_purchase':
                site=self.codec.storage_sites.index(self.codec.plan_codec.sites[j])
                for name in ('soc','phys_bess_q'):
                    ids=(self.base.maps[name][:,site,:] if name=='soc' else self.base.maps[name][:,:,site]).ravel()
                    fields.append(Field(name,ids[ids>=0],'direct' if name=='soc' else 'relative'))
            elif family in ('cap_choice','reg_choice','retro_choice','bess_choice'):
                name={'cap_choice':'phys_cap_state','reg_choice':'phys_tap','retro_choice':'existing_q','bess_choice':'phys_bess_q'}[family]
                site=int(self.codec.unit_site[j]) if family=='bess_choice' else j
                ids=self.base.maps[name][:,:,site].ravel(); fields.append(Field(name,ids[ids>=0]))
                if family=='bess_choice':
                    ids=self.base.maps['soc'][:,site,:].ravel(); fields.append(Field('soc',ids[ids>=0],'direct'))
        cycle=self.base.coupled_operation(fields)
        b=WindowBlock(reference,fields,dict(scope='investment_joint',family=family,
            site_index=j,coupled_cycle=cycle,risk_role='mandatory_asset',
            complete_SOC_cycles=True,auxiliary_optimizer=False),self.base.mass,self.local_radius)
        return b

    def temporal(self,reference):
        st=self.codec.study; family=('soc','conventional','ev')[self.temporal_visits%3]
        self.temporal_visits+=1; self.family_visits[family]+=1
        k,day=divmod(self.base.draw('risk_temporal_'+family,st.K*len(st.cycles)),len(st.cycles))
        ids=self.base.maps[family][k][:,st.cycles[day]].ravel()
        fields=[Field(family,ids[ids>=0],'direct')]
        # Keep every state of this day, and all coupled operating fields.
        cases=[(k,s) for s in st.cycles[day]]
        fields.extend(self.base.joint_fields(cases))
        b=WindowBlock(reference,fields,dict(scope='temporal_cycle',family=family,
            stage=k,cycle=day,case_groups=[cases],risk_role='mandatory_temporal',
            complete_SOC_cycles=True,auxiliary_optimizer=False),self.base.mass,self.local_radius)
        return b

    def tree(self,reference):
        st=self.codec.study
        constrained=any((self.record or {}).get('normalized_violations',{}).get(n,0)>0 for n in ('SAIDI','SAIFI'))
        stage=self._worst_stage(reference) if constrained else self.base.draw('risk_tree_stage',st.K)
        if self.options.risk_native_reliability:
            asset=JointTreeCableBlock(self.codec,self.base,reference,stage,self.base.mass,self.local_radius)
        else:
            asset=TreeExchangeBlock(self.codec,reference,[],dict(stage=stage,scope='native_single_tree',experimental=False),self.base.mass,self.local_radius)
        cases=[(stage,s) for s in st.cycles[self.base.draw('risk_tree_cycle',len(st.cycles))]]
        b=SlackExchangeBlock(CoordinatedBlock(self.codec,self.base,reference,[cases],self.base.mass,self.local_radius,asset))
        b.metadata.update(risk_role='joint_reliability' if self.options.risk_native_reliability else 'protected_tree_operation',allocation='worst_reliability_stage' if constrained else 'protected_tree_stage')
        self.family_visits['topology']+=1
        return b

    def choose(self,reference,contexts):
        if self.coverage_preflight:return super().choose(reference,contexts)
        if self._tier()!='feasible' and not self.options.risk_native_reliability:
            self.original_feasibility_visits+=1
            return super().choose(reference,contexts)
        self.pq_generation_budget=None
        slot=self.risk_step%8; self.risk_step+=1
        if slot==0:
            constrained=any((self.record or {}).get('normalized_violations',{}).get(n,0)>0 for n in ('SAIDI','SAIFI'))
            b=self.tree(reference) if constrained else self.operation(reference)
        elif slot==1:b=self.asset(reference)
        elif slot==2:
            eligible=[(n,r) for n,r in self.recent.items() if r['tier']==self._tier() and r['gain']>0 and self.risk_step-r['step']<=16]
            if eligible:
                n,_=max(eligible,key=lambda x:(x[1]['gain']/max(1,x[1]['new_AC']),-x[0]))
                return contexts[n],None
            b=self.operation(reference)
        elif slot==3:
            old,b=ConsolidatedScheduler.choose(self,reference,contexts)
            if old is not None:return old,None
            b.metadata['risk_role']='protected_original_broad'
        elif slot==4:b=self.operation(reference)
        elif slot==5:b=self.tree(reference)
        elif slot==6 and self._tier()=='feasible':
            old,b=self.choose_pq(reference,contexts)
            if old is not None:return old,None
            b.metadata['risk_role']='protected_local_PQ'
        else:b=self.temporal(reference)
        b.metadata.update(risk_scheduler='native_risk_v1',scheduler_slot=slot,
            protected_cycle_length=8,auxiliary_optimizer=False)
        return None,b

    def observe_charge(self,context,origin,new_ac):
        s=self.current_stats.setdefault(context.number,Counter())
        s['new_AC']+=new_ac; s['all_calls']+=1

    def observe(self,context,record,previous,origin):
        super().observe(context,record,previous,origin)
        if origin!='native_offspring':return
        s=self.current_stats.setdefault(context.number,Counter())
        s['native_calls']+=1; s['feasible']+=record.get('status')=='feasible'
        gain=0.
        if previous and previous['status']==record['status']=='feasible':
            a=previous['metrics']['selection_objective']; b=record['metrics']['selection_objective']
            gain=max(0.,a-b)/max(1.,abs(a))
        elif previous and previous['status']==record['status']=='violated':
            a=previous['normalized_violation']; b=record['normalized_violation']
            gain=max(0.,a-b)/max(1.,abs(a))
        s['gain']+=gain
        for group in context.block.metadata.get('case_groups',[]):
            for c in group:self.case_native_gain[tuple(c)]+=gain

    def visit_generations(self,context,default):
        if 'risk_role' not in context.block.metadata:
            return super().visit_generations(context,default)
        if context.block.metadata.get('risk_role') in ('small_operation','protected_local_PQ'):
            previous=self.recent.get(context.number,{})
            return min(default,8 if previous.get('gain',0)>0 else 2)
        return default

    def finish_visit(self,context):
        super().finish_visit(context)
        s=dict(self.current_stats.pop(context.number,{}))
        self.recent[context.number]=dict(gain=s.get('gain',0.),new_AC=s.get('new_AC',0),
            tier=self._tier(),step=self.risk_step,native_calls=s.get('native_calls',0),all_calls=s.get('all_calls',0))
        if context.block.metadata.get('risk_role')!='small_operation' or not s.get('native_calls'):return
        before=(self.group_width,self.local_radius)
        if s.get('feasible',0)/s['native_calls']<.5:
            self.group_width=max(1,self.group_width//2);self.local_radius=max(.005,self.local_radius/2)
        elif s.get('gain',0)>0:
            self.group_width=min(8,self.group_width*2);self.local_radius=min(.25,self.local_radius*2)
        else:
            self.local_radius=min(.25,self.local_radius*2)
        self.group_events.append(dict(context=context.number,before=before,
            after=(self.group_width,self.local_radius),evidence=s))

    def report(self):
        out=super().report()
        out.update(risk_scheduler='native_risk_v1',risk_visits=self.risk_step,
            feasibility_policy='experimental_joint_reliability' if self.options.risk_native_reliability else 'unchanged_original_until_feasible',
            original_feasibility_visits=self.original_feasibility_visits,
            mandatory_families=dict(self.family_visits),protected_case_sweeps=self.protected_sweeps,
            protected_cases_current_sweep=len(self.protected_case_seen),group_events=self.group_events,
            recent_native_productivity=self.recent,native_auxiliary_calls=0,
            resource_allocation='recent native rank improvement / charged new AC; protected coverage',
            acceptance='unchanged full evaluator and comparator; current beta and alpha')
        return out
