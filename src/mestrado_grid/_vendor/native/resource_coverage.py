"""Isolated direct resource coverage. No fitness/AC calls or preferred technology.

First a seeded sweep; subsequently one resource visit every two outer visits.
All sites have finite seeded decks. Site opportunities, native physical changes,
operating changes and accepted improvements are separate, auditable quantities.
Engines and the evaluator are unchanged. Unscored stratification is initialization,
never attributed to native offspring and never forces an investment's acceptance.
"""
import copy
from collections import Counter
import numpy as np
from effective_scheduler import EffectiveScheduler,ContextSubset
from meta_planning.initial_resource_search import InitialResourceScheduler,resource_tasks
from meta_planning.native_blocks import Field
from meta_planning.guided_blocks import WindowBlock,TreeExchangeBlock
from meta_planning.consolidated_search import CoordinatedBlock
from meta_planning.balanced_blocks import physical_key
from meta_planning.temporal import storage_dispatch
from meta_planning.schema import fingerprint


def primary(codec,meta,c):
    task=meta['resource_task'];family=task.split(':')[0];j=meta.get('site_index');k=meta.get('stage')
    if family=='storage_purchase':
        bus=codec.plan_codec.sites[j];ids={u['id'] for u in codec.study.units if u['bus']==bus}
        return sorted(tuple(v) for v in c.plan.storage_buys if v[0] in ids)
    if family=='line_schedule':return sorted(tuple(v) for v in c.plan.line_buys if v[0]==codec.plan_codec.lines[j]['id'])
    if family=='new_capacity':return c.new_capacity[:,j]
    if family in ('cap_choice','reg_choice','retro_choice'):return c.physical[family][j]
    if family=='topology':return c.plan.canonical().trees[k]
    if family=='existing_PQ':return [c.operation.existing_p[:,:,j],c.operation.existing_q[:,:,j]]
    name='cut' if task=='service' else task
    return getattr(c.operation,name)[:,j,:]


def effect(codec,meta,reference,c):
    task=meta['resource_task'];family=task.split(':')[0];j=meta.get('site_index')
    old=primary(codec,meta,reference);new=primary(codec,meta,c)
    a=np.asarray(old);b=np.asarray(new)
    if a.shape==b.shape and a.dtype.kind in 'biuf' and b.dtype.kind in 'biuf':
        changed=bool(np.any(np.abs(b-a)>1e-9))
    else:changed=fingerprint(old)!=fingerprint(new)
    operated=changed
    if family=='storage_purchase':
        bus=codec.plan_codec.sites[j];units=[i for i,u in enumerate(codec.study.units) if u['bus']==bus]
        charge,discharge=storage_dispatch(codec.study,c)
        operated=bool(new) and bool(np.any(charge[:,units,:]>1e-10)) and bool(np.any(discharge[:,units,:]>1e-10))
    elif family=='new_capacity':operated=bool(np.any(c.operation.new_p[:,:,j]>1e-10))
    return dict(changed=bool(changed),operated=bool(operated),signature=fingerprint(new))


def resource_initialization(block,codec,sources,n,rng):
    """Anchor + stratified, unscored alternatives of the selected primary decision.

Archives differing only in operation cannot occupy all resource alternatives.
Categorical keys span their full domain; remaining operation stays local. Identity
checks use decoded controls only, with no objective/constraint scoring or repair.
"""
    ref=codec.decode(block.reference)[0];anchor=block.project(block.reference)
    rows=[anchor.copy()];origins=['context_retest'];identities={physical_key(ref)}
    signatures={fingerprint(primary(codec,block.metadata,ref))};prefix=block.prefix
    cats=list(getattr(block,'categorical_indices',[]))
    strata=np.array([(rng.permutation(n-1)+rng.random(n-1))/(n-1) for _ in range(block.dimension)]).T
    for attempt in range(64*(n-1)):
        if len(rows)==n:break
        slot=attempt%(n-1);z=anchor.copy()
        # The enabling asset/control receives full normalized coverage; its
        # declared decoder radius still bounds physical continuous increments.
        z[:prefix]=strata[slot,:prefix] if attempt<n-1 else rng.random(prefix)
        z[prefix:]=np.clip(anchor[prefix:]+rng.uniform(-.08,.08,block.dimension-prefix),0,1)
        if cats:z[cats]=strata[slot,cats] if attempt<n-1 else rng.random(len(cats))
        c=codec.decode(block.decode(z))[0];e=effect(codec,block.metadata,ref,c);identity=physical_key(c)
        if not e['changed'] or identity in identities:continue
        # BESS alternatives must include a real complete charge/discharge cycle.
        if block.metadata['resource_task']=='storage_purchase' and not e['operated']:continue
        identities.add(identity);signatures.add(e['signature']);rows.append(z.copy());origins.append('diversity_initialization')
    if len(rows)<n:raise ValueError('Resource representation lacks decoded diversity: '+block.metadata['resource_task'])
    return np.asarray(rows),origins,dict(physical_unique=len(identities),selected_resource_unique=len(signatures),
        repeated_fallback=0,generated_initial_members=n-1,identity_only_probes=attempt+1,
        unchanged_anchor_preserved=True,auxiliary_evaluations=0)


class ResourceScheduler(InitialResourceScheduler,EffectiveScheduler):
    def __init__(self,*args):
        super().__init__(*args)
        self.resource_cursor=0;self.outer_after_sweep=0;self.resource_visits=[];self.resource_stats={}
        self.resource_totals={t:Counter() for t in self.initial_order};self.resource_sites={t:set() for t in self.initial_order}
        self.resource_changed_sites={t:set() for t in self.initial_order};self.resource_native_signatures={t:set() for t in self.initial_order}
        self.resource_family_attempts=Counter()
        self.resource_contexts={}
        self.resource_seen_candidates={}

    def _choose_original(self,reference,original):
        positions=[n for n in original.positions if not original.contexts[n].block.metadata.get('resource_coverage')]
        return EffectiveScheduler._choose_original(self,reference,ContextSubset(original.contexts,positions))

    def resource_block(self,reference,task):
        st=self.codec.study;family=task.split(':')[0]
        if family in ('line_schedule','storage_purchase','cap_choice','reg_choice','retro_choice','new_capacity'):
            b=self.asset_initial(reference,task)
            radius=(.02,.1,.5,1.)[self.resource_family_attempts[task]%4]
            b.asset.radius=radius;b.asset.metadata['radius']=radius
            # An asset acts in all stages/cases after its declared entry date;
            # keep operations coupled across all representative cycles.
            groups=[[(k,s) for s in cy] for k in range(st.K) for cy in st.cycles]
            b=CoordinatedBlock(self.codec,self.base,reference,groups,self.base.mass,.02,b.asset)
            b.metadata.update(family=family,site_index=b.asset.metadata['site_index'])
        else:
            k=self.draw_initial('stage:'+task,range(st.K));day=self.draw_initial('day:'+task,range(len(st.cycles)))
            cases=[(k,s) for s in st.cycles[day]]
            if family=='topology':
                asset=TreeExchangeBlock(self.codec,reference,[],dict(scope='native_tree',stage=k),self.base.mass,1.)
                asset.categorical_indices=[0,1];j=k
            else:
                if family=='existing_PQ':
                    tech=task.split(':')[1];js=[i for i,g in enumerate(st.old) if g['technology']==tech];j=self.draw_initial('site:'+task,js)
                    fields=[self.field(name,self.base.maps[name][k,st.cycles[day],j],'direct') for name in ('existing_p','existing_q')]
                else:
                    name='cut' if task=='service' else task
                    js=[j for j in range(st.B) if np.any(self.base.maps[name][:,j,:]>=0)];j=self.draw_initial('site:'+task,js)
                    fields=[self.field(name,self.base.maps[name][k,j,st.cycles[day]],'direct')]
                asset=WindowBlock(reference,[f for f in fields if len(f.indices)],dict(scope='resource_cycle',family=task,site_index=j),self.base.mass,.1)
            b=CoordinatedBlock(self.codec,self.base,reference,[cases],self.base.mass,.02,asset)
            b.metadata.update(family=task,site_index=j,stage=k,cycle=day)
        self.resource_family_attempts[task]+=1
        b.metadata.update(resource_coverage=True,resource_task=task,resource_visit=self.resource_cursor,
            risk_role='protected_resource',auxiliary_optimizer=False,
            installation_forced=False,selection='unchanged full evaluator/FO/CVaR/constraints',
            coverage_contract='seeded family/site opportunities; completion requires native selected-resource physical change',
            complete_SOC_cycles=family=='storage_purchase')
        b.metadata['resource_context_key']=(task,b.metadata.get('site_index'),b.metadata.get('stage'),
            b.metadata.get('cycle'),getattr(b.asset,'radius',None))
        return b

    def choose(self,reference,contexts):
        count=len(self.initial_order)
        resource=self.resource_cursor<count or self.outer_after_sweep%2==0
        if self.resource_cursor>=count:self.outer_after_sweep+=1
        if not resource:return EffectiveScheduler.choose(self,reference,contexts)
        task=self.initial_order[self.resource_cursor%count];b=self.resource_block(reference,task);self.resource_cursor+=1
        key=tuple(b.metadata['resource_context_key']);number=self.resource_contexts.get(key)
        if number is not None:
            context=contexts[number]
            if tuple(context.block.metadata['resource_context_key'])!=key:raise ValueError('Resource context identity mismatch')
            # The persistent runner retains native populations/memories and
            # re-evaluates every inherited fitness in the new outside context.
            return context,None
        return None,b

    def observe_candidate(self,context,full,record,origin):
        super().observe_candidate(context,full,record,origin)
        if not context or not context.block.metadata.get('resource_coverage'):return
        meta=context.block.metadata;task=meta['resource_task'];s=self.resource_stats.setdefault(context.number,Counter())
        ref=self.codec.decode(context.block.reference)[0];c=self.codec.decode(full)[0];e=effect(self.codec,meta,ref,c)
        seen=self.resource_seen_candidates.setdefault(context.number,set())
        novel=c.candidate_id not in seen
        seen.add(c.candidate_id)
        s['evaluations']+=1;s['physical_changes']+=e['changed'];s['feasible']+=record['status']=='feasible'
        if origin!='native_offspring':return
        s['native_proposals']+=1;s['native_physical_changes']+=e['changed'];s['native_operating_changes']+=e['changed'] and e['operated']
        s['native_repeated_candidates']+=not novel
        s['native_novel_physical_changes']+=novel and e['changed']
        s['native_novel_operating_changes']+=novel and e['changed'] and e['operated']
        if novel and e['changed']:
            self.resource_changed_sites[task].add(meta.get('site_index'));self.resource_native_signatures[task].add(e['signature'])

    def finish_visit(self,context):
        # Skip the old one-shot completion bookkeeping; retain all effective/risk
        # learning and the original native context memories unchanged.
        EffectiveScheduler.finish_visit(self,context)
        meta=context.block.metadata
        if not meta.get('resource_coverage'):return
        task=meta['resource_task'];s=self.resource_stats.pop(context.number,Counter());self.resource_totals[task].update(s)
        self.resource_contexts[tuple(meta['resource_context_key'])]=context.number
        self.resource_sites[task].add(meta.get('site_index'))
        self.resource_visits.append(dict(task=task,site_index=meta.get('site_index'),counts=dict(s)))

    def visit_generations(self,context,default):
        if context.block.metadata.get('resource_coverage'):return min(2,default)
        return EffectiveScheduler.visit_generations(self,context,default)

    def report(self):
        out=EffectiveScheduler.report(self)
        out['protected_resources']=dict(order=self.initial_order,visits=self.resource_visits,
            initial_opportunity_sweep_complete=self.resource_cursor>=len(self.initial_order),
            native_physical_family_coverage_complete=all(v['native_novel_physical_changes']>0 for v in self.resource_totals.values()),
            families={t:dict(v,sites_offered=sorted(self.resource_sites[t]),sites_changed_natively=sorted(self.resource_changed_sites[t]),
                native_primary_signatures=len(self.resource_native_signatures[t])) for t,v in self.resource_totals.items()},
            recurring_outer_resource_fraction=.5,maximum_resource_visits_between_same_family=len(self.initial_order),
            interpretation='Finite opportunities per enabled site; no guarantee of feasibility, improvement or exhaustive combinations',
            auxiliary_evaluations=0)
        return out
