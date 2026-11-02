"""Experimental per-engine construction; no auxiliary search or historical vector.

The evaluator and monetary ledger remain unchanged. Construction selection is
an explicit extra annotation removed by independent re-evaluation at handoff.
Physical preference: load cut + connected renewable curtailment (MWh/year),
equal weights, after strict feasibility / measured violation. Availability
loss is reported separately and is never called curtailment reduction.
"""
import copy
import time
from collections import Counter
from pathlib import Path
import numpy as np
from . import economics,physical
from .consolidated_search import ConsolidatedScheduler,CoordinatedBlock
from .guided_blocks import TreeExchangeBlock
from .prepared_search import SearchState,write,digest,select_population
from .evaluator import Evaluator,rank
from .search_deadline import SearchDeadlineReached


def energies(st,c):
    w=economics.weights(st);w=st.c['annual_hours']*(w/w.sum())[:,None]*st.probabilities[None,:]
    old=np.array([g['technology'] in ('PV','WT') for g in st.old],bool)
    new=np.array([g['technology'] in ('PV','WT') for g in st.new],bool)
    raw=st.existing_bounds()[1];connected=physical.existing_bounds(st,c)[1]
    av=np.array([[g['availability'][str(s)] for g in st.new] for s in range(st.S)])
    cap=st.active_capacity(c)[:,None,:]
    newraw=cap*av[None,:,:];newconnected=cap*physical.availability(st)[None,:,:]
    curtailed=np.maximum(0,connected[:,:,old]-c.operation.existing_p[:,:,old]).sum(2)
    curtailed+=np.maximum(0,newconnected[:,:,new]-c.operation.new_p[:,:,new]).sum(2)
    unavailable=np.maximum(0,raw[:,:,old]-connected[:,:,old]).sum(2)
    unavailable+=np.maximum(0,newraw[:,:,new]-newconnected[:,:,new]).sum(2)
    cut=c.operation.cut.sum(1)
    return dict(policy='native_initial_energy_v1',load_cut_MWh_year=float(np.sum(w*cut)),
        connected_curtailment_MWh_year=float(np.sum(w*curtailed)),
        renewable_unavailable_MWh_year=float(np.sum(w*unavailable)),
        weighted_case_energy=(w*(cut+curtailed)).tolist(),
        weights=dict(load_cut=1.,connected_curtailment=1.),
        units='MWh/year equivalent, normalized stage VP weights',
        rule='strict feasibility; measured violation before energy; official FO only feasible tie breaker')


def thin_history(record):
    out={k:copy.deepcopy(v) for k,v in record.items() if k not in ('metrics','case_feedback')}
    m=record.get('metrics')
    out['metrics']=None if m is None else {k:copy.deepcopy(v) for k,v in m.items()
        if k not in ('finance','financial_views','EAC')}
    if m is not None:out['metrics']['finance']={k:copy.deepcopy(v) for k,v in m['finance'].items()
        if k in ('objective','PC1','expected_operation','CVaR_total','CVaR_operation')}
    out['history_layout']='compact scalar ledger; full independent final validation retained; no change to in-memory ranking or guidance'
    return out


class ConstructionState(SearchState):
    selection_description='experimental construction: feasibility, normalized violation, then load cut plus connected renewable curtailment; original financial ledger retained'

    def decorate_record(self,record,candidate):
        record['construction_selection']=energies(self.study,candidate)
        return record

    def history_record(self,record):return thin_history(record)

    def evaluate(self,x,event):
        record=super().evaluate(x,event)
        # Exact memoization only. Eviction changes runtime, never scores. AC
        # cases already use an independent bounded exact cache.
        while len(self.cache)>256:self.cache.pop(next(iter(self.cache)))
        if record['status']=='feasible' and not hasattr(self,'first_feasible'):
            self.first_feasible=dict(evaluation=self.evaluations,event=copy.deepcopy(event),
                candidate_id=record['candidate_id'],elapsed_seconds=time.monotonic()-self.start)
        return record


class ConstructionScheduler(ConsolidatedScheduler):
    """Alternate constraint/energy scopes and the complete existing scheduler.

    No candidate is screened, scored or optimized here. Topology edge choices
    and P/Q/service coordinates are selected by the selected native engine.
    Current reliability violations cause more topology visits, never a fixed
    topology/site. Energy scopes prioritize measured current-case unused MWh.
    """
    def __init__(self,*args):
        super().__init__(*args);self.construction_visits=0;self.special_visits=0
        self.record={};self.energy_visits=Counter()

    def update(self,record):
        self.record=record;super().update(record)

    def report(self):
        out=super().report()
        out['note']='Experimental constraint/energy scope allocation; original evaluator and all constraints retained. Construction selection differs explicitly from phase-two financial selection.'
        out['construction_special_visits']=self.special_visits
        out['auxiliary_optimizer']=False
        return out

    def choose(self,reference,contexts):
        self.construction_visits+=1
        if self.construction_visits%2==0:return super().choose(reference,contexts)
        st=self.codec.study;self.special_visits+=1
        radius=self.options.radii[(self.special_visits-1)%len(self.options.radii)]
        violations=self.record.get('normalized_violations',{})
        if any(violations.get(k,0)>0 for k in ('SAIFI','SAIDI')):
            stage=self.protected.draw('construction_tree_stage',st.K)
            asset=TreeExchangeBlock(self.codec,reference,[],dict(scope='construction_tree',stage=stage,
                allocation='measured reliability violation; seeded stage deck'),self.base.mass,radius)
            # Two pure exchanges then one jointly supplied exchange. Pure moves
            # avoid gratuitously perturbing all dispatch controls during repair.
            if self.special_visits%3:return None,asset
            block=CoordinatedBlock(self.codec,self.base,reference,
                [[(stage,s) for s in range(st.S)]],self.base.mass,radius,asset)
        else:
            energy=np.asarray(self.record.get('construction_selection',{}).get('weighted_case_energy',np.zeros((st.K,st.S))))
            # An AC violation takes priority over the energy opportunity.
            bad={(int(r['stage']),int(r['state_index'])):sum(r.get('violations',{}).values())
                 for r in self.record.get('case_feedback',[])}
            cases=[(k,s) for k in range(st.K) for s in range(st.S)]
            case=max(cases,key=lambda c:(bad.get(c,0),energy[c]/(1+self.energy_visits[c])))
            self.energy_visits[case]+=1
            block=CoordinatedBlock(self.codec,self.base,reference,[[case]],self.base.mass,radius)
        block.metadata['allocation']='construction measured constraint/energy; native values; alternating broad eligibility'
        return None,block


def is_feasible(record,expected):
    return (record is not None and record['status']=='feasible'
        and record['approved_cases']==expected and record['expected_cases']==expected
        and record['normalized_violation']==0)


class FirstFeasibleReached(SearchDeadlineReached):pass


def construct(study,method,output,seed=96101,population=8,options=None):
    from .engines import METHODS
    if method not in METHODS:raise ValueError('Unknown motor')
    if type(population) is not int or population<4 or population%2:raise ValueError('Even population >=4 required')
    dst=Path(output)
    state=ConstructionState(study,seed,population,dst,initialization='mat_original')
    expected=study.K*study.S
    def check():
        if is_feasible(state.best,expected):raise FirstFeasibleReached()
    try:
        state.status(phase='fresh_MAT_initialization')
        for i,x in enumerate(state.initial):
            state.evaluate(x,dict(origin='initialization',member=i,method=method,
                component='original_MAT_resource_aware_unscored_initialization'))
            if is_feasible(state.best,expected):break
        state.initial_best=copy.deepcopy(state.best)
        state.check_budget=check
        if not is_feasible(state.best,expected):
            cfg=dict(strategy='native_construction',coverage_contract='partial_pilot',
                generations_per_visit=4,radii=[.02,.1,.5,1.],economic_allocation=False)
            cfg.update(options or {})
            if cfg['strategy']!='native_construction':raise ValueError('Construction strategy must be explicit')
            state.native(method,None,.25,options=cfg)
        assert is_feasible(state.best,expected),'Search ended without an independently auditable feasible seed'
        search_seconds=time.monotonic()-state.start
        if hasattr(state,'persistent_diagnostics'):
            state.persistent_diagnostics['termination_reason']='first_feasible_found'
            write(dst/'persistent_diagnostics.json',state.persistent_diagnostics)
        state.status(phase='independent_handoff_audit')
        pool=[b for b in state.bank if is_feasible(b['record'],expected)]
        pop,records,diversity=select_population(pool,population)
        np.save(dst/'engine_population.npy',pop,allow_pickle=False)
        members=[];audit_started=time.monotonic()
        for i,(x,record) in enumerate(zip(pop,records)):
            c,_=state.codec.decode(x);v=Evaluator(study).evaluate(c)
            assert is_feasible(v,expected) and v['candidate_id']==record['candidate_id']
            assert not any(v['normalized_violations'].values())
            assert abs(v['metrics']['selection_objective']-record['metrics']['selection_objective'])<1e-5
            write(dst/f'population_validation_{i}.json',v)
            members.append(dict(index=i,candidate_id=c.candidate_id,plan_id=c.plan_id,
                status=v['status'],approved_cases=v['approved_cases'],normalized_violation=v['normalized_violation']))
        hand=dict(verified=True,method=method,study_id=study.input_id,population_sha256=digest(pop),
            members=members,diversity=diversity,first_feasible=state.first_feasible,
            selection_before='native_initial_energy_v1',selection_after='original Castro-aligned FO',
            auxiliary_optimizer=False,historical_candidate=False,
            reset_engine_fitness_and_memories=True,unselected_archive_transferred=False,
            search_seconds=search_seconds,audit_seconds=time.monotonic()-audit_started,
            search_proposals=state.evaluations,new_search_AC_cases=state.ev.case_misses,
            independent_audit_AC_cases=population*expected)
        write(dst/'handoff_audit.json',hand)
        state.save_best()
        write(dst/'summary.json',dict(completed=True,method=method,first_feasible=state.first_feasible,
            initial_best=state.initial_best,best=state.best,evaluations=state.evaluations,
            counters=dict(state.counters),timing=hand,selection=state.selection_description))
        state.status('completed','first_feasible_independently_audited')
        return hand
    except Exception:
        import traceback
        write(dst/'error.json',dict(error=traceback.format_exc()))
        state.status('failed','construction_or_handoff_audit');raise
