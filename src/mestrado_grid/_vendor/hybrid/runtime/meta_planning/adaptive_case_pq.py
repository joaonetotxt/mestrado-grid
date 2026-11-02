"""Opt-in scheduling only: no candidate oracle, repair or new objective.

All P/Q values still come from Engine. Window decisions use charged native
observations. Recentring creates a DECLARED fresh native context; old fitness,
velocities and memories are never transferred to a changed coordinate system.
Two out of three scheduling slots retain the previous unrestricted scheduler.
"""
from collections import Counter
import numpy as np
from .active_search import ActiveScheduler
from .case_pq_search import CasePQScheduler, case_block, free_resources
from .evaluator import rank


class AdaptiveCasePQScheduler(CasePQScheduler):
    MIN_RADIUS=.005
    INITIAL_RADIUS=.02
    FULL_PERIOD=12

    def __init__(self,*args):
        super().__init__(*args)
        self.focus_count=0
        self.sweep_seen=set()
        self.sweep=0
        self.native_stats={}
        self.finished={}
        self.window_events=[]
        self.coverage_counts=Counter()
        self.pq_generation_budget=None

    @staticmethod
    def _pq(context):
        return context.block.metadata.get('scope')=='case_native_PQ'

    def choose(self,reference,contexts):
        # An old scheduler slot keeps its own visit budget, even when that
        # scheduler elects to revisit an existing P/Q context.
        self.pq_generation_budget=None
        if self.coverage_preflight or (self.record or {}).get('status')!='feasible':
            return ActiveScheduler.choose(self,reference,contexts)
        slot=self.pq_visit%3
        self.pq_visit+=1
        if slot:return ActiveScheduler.choose(self,reference,contexts)
        return self.choose_pq(reference,contexts)

    def choose_pq(self,reference,contexts):
        """Focused slot, also used by the explicitly P/Q-only isolated pilot."""
        j=self.focus_count
        self.focus_count+=1
        fast=self.options.pq_fast_coverage
        self.pq_generation_budget=2 if fast else None
        revisit=(j%3==2) if fast else (j%2==1)
        prior=next((c for c in reversed(contexts) if self._pq(c)),None)
        force_full=self.options.pq_adaptive_windows and (j+1)%self.FULL_PERIOD==0
        if revisit and prior is not None:
            stats=self.finished.get(prior.number,{})
            if fast:self.pq_generation_budget=8 if stats.get('gains',0)>0 else 2
            if not self.options.pq_adaptive_windows:
                prior.block.metadata.update(pq_role='native_memory_revisit',
                    productive_previous_visit=stats.get('gains',0)>0)
                return prior,None
            case=tuple(prior.block.metadata['case_groups'][0][0])
            selected=[tuple(x) for x in prior.block.metadata['selected_resources']]
            stats=self.finished.get(prior.number,{})
            radius,reason=self.next_radius(prior.block.radius,stats,force_full)
            # Always reanchor a changed representation to the fully evaluated
            # incumbent. No clipped old personal bests or stale fitness survive.
            block=case_block(self.codec,self.base,reference,case,selected,radius,self.base.mass)
            if not block.dimension:return self._new(reference,force_full)
            block.metadata.update(self._metadata('adaptive_revisit'),
                reframed_from_context=prior.number,window_reason=reason,
                native_memories='reset; initialization and reevaluation charged',
                productive_previous_visit=stats.get('gains',0)>0)
            self.window_events.append(dict(context=prior.number,radius_before=prior.block.radius,
                radius_after=radius,reason=reason,native_observations=dict(stats)))
            return None,block
        return self._new(reference,force_full)

    @classmethod
    def next_radius(cls,radius,stats,force_full=False):
        if force_full:return 1.,'periodic_full_domain'
        calls=stats.get('calls',0)
        if not calls:return radius,'no_native_evidence'
        # A temporary global probe does not permanently replace local search.
        if radius==1.:return .1,'return_from_full_domain'
        if stats.get('feasible',0)/calls<.5:
            return max(cls.MIN_RADIUS,radius*.5),'majority_infeasible'
        if stats.get('boundary_gains',0) or stats.get('gains',0):
            return min(1.,radius*2),'native_improvement_expand'
        return min(1.,radius*2),'feasible_stagnation_expand'

    def _metadata(self,role):
        return dict(scheduler='adaptive_case_pq_v2',pq_role=role,
            protected_original_scheduler_fraction=2/3,
            coverage_policy='two fresh cases, one revisit' if self.options.pq_fast_coverage else 'baseline new/revisit',
            adaptive_windows=self.options.pq_adaptive_windows,
            allocation_bias='declared case coverage and charged native feedback; no prescribed resource values')

    def _new(self,reference,force_full):
        st=self.codec.study
        all_cases={divmod(i,st.S) for i in range(st.K*st.S)}
        fast=self.options.pq_fast_coverage
        if fast and self.sweep_seen==all_cases:
            self.sweep_seen.clear();self.sweep+=1
        available=all_cases-self.sweep_seen if fast else all_cases
        adaptive=self.pq_new%2==1 and self.feedback and self.options.economic_allocation
        if adaptive:
            case=max(sorted(available),key=lambda ks:self.feedback.get(ks,{}).get('economic_opportunity',0)/(1+self.pq_case_visits.get(ks,0)))
        else:
            # Skips only cases already served in this sweep. Every stable free
            # case receives one fresh visit before another sweep begins.
            while True:
                case=divmod(self.pq_decks.draw('pq_states',st.K*st.S),st.S)
                if case in available:break
        self.sweep_seen.add(case)
        self.pq_case_visits[case]=self.pq_case_visits.get(case,0)+1
        rows=free_resources(self.codec,self.base,reference,case)
        if not rows:
            self.coverage_counts['cases_without_free_PQ']+=1
            return ActiveScheduler.choose(self,reference,[])
        domain=tuple((r['kind'],r['index']) for r in rows)
        key=('pq_sources',case,domain)
        selected=[]
        while len(selected)<min(8,len(domain)):
            item=domain[self.pq_decks.draw(key,len(domain))]
            if item not in selected:selected.append(item)
        radius=(.02,.05,.1,.25,1.)[(self.pq_case_visits[case]-1)%5]
        if force_full:radius=1.
        block=case_block(self.codec,self.base,reference,case,selected,radius,self.base.mass)
        block.metadata.update(self._metadata('fresh_case'),coverage_sweep=self.sweep,
            allocation='economic_unvisited_case' if adaptive else 'protected_seed_deck',
            window_reason='periodic_full_domain' if force_full else 'case_visit_radius',
            productive_previous_visit=False)
        self.pq_new+=1
        return None,block

    def observe(self,context,record,previous,origin):
        super().observe(context,record,previous,origin)
        if not self._pq(context) or origin!='native_offspring':return
        stats=self.native_stats.setdefault(context.number,Counter())
        stats['calls']+=1
        stats['feasible']+=record.get('status')=='feasible'
        improved=previous is not None and rank(record)<rank(previous)
        stats['gains']+=improved
        context.pq_last_native_improved=improved

    def observe_candidate(self,context,full,record,origin):
        super().observe_candidate(context,full,record,origin)
        if self._pq(context) and origin=='native_offspring' and getattr(context,'pq_last_native_improved',False):
            ids=context.controlled;a=context.block.reference[ids];delta=full[ids]-a
            reach=np.where(delta<0,np.minimum(a,context.block.radius),np.minimum(1-a,context.block.radius))
            fraction=np.divide(abs(delta),reach,out=np.zeros_like(delta),where=reach>0)
            self.native_stats[context.number]['boundary_gains']+=bool(np.any(fraction>=.8))

    def visit_generations(self,context,default):
        if self.pq_generation_budget is None or not self._pq(context):return default
        return self.pq_generation_budget

    def finish_visit(self,context):
        if self._pq(context):
            self.finished[context.number]=dict(self.native_stats.pop(context.number,{}))

    def report(self):
        out=super().report()
        out.update(focused_slots=self.focus_count,coverage_sweep=self.sweep,
            fresh_cases_current_sweep=len(self.sweep_seen),window_events=self.window_events,
            adaptive_window_policy='native feedback only; reframe resets memories and charges initialization',
            full_window_every_focused_slots=self.FULL_PERIOD,
            coverage_diagnostics=dict(self.coverage_counts))
        return out
