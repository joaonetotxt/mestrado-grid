"""Isolated scheduling experiment. Never installed in a production campaign.

Only visit order and case/mode allocation change. Native blocks, proposal
operators, comparator, feasibility, CVaR and physical/financial evaluation are
imported unchanged. Counts are opportunity guarantees, not wall time guarantees.
"""
import copy
from collections import Counter
import numpy as np
from selective_scheduler import SelectiveScheduler
from selective_components import Coverage,SpatialBlock,PriorityTreeBlock
from effective_scheduler import EffectiveScheduler,Adapter

VERSION='early_selective_order_isolated_v1'

class EarlyCoverage(Coverage):
    # Same multiset as the old five-sweep cycle: two broad P, local P, PQ, Q.
    modes=('PQ','wide','Q','local','wide')
    def __init__(self,K,S,seed):
        super().__init__(K,S,seed)
        deck=np.random.default_rng(np.random.SeedSequence([seed,19617])).permutation(len(self.domain))
        self.offsets={self.domain[int(j)]:i%5 for i,j in enumerate(deck)}
        self.mode_slot=0

    @classmethod
    def from_previous(cls,old):
        new=cls(max(k for k,s in old.domain)+1,max(s for k,s in old.domain)+1,96101)
        for key,value in vars(old).items():setattr(new,key,copy.deepcopy(value))
        # Keep outstanding cases, observed counts and RNG; do not replay a sweep.
        return new

    def next(self,feedback):
        if not self.sweep:self.sweep=[self.domain[i] for i in self.rng.permutation(len(self.domain))]
        target=self.mode_slot%5;self.mode_slot+=1
        # Buckets differ by at most one case in a fresh complete sweep. Empty
        # buckets near the end defer the requested mode; no case is discarded.
        for shift in range(5):
            slot=(target+shift)%5
            eligible=[c for c in self.sweep if (self.offsets[c]+self.counts[c])%5==slot]
            if eligible:break
        case=max(eligible,key=lambda c:feedback.get(c,{}).get('economic_opportunity',0)) if self.steps%2==0 else eligible[0]
        self.sweep.remove(case);mode=self.modes[slot]
        self.counts[case]+=1;self.mode_counts[case,mode]+=1;self.steps+=1
        return case,mode

class EarlySelectiveScheduler(SelectiveScheduler):
    def bind_stream(self,method,beta,regime='normais'):
        super().bind_stream(method,beta,regime)
        self.enable_early_order()
        return self

    def enable_early_order(self):
        if hasattr(self,'early_version'):raise ValueError('Already configured; no reset')
        self.early_version=VERSION;self.early_step=0;self.early_tree_slots=0
        self.selective_coverage=EarlyCoverage.from_previous(self.selective_coverage)
        return self

    def choose(self,reference,contexts):
        # Existing stream phase, unfinished resource deck, site decks and contexts
        # stay intact. Rotate only the next warmup cycle to F,R,R.
        step=self.allocation_step;self.allocation_step+=1
        if self.initial_pending:
            resource=self.early_step%3!=0;self.early_step+=1
        else:
            resource=self.post_sweep_step%2==0;self.post_sweep_step+=1
        if resource:
            if self.initial_pending:task=self.initial_pending.pop(0)
            else:
                task=self.interleaved_order[self.repeat_cursor%len(self.interleaved_order)];self.repeat_cursor+=1
            block=self.resource_block(reference,task);self.resource_cursor+=1
            block.metadata.update(allocation_version=VERSION,stream_phase=self.resource_phase,allocation_step=step)
            self.allocation_trace.append(dict(step=step,kind='resource',task=task))
            key=tuple(block.metadata['resource_context_key']);number=self.resource_contexts.get(key)
            if number is not None:
                context=contexts[number]
                if tuple(context.block.metadata['resource_context_key'])!=key:raise ValueError('Resource identity mismatch')
                return context,None
            return None,block
        self.allocation_trace.append(dict(step=step,kind='effective_or_original'))
        context,block=EffectiveScheduler.choose(self,reference,contexts)
        if block is None:return context,block
        scope=block.metadata.get('scope')
        if scope in ('effective_tree','effective_chronological_tree'):
            self.selective_tree_slots+=1;self.early_tree_slots+=1
            if self.early_tree_slots%3==1:
                block=PriorityTreeBlock(self.codec,self.base,reference,getattr(self,'paired_seed',96101)+self.selective_tree_slots)
                block.metadata.update(selective_version=VERSION,original_slot=scope,reliability_only=True)
                self.selective_counts['priority_tree']+=1
            else:self.selective_counts['retained_tree']+=1
            return None,block
        if scope!='effective_channels':return context,block
        case,mode=self.selective_coverage.next(self.feedback);row=self.feedback.get(case,{})
        service=row.get('hourly_cut',0)>row.get('hourly_curtailment',0)
        regional=self.scenario_regime=='normais' and mode in ('local','PQ')
        if regional:
            block=SpatialBlock(self.codec,self.base,reference,case,'P' if mode=='local' else 'PQ',1.,self.selective_coverage.steps-1,4,service)
            self.selective_counts['regional']+=1
        else:
            broadmode='Q' if mode=='Q' else ('P_service' if service else 'P')
            block=Adapter(self.codec,self.base,reference,'channels',case=case,mode=broadmode,radius=.1 if broadmode in ('Q','P_service') else 1.)
            self.selective_counts['broad_'+broadmode]+=1
        block.metadata.update(selective_version=VERSION,fair_case=case,coverage_requested_mode=mode,regional_enabled=regional,scenario_regime=self.scenario_regime)
        return None,block

    def report(self):
        out=super().report()
        out['allocation_policy'].update(version=VERSION,warmup_sequence=['effective_or_original','resource','resource'],new_phase_origin='explicit complete-boundary migration or fresh start',early_steps=self.early_step)
        out['selective'].update(early_version=VERSION,mode_schedule='Five mode buckets interleaved from first sweep; cases once per 96 focused visits; full per-case mode cycle in five sweeps',reliability_first='Priority, retained single-stage, retained chronological, repeating')
        return out

def migrate_isolated(scheduler,complete_visit_boundary):
    if complete_visit_boundary is not True:raise ValueError('Complete visit boundary required')
    if type(scheduler) is not SelectiveScheduler:raise ValueError('Unsupported source scheduler')
    scheduler.__class__=EarlySelectiveScheduler
    return scheduler.enable_early_order()
