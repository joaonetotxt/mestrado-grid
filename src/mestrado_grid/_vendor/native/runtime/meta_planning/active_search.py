"""Declared allocation variant; only native engines choose candidate coordinates.

One third of visits retain the previous unrestricted scheduler. The remaining
visits address incumbent reliability constraints or joint multi-state operation.
No neighbor is scored, filtered, or optimized before the official evaluator.
"""
import numpy as np
from .consolidated_search import ConsolidatedScheduler, CoordinatedBlock
from .guided_blocks import TreeExchangeBlock
from .topology import reliability


class ActiveScheduler(ConsolidatedScheduler):
    def __init__(self,*args):
        super().__init__(*args)
        self.active_visits=0;self.focus_visits=0;self.stage_cache={}

    def choose(self,reference,contexts):
        if self.coverage_preflight:return super().choose(reference,contexts)
        slot=self.active_visits%3;self.active_visits+=1
        if slot==2:return super().choose(reference,contexts)
        st=self.codec.study;record=self.record or {}
        violations=record.get('normalized_violations',{})
        constrained=any(violations.get(n,0)>0 for n in ('SAIFI','SAIDI'))
        j=self.focus_visits;self.focus_visits+=1
        radius=self.options.radii[(j//2)%len(self.options.radii)]
        if constrained:
            c=self.codec.decode(reference)[0]
            if c.plan_id not in self.stage_cache:
                rows=reliability(st.data,c.plan)
                limits=st.data['config']
                def violation(row):
                    k=-1 if st.data['investment_mode']=='single' else row['stage']
                    return sum(max(0.,row[n]-limits[n+'_limits'][k])/max(1e-12,limits[n+'_limits'][k]) for n in ('SAIFI','SAIDI'))
                self.stage_cache={c.plan_id:max(rows,key=violation)['stage']}
            k=self.stage_cache[c.plan_id]
            asset=TreeExchangeBlock(self.codec,reference,[],dict(scope='active_reliability_tree',stage=k),self.base.mass,radius)
            block=asset if slot==0 else CoordinatedBlock(self.codec,self.base,reference,
                [[(k,s) for s in range(st.S)]],self.base.mass,radius,asset)
            mode='incumbent_worst_reliability_stage'
        else:
            # Revisit the latest joint operation context, preserving native
            # memories and refreshing every fitness under its new surroundings.
            if slot==1:
                for context in reversed(contexts):
                    if context.block.metadata.get('allocation')=='multi_state_native_operation':
                        return context,None
            groups=[[(k,s) for s in range(st.S)] for k in range(st.K)]
            block=CoordinatedBlock(self.codec,self.base,reference,groups,self.base.mass,radius)
            mode='multi_state_native_operation'
        block.metadata.update(scheduler='active_constraints_v1',allocation=mode,
            protected_broad_fraction=1/3,auxiliary_optimizer=False,
            pre_scored_neighbors=0)
        if isinstance(block,CoordinatedBlock):
            from .slack_exchange import SlackExchangeBlock
            block=SlackExchangeBlock(block)
        return None,block

    def report(self):
        out=super().report()
        out.update(active_visits=self.active_visits,focus_visits=self.focus_visits,
            adaptation='declared search bias; no fixed bus, edge or technology',
            eligibility='all original families retained by protected broad scheduler')
        return out
