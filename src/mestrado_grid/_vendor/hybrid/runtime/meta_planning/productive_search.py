"""Budgeted coarse-to-fine native search, with explicit economic allocation.

No setpoint optimizer, line preference, witness solution, price modification or
extra AC probe. Native engines choose values; all initialization is accounted.
The all-case/family contract remains a finite coverage contract, not optimality.
"""
import numpy as np
from .guided_blocks import GuidedScheduler,WindowBlock,TreeExchangeBlock
from .measured_search import PhysicalScheduler
from .native_blocks import Block,Field


class ProductiveScheduler(PhysicalScheduler):
    """Protect broad operation/investment trials from unlimited tree polishing.

    Every 8 visits: one coarse cycle, two economic service blocks, one protected
    seed-deck service block, two outcome-ranked revisits and two complementary
    blocks. Complementary domains cycle without a selected site or technology.
    A topology block already includes native dispatch; it cannot recursively
    claim a visit for every distinct tree it generated.
    """
    SCOPES=('investment_joint','topology','temporal_cycle','full_native',
            'state_direct','state_family','coarse_joint')

    def __init__(self,codec,seed,mass,options):
        super().__init__(codec,seed,mass,options)
        self.coarse_visits=0;self.productivity={}

    def observe_candidate(self,context,full,record,origin):
        # Candidate physics still evaluated normally. No implicit work queue.
        return

    def observe(self,context,record,previous,origin):
        super().observe(context,record,previous,origin)
        d=self.productivity.setdefault(context.number,dict(calls=0,gain=0.,native_gain=0.))
        d['calls']+=1
        if previous and previous['status']==record['status']=='feasible':
            g=max(0.,previous['metrics']['selection_objective']-record['metrics']['selection_objective'])
            d['gain']+=g
            if origin=='native_offspring':d['native_gain']+=g

    def choose(self,reference,contexts):
        slot=self.step%8;self.step+=1
        if slot==0:
            fields=[];cycle=self.base.coupled_operation(fields)
            radius=self.options.radii[self.coarse_visits//len(self.codec.study.cycles)%len(self.options.radii)]
            self.coarse_visits+=1
            groups=[[(k,s) for s in self.codec.study.cycles[cycle]] for k in range(self.codec.study.K)]
            return None,WindowBlock(reference,fields,dict(scope='coarse_cycle_joint',scheduler='productive',
                coupled_cycle=cycle,case_groups=groups,allocation='protected_multistage_operation'),self.base.mass,radius)
        if slot in (2,6) and contexts:
            # Use actual global gain per TOTAL charged call, including context
            # refresh and diversification. Never silently credit this to native.
            if self.coverage_preflight or self.revisit%4==3:
                chosen=contexts[self.revisit%len(contexts)]
            else:
                rates=[self.productivity.get(c.number,{}).get('gain',0.)/max(1,self.productivity.get(c.number,{}).get('calls',0)) for c in contexts]
                scale=max(max(rates),1e-15)
                def score(c):
                    d=self.productivity.get(c.number,{})
                    return d.get('gain',0.)/max(1,d.get('calls',0))/scale+.1*np.sqrt(np.log(self.step+1)/(1+d.get('calls',0)))
                chosen=max(contexts,key=score)
            self.revisit+=1
            return chosen,None
        if slot in (3,7):
            scope=self.SCOPES[self.complement%len(self.SCOPES)];self.complement+=1
            if scope=='topology':
                k=self.base.draw('exchange_stage',self.codec.study.K);fields=[];cycle=self.base.coupled_operation(fields,[k])
                return None,TreeExchangeBlock(self.codec,reference,fields,dict(scope='topology',stage=k,
                    coupled_cycle=cycle,scheduler='productive',allocation='bounded_native_tree_and_dispatch'),self.base.mass,self.options.radii[0])
            if scope=='full_native':
                return None,Block(reference,[Field('all_eligible_coordinates',np.arange(self.codec.dimension),'direct',False)],dict(scope=scope),self.base.mass)
            self.base.count=3;self.base.complement_visits=self.base.COMPLEMENTS.index(scope)
            block=self.base.next(reference)
            return None,WindowBlock(reference,block.fields,dict(block.metadata,scheduler='productive'),self.base.mass,self.options.radii[0])
        if slot in (1,5) and self.feedback and not self.coverage_preflight:
            signal='economic_cut' if slot==1 else 'economic_curtailment'
            if not any(row.get(signal,0.)>0 for row in self.feedback.values()):signal='economic_opportunity'
            case=max(self.feedback,key=lambda c:self.feedback[c].get(signal,0.)/(1+self.case_visits[c]))
            # Retain cheap isolated restoration as well as measured joint
            # generation/service. A joint high-dimensional move is not required
            # when no dispatch alteration is needed. Neither face fixes signs.
            if slot==1:return None,self.service(reference,case,'economic_joint_service')
            return None,GuidedScheduler.service(self,reference,case,'diagnostic_curtailment')
        if self.coverage_preflight and slot in (1,5):
            return None,GuidedScheduler.service(self,reference,(0,0),'reserved_adaptive_slot')
        index=self.base.case_order[self.uniform%len(self.base.case_order)];self.uniform+=1
        case=divmod(index,self.codec.study.S)
        return None,GuidedScheduler.service(self,reference,case,'uniform_coverage')

    def report(self):
        out=super().report()
        out.update(productivity=self.productivity,coarse_visits=self.coarse_visits,
            topology_budget='fixed scheduled visits with coupled native dispatch; no recursive polish queue')
        return out
