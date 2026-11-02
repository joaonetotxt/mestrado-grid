"""Conservative exact cache across changes confined to other network stages.

No approximation of AC or financial terms. Only the case key is decomposed;
portfolio costs, chronology, temporal limits and reliability always recompute.
Capacity, BESS purchases and physical choices deliberately remain global keys.
"""
from meta_planning.case_cache import CaseCachedEvaluator
from meta_planning.topology import line_schedule
from meta_planning.schema import fingerprint

class StageCachedEvaluator(CaseCachedEvaluator):
    def evaluate(self,candidate):
        self.stage_ids=None
        try:
            installed,_=line_schedule(self.study.data,candidate.plan)
            self.stage_ids=[fingerprint(dict(
                tree=candidate.plan.trees[k],
                installed=sorted((e,option) for (e,stage),option in installed.items() if stage==k),
                storage_buys=candidate.plan.storage_buys,
                new_capacity=candidate.new_capacity,
                physical_installations=None if candidate.physical is None else {
                    n:v for n,v in candidate.physical.items() if n.endswith('_choice')}
            )) for k in range(self.study.K)]
        except (ValueError,KeyError,TypeError):
            pass  # Original evaluator owns structural failures.
        return super().evaluate(candidate)

    def _case(self,candidate,k,s,topology,charge,discharge,bounds):
        assert self.stage_ids is not None
        self.current_plan_id=self.stage_ids[k]
        return super()._case(candidate,k,s,topology,charge,discharge,bounds)
