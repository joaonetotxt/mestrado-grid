"""Identical seed-based resource phase for all eight methods in a paired cell."""
from collections import Counter
import numpy as np
from early_policy import EarlySelectiveScheduler,EarlyCoverage,VERSION
from meta_planning.persistent_search import make_scheduler as original_factory

class PairedScheduler(EarlySelectiveScheduler):
 def bind_pair(self,method,beta,regime,seed):
  if regime not in ('normal','extremo'):raise ValueError(regime)
  self.allocation_version='paired_full_resource_v1';self.stream_method=method;self.stream_beta=float(beta)
  self.resource_phase=int(np.random.default_rng(np.random.SeedSequence([seed,round(beta*10),6161])).integers(len(self.initial_order)))
  self.interleaved_order=self.initial_order[self.resource_phase:]+self.initial_order[:self.resource_phase]
  self.initial_pending=list(self.interleaved_order)
  self.allocation_step=0;self.post_sweep_step=0;self.repeat_cursor=0;self.allocation_trace=[]
  self.selective_version=VERSION;self.scenario_regime={'normal':'normais','extremo':'extremos'}[regime]
  self.selective_tree_slots=0;self.selective_counts=Counter()
  self.early_version=VERSION;self.early_step=0;self.early_tree_slots=0
  self.selective_coverage=EarlyCoverage(self.codec.study.K,self.codec.study.S,seed)
  self.paired_seed=seed
  return self
 def report(self):
  out=super().report();out['allocation_policy'].update(unequal_first_families_declared=False,paired_seed=self.paired_seed,phase_equal_across_methods=True)
  return out

def install(method,beta,regime):
 from meta_planning import persistent_search as p
 def factory(codec,seed,mass,options):
  if options.protected_resource_coverage:
   return PairedScheduler(codec,seed,mass,options).bind_pair(method,beta,regime,seed)
  return original_factory(codec,seed,mass,options)
 p.make_scheduler=factory
