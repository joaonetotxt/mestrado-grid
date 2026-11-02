"""Persistent block adaptation of L-SHADE: LPSR uses the whole trial NFE clock.

All local contexts start from the same eight-member initialization rule. After
native selection, population size falls from eight to four against the shared
global budget. F/CR memories and position-only archive survive revisits; all
fitness is refreshed when outside coordinates change. This is a declared
block/constrained adaptation, not a reproduction of CEC's 18D population.
"""
from types import SimpleNamespace
import numpy as np
from new_engines import LSHADE

class PersistentLSHADE(LSHADE):
 def __init__(self,initial,seed):
  super().__init__(initial,seed,2**63-1)
  self.config=SimpleNamespace(method='L-SHADE',parameters=dict(memory=6,p=.11,archive_rate=2.6,minimum_population=4,lpsr_clock='global_charged_trial_calls'))
  self.budget_progress=0.;self.global_progress=0.;self.velocity=None
 def set_budget_progress(self,progress):self.budget_progress=float(progress)
 def _propose(self):return self.propose()
 def accept(self):
  # Parent implements exact success history, archive and LPSR. Supply its linear
  # clock without changing the true local evaluation counter or random stream.
  actual=self.evaluations;budget=self.budget
  self.evaluations=self.global_progress;self.budget=1.
  try:super().accept()
  finally:self.evaluations=actual;self.budget=budget
