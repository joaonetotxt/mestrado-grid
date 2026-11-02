"""Exact reuse of unchanged AC cases; all portfolio constraints are reevaluated.

Keys include the full investment plan, current-case controls and the actual
charge/discharge calculated from the entire SOC cycle. No rounded keys.
"""
from collections import OrderedDict
import copy, hashlib, sys
import numpy as np
from .evaluator import Evaluator
from .search_arrays import array_candidate

class CaseCachedEvaluator(Evaluator):
    def __init__(self,study,policy=None,max_cases=512,max_bytes=None):
        super().__init__(study,policy);self.cache=OrderedDict();self.max_cases=max_cases
        self.case_hits=0;self.case_misses=0;self.current_plan_id=None
        if type(max_cases) is not int or max_cases<1:raise ValueError('Positive case cache size required')
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes<1):raise ValueError('Positive byte limit required')
        self.max_bytes=max_bytes;self.cache_bytes=0;self.entry_bytes={};self.evictions=0

    @staticmethod
    def resident_size(value):
        """Conservative owned-object estimate; no rounding of physical keys."""
        if isinstance(value,np.ndarray):return sys.getsizeof(value)+value.nbytes
        if isinstance(value,dict):return sys.getsizeof(value)+sum(CaseCachedEvaluator.resident_size(k)+CaseCachedEvaluator.resident_size(v) for k,v in value.items())
        if isinstance(value,(list,tuple)):return sys.getsizeof(value)+sum(CaseCachedEvaluator.resident_size(v) for v in value)
        return sys.getsizeof(value)

    def cache_report(self):
        return dict(hits=self.case_hits,misses=self.case_misses,evictions=self.evictions,
            entries=len(self.cache),estimated_bytes=self.cache_bytes,max_cases=self.max_cases,max_bytes=self.max_bytes,
            exact_keys=True,rounded_keys=False,global_constraints_reevaluated=True)

    def evaluate(self,candidate):
        self.current_plan_id=candidate.plan_id
        return super().evaluate(array_candidate(candidate))

    def _case(self,candidate,k,s,topology,charge,discharge,bounds):
        h=hashlib.sha256(f'{self.signature}/{self.current_plan_id}/{k}/{s}'.encode())
        def add(x):h.update(np.asarray(x,dtype='<f8').tobytes())
        op=candidate.operation
        for name in ('existing_p','existing_q','new_p'):add(getattr(op,name)[k,s])
        for name in ('conventional','ev','cut'):add(getattr(op,name)[k,:,s])
        add(charge[k,:,s]);add(discharge[k,:,s])
        if candidate.physical is not None:
            for name in ('new_q','bess_q','cap_state','tap'):add(candidate.physical[name][k][s])
        key=h.digest()
        if key in self.cache:
            self.case_hits+=1;self.cache.move_to_end(key);return copy.deepcopy(self.cache[key])
        self.case_misses+=1;row=super()._case(candidate,k,s,topology,charge,discharge,bounds)
        self.cache[key]=copy.deepcopy(row)
        size=self.resident_size(self.cache[key]);self.entry_bytes[key]=size;self.cache_bytes+=size
        while len(self.cache)>self.max_cases or (self.max_bytes is not None and self.cache_bytes>self.max_bytes):
            discarded,_=self.cache.popitem(last=False)
            self.cache_bytes-=self.entry_bytes.pop(discarded);self.evictions+=1
        return row
