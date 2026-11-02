"""Decoded, visit-relative evidence. Never participates in selection."""
from collections import Counter
import hashlib
import numpy as np
from .evaluator import rank
from .schema import fingerprint


def decoded_change(a, b, atol=1e-9):
    """Compare actual controls, not latent keys or inherited best-plan changes.

    Exact identity and numerical movement are separate: atol is diagnostic only.
    Case indices are zero-based. Investments do not count as dispatch changes.
    """
    families = []; cases = set(); digest = hashlib.sha256()
    resources = []; stages = set()
    for name, old in vars(a.operation).items():
        delta = np.asarray(getattr(b.operation, name)) - np.asarray(old)
        mask = abs(delta) > atol
        if not mask.any(): continue
        families.append(name)
        digest.update(name.encode()); digest.update(np.where(mask, delta, 0.).astype('<f8').tobytes())
        if delta.ndim == 3:
            case_mask = mask.any(axis=2 if name in ('existing_p','existing_q','new_p') else 1)
            cases.update(map(tuple, np.argwhere(case_mask).tolist()))
    delta = np.asarray(b.new_capacity) - np.asarray(a.new_capacity)
    mask = abs(delta) > atol
    if mask.any():
        families.append('new_capacity'); resources = np.flatnonzero(mask.any(axis=0)).tolist()
        stages.update(np.flatnonzero(mask.any(axis=1)).tolist())
        digest.update(np.where(mask, delta, 0.).astype('<f8').tobytes())
    for name in ('trees','line_buys','storage_buys'):
        old = getattr(a.plan.canonical(), name); new = getattr(b.plan.canonical(), name)
        if old != new:
            families.append(name); digest.update(fingerprint([old,new]).encode())
    for name in sorted(set(a.physical or {}) | set(b.physical or {})):
        old = np.asarray((a.physical or {}).get(name, [])); new = np.asarray((b.physical or {}).get(name, []))
        if old.shape != new.shape: raise ValueError('Physical shape changed: '+name)
        if old.dtype.kind not in 'biuf' or new.dtype.kind not in 'biuf':
            if not np.array_equal(old,new):raise ValueError('Physical metadata changed: '+name)
            continue
        mask = abs(new-old) > atol
        if not mask.any(): continue
        families.append(name); digest.update(name.encode()); digest.update(np.where(mask,new-old,0.).astype('<f8').tobytes())
        if mask.ndim == 3:
            # Physical controls use K,S,N except installation choices (K,N).
            cases.update(map(tuple,np.argwhere(mask.any(axis=2)).tolist()))
    return dict(changed=bool(families),families=families,dispatch_cases=sorted(cases),
                capacity_resources=resources,capacity_stages=sorted(stages),
                delta_identity=digest.hexdigest(),diagnostic_atol=atol)


class EffectiveCoverage:
    def __init__(self,study=None):
        self.study=study
        self.groups={}; self.included=set(); self.individual=set(); self.changed=set()
        self.resources=set(); self.stages=set()

    def register_block(self, block):
        cases={tuple(c) for group in block.get('case_groups',[]) for c in group}
        self.included.update(cases)
        if len(cases)==1:self.individual.update(cases)

    def observe(self, block, origin, identity, change, record, previous):
        family=(block.get('asset') or {}).get('scope',block['scope'])
        key=family+'|'+origin
        g=self.groups.setdefault(key,dict(counts=Counter(),identities=set(),deltas=set(),gain=0.))
        c=g['counts']; c['proposals']+=1; g['identities'].add(identity)
        c['physically_changed']+=int(change['changed'])
        feasible=record['status']=='feasible'; c['feasible']+=int(feasible)
        accepted=previous is not None and rank(record)<rank(previous)
        c['accepted']+=int(accepted)
        if change['changed']:
            g['deltas'].add(change['delta_identity']); c['changed_feasible']+=int(feasible)
        if accepted and feasible and previous['status']=='feasible':
            g['gain']+=previous['metrics']['selection_objective']-record['metrics']['selection_objective']
        self.changed.update(map(tuple,change['dispatch_cases']))
        self.resources.update(change['capacity_resources']);self.stages.update(change['capacity_stages'])

    def report(self):
        rows={k:dict(v['counts'],unique_candidates=len(v['identities']),
                    unique_visit_relative_changes=len(v['deltas']),gain_FO=v['gain']) for k,v in self.groups.items()}
        locations=[] if self.study is None else [dict(index=j,bus=self.study.new[j]['bus'],technology=self.study.new[j]['technology']) for j in sorted(self.resources)]
        return dict(capacity_locations=locations,included_cases=sorted(self.included),individually_targeted_cases=sorted(self.individual),
                    physically_changed_dispatch_cases=sorted(self.changed),capacity_resources=sorted(self.resources),
                    capacity_stages=sorted(self.stages),by_family_and_origin=rows,
                    note='Changes relative to frozen visit reference; inherited changes excluded. Repeated proposals count as calls, not unique trials. Dispatch coverage is not investment coverage. Immediate origin is not causal isolation.')
