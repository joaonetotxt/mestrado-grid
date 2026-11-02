"""Seeded candidate construction using catalogue rules only; no saved vectors."""
import numpy as np
from .temporal import demand_bounds


def initialize(codec,seed=0,mode='constructive'):
    if mode not in ('constructive','random','physical','resource','mat_original'):
        raise ValueError('Unknown initialization mode')
    rng=np.random.default_rng(seed)
    if mode in ('physical','resource','mat_original'):
        if mode=='physical':
            from .construction import physical_start as build
        elif mode=='mat_original':
            from .construction import original_mat_start as build
        else:
            from .resource_construction import resource_start as build
        candidate,details=build(codec,rng)
        return candidate,codec.encode(candidate),dict(seed=seed,mode=mode,origin='catalogue_rules_and_seed_only',repairs=details)
    blocks=codec.unpack(rng.random(codec.dimension))
    if mode=='constructive':
        st=codec.study;pc=codec.plan_codec
        # Original physical network is a rule-based starting catalogue, not an
        # optimized historical plan. Missing connectivity purchases remain logged.
        blocks['plan'][:len(pc.lines)]=0.
        blocks['new_capacity']*=.1
        blocks['existing_p']*=.5
        lo,hi,qlo,qhi=st.existing_bounds()
        blocks['existing_q']=np.clip(np.divide(-qlo,qhi-qlo,out=np.zeros(qlo.shape),where=qhi>qlo),0,1)
        blocks['soc']=.45+.1*blocks['soc']
        clo,chi,elo,ehi=demand_bounds(st)
        for name,base,low,high in [('conventional',st.base_load,clo,chi),('ev',st.base_ev,elo,ehi)]:
            blocks[name]=np.clip(np.divide(base-low,high-low,out=np.zeros(base.shape),where=high>low)+.05*(blocks[name]-.5),0,1)
        blocks['cut']*=.05
    candidate,repairs=codec.decode(codec.pack(blocks))
    # Algorithms must receive the position corresponding to the evaluated
    # physical candidate, not a pre-repair position paired with another fitness.
    return candidate,codec.encode(candidate),dict(seed=seed,mode=mode,origin='catalogue_rules_and_seed_only',repairs=repairs)
